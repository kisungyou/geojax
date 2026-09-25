"""Reproduce the final strict documentation gate in an isolated source copy.

Preparation imports no numerical libraries and starts no kernels. Execution is
explicit, uses the authored 300-second cell limit, and retains kernel outputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time


REPO = Path(__file__).resolve().parents[5]
EVIDENCE = Path(__file__).resolve().parent
EXCLUDED_PARTS = {"__pycache__", ".ipynb_checkpoints", "_build", ".DS_Store"}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs() -> list[str]:
    paths = ["README.md", "Makefile", "pyproject.toml", "LICENSE"]
    for folder in ("geojax", "docs"):
        paths.extend(
            str(path.relative_to(REPO))
            for path in (REPO / folder).rglob("*")
            if path.is_file()
            and not EXCLUDED_PARTS.intersection(path.parts)
            and path.suffix not in {".pyc", ".pyo"}
        )
    return sorted(paths)


def hashes(base: Path, paths: list[str]) -> dict[str, str | None]:
    return {name: digest(base / name) if (base / name).is_file() else None for name in paths}


def prepare() -> None:
    if (EVIDENCE / "prepared.json").exists():
        raise SystemExit("Preparation already exists; retain its evidence before making another run.")
    work = Path(tempfile.mkdtemp(prefix="geojax-docs-final-20260925-"))
    source = work / "source"
    paths = inputs()
    before = hashes(REPO, paths)
    for name in paths:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
    if hashes(source, paths) != before or hashes(REPO, paths) != before:
        raise SystemExit("Source changed during snapshot preparation.")
    conf = (source / "docs/conf.py").read_text()
    if not all(value in conf for value in (
        'nb_execution_mode = "force"',
        "nb_execution_timeout = 300",
        "nb_execution_raise_on_error = True",
    )):
        raise SystemExit("Authored notebook execution policy changed; inspect before proceeding.")
    tutorials = {}
    for path in sorted((source / "docs/tutorials").glob("*.md")):
        count = len(re.findall(r"^```\{code-cell\}(?: (?:python|ipython3))?\s*$", path.read_text(), re.M))
        if count:
            tutorials[path.stem] = count
    if len(tutorials) != 25 or sum(tutorials.values()) != 97:
        raise SystemExit(f"Unexpected tutorial inventory: {tutorials}")
    historical = json.loads((REPO / "audit/2026-09-10/corrections/docs-source-manifest.json").read_text())
    write_json(EVIDENCE / "source_hashes_before.json", before)
    write_json(EVIDENCE / "historical_source_comparison.json", {
        "historical_input_count": len(historical),
        "current_input_count": len(before),
        "changed_historical_inputs": sorted(name for name, value in historical.items() if before.get(name) != value),
        "additional_inputs": sorted(set(before) - set(historical)),
        "missing_historical_inputs": sorted(set(historical) - set(before)),
    })
    prepared = {
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPO),
        "work_directory": str(work),
        "source_directory": str(source),
        "site_directory": str(work / "site"),
        "python": sys.executable,
        "python_version": sys.version,
        "versions": {name: version(name) for name in (
            "jax", "jaxlib", "numpy", "scipy", "sphinx", "myst-nb", "matplotlib",
            "scikit-learn", "networkx", "pydata-sphinx-theme", "ipykernel",
        )},
        "execution_mode": "force",
        "cell_timeout_seconds": 300,
        "tutorial_code_cells": tutorials,
        "command": [sys.executable, "-m", "sphinx", "-E", "-a", "-W", "--keep-going", "-b", "html", str(source / "docs"), str(work / "site")],
        "source_manifest_sha256": digest(EVIDENCE / "source_hashes_before.json"),
        "status": "prepared; no kernels started",
    }
    write_json(EVIDENCE / "prepared.json", prepared)
    print(json.dumps({"status": prepared["status"], "work_directory": str(work), "inputs": len(paths)}), flush=True)


def execute() -> None:
    prepared = json.loads((EVIDENCE / "prepared.json").read_text())
    before = json.loads((EVIDENCE / "source_hashes_before.json").read_text())
    paths = sorted(before)
    source = Path(prepared["source_directory"])
    site = Path(prepared["site_directory"])
    if hashes(REPO, paths) != before or hashes(source, paths) != before:
        raise SystemExit("Prepared inputs no longer match the current source or snapshot.")
    if (EVIDENCE / "strict_sphinx.log").exists():
        raise SystemExit("Strict build already started; preserve the existing attempt.")
    started = time.monotonic()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (EVIDENCE / "strict_sphinx.log").open("w") as log:
        completed = subprocess.run(prepared["command"], cwd=source, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    log_text = (EVIDENCE / "strict_sphinx.log").read_text()
    html = {"returncode": None}
    if completed.returncode == 0:
        for name in ("geojax-gj-mark.png", "geojax-gj-mark-dark.png", "geojax-gj-favicon.png"):
            shutil.copy2(source / "docs/_static/brand" / name, site / "_static" / name)
        html_command = [sys.executable, str(source / "docs/audit_html.py"), str(site)]
        with (EVIDENCE / "html_audit.log").open("w") as log:
            html_run = subprocess.run(html_command, cwd=source, stdout=log, stderr=subprocess.STDOUT)
        html = {"command": html_command, "returncode": html_run.returncode}
    outputs = []
    findings = []
    text_outputs = []
    # MyST-NB stores downloadable executed notebooks in the output's sibling folder.
    candidates = list(Path(prepared["work_directory"]).rglob("*.ipynb"))
    for name, expected_cells in prepared["tutorial_code_cells"].items():
        matches = [path for path in candidates if path.stem == name and ".jupyter_cache" not in path.parts]
        if len(matches) != 1:
            findings.append({"tutorial": name, "problem": "Expected exactly one executed notebook", "matches": [str(path) for path in matches]})
            continue
        path = matches[0]
        notebook = json.loads(path.read_text())
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        counts = [cell.get("execution_count") for cell in code]
        if len(code) != expected_cells or counts != list(range(1, expected_cells + 1)):
            findings.append({"tutorial": name, "problem": "Code cells or execution counts do not match", "execution_counts": counts})
        for index, cell in enumerate(code, 1):
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    findings.append({"tutorial": name, "cell": index, "problem": "Notebook error output", "output": output})
                parts = []
                if "text" in output:
                    parts.append(output["text"])
                for mime, value in output.get("data", {}).items():
                    if mime.startswith("text/") or mime == "application/json":
                        parts.append(value if isinstance(value, (list, str)) else json.dumps(value))
                for part in parts:
                    text = "".join(part) if isinstance(part, list) else part
                    text_outputs.append({"tutorial": name, "cell": index, "type": output["output_type"], "text": text})
                    if re.search(r"(?<![A-Za-z_])(?:[+-]?inf(?:inity)?|nan)(?![A-Za-z_])", text, re.I):
                        findings.append({"tutorial": name, "cell": index, "problem": "Potential nonfinite token requires review", "text": text})
        outputs.append({"tutorial": name, "notebook": str(path), "sha256": digest(path), "code_cells": len(code), "execution_counts": counts, "metadata": notebook.get("metadata", {})})
    write_json(EVIDENCE / "executed_notebooks.json", outputs)
    write_json(EVIDENCE / "text_outputs.json", text_outputs)
    write_json(EVIDENCE / "notebook_output_audit.json", {"scope": "All code-cell execution counts, notebook error outputs, and textual nonfinite-token screening; not a tensor-level finiteness proof.", "notebooks": len(outputs), "code_cells": sum(item["code_cells"] for item in outputs), "findings": findings})
    after = hashes(REPO, paths)
    write_json(EVIDENCE / "source_hashes_after.json", after)
    changed = sorted(name for name in paths if before[name] != after[name])
    staged_after = hashes(source, paths)
    result = {
        **prepared,
        "status": "passed" if completed.returncode == 0 and html["returncode"] == 0 and not findings and not changed else "review required",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "sphinx_returncode": completed.returncode,
        "sphinx_warnings_as_errors": True,
        "sphinx_warning_lines": [line for line in log_text.splitlines() if re.search(r"\b(?:WARNING|ERROR):", line)],
        "elapsed_seconds": elapsed,
        "cpu_seconds": (usage_after.ru_utime + usage_after.ru_stime) - (usage_before.ru_utime + usage_before.ru_stime),
        "completed_notebook_log_count": log_text.count("Executed notebook in "),
        "executed_notebooks": len(outputs),
        "executed_code_cells": sum(item["code_cells"] for item in outputs),
        "output_findings_count": len(findings),
        "html_audit": html,
        "original_source_unchanged": not changed,
        "changed_original_inputs": changed,
        "snapshot_files_changed_by_execution": sorted(name for name in paths if before[name] != staged_after[name]),
        "executed_notebook_outputs_retained": True,
        "harness_sha256": digest(Path(__file__)),
    }
    write_json(EVIDENCE / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "sphinx_returncode", "executed_notebooks", "executed_code_cells", "elapsed_seconds", "original_source_unchanged")}), flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else execute()
