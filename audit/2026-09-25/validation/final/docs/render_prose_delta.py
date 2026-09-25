"""Strictly render the final prose-only correction without rerunning tutorials."""

from __future__ import annotations

from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

from run_docs_validation import EVIDENCE, REPO, digest, hashes, inputs, write_json


def main() -> None:
    full = json.loads((EVIDENCE / "result.json").read_text())
    if full["status"] != "passed":
        raise SystemExit("The complete notebook execution must pass first.")
    old = json.loads((EVIDENCE / "source_hashes_before.json").read_text())
    current = hashes(REPO, inputs())
    differences = sorted(name for name in set(old) | set(current) if old.get(name) != current.get(name))
    if differences != ["docs/guide/learning.md"]:
        raise SystemExit(f"Unexpected changes after execution: {differences}")
    previous_guide = (Path(full["source_directory"]) / "docs/guide/learning.md").read_text()
    current_guide = (REPO / "docs/guide/learning.md").read_text()
    previous_sentence = "Custom retractions that cannot be differentiated in their step multiplier use the previous Armijo strategy without the derivative-based safeguard."
    current_sentence = "Custom retractions that encounter JAX tracing errors from Python or NumPy scalar conversions use the previous Armijo strategy without the derivative-based safeguard; unrelated errors still propagate."
    if " ".join(previous_guide.split()).replace(previous_sentence, current_sentence) != " ".join(current_guide.split()):
        raise SystemExit("The guide differs beyond the agreed compatibility clarification.")
    (EVIDENCE / "prose_delta.patch").write_text("".join(difflib.unified_diff(
        previous_guide.splitlines(keepends=True), current_guide.splitlines(keepends=True),
        fromfile="executed-snapshot/docs/guide/learning.md", tofile="final/docs/guide/learning.md",
    )))
    work = Path(tempfile.mkdtemp(prefix="geojax-docs-prose-final-20260925-"))
    source = work / "source"
    site = work / "site"
    for name in current:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
    metadata = []
    for name in ("grassmann_pca", "solver_comparison", "sphere_eigenvector", "torus_geodesics"):
        path = source / f"docs/tutorials/{name}.md"
        text = path.read_text()
        target = "  display_name: Python 3\n"
        if text.count(target) != 1 or "  language: python" in text.split("---", 2)[1]:
            raise SystemExit(f"Unexpected lexer metadata in {path}")
        path.write_text(text.replace(target, target + "  language: python\n", 1))
        metadata.append({"file": str(path.relative_to(source)), "change": "Temporary-copy kernelspec.language=python only; no code or prose changes", "sha256": digest(path)})
    command = [sys.executable, "-m", "sphinx", "-E", "-a", "-W", "--keep-going", "-b", "html", "-D", "nb_execution_mode=off", str(source / "docs"), str(site)]
    started = time.monotonic()
    with (EVIDENCE / "prose_delta_sphinx.log").open("w") as log:
        completed = subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT)
    html_command = [sys.executable, str(source / "docs/audit_html.py"), str(site)]
    html_code = None
    if completed.returncode == 0:
        for name in ("geojax-gj-mark.png", "geojax-gj-mark-dark.png", "geojax-gj-favicon.png"):
            shutil.copy2(source / "docs/_static/brand" / name, site / "_static" / name)
        with (EVIDENCE / "prose_delta_html_audit.log").open("w") as log:
            html_code = subprocess.run(html_command, cwd=source, stdout=log, stderr=subprocess.STDOUT).returncode
    after = hashes(REPO, list(current))
    log_text = (EVIDENCE / "prose_delta_sphinx.log").read_text()
    result = {
        "scope": "Final compatibility clarification only; no tutorial code/package changes or notebook execution. Complete final-package tutorial execution remains in result.json.",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "site": str(site),
        "sphinx_returncode": completed.returncode,
        "html_command": html_command,
        "html_returncode": html_code,
        "elapsed_seconds": time.monotonic() - started,
        "changes_after_complete_execution": differences,
        "exact_agreed_prose_delta_verified": True,
        "temporary_metadata_changes": metadata,
        "current_source_hashes": current,
        "original_inputs_unchanged_during_render": current == after,
        "tutorials_executed": False,
        "execution_log_count": log_text.count("Executed notebook in "),
        "sphinx_warning_lines": [line for line in log_text.splitlines() if re.search(r"\b(?:WARNING|ERROR):", line)],
        "learning_guide_exactly_matches_authored_source": digest(source / "docs/guide/learning.md") == current["docs/guide/learning.md"],
    }
    result["status"] = "passed" if completed.returncode == 0 and html_code == 0 and current == after and result["execution_log_count"] == 0 else "failed"
    write_json(EVIDENCE / "prose_delta_result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "sphinx_returncode", "html_returncode", "elapsed_seconds")}))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
