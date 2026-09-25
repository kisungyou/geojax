"""Replay the two unchanged independent audit scripts against the final wheel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import faulthandler
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import tempfile
import time
import traceback


REPOSITORY = Path(__file__).resolve().parents[2]
OUTPUT = REPOSITORY / "audit/2026-09-25/validation/final/current_dependencies/independent"
SCRIPTS = (
    "audit/2026-09-10/statistical_crosscheck.py",
    "audit/2026-09-10/reproduce_limitations.py",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_supplement():
    spec = importlib.util.spec_from_file_location(
        "current_dependencies", REPOSITORY / "audit/2026-09-25/run_current_dependencies.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(script, evidence):
    faulthandler.enable()
    if os.name == "posix":
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    supplement = load_supplement()
    runner = supplement.load_runner()
    package, package_hashes = runner._installed_package(REPOSITORY)
    before = sha256(script)
    result = {"script": str(script), "script_sha256_before": before, "status": "running"}
    try:
        # Execute the original script bytes unchanged, retaining its own assertions.
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        traceback.print_exc()
    finally:
        result["script_sha256_after"] = sha256(script)
        try:
            runner._check_loaded_modules(package)
            after_package, after_hashes = runner._installed_package(REPOSITORY)
            if (before != result["script_sha256_after"] or after_package != package
                    or after_hashes != package_hashes):
                raise RuntimeError("script or installed package changed during execution")
            result["installed_module_locations_verified"] = True
            result["installed_package_hash_count"] = len(package_hashes)
        except Exception as error:
            result.update(status="failed", integrity_error=str(error))
        runner._write_json(evidence, result)
    return 0 if result["status"] == "passed" else 1


def reported_checks(script, log):
    output = log.read_text()
    # The runner's first line records its command; the unchanged scripts emit one JSON object.
    body = output.split("\n", 1)[1]
    decoder = json.JSONDecoder()
    parsed = None
    for index, character in enumerate(body):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(body[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and ("fanova" in candidate or "results" in candidate):
            parsed = candidate
            break
    if parsed is None:
        raise RuntimeError(f"{script} did not emit its expected JSON evidence")
    if script.endswith("statistical_crosscheck.py"):
        if len(parsed["permutation_checks"]) != 2:
            raise RuntimeError("expected two exact permutation calibrations")
        for calibration in parsed["permutation_checks"]:
            rates = calibration["exact_conditional_rejection_rates"]
            if calibration["partitions"] != 70 or set(rates) != {"0.01", "0.05", "0.1", "0.2"}:
                raise RuntimeError("incomplete exact permutation calibration")
            if any(rate > float(alpha) + 1e-14 for alpha, rate in rates.items()):
                raise RuntimeError("exact permutation p-values failed super-uniformity")
        return {"fanova_reference_comparisons": 2, "permutation_statistics": 2,
                "partitions_per_statistic": 70, "calibration_thresholds_per_statistic": 4,
                "output": parsed}
    if len(parsed["results"]) != 11 or not all(item["passed"] for item in parsed["results"]):
        raise RuntimeError("one or more original analytic audit checks did not pass")
    if parsed["x64"] is not True:
        raise RuntimeError("the independent analytic script did not enable float64")
    return {"original_analytic_checks_passed": 11, "output": parsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--evidence", type=Path)
    arguments = parser.parse_args()
    if arguments.worker:
        if arguments.evidence is None:
            parser.error("--worker requires --evidence")
        return worker(arguments.worker, arguments.evidence)
    supplement = load_supplement()
    runner = supplement.load_runner()
    before = supplement.inventory(runner)
    completed = json.loads((supplement.OUTPUT / "results.json").read_text())
    if completed["status"] != "passed" or completed["environment"] != before:
        raise RuntimeError("the current-dependency suite must pass on this unchanged environment first")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError("preserve existing independent replay evidence")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    script_hashes = {script: sha256(REPOSITORY / script) for script in SCRIPTS}
    harness_hash = sha256(Path(__file__))
    summary = {"status": "running", "environment_before": before,
               "script_hashes_before": script_hashes, "harness_sha256": harness_hash,
               "started_utc": datetime.now(timezone.utc).isoformat(), "files": []}
    runner._write_json(OUTPUT / "results.json", summary)
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS"):
        environment.pop(key, None)
    environment.update(PYTHONHASHSEED="0", PYTHONUNBUFFERED="1",
                       GEOJAX_TEST_X64="1", JAX_ENABLE_X64="1")
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="geojax-independent-replays-") as temporary:
            work = Path(temporary).resolve()
            if work == REPOSITORY or REPOSITORY in work.parents:
                raise RuntimeError("working directory must be outside the checkout")
            summary["working_directory"] = str(work)
            for script in SCRIPTS:
                name = Path(script).stem
                log = OUTPUT / f"{name}.log"
                evidence = OUTPUT / f"{name}.integrity.json"
                command = [sys.executable, "-I", str(Path(__file__).resolve()),
                           "--worker", str(REPOSITORY / script), "--evidence", str(evidence)]
                print(f"Independent float64 replay: {script}", flush=True)
                row = runner._run_process(command, work, environment, log, 600)
                row["script"] = script
                summary["files"].append(row)
                if row["timed_out"] or row["returncode"] != 0:
                    raise RuntimeError(f"{script} failed or exceeded its 600-second wall deadline")
                integrity = json.loads(evidence.read_text())
                if integrity["status"] != "passed":
                    raise RuntimeError(f"{script} failed its installed-module integrity check")
                row["integrity"] = integrity
                row["checks"] = reported_checks(script, log)
                runner._write_json(OUTPUT / f"{name}.json", row["checks"]["output"])
                runner._write_json(OUTPUT / "results.json", summary)
        summary["status"] = "passed"
    except Exception as error:
        summary.update(status="failed", error=str(error))
        traceback.print_exc()
    finally:
        try:
            after = supplement.inventory(runner)
            after_scripts = {script: sha256(REPOSITORY / script) for script in SCRIPTS}
            summary.update(environment_after=after, script_hashes_after=after_scripts)
            unchanged = (before == after and script_hashes == after_scripts
                         and harness_hash == sha256(Path(__file__)))
            summary["source_wheel_environment_scripts_unchanged"] = unchanged
            if not unchanged:
                summary.update(status="failed", error="source, wheel, environment, or scripts changed")
        except Exception as error:
            summary.update(status="failed", integrity_error=str(error))
        summary["seconds"] = round(time.monotonic() - started, 3)
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        runner._write_json(OUTPUT / "results.json", summary)
    print(f"{summary['status']}: {OUTPUT / 'results.json'}", flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
