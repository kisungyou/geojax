"""Replay the September 25 installed-wheel matrix using prepared tox environments.

The supported public entry point is ``make test-matrix-parallel``. This audit
driver additionally records a single durable summary of the ten local runs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import configparser
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parents[2]
LOCK = threading.Lock()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.jobs < 1:
        parser.error("--jobs must be positive")
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output directory must be empty")
    config = configparser.ConfigParser()
    config.read(ROOT / "tox.ini")
    names = config["tox"]["env_list"].split()
    # Exercise the oldest dependency floor and newest Python first.
    priority = ["py311-min-float64", "py314-stable-float32"]
    names = priority + [name for name in names if name not in priority]
    runner = ROOT / "scripts" / "run_test_suite.py"
    summary = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": arguments.jobs,
        "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "environments": names,
        "runs": {},
    }
    write_json(output / "matrix_results.json", summary)
    started = time.monotonic()

    def run(name):
        command = [
            str(ROOT / ".tox" / name / "bin" / "python"),
            "-I",
            str(runner),
            "--repository",
            str(ROOT),
            "--output",
            str(output / name),
            "--timeout",
            "600",
            "--trace-after",
            "90",
        ]
        environment = dict(os.environ)
        environment["GEOJAX_TEST_X64"] = "1" if name.endswith("float64") else "0"
        for key in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS"):
            environment.pop(key, None)
        entry = {"status": "running", "command": command}
        with LOCK:
            summary["runs"][name] = entry
            write_json(output / "matrix_results.json", summary)
            print(f"Starting {name}", flush=True)
        with (output / f"{name}.progress.log").open("w") as log:
            result = subprocess.run(
                command, env=environment, cwd="/private/tmp", stdout=log, stderr=subprocess.STDOUT
            )
        result_file = output / name / "results.json"
        details = json.loads(result_file.read_text()) if result_file.exists() else {}
        entry.update(
            status="passed"
            if result.returncode == 0 and details.get("status") == "passed"
            else "failed",
            returncode=result.returncode,
            results=str(result_file),
            totals=details.get("totals"),
            seconds=details.get("seconds"),
            error=details.get("error"),
        )
        with LOCK:
            write_json(output / "matrix_results.json", summary)
            print(f"{name}: {entry['status']} {entry.get('totals')}", flush=True)
        return entry

    with ThreadPoolExecutor(max_workers=arguments.jobs) as executor:
        futures = [executor.submit(run, name) for name in names]
        for future in as_completed(futures):
            future.result()
    summary["status"] = (
        "passed" if all(row["status"] == "passed" for row in summary["runs"].values()) else "failed"
    )
    summary["seconds"] = round(time.monotonic() - started, 3)
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "matrix_results.json", summary)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
