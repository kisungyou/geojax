"""Prepare, then explicitly run the final-wheel current-dependency supplement.

Use the current-optional-env interpreter with -I. The default action only checks
metadata and file hashes; numerical tests require the explicit --run flag.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback


REPOSITORY = Path(__file__).resolve().parents[2]
OUTPUT = REPOSITORY / "audit/2026-09-25/validation/final/current_dependencies"
TESTS = {
    "tests/test_optional_transport.py": 5,
    "tests/test_linesearch.py": 40,
    "tests/test_shape_audit_fixes.py": 25,
}
EXPECTED_VERSIONS = {
    "geojax": "0.2.0",
    "jax": "0.11.2",
    "jaxlib": "0.11.2",
    "numpy": "2.5.3",
    "scipy": "1.18.1",
    "ott-jax": "0.6.0",
    "pytest": "9.0.3",
    "pytest-cov": "7.1.0",
    "coverage": "7.16.1",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "geojax_verification_runner", REPOSITORY / "scripts/run_test_suite.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory(runner):
    package, hashes = runner._installed_package(REPOSITORY)
    if len(hashes) != 67:
        raise RuntimeError(f"expected 67 installed package hashes, found {len(hashes)}")
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    if versions != EXPECTED_VERSIONS:
        raise RuntimeError(f"unexpected dependency versions: {versions}")
    if not sys.flags.isolated:
        raise RuntimeError("run this harness with Python -I")
    return {
        "executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "installed_package": str(package),
        "installed_package_hashes": hashes,
        "versions": versions,
        "source_hashes": runner._source_hashes(REPOSITORY),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute the prepared tests")
    arguments = parser.parse_args()
    runner = load_runner()
    before = inventory(runner)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prepared_path = OUTPUT / "prepared.json"
    plan = {
        **before,
        "status": "prepared_only",
        "tests": TESTS,
        "precisions": {"float64": "1", "float32": "0"},
        "expected_tests_per_precision": 70,
        "expected_skips": 0,
        "timeout_seconds_per_file": 600,
        "scope": "supplemental selected tests; no whole-package coverage claim",
    }
    if not arguments.run:
        if prepared_path.exists():
            previous = json.loads(prepared_path.read_text())
            previous.pop("prepared_utc", None)
            if previous != plan:
                raise RuntimeError("existing prepared evidence differs; preserve and investigate it")
        else:
            runner._write_json(prepared_path, {**plan, "prepared_utc": utc_now()})
        print(f"Prepared only; no numerical tests run. Evidence: {prepared_path}")
        return 0
    if not prepared_path.exists():
        raise RuntimeError("prepare this harness before executing it")
    previous = json.loads(prepared_path.read_text())
    previous.pop("prepared_utc", None)
    if previous != plan:
        raise RuntimeError("source, wheel, harness, or dependencies changed since preparation")
    results_path = OUTPUT / "results.json"
    if results_path.exists():
        raise RuntimeError("results already exist; preserve prior evidence rather than overwrite")
    summary = {
        "status": "running",
        "started_utc": utc_now(),
        "environment": before,
        "scope": plan["scope"],
        "precisions": {},
    }
    runner._write_json(results_path, summary)
    started = time.monotonic()
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS"):
        environment.pop(key, None)
    environment.update(PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
    try:
        with tempfile.TemporaryDirectory(prefix="geojax-current-dependencies-") as temporary:
            work = Path(temporary).resolve()
            if work == REPOSITORY or REPOSITORY in work.parents:
                raise RuntimeError("working directory must be outside the checkout")
            summary["working_directory"] = str(work)
            for precision, x64 in plan["precisions"].items():
                directory = OUTPUT / precision
                directory.mkdir()
                coverage = directory / "coverage"
                coverage.mkdir()
                environment["GEOJAX_TEST_X64"] = x64
                environment["JAX_ENABLE_X64"] = x64
                rows = []
                summary["precisions"][precision] = {"status": "running", "files": rows}
                for index, (test, expected) in enumerate(TESTS.items(), 1):
                    name = f"{index:03d}-{Path(test).stem}"
                    manifest = directory / f"{name}.collection.json"
                    junit = directory / f"{name}.xml"
                    data_file = coverage / f".coverage.{index:03d}"
                    environment["COVERAGE_FILE"] = str(data_file)
                    command = [
                        sys.executable, "-I", str(REPOSITORY / "scripts/run_test_suite.py"),
                        "--repository", str(REPOSITORY), "--worker", str(REPOSITORY / test),
                        "--manifest", str(manifest), "--junit", str(junit),
                    ]
                    summary["active_file"] = f"{precision}/{test}"
                    runner._write_json(results_path, summary)
                    print(f"{precision}: {test} ({expected} required tests)", flush=True)
                    row = runner._run_process(
                        command, work, environment, directory / f"{name}.log", 600
                    )
                    row["file"] = test
                    rows.append(row)
                    if row["timed_out"]:
                        raise RuntimeError(f"{precision}/{test} exceeded 600 seconds")
                    if not manifest.exists() or not junit.exists():
                        raise RuntimeError(f"{precision}/{test} produced incomplete evidence")
                    row.update(runner._validate_result(
                        row["returncode"], json.loads(manifest.read_text()), junit
                    ))
                    if (row["collected"] != expected or row["junit_cases"] != expected
                            or row.get("passed", 0) != expected or row.get("skipped", 0)
                            or row["collection_skips"]):
                        raise RuntimeError(f"{precision}/{test} did not execute every required test")
                    if not data_file.exists():
                        raise RuntimeError(f"{precision}/{test} produced no coverage data")
                    runner._write_json(results_path, summary)
                totals = dict(sum((Counter({
                    key: row.get(key, 0)
                    for key in ("collected", "junit_cases", "passed", "skipped")
                }) for row in rows), Counter()))
                summary["precisions"][precision].update(status="passed", totals=totals)
                runner._write_json(directory / "results.json", summary["precisions"][precision])
                runner._write_json(results_path, summary)
        summary["status"] = "passed"
    except Exception as error:
        summary.update(status="failed", error=str(error))
        traceback.print_exc()
    finally:
        try:
            after = inventory(runner)
            runner._write_json(OUTPUT / "environment_after.json", after)
            summary["source_wheel_environment_unchanged"] = before == after
            if before != after:
                summary.update(status="failed", error="source, wheel, or environment changed")
        except Exception as error:
            summary.update(status="failed", final_integrity_error=str(error))
        summary.pop("active_file", None)
        summary["seconds"] = round(time.monotonic() - started, 3)
        summary["finished_utc"] = utc_now()
        runner._write_json(results_path, summary)
    print(f"{summary['status']}: {results_path}", flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
