"""Verify the installed wheel, running each test module in a fresh process.

Run with the environment's Python, for example::

    python -I scripts/run_test_suite.py --repository . --output /tmp/geojax-tests

Every test module is required. Separate processes bound JAX compilation memory;
coverage thresholds apply to their combined data, never to an individual module.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import faulthandler
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _hashes(root: Path, paths: list[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def _source_hashes(repository: Path) -> dict[str, str]:
    paths = [
        path
        for directory in ("geojax", "tests", "scripts")
        for path in (repository / directory).rglob("*.py")
    ]
    paths += [repository / "geojax" / "py.typed"]
    paths += [repository / name for name in ("pyproject.toml", "tox.ini")]
    paths += list((repository / ".github" / "workflows").glob("*.yml"))
    return _hashes(repository, paths)


def _installed_package(repository: Path) -> tuple[Path, dict[str, str]]:
    specification = importlib.util.find_spec("geojax")
    if specification is None or specification.origin is None:
        raise RuntimeError("the GeoJAX wheel is not installed")
    package = Path(specification.origin).resolve().parent
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    if purelib not in package.parents or package == repository / "geojax":
        raise RuntimeError(f"expected installed wheel in {purelib}, found {package}")
    expected = _hashes(
        repository / "geojax",
        list((repository / "geojax").rglob("*.py")) + [repository / "geojax" / "py.typed"],
    )
    actual = _hashes(package, list(package.rglob("*.py")) + [package / "py.typed"])
    if expected != actual:
        differences = sorted(
            key for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key)
        )
        raise RuntimeError(f"installed wheel differs from source: {differences}")
    return package, actual


def _check_loaded_modules(package: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name == "geojax" or name.startswith("geojax."):
            filename = getattr(module, "__file__", None)
            if filename is not None and package not in Path(filename).resolve().parents:
                raise RuntimeError(f"module {name} came from outside the wheel: {filename}")


def _run_process(
    command: list[str], cwd: Path, environment: dict[str, str], log: Path, timeout: float
) -> dict:
    started = time.monotonic()
    with log.open("w") as stream:
        stream.write("Command: " + json.dumps(command) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stream.write(f"\nWall timeout after {timeout:g} seconds; terminating process group.\n")
            stream.flush()
            if os.name == "posix":
                # Workers register this signal for a final stack snapshot.
                if "--worker" in command:
                    os.kill(process.pid, signal.SIGUSR1)
                    time.sleep(0.1)
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                returncode = process.wait()
    return {
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "seconds": round(time.monotonic() - started, 3),
        "log": str(log),
    }


class _CollectionRecorder:
    """Record collected IDs and runtime reports independently of pytest's exit code."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {
            "nodeids": [],
            "collection_reports": [],
            "runtime_reports": [],
            "deselected": [],
        }

    def pytest_deselected(self, items):
        self.data["deselected"].extend(item.nodeid for item in items)

    def pytest_collection_finish(self, session):
        self.data["nodeids"] = [item.nodeid for item in session.items]
        _write_json(self.path, self.data)

    def pytest_collectreport(self, report):
        if report.outcome != "passed":
            self.data["collection_reports"].append(
                {"nodeid": report.nodeid, "outcome": report.outcome, "reason": str(report.longrepr)}
            )

    def pytest_runtest_logreport(self, report):
        self.data["runtime_reports"].append(
            {
                "nodeid": report.nodeid,
                "when": report.when,
                "outcome": report.outcome,
                "seconds": report.duration,
            }
        )

    def pytest_sessionfinish(self, session, exitstatus):
        self.data["exitstatus"] = int(exitstatus)
        _write_json(self.path, self.data)


def _worker(arguments: argparse.Namespace) -> int:
    import pytest

    faulthandler.enable()
    if os.name == "posix":
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    package, _ = _installed_package(arguments.repository)
    recorder = _CollectionRecorder(arguments.manifest)
    result = pytest.main(
        [
            str(arguments.worker),
            f"--rootdir={arguments.repository}",
            "-o",
            "pythonpath=",
            "-o",
            "addopts=",
            "-o",
            f"faulthandler_timeout={arguments.trace_after}",
            "--maxfail=1",
            "-v",
            "--durations=0",
            f"--junitxml={arguments.junit}",
            "--cov=geojax",
            f"--cov-config={arguments.repository / 'pyproject.toml'}",
            "--cov-report=",
            "--cov-fail-under=0",
        ],
        plugins=[recorder],
    )
    _check_loaded_modules(package)
    return int(result)


def _validate_result(returncode: int, manifest: dict, junit: Path) -> dict:
    """Accept exit 5 only for a recorded, explained module-level collection skip."""
    nodeids = manifest["nodeids"]
    if manifest.get("deselected"):
        raise RuntimeError("the complete suite may not deselect tests")
    collection = manifest["collection_reports"]
    if len(nodeids) != len(set(nodeids)):
        raise RuntimeError("duplicate collected test IDs")
    if manifest.get("exitstatus") != returncode:
        raise RuntimeError("pytest exit status disagrees with its collection manifest")
    skipped_collection = [report for report in collection if report["outcome"] == "skipped"]
    if any(report["outcome"] != "skipped" for report in collection):
        raise RuntimeError("test collection failed")
    if returncode == 5:
        if (
            nodeids
            or not skipped_collection
            or any(not report["reason"] for report in skipped_collection)
        ):
            raise RuntimeError("empty collection without a documented module-level skip")
    elif returncode != 0 or not nodeids:
        raise RuntimeError(f"pytest failed or collected no tests (exit {returncode})")
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    if len(cases) != len(nodeids) + len(skipped_collection):
        raise RuntimeError("JUnit test count disagrees with the collection manifest")
    counts = Counter()
    for case in cases:
        outcome = next(
            (name for name in ("failure", "error", "skipped") if case.find(name) is not None),
            "passed",
        )
        counts[outcome] += 1
    if counts["failure"] or counts["error"]:
        raise RuntimeError("JUnit recorded a failure despite a successful pytest exit")
    runtime: dict[str, dict] = {}
    for report in manifest["runtime_reports"]:
        phases = runtime.setdefault(report["nodeid"], {})
        if report["when"] in phases:
            raise RuntimeError("duplicate test phase report")
        phases[report["when"]] = report["outcome"]
    if set(runtime) != set(nodeids):
        raise RuntimeError("collected tests were not all executed")
    for phases in runtime.values():
        if phases.get("teardown") != "passed" or not (
            phases.get("setup") == "passed"
            and phases.get("call") in {"passed", "skipped"}
            or phases.get("setup") == "skipped"
        ):
            raise RuntimeError("incomplete test execution")
    return {
        "collected": len(nodeids),
        "junit_cases": len(cases),
        **dict(counts),
        "collection_skips": skipped_collection,
    }


def _main(arguments: argparse.Namespace) -> int:
    repository = arguments.repository
    output = arguments.output.resolve()
    if arguments.unique_output:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output = output / f"{stamp}-{os.getpid()}"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory must be empty to avoid mixing runs: {output}")
    output.mkdir(parents=True, exist_ok=True)
    workspace = tempfile.TemporaryDirectory(prefix="geojax-verification-")
    work = Path(workspace.name).resolve()
    if work == repository or repository in work.parents:
        raise RuntimeError("test working directory must be outside the checkout")
    coverage_directory = output / "coverage"
    coverage_directory.mkdir()
    tests = sorted((repository / "tests").rglob("test_*.py"))
    if not tests:
        raise RuntimeError("no test files found")
    before = _source_hashes(repository)
    _write_json(output / "source_hashes_before.json", before)
    summary: dict = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "working_directory": str(work),
        "precision_x64": os.environ.get("GEOJAX_TEST_X64", "1"),
        "test_files": [str(path.relative_to(repository)) for path in tests],
        "files": [],
        "checks": [],
    }
    _write_json(output / "results.json", summary)
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS"):
        environment.pop(key, None)
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    try:
        package, package_hashes = _installed_package(repository)
        summary["installed_package"] = str(package)
        summary["versions"] = {
            name: importlib.metadata.version(name)
            for name in (
                "geojax",
                "jax",
                "jaxlib",
                "numpy",
                "scipy",
                "pytest",
                "pytest-cov",
                "coverage",
            )
        }
        _write_json(output / "installed_package_hashes.json", package_hashes)
        coverage_files = []
        for index, test in enumerate(tests, 1):
            name = f"{index:03d}-{test.stem}"
            manifest_path = output / f"{name}.collection.json"
            junit = output / f"{name}.xml"
            data_file = coverage_directory / f".coverage.{index:03d}"
            environment["COVERAGE_FILE"] = str(data_file)
            command = [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                "--repository",
                str(repository),
                "--worker",
                str(test),
                "--manifest",
                str(manifest_path),
                "--junit",
                str(junit),
                "--trace-after",
                str(arguments.trace_after),
            ]
            print(f"[{index}/{len(tests)}] {test.name}", flush=True)
            summary["active_file"] = str(test.relative_to(repository))
            _write_json(output / "results.json", summary)
            result = _run_process(
                command, work, environment, output / f"{name}.log", arguments.timeout
            )
            result["file"] = str(test.relative_to(repository))
            summary["files"].append(result)
            if result["timed_out"]:
                raise RuntimeError(
                    f"{test.name} exceeded its {arguments.timeout:g}-second wall timeout"
                )
            if not manifest_path.exists() or not junit.exists():
                raise RuntimeError(f"{test.name} did not produce complete collection/JUnit records")
            result.update(
                _validate_result(result["returncode"], json.loads(manifest_path.read_text()), junit)
            )
            if not data_file.exists():
                raise RuntimeError(f"{test.name} did not produce coverage data")
            coverage_files.append(str(data_file))
            _write_json(output / "results.json", summary)
        environment["COVERAGE_FILE"] = str(output / ".coverage")
        base = [sys.executable, "-I", "-m", "coverage"]
        config = f"--rcfile={repository / 'pyproject.toml'}"
        checks = [
            ("coverage-combine", ["combine", config, "--keep", *coverage_files]),
            (
                "coverage-json",
                ["json", config, "--fail-under=0", "-o", str(output / "coverage.json")],
            ),
            ("coverage-global", ["report", config, "--precision=4", "--fail-under=85"]),
            (
                "coverage-learning",
                [
                    "report",
                    config,
                    "--precision=4",
                    "--include=*/geojax/learning/*",
                    "--fail-under=95",
                ],
            ),
        ]
        for name, options in checks:
            result = _run_process(base + options, work, environment, output / f"{name}.log", 120)
            result["name"] = name
            summary["checks"].append(result)
            if result["returncode"] != 0 or result["timed_out"]:
                raise RuntimeError(f"{name} failed; see {result['log']}")
        summary["status"] = "passed"
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = str(error)
        traceback.print_exc()
    finally:
        after = _source_hashes(repository)
        _write_json(output / "source_hashes_after.json", after)
        summary["source_unchanged"] = before == after
        try:
            _installed_package(repository)
        except Exception as error:
            summary["status"] = "failed"
            summary["installed_package_error"] = str(error)
        if before != after:
            summary["status"] = "failed"
            summary["changed_sources"] = sorted(
                key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
            )
        summary.pop("active_file", None)
        summary["seconds"] = round(time.monotonic() - started, 3)
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        summary["totals"] = dict(
            sum(
                (
                    Counter(
                        {
                            key: row.get(key, 0)
                            for key in ("collected", "junit_cases", "passed", "skipped")
                        }
                    )
                    for row in summary["files"]
                ),
                Counter(),
            )
        )
        _write_json(output / "results.json", summary)
        workspace.cleanup()
    print(f"{summary['status']}: {output / 'results.json'}", flush=True)
    return 0 if summary["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", "--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--unique-output",
        action="store_true",
        help="preserve prior runs in unique timestamped output subdirectories",
    )
    parser.add_argument("--timeout", type=float, default=600, help="wall seconds per test file")
    parser.add_argument(
        "--trace-after",
        type=float,
        default=90,
        help="seconds per test before dumping thread stacks",
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--junit", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    arguments.repository = arguments.repository.resolve()
    if arguments.timeout <= 0 or arguments.trace_after <= 0:
        parser.error("timeouts must be positive")
    if arguments.worker:
        if arguments.manifest is None or arguments.junit is None:
            parser.error("worker requires manifest and JUnit paths")
        return _worker(arguments)
    if arguments.output is None:
        parser.error("--output is required")
    return _main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
