"""Guard against incomplete verification being reported as a successful suite."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_test_suite", ROOT / "scripts" / "run_test_suite.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _manifest(nodeids, exitstatus=0):
    return {
        "nodeids": nodeids,
        "exitstatus": exitstatus,
        "collection_reports": [],
        "runtime_reports": [
            {"nodeid": node, "when": phase, "outcome": "passed"}
            for node in nodeids
            for phase in ("setup", "call", "teardown")
        ],
    }


def test_runner_rejects_empty_collection_without_skip(tmp_path):
    junit = tmp_path / "tests.xml"
    junit.write_text('<testsuites><testsuite tests="0" /></testsuites>')
    with pytest.raises(RuntimeError, match="empty collection"):
        runner._validate_result(5, _manifest([], 5), junit)


def test_runner_requires_every_collected_test_to_run(tmp_path):
    junit = tmp_path / "tests.xml"
    junit.write_text(
        '<testsuites><testsuite><testcase name="a" /><testcase name="b" /></testsuite></testsuites>'
    )
    manifest = _manifest(["test_file.py::a", "test_file.py::b"])
    manifest["runtime_reports"] = manifest["runtime_reports"][:3]
    with pytest.raises(RuntimeError, match="not all executed"):
        runner._validate_result(0, manifest, junit)
    with pytest.raises(RuntimeError, match="JUnit test count"):
        runner._validate_result(0, _manifest(["test_file.py::a"]), junit)


@pytest.mark.parametrize(
    "source,expected_returncode,expected_count",
    [
        ("def test_success():\n    assert True\n", 0, 1),
        (
            "import pytest\npytest.skip('optional backend unavailable', allow_module_level=True)\n",
            5,
            0,
        ),
        (
            "import pytest\n@pytest.mark.skip(reason='precision contract')\ndef test_skip(): pass\n",
            0,
            1,
        ),
    ],
)
def test_runner_records_real_pytest_collection_and_skips(
    tmp_path, source, expected_returncode, expected_count
):
    test_file = tmp_path / "test_synthetic.py"
    test_file.write_text(source)
    manifest_path = tmp_path / "collection.json"
    junit = tmp_path / "tests.xml"
    program = (
        "import importlib.util, pytest; "
        f"spec=importlib.util.spec_from_file_location('runner', {str(ROOT / 'scripts' / 'run_test_suite.py')!r}); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "from pathlib import Path; "
        f"raise SystemExit(pytest.main([{str(test_file)!r}, '--confcutdir={tmp_path}', "
        f"'--rootdir={tmp_path}', '--junitxml={junit}', '-o', 'addopts='], "
        f"plugins=[module._CollectionRecorder(Path({str(manifest_path)!r}))]))"
    )
    environment = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    result = runner._run_process(
        [sys.executable, "-I", "-c", program], tmp_path, environment, tmp_path / "pytest.log", 30
    )
    assert result["returncode"] == expected_returncode, (tmp_path / "pytest.log").read_text()
    validated = runner._validate_result(
        result["returncode"], json.loads(manifest_path.read_text()), junit
    )
    assert validated["collected"] == expected_count
    assert validated["junit_cases"] == 1


def test_runner_terminates_a_timed_out_process(tmp_path):
    result = runner._run_process(
        [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
        tmp_path,
        dict(os.environ),
        tmp_path / "timeout.log",
        0.1,
    )
    assert result["timed_out"]
    assert result["returncode"] != 0
    assert result["seconds"] < 10
    assert "Wall timeout" in (tmp_path / "timeout.log").read_text()


def test_runner_verifies_wheel_bytes_inside_tox_directory(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    source = repository / "geojax"
    installed = repository / ".tox" / "example" / "site-packages" / "geojax"
    for package in (source, installed):
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("VERSION = 'test'\n")
        (package / "py.typed").touch()
    from types import SimpleNamespace

    monkeypatch.setattr(
        runner.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(installed / "__init__.py")),
    )
    monkeypatch.setattr(runner.sysconfig, "get_path", lambda name: str(installed.parent))
    assert runner._installed_package(repository)[0] == installed
    (installed / "__init__.py").write_text("VERSION = 'stale'\n")
    with pytest.raises(RuntimeError, match="differs from source"):
        runner._installed_package(repository)
    monkeypatch.setattr(
        runner.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(source / "__init__.py")),
    )
    with pytest.raises(RuntimeError, match="expected installed wheel"):
        runner._installed_package(repository)


def test_runner_disables_background_watchdog_even_if_configured(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\nfaulthandler_timeout = 0.001\n"
    )
    (tmp_path / "conftest.py").write_text(
        "import faulthandler\n"
        "def pytest_configure(config):\n"
        "    def forbidden_watchdog(*args, **kwargs):\n"
        "        raise AssertionError('background traceback watchdog must remain disabled')\n"
        "    faulthandler.dump_traceback_later = forbidden_watchdog\n"
    )
    test_file = tmp_path / "test_synthetic.py"
    test_file.write_text(
        "import faulthandler, time\n"
        "def test_success():\n"
        "    assert faulthandler.is_enabled()\n"
        "    time.sleep(0.02)\n"
    )
    manifest_path = tmp_path / "collection.json"
    junit = tmp_path / "tests.xml"
    program = (
        "import importlib.util; from argparse import Namespace; from pathlib import Path; "
        f"spec=importlib.util.spec_from_file_location('runner', {str(ROOT / 'scripts' / 'run_test_suite.py')!r}); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        # Isolate diagnostic behavior; installed-wheel validation has its own tests.
        "module._installed_package=lambda repository: (repository, {}); "
        "module._check_loaded_modules=lambda package: None; "
        f"raise SystemExit(module._worker(Namespace(repository=Path({str(tmp_path)!r}), "
        f"worker=Path({str(test_file)!r}), manifest=Path({str(manifest_path)!r}), "
        f"junit=Path({str(junit)!r}))))"
    )
    environment = dict(os.environ, COVERAGE_FILE=str(tmp_path / ".coverage"))
    environment.pop("PYTEST_ADDOPTS", None)
    result = runner._run_process(
        [sys.executable, "-I", "-c", program], tmp_path, environment, tmp_path / "pytest.log", 30
    )
    assert result["returncode"] == 0, (tmp_path / "pytest.log").read_text()
    assert not result["timed_out"]
    validated = runner._validate_result(0, json.loads(manifest_path.read_text()), junit)
    assert validated["passed"] == 1


@pytest.mark.skipif(os.name != "posix", reason="signal snapshots are supported on POSIX")
def test_runner_requests_snapshot_only_after_wall_timeout(tmp_path):
    program = (
        "import faulthandler, signal, time; "
        "faulthandler.register(signal.SIGUSR1, all_threads=True); "
        "time.sleep(30)"
    )
    log = tmp_path / "timeout.log"
    result = runner._run_process(
        [sys.executable, "-I", "-c", program, "--worker"], tmp_path, dict(os.environ), log, 0.5
    )
    assert result["timed_out"] and result["returncode"] != 0
    text = log.read_text()
    assert text.index("Wall timeout") < text.index("Current thread")
    assert result["seconds"] < 10
