"""Audit the real release recipe in throwaway annotated-tag Git fixtures.

No tags or Git state in the real repository are changed. Expensive targets
are replaced only inside the fixtures; release-check and its source verifier
are copied verbatim from the recorded source.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "audit/2026-09-25/validation/release_source_guard"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def fixture(name: str, makefile: Path, dirty: bool) -> dict:
    directory = Path(tempfile.mkdtemp(prefix=f"geojax-release-guard-{name}-"))
    (directory / "scripts").mkdir()
    shutil.copy2(ROOT / "scripts/check_release_source.py", directory / "scripts/check_release_source.py")
    shutil.copy2(ROOT / "pyproject.toml", directory / "pyproject.toml")
    shutil.copy2(makefile, directory / "Makefile.production")
    image = directory / "docs/_static/tutorials/dimension-reduction.png"
    image.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "docs/_static/tutorials/dimension-reduction.png", image)
    (directory / ".gitignore").write_text(".audit-output/\n")
    (directory / "mock_target.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "target, dirty = sys.argv[1:]\n"
        "out = Path('.audit-output'); out.mkdir(exist_ok=True)\n"
        "(out / target).write_text('called\\n')\n"
        "if target == 'website' and dirty == '1':\n"
        "    path = Path('docs/_static/tutorials/dimension-reduction.png')\n"
        "    path.write_bytes(path.read_bytes() + b'fixture mutation')\n"
    )
    # GNU make reads the real release recipe from Makefile.production. Only
    # its expensive prerequisites are overridden, and recursive make keeps
    # using this same wrapper in the fixture.
    (directory / "Makefile").write_text(
        "include Makefile.production\n"
        "quality test-matrix website package-check:\n"
        "\t$(PYTHON) mock_target.py $@ $(DIRTY_WEBSITE)\n"
    )
    setup_commands = [
        ["git", "init", "-q"],
        ["git", "config", "user.name", "GeoJAX audit fixture"],
        ["git", "config", "user.email", "audit@example.invalid"],
        ["git", "add", "."],
        ["git", "-c", "commit.gpgsign=false", "commit", "-qm", "Isolated release guard fixture"],
        ["git", "-c", "tag.gpgsign=false", "tag", "-a", "v0.2.0", "-m", "Local audit fixture only"],
    ]
    for command in setup_commands:
        subprocess.run(command, cwd=directory, check=True, capture_output=True, text=True)
    command = ["make", "release-check", f"PYTHON={sys.executable}", f"DIRTY_WEBSITE={int(dirty)}"]
    started = time.monotonic()
    completed = subprocess.run(command, cwd=directory, capture_output=True, text=True)
    log = OUT / f"{name}.log"
    log.write_text(completed.stdout + completed.stderr)
    packaged = (directory / ".audit-output/package-check").exists()
    check_count = completed.stdout.count("scripts/check_release_source.py")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=directory, capture_output=True, text=True, check=True).stdout
    expected_success = name != "final-dirty"
    assert (completed.returncode == 0) == expected_success, (name, completed.returncode, log)
    assert packaged == expected_success, (name, packaged)
    assert (directory / ".audit-output/website").exists()
    assert check_count == (1 if name == "before-dirty" else 2), (name, check_count)
    if dirty:
        assert "docs/_static/tutorials/dimension-reduction.png" in status
    else:
        assert not status
    if name == "final-dirty":
        assert "release builds require a clean working tree" in completed.stderr
    return {
        "scenario": name,
        "fixture_directory": str(directory),
        "setup_commands": setup_commands,
        "command": command,
        "exit_code": completed.returncode,
        "seconds": time.monotonic() - started,
        "source_check_invocations": check_count,
        "website_ran": True,
        "tracked_png_modified": dirty,
        "package_check_ran": packaged,
        "git_status_after": status,
        "actual_release_recipe_sha256": digest(makefile),
        "actual_source_verifier_sha256": digest(directory / "scripts/check_release_source.py"),
        "expected_outcome_verified": True,
        "log": str(log.relative_to(ROOT)),
    }


def installed_snapshot(python: Path, cwd: Path) -> dict:
    program = """
from pathlib import Path
import hashlib, importlib.util, importlib.metadata, json, sysconfig
spec = importlib.util.find_spec('geojax')
package = Path(spec.origin).resolve().parent
purelib = Path(sysconfig.get_path('purelib')).resolve()
assert purelib in package.parents
hashes = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(list(package.rglob('*.py')) + [package/'py.typed'])}
print(json.dumps({'package':str(package),'hashes':hashes,'versions':{name:importlib.metadata.version(name) for name in ('geojax','jax','jaxlib','numpy','pytest','pytest-cov','coverage')}}))
"""
    result = subprocess.run([str(python), "-I", "-c", program], cwd=cwd, capture_output=True, text=True, check=True)
    snapshot = json.loads(result.stdout)
    expected = {str(path.relative_to(ROOT / "geojax")): digest(path) for path in sorted(list((ROOT / "geojax").rglob("*.py")) + [ROOT / "geojax/py.typed"])}
    assert snapshot["hashes"] == expected
    return snapshot


def release_tests() -> list[dict]:
    import configparser

    config = configparser.ConfigParser()
    config.read(ROOT / "tox.ini")
    names = config["tox"]["env_list"].split()
    assert len(names) == 10
    cwd = Path(tempfile.mkdtemp(prefix="geojax-release-contract-tests-"))
    results = []
    package_before = {str(path.relative_to(ROOT)): digest(path) for path in (ROOT / "geojax").rglob("*.py")}
    frozen_paths = [ROOT / "tests/test_release_contract.py", *sorted((ROOT / "scripts").glob("*.py")), ROOT / "tox.ini", ROOT / ".github/workflows/ci.yml", ROOT / "Makefile", ROOT / "RELEASING.md", ROOT / "CHANGELOG.md"]
    tool_before = {str(path.relative_to(ROOT)): digest(path) for path in frozen_paths}
    for name in names:
        python = ROOT / ".tox" / name / "bin/python"
        before = installed_snapshot(python, cwd)
        xml = OUT / f"{name}.xml"
        command = [str(python), "-I", "-m", "pytest", str(ROOT / "tests/test_release_contract.py"), f"--rootdir={ROOT}", "-o", "pythonpath=", "-o", "addopts=", "-o", "faulthandler_timeout=0", f"--junitxml={xml}", "-q"]
        environment = dict(os.environ)
        for key in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "COVERAGE_FILE"):
            environment.pop(key, None)
        environment["GEOJAX_TEST_X64"] = "1" if name.endswith("float64") else "0"
        environment["PYTHONHASHSEED"] = "0"
        started = time.monotonic()
        with (OUT / f"{name}.log").open("w") as stream:
            result = subprocess.run(command, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=60)
        after = installed_snapshot(python, cwd)
        assert before == after, name
        cases = list(ET.parse(xml).getroot().iter("testcase"))
        assert result.returncode == 0 and len(cases) == 6, name
        assert not any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")), name
        results.append({"environment": name, "command": command, "exit_code": result.returncode, "seconds": time.monotonic() - started, "passed": 6, "skipped": 0, "installed_package_unchanged": True, "installed_package": before})
        write_json(OUT / "release_contract_runs.json", results)
        print(f"{name}: 6 passed, package unchanged", flush=True)
    assert package_before == {str(path.relative_to(ROOT)): digest(path) for path in (ROOT / "geojax").rglob("*.py")}
    assert tool_before == {str(path.relative_to(ROOT)): digest(path) for path in frozen_paths}
    write_json(OUT / "frozen_inputs.json", {"package": package_before, "tools_and_release_test": tool_before, "unchanged": True})
    return results


def main() -> None:
    old = OUT / "Makefile.before"
    new = ROOT / "Makefile"
    old_text, new_text = old.read_text(), new.read_text()
    assert new_text == old_text.replace("\t$(MAKE) website\n\t$(MAKE) package-check\n", "\t$(MAKE) website\n\t$(MAKE) release-source-check\n\t$(MAKE) package-check\n")
    scenarios = [fixture("before-dirty", old, True), fixture("final-clean", new, False), fixture("final-dirty", new, True)]
    write_json(OUT / "fixture_results.json", {"status": "passed", "scenarios": scenarios})
    runs = release_tests()
    write_json(OUT / "result.json", {
        "status": "passed",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "change": "A second clean annotated-tag source check runs after documentation and before artifact construction.",
        "makefile_before_sha256": digest(old),
        "makefile_final_sha256": digest(new),
        "website_recipe_unchanged": old_text.split("website:\n", 1)[1].split("\nserve:", 1)[0] == new_text.split("website:\n", 1)[1].split("\nserve:", 1)[0],
        "fixture_scenarios": scenarios,
        "release_contract_tests": {"environments": len(runs), "passed_per_environment": 6, "total_passed": 60, "skipped": 0, "evidence": "release_contract_runs.json", "scope": "Separate repeated checks of the unchanged six release-contract tests; do not add to full-matrix unique test counts."},
        "real_repository_git_state_modified": False,
        "package_tests_scripts_ci_tox_unchanged": True,
        "harness_sha256": digest(Path(__file__)),
    })
    print("Release guard scenarios and all ten unchanged release-contract runs passed.", flush=True)


if __name__ == "__main__":
    main()
