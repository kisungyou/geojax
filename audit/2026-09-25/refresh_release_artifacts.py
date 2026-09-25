"""Refresh only the final source archive after release-documentation changes."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "audit/2026-09-25/validation/final"
DIST = ROOT / "dist/audit-2026-09-25-final"
WHEEL = DIST / "geojax-0.2.0-py3-none-any.whl"
SDIST = DIST / "geojax-0.2.0.tar.gz"
WHEEL_SHA256 = "dd7ea7fc391ec2e0cc9c3a81757a94fb89b09a3f5401ed8ca7f4fbc62b20fc5e"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_hashes():
    spec = importlib.util.spec_from_file_location("runner", ROOT / "scripts/run_test_suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hashes = module._source_hashes(ROOT)
    for name in ("Makefile", "RELEASING.md", "CHANGELOG.md", "README.md", "MANIFEST.in"):
        hashes[name] = sha(ROOT / name)
    return hashes


def run(command, log, cwd):
    start = time.monotonic()
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(key, None)
    with log.open("w") as stream:
        stream.write("Command: " + json.dumps(command) + "\n")
        stream.flush()
        result = subprocess.run(
            command, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=600
        )
    record = {
        "command": command,
        "returncode": result.returncode,
        "seconds": round(time.monotonic() - start, 3),
        "log": str(log),
    }
    if result.returncode:
        raise RuntimeError(f"command failed; see {log}")
    return record


def package_files(wheel):
    with zipfile.ZipFile(wheel) as archive:
        return {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name.startswith("geojax/") and not name.endswith("/")
        }


def main():
    if sha(WHEEL) != WHEEL_SHA256:
        raise RuntimeError("the principal matrix wheel has changed")
    before = source_hashes()
    history = OUTPUT / "before_release_guard"
    history.mkdir()
    preserved = {}
    for name in (
        "package_provenance.json",
        "artifact_provenance_check.log",
        "final_package_metadata_check.log",
        "artifact_smoke.json",
        "final_sdist_smoke.json",
        "final_sdist_smoke.log",
        "sdist_rebuilt_wheel.json",
        "sdist_rebuild.log",
        "sdist_build_after_guide.log",
        "build.log",
    ):
        source = OUTPUT / name
        if source.exists():
            shutil.copy2(source, history / name)
            preserved[name] = sha(source)
    old_directory = ROOT / "dist/audit-2026-09-25-before-release-guard"
    old_directory.mkdir()
    shutil.copy2(SDIST, old_directory / SDIST.name)
    write(
        history / "preservation.json",
        {
            "preserved_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_sha256": preserved,
            "original_source_archive": str(old_directory / SDIST.name),
            "source_archive_sha256": sha(SDIST),
            "matrix_wheel_sha256": sha(WHEEL),
            "note": "Preserved JSON records retain paths relative to their original final/ location.",
        },
    )
    write(OUTPUT / "release_guard_source_hashes_before.json", before)
    summary = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "commands": [],
        "preserved_evidence": str(history),
        "matrix_wheel_sha256": WHEEL_SHA256,
    }
    write(OUTPUT / "release_guard_artifact_refresh.json", summary)
    try:
        summary["commands"].append(
            run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--sdist",
                    "--no-isolation",
                    "--outdir",
                    str(DIST),
                    str(ROOT),
                ],
                OUTPUT / "sdist_build_after_release_guard.log",
                ROOT,
            )
        )
        if sha(WHEEL) != WHEEL_SHA256:
            raise RuntimeError("source archive build changed the matrix wheel")
        summary["commands"].append(
            run(
                [sys.executable, "-I", str(ROOT / "audit/2026-09-25/verify_final_artifacts.py")],
                OUTPUT / "artifact_provenance_check.log",
                Path("/private/tmp"),
            )
        )
        with tempfile.TemporaryDirectory(prefix="geojax-release-sdist-rebuild-") as temporary:
            temporary = Path(temporary)
            source = temporary / "source"
            source.mkdir()
            with tarfile.open(SDIST) as archive:
                archive.extractall(source, filter="data")
            extracted = list(source.iterdir())
            if len(extracted) != 1 or not extracted[0].is_dir():
                raise RuntimeError("source archive lacks one top-level source directory")
            rebuilt_directory = temporary / "rebuilt"
            build = run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--wheel",
                    "--no-isolation",
                    "--outdir",
                    str(rebuilt_directory),
                    str(extracted[0]),
                ],
                OUTPUT / "sdist_rebuild.log",
                temporary,
            )
            summary["commands"].append(build)
            (rebuilt,) = rebuilt_directory.glob("*.whl")
            package = package_files(WHEEL)
            if len(package) != 67 or package_files(rebuilt) != package:
                raise RuntimeError(
                    "independently rebuilt wheel differs from the 67 matrix package files"
                )
            retained = ROOT / "dist/audit-2026-09-25-rebuilt-after-release-guard"
            retained.mkdir()
            shutil.copy2(rebuilt, retained / rebuilt.name)
            write(
                OUTPUT / "sdist_rebuilt_wheel.json",
                {
                    "status": "passed",
                    "source_archive_sha256": sha(SDIST),
                    "matrix_wheel_sha256": sha(WHEEL),
                    "rebuilt_wheel_sha256": sha(rebuilt),
                    "rebuilt_wheel_path": str(retained / rebuilt.name),
                    "identical_python_modules": sum(name.endswith(".py") for name in package),
                    "identical_typing_marker": "geojax/py.typed" in package,
                    "module_hashes": package,
                    "build_returncode": build["returncode"],
                    "build_command": build["command"],
                    "build_seconds": build["seconds"],
                    "scope": "Independent build from the refreshed source archive; no checkout package imports or numerical tests.",
                },
            )
        summary.update(
            status="archive_verified_and_rebuilt_awaiting_smoke", sdist_sha256=sha(SDIST)
        )
    finally:
        after = source_hashes()
        write(OUTPUT / "release_guard_source_hashes_after.json", after)
        summary["frozen_sources_unchanged"] = before == after
        summary["matrix_wheel_unchanged"] = sha(WHEEL) == WHEEL_SHA256
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if before != after or not summary["matrix_wheel_unchanged"]:
            summary["status"] = "failed"
        write(OUTPUT / "release_guard_artifact_refresh.json", summary)
    if not summary["frozen_sources_unchanged"] or not summary["matrix_wheel_unchanged"]:
        raise RuntimeError("frozen source or matrix wheel changed")
    print(json.dumps({key: value for key, value in summary.items() if key != "commands"}, indent=2))


if __name__ == "__main__":
    main()
