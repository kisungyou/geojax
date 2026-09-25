"""Reconcile the refreshed source archive's completed packaging evidence."""

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "refresh", Path(__file__).with_name("refresh_release_artifacts.py")
)
refresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refresh)
output = refresh.OUTPUT


def read(name):
    return json.loads((output / name).read_text())


smoke = read("final_sdist_smoke.json")
rebuilt = read("sdist_rebuilt_wheel.json")
provenance = read("package_provenance.json")
earlier_wheel = read("before_guide_clarification/artifact_smoke.json")
history = read("before_release_guard/preservation.json")
assert smoke["exit_code"] == 0
assert earlier_wheel["exit_code"] == 0
assert earlier_wheel["artifacts"][refresh.WHEEL.name] == refresh.WHEEL_SHA256
assert refresh.sha(refresh.WHEEL) == refresh.WHEEL_SHA256 == rebuilt["matrix_wheel_sha256"]
assert refresh.sha(refresh.SDIST) == smoke["sdist_sha256"] == rebuilt["source_archive_sha256"]
assert provenance["artifacts"][refresh.SDIST.name]["sha256"] == smoke["sdist_sha256"]
assert rebuilt["module_hashes"] == provenance["package_hashes"]
assert len(rebuilt["module_hashes"]) == 67
assert refresh.source_hashes() == read("release_guard_source_hashes_before.json")
assert all(
    refresh.sha(output / "before_release_guard" / name) == digest
    for name, digest in history["evidence_sha256"].items()
)
assert refresh.sha(Path(history["original_source_archive"])) == history["source_archive_sha256"]
record = {
    "status": "passed",
    "wheel": {
        "sha256": refresh.WHEEL_SHA256,
        "clean_install_exit_code": earlier_wheel["exit_code"],
        "evidence": "before_guide_clarification/artifact_smoke.json",
        "unchanged_from_all_matrix_installations": True,
    },
    "sdist": {
        "sha256": smoke["sdist_sha256"],
        "clean_install_exit_code": smoke["exit_code"],
        "evidence": "final_sdist_smoke.json",
        "wall_seconds": smoke["wall_seconds"],
    },
    "source_archive_rebuild_evidence": "sdist_rebuilt_wheel.json",
    "prior_evidence": "before_release_guard/preservation.json",
    "initial_network_restricted_attempt": "release_guard_smoke_network_attempt/final_sdist_smoke.json",
    "provenance": (
        "The tested matrix wheel is unchanged. After the release-only Makefile, RELEASING.md, "
        "and CHANGELOG.md edits, only the source archive was rebuilt. Every included checkout "
        "file and strict package metadata passed verification. A fresh isolated installation "
        "passed dependency/import/version/typing checks, and an independent wheel rebuilt from "
        "the archive has the same 66 Python modules plus py.typed as the matrix wheel."
    ),
}
refresh.write(output / "artifact_smoke.json", record)
summary = read("release_guard_artifact_refresh.json")
summary.update(
    status="passed",
    sdist_clean_install_exit_code=smoke["exit_code"],
    sdist_clean_install_seconds=smoke["wall_seconds"],
    sdist_clean_install_evidence="final_sdist_smoke.json",
    source_archive_file_count=len(provenance["sdist_member_hashes"]),
    source_archive_compared_checkout_file_count=provenance["sdist_compared_checkout_file_count"],
    identical_package_file_count=67,
    prior_evidence_unchanged=True,
    release_file_hashes={
        name: refresh.sha(refresh.ROOT / name)
        for name in ("Makefile", "RELEASING.md", "CHANGELOG.md")
    },
    frozen_sources_unchanged=True,
    matrix_wheel_unchanged=True,
    finished_utc=datetime.now(timezone.utc).isoformat(),
)
refresh.write(output / "release_guard_source_hashes_after.json", refresh.source_hashes())
refresh.write(output / "release_guard_artifact_refresh.json", summary)
print(json.dumps(record, indent=2))
