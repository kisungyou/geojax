from __future__ import annotations

from pathlib import Path
import re
import tomllib

import geojax


ROOT = Path(__file__).resolve().parents[1]


def _match(pattern: str, path: Path) -> str:
    match = re.search(pattern, path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    assert match is not None, f"{pattern!r} was not found in {path.name}"
    return match.group(1)


def test_release_versions_agree():
    project_version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    docs_version = _match(r'^release = "([^"]+)"$', ROOT / "docs" / "conf.py")
    citation_version = _match(r"^version: ([^\s]+)$", ROOT / "CITATION.cff")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    changelog_match = re.search(
        r"^## ([0-9][^\s]+) - (Unreleased|\d{4}-\d{2}-\d{2})$",
        changelog,
        flags=re.MULTILINE,
    )
    assert changelog_match is not None
    changelog_version, release_marker = changelog_match.groups()
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation_date = re.search(
        r"^date-released: (\d{4}-\d{2}-\d{2})$", citation, flags=re.MULTILINE
    )

    assert project_version == geojax.__version__
    assert project_version == docs_version
    assert project_version == citation_version
    assert project_version == changelog_version
    if release_marker == "Unreleased":
        assert citation_date is None
    else:
        assert citation_date is not None
        assert citation_date.group(1) == release_marker


def test_release_guide_uses_current_version():
    release_guide = (ROOT / "RELEASING.md").read_text(encoding="utf-8")
    version = geojax.__version__

    assert f"dist/{version}" in release_guide
    assert f"geojax=={version}" in release_guide
    assert f"v{version}" in release_guide
