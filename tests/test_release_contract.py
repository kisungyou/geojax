from __future__ import annotations

import configparser
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
    citation_date = re.search(r"^date-released: (\d{4}-\d{2}-\d{2})$", citation, flags=re.MULTILINE)

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


def test_pep561_marker_is_packaged():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_data = pyproject["tool"]["setuptools"]["package-data"]["geojax"]

    assert (ROOT / "geojax" / "py.typed").is_file()
    assert "py.typed" in package_data


def test_supported_python_matrix_is_consistent():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    supported = tuple(
        line.strip()
        for line in (ROOT / ".python-versions").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    classifiers = set(pyproject["project"]["classifiers"])

    parser = configparser.ConfigParser()
    parser.read(ROOT / "tox.ini", encoding="utf-8")
    environments = set(parser["tox"]["env_list"].split())

    assert supported == ("3.11", "3.12", "3.13", "3.14")
    assert pyproject["project"]["requires-python"] == ">=3.11"
    assert "include .python-versions" in manifest
    for version in supported:
        assert f"Programming Language :: Python :: {version}" in classifiers
        factor = version.replace(".", "")
        assert f"py{factor}-stable-float32" in environments
        assert f"py{factor}-stable-float64" in environments

    assert "py311-min-float32" in environments
    assert "py311-min-float64" in environments
    assert not any("-latest-" in environment for environment in environments)
    assert parser["testenv"]["uv_python_preference"] == "only-managed"
    assert "COVERAGE_FILE={env_tmp_dir}/.coverage" in parser["testenv"]["set_env"]
    assert "PYTHONHASHSEED=0" in parser["testenv"]["set_env"]
