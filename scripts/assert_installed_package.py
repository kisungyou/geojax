"""Fail when GeoJAX resolves to the checkout instead of an installed package."""

from __future__ import annotations

import argparse
from pathlib import Path
from sysconfig import get_path

import geojax


def assert_installed_package(
    package_file: Path,
    site_packages: Path,
    source_package: Path,
) -> None:
    """Require ``package_file`` below ``site_packages`` and outside ``source_package``."""
    loaded = package_file.resolve()
    purelib = site_packages.resolve()
    source = source_package.resolve()
    if purelib not in loaded.parents or source in loaded.parents:
        raise RuntimeError(f"expected installed wheel, imported {loaded}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    arguments = parser.parse_args()
    assert_installed_package(
        Path(geojax.__file__),
        Path(get_path("purelib")),
        arguments.repository / "geojax",
    )
    print(Path(geojax.__file__).resolve())


if __name__ == "__main__":
    main()
