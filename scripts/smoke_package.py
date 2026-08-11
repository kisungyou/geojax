"""Install built GeoJAX artifacts outside the source tree and smoke-test them."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _interpreter(environment: Path) -> Path:
    if sys.platform == "win32":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _artifacts(directory: Path) -> list[Path]:
    wheels = sorted(directory.rglob("*.whl"))
    source_distributions = sorted(directory.rglob("*.tar.gz"))
    if len(wheels) != 1 or len(source_distributions) != 1:
        raise RuntimeError(
            "package smoke test requires exactly one wheel and one .tar.gz source distribution; "
            f"found {len(wheels)} wheel(s) and {len(source_distributions)} source distribution(s)."
        )
    return [wheels[0], source_distributions[0]]


def _check_artifact(artifact: Path, expected_version: str) -> None:
    with artifact.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(f"sha256  {digest}  {artifact.name}")
    with tempfile.TemporaryDirectory(prefix="geojax-package-smoke-") as temporary:
        root = Path(temporary)
        environment = root / "environment"
        subprocess.run(
            [sys.executable, "-m", "venv", str(environment)],
            check=True,
        )
        python = _interpreter(environment)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--force-reinstall",
                str(artifact.resolve()),
            ],
            check=True,
            cwd=root,
        )
        subprocess.run(
            [str(python), "-m", "pip", "check"],
            check=True,
            cwd=root,
        )
        smoke = """
from importlib.metadata import version
from pathlib import Path
import sys

import geojax
from geojax.geometry import Sphere

prefix = Path(sys.prefix).resolve()
package = Path(geojax.__file__).resolve().parent
assert prefix in package.parents, f"GeoJAX was not imported from the smoke environment: {package}"
assert (package / "py.typed").is_file(), "the wheel is missing geojax/py.typed"
assert version("geojax") == sys.argv[2], (version("geojax"), sys.argv[2])
assert Sphere(3).shape == (3,)
print(f"{sys.argv[1]}: imported GeoJAX {version('geojax')} from {package}")
"""
        subprocess.run(
            [str(python), "-I", "-c", smoke, artifact.name, expected_version],
            check=True,
            cwd=root,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="directory containing one wheel and one sdist")
    arguments = parser.parse_args()
    with (ROOT / "pyproject.toml").open("rb") as stream:
        expected_version = tomllib.load(stream)["project"]["version"]
    for artifact in _artifacts(arguments.directory.resolve()):
        _check_artifact(artifact, expected_version)


if __name__ == "__main__":
    main()
