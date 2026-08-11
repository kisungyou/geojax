"""Require release artifacts to be built from a clean, annotated version tag."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError("release builds require a clean working tree")

    with (ROOT / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    expected_tag = f"v{version}"
    expected_ref = f"refs/tags/{expected_tag}"
    try:
        tagged_commit = _git("rev-parse", f"{expected_ref}^{{commit}}")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"release commit must carry annotated tag {expected_tag}") from exc
    if tagged_commit != _git("rev-parse", "HEAD"):
        raise RuntimeError(f"{expected_tag} must identify the release commit at HEAD")
    if _git("cat-file", "-t", expected_ref) != "tag":
        raise RuntimeError(f"{expected_tag} must be signed or annotated, not lightweight")

    print(f"release source verified at {expected_tag} ({tagged_commit})")


if __name__ == "__main__":
    main()
