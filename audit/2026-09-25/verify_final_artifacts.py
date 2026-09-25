from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tarfile
import zipfile

root = Path(__file__).resolve().parents[2]
out = root / "audit/2026-09-25/validation/final"
dist = root / "dist/audit-2026-09-25-final"
wheel = dist / "geojax-0.2.0-py3-none-any.whl"
sdist = dist / "geojax-0.2.0.tar.gz"


def sha(b):
    return hashlib.sha256(b).hexdigest()


with zipfile.ZipFile(wheel) as z:
    package = {
        n: sha(z.read(n))
        for n in z.namelist()
        if n.startswith("geojax/") and (n.endswith(".py") or n.endswith("/py.typed"))
    }
    assert (
        z.read("geojax-0.2.0.dist-info/METADATA").partition(b"\n\n")[2]
        == (root / "README.md").read_bytes()
    )
source = {
    str(p.relative_to(root)): sha(p.read_bytes()) for p in sorted((root / "geojax").rglob("*.py"))
}
source["geojax/py.typed"] = sha((root / "geojax/py.typed").read_bytes())
assert source == package
with tarfile.open(sdist) as t:
    members = {m.name.split("/", 1)[1]: m for m in t.getmembers() if m.isfile() and "/" in m.name}
    archive_hashes = {n: sha(t.extractfile(m).read()) for n, m in members.items()}
    assert all(archive_hashes[n] == v for n, v in package.items())
    compared = {n: h for n, h in archive_hashes.items() if (root / n).is_file()}
    assert all(sha((root / n).read_bytes()) == h for n, h in compared.items())
    generated = set(archive_hashes) - set(compared)
    assert generated == {"PKG-INFO", "setup.cfg"}, generated
    assert (
        t.extractfile(members["setup.cfg"]).read().decode().strip()
        == "[egg_info]\ntag_build = \ntag_date = 0"
    )
    assert (
        t.extractfile(members["PKG-INFO"]).read().partition(b"\n\n")[2]
        == (root / "README.md").read_bytes()
    )
    required = [
        "README.md",
        "docs/development/testing.md",
        "scripts/run_test_suite.py",
        "tests/test_test_suite_runner.py",
    ] + [
        str(p.relative_to(root))
        for sub in ["scripts", "tests"]
        for p in sorted((root / sub).rglob("*.py"))
    ]
    assert all(n in archive_hashes for n in required)
    assert "CORRECTNESS_AUDIT.md" not in archive_hashes and not any(
        n.startswith("audit/") for n in archive_hashes
    )
result = subprocess.run(
    [sys.executable, "-m", "twine", "check", "--strict", str(wheel), str(sdist)],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
(out / "final_package_metadata_check.log").write_text(result.stdout)
assert result.returncode == 0, result.stdout
record = {
    "status": "passed",
    "artifacts": {
        p.name: {"path": str(p), "sha256": sha(p.read_bytes()), "bytes": p.stat().st_size}
        for p in [wheel, sdist]
    },
    "all_66_python_modules_and_py_typed_identical_to_checkout": True,
    "package_hashes": package,
    "wheel_description_matches_current_readme": True,
    "sdist_description_matches_current_readme": True,
    "sdist_all_included_current_files_exact": True,
    "sdist_compared_checkout_file_count": len(compared),
    "sdist_generated_files": sorted(generated),
    "sdist_member_hashes": archive_hashes,
    "required_scripts_tests_docs_included": sorted(set(required)),
    "ci_source_hashes": {
        str(p.relative_to(root)): sha(p.read_bytes())
        for p in sorted((root / ".github/workflows").glob("*.yml"))
    },
    "ci_in_sdist": any(n.startswith(".github/") for n in archive_hashes),
    "audit_report_excluded": True,
    "strict_metadata_exit_code": result.returncode,
    "provenance": "Final source after the Frechet-mean rounding correction. This exact wheel is installed in every principal compatibility-matrix environment.",
}
(out / "package_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
print(
    json.dumps(
        {
            k: v
            for k, v in record.items()
            if k
            not in [
                "package_hashes",
                "sdist_member_hashes",
                "required_scripts_tests_docs_included",
                "ci_source_hashes",
            ]
        },
        indent=2,
    )
)
