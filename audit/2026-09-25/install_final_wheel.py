from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

root = Path(__file__).resolve().parents[2]
wheel = root / "dist/audit-2026-09-25-final/geojax-0.2.0-py3-none-any.whl"
out = root / "audit/2026-09-25/validation/final"
scratch = Path("/private/tmp/geojax-validation-sep25")
envs = [
    "py311-min-float32",
    "py311-min-float64",
    "py311-stable-float32",
    "py311-stable-float64",
    "py312-stable-float32",
    "py312-stable-float64",
    "py313-stable-float32",
    "py313-stable-float64",
    "py314-stable-float32",
    "py314-stable-float64",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


source = {
    str(p.relative_to(root)): sha(p.read_bytes()) for p in sorted((root / "geojax").rglob("*.py"))
}
source["geojax/py.typed"] = sha((root / "geojax/py.typed").read_bytes())
with zipfile.ZipFile(wheel) as z:
    actual = {
        name: sha(z.read(name))
        for name in z.namelist()
        if name.startswith("geojax/") and (name.endswith(".py") or name.endswith("/py.typed"))
    }
    assert actual == source, {
        "missing": sorted(set(source) - set(actual)),
        "unexpected": sorted(set(actual) - set(source)),
        "changed": [p for p in source.keys() & actual.keys() if source[p] != actual[p]],
    }
    wheel_metadata = {
        name: z.read(name).decode()
        for name in z.namelist()
        if name.endswith(("/METADATA", "/WHEEL", "/top_level.txt"))
    }
metadata = subprocess.run(
    [sys.executable, "-m", "twine", "check", "--strict", str(wheel)],
    cwd=scratch,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
(out / "wheel_metadata_check.log").write_text(metadata.stdout)
assert metadata.returncode == 0, metadata.stdout
record = {
    "wheel": str(wheel),
    "wheel_sha256": sha(wheel.read_bytes()),
    "python_module_count": sum(p.endswith(".py") for p in source),
    "includes_py_typed": "geojax/py.typed" in source,
    "source_and_wheel_match": True,
    "source_hashes": source,
    "metadata_check_exit": metadata.returncode,
    "wheel_metadata": wheel_metadata,
}
(out / "wheel_integrity.json").write_text(json.dumps(record, indent=2) + "\n")
print(
    "Wheel verified: "
    + str(record["python_module_count"])
    + " Python modules plus py.typed match source; strict metadata passed.",
    flush=True,
)
probe = """import hashlib,importlib.metadata as m,json,pathlib,platform,sys
import geojax
root=pathlib.Path(geojax.__file__).resolve().parent
files={"geojax/"+str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.py")}
files["geojax/py.typed"]=hashlib.sha256((root/"py.typed").read_bytes()).hexdigest()
print(json.dumps(dict(python=platform.python_version(),python_executable=sys.executable,prefix=sys.prefix,geojax_path=str(root),geojax_version=m.version("geojax"),versions={n:m.version(n) for n in ["jax","jaxlib","numpy","scipy","ml-dtypes","opt-einsum","pytest","pytest-cov","coverage"]},all_distributions={d.metadata["Name"].lower().replace("_","-"):d.version for d in m.distributions()},installed_hashes=files)))
"""
results = []
for name in envs:
    start = time.monotonic()
    python = root / ".tox" / name / "bin/python"
    before = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            'import importlib.metadata as m,json; print(json.dumps({d.metadata["Name"].lower().replace("_","-"):d.version for d in m.distributions()}))',
        ],
        cwd=scratch,
        text=True,
        capture_output=True,
        check=True,
    )
    old = json.loads(before.stdout)
    environment = os.environ.copy()
    environment.update(UV_CACHE_DIR=str(scratch / "uv-cache"))
    install = subprocess.run(
        [
            "/Users/kisung/.local/bin/uv",
            "pip",
            "install",
            "--offline",
            "--python",
            str(python),
            "--no-deps",
            "--reinstall",
            str(wheel),
        ],
        cwd=scratch,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (out / (name + "_wheel_install.log")).write_text(install.stdout)
    assert install.returncode == 0, install.stdout
    verified = subprocess.run(
        [str(python), "-I", "-c", probe], cwd=scratch, text=True, capture_output=True, check=True
    )
    value = json.loads(verified.stdout)
    assert value["installed_hashes"] == source, (name, "installed source mismatch")
    assert Path(value["geojax_path"]).is_relative_to(root / ".tox" / name), (
        name,
        "shadowing",
        value["geojax_path"],
    )
    new = value["all_distributions"]
    old.pop("geojax", None)
    non_package = dict(new)
    non_package.pop("geojax", None)
    assert old == non_package, (name, "dependency change", old, non_package)
    result = {
        "environment": name,
        "install_exit": install.returncode,
        "wall_seconds": time.monotonic() - start,
        "dependencies_unchanged": True,
        "installed_package_matches_wheel": True,
        **value,
    }
    (out / (name + "_installed_package.json")).write_text(json.dumps(result, indent=2) + "\n")
    results.append(
        {k: v for k, v in result.items() if k not in ["installed_hashes", "all_distributions"]}
    )
    (out / "wheel_install_matrix.json").write_text(
        json.dumps({"wheel_sha256": record["wheel_sha256"], "environments": results}, indent=2)
        + "\n"
    )
    print(name + ": installed, versions unchanged, all package hashes match.", flush=True)
print("All ten environments ready.", flush=True)
