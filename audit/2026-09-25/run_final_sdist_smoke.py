import hashlib
import json
import os
import pathlib
import resource
import subprocess
import sys
import time
import tomllib

root = pathlib.Path(__file__).resolve().parents[2]
out = root / "audit/2026-09-25/validation/final"
artifact = root / "dist/audit-2026-09-25-final/geojax-0.2.0.tar.gz"
version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
code = 'import pathlib,runpy,sys; helper=runpy.run_path(sys.argv[1]); helper["_check_artifact"](pathlib.Path(sys.argv[2]),sys.argv[3])'
command = [
    sys.executable,
    "-u",
    "-I",
    "-c",
    code,
    str(root / "scripts/smoke_package.py"),
    str(artifact),
    version,
]
env = os.environ.copy()
env.update(
    PIP_INDEX_URL="https://pypi.org/simple",
    PIP_EXTRA_INDEX_URL="",
    PIP_CACHE_DIR="/private/tmp/geojax-validation-sep25/pip-cache",
    PYTHONUNBUFFERED="1",
)
start = time.monotonic()
with (out / "final_sdist_smoke.log").open("w") as log:
    r = subprocess.run(
        command,
        cwd="/private/tmp/geojax-validation-sep25",
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
d = {
    "command": command,
    "exit_code": r.returncode,
    "wall_seconds": time.monotonic() - start,
    "cpu_user_seconds": usage.ru_utime,
    "cpu_system_seconds": usage.ru_stime,
    "max_rss_bytes": usage.ru_maxrss,
    "artifact": str(artifact),
    "sdist_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    "package_index": "https://pypi.org/simple",
    "scope": "Final sdist-only fresh install, pip dependency check, installed import, package version, and py.typed using existing smoke helper. No redundant wheel reinstall or numerical suite.",
}
(out / "final_sdist_smoke.json").write_text(json.dumps(d, indent=2) + "\n")
print(json.dumps(d, indent=2), flush=True)
raise SystemExit(r.returncode)
