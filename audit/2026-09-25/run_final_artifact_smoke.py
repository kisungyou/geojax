import hashlib
import json
import os
import pathlib
import resource
import subprocess
import sys
import time

root = pathlib.Path(__file__).resolve().parents[2]
out = root / "audit/2026-09-25/validation/final"
dist = root / "dist/audit-2026-09-25-final"
command = [sys.executable, "-u", str(root / "scripts/smoke_package.py"), str(dist)]
environment = os.environ.copy()
environment.update(
    PIP_INDEX_URL="https://pypi.org/simple",
    PIP_EXTRA_INDEX_URL="",
    PIP_CACHE_DIR="/private/tmp/geojax-validation-sep25/pip-cache",
    PYTHONUNBUFFERED="1",
)
start = time.monotonic()
with (out / "artifact_smoke.log").open("w") as log:
    result = subprocess.run(
        command,
        cwd="/private/tmp/geojax-validation-sep25",
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
record = {
    "command": command,
    "exit_code": result.returncode,
    "wall_seconds": time.monotonic() - start,
    "cpu_user_seconds": usage.ru_utime,
    "cpu_system_seconds": usage.ru_stime,
    "max_rss_bytes": usage.ru_maxrss,
    "artifacts": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(dist.iterdir())
        if p.is_file()
    },
    "package_index": "https://pypi.org/simple",
}
(out / "artifact_smoke.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2), flush=True)
raise SystemExit(result.returncode)
