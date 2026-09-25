"""Independently validate the ten-environment audit evidence without running tests.

Exit 0 means every selected complete run is verified, 2 means runs are pending,
and 1 means evidence is invalid. Historical failed/incomplete runs are not reused.
"""

from __future__ import annotations

import argparse
from collections import Counter
import configparser
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import tomllib
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "audit/2026-09-25/validation"
ENVIRONMENTS = {
    f"py{python}-{dependency}-{precision}"
    for python, dependency in [
        ("311", "min"),
        ("311", "stable"),
        ("312", "stable"),
        ("313", "stable"),
        ("314", "stable"),
    ]
    for precision in ["float32", "float64"]
}
FIRST_BATCH = {
    "py314-stable-float64",
    "py311-stable-float32",
    "py312-stable-float64",
    "py313-stable-float32",
    "py314-stable-float32",
}
EXCLUDED_EXAMPLES = {"examples/largest_eigenvector.py", "examples/spd_frechet_mean.py"}
EXPECTED_COLLECTED = 1147
OPTIONAL_SKIP = "tests/test_optional_transport.py"
PRECISION_SKIP = (
    "tests/test_learning_embedding_metric.py::"
    "test_rmml_zero_regularization_repairs_degenerate_scatter[float64]"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_hashes():
    paths = [
        p for directory in ["geojax", "tests", "scripts"] for p in (ROOT / directory).rglob("*.py")
    ]
    paths += [ROOT / "geojax/py.typed", ROOT / "pyproject.toml", ROOT / "tox.ini"]
    paths += list((ROOT / ".github/workflows").glob("*.yml"))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def canonical_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def expected_dependencies(name):
    config = configparser.ConfigParser()
    config.read(ROOT / "tox.ini")
    factors = set(name.split("-"))
    pins = {}
    for raw in config["testenv"]["deps"].splitlines():
        requirement = raw.strip()
        if not requirement:
            continue
        if ":" in requirement:
            condition, requirement = requirement.split(":", 1)
            if not set(condition.strip().split("-")) <= factors:
                continue
        parts = requirement.strip().split("==")
        require(len(parts) == 2 and all(parts), "tox dependencies must be exact pins")
        package, version = canonical_name(parts[0]), parts[1]
        require(package not in pins, f"duplicate dependency pin: {name}/{package}")
        pins[package] = version
    require(
        set(pins)
        == {
            "jax",
            "jaxlib",
            "numpy",
            "scipy",
            "ml-dtypes",
            "opt-einsum",
            "pytest",
            "pytest-cov",
            "coverage",
        },
        f"unexpected pinned dependency set: {name}",
    )
    return pins


def installed_versions(site_packages):
    # Read distribution metadata only; do not import numerical packages or start workers.
    result = {}
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        name = canonical_name(distribution.metadata["Name"])
        require(name not in result, f"duplicate installed distribution: {name}")
        result[name] = distribution.version
    return result


def dependency_evidence(name, summary, installed, expected_package):
    pins = expected_dependencies(name)
    installation_path = BASE / "final" / f"{name}_installed_package.json"
    installation = read(installation_path)
    require(
        installation["environment"] == name
        and installation["install_exit"] == 0
        and installation["dependencies_unchanged"] is True
        and installation["installed_package_matches_wheel"] is True,
        f"invalid final wheel installation evidence: {name}",
    )
    require(
        Path(installation["geojax_path"]) == installed
        and installation["installed_hashes"]
        == {"geojax/" + key: value for key, value in expected_package.items()},
        f"installation provenance does not match final source: {name}",
    )
    require(installation["versions"] == pins, f"installation dependency pins differ: {name}")
    live = installed_versions(installed.parent)
    require(live == installation["all_distributions"], f"installed dependencies changed: {name}")
    require(
        all(live.get(key) == value for key, value in pins.items()),
        f"wrong installed dependency versions: {name}",
    )
    package_version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    expected_recorded = {
        key: value for key, value in pins.items() if key not in {"ml-dtypes", "opt-einsum"}
    }
    expected_recorded["geojax"] = package_version
    require(summary["versions"] == expected_recorded, f"wrong recorded dependency versions: {name}")
    require(
        live["geojax"] == installation["geojax_version"] == package_version,
        f"wrong installed package version: {name}",
    )
    require(
        summary["python"].split()[0] == installation["python"],
        f"Python differs from installation evidence: {name}",
    )
    return {
        "pinned": pins,
        "all_distributions": live,
        "installation_evidence": str(installation_path.relative_to(ROOT)),
    }


def expected_collection():
    path = BASE / "final/collection.log"
    nodeids = [
        line for line in path.read_text().splitlines() if line.startswith("tests/") and "::" in line
    ]
    require(
        len(nodeids) == len(set(nodeids)) == EXPECTED_COLLECTED,
        "final collection reference is incomplete or contains duplicate IDs",
    )
    result = {}
    for nodeid in nodeids:
        result.setdefault(nodeid.split("::", 1)[0], []).append(nodeid)
    result[OPTIONAL_SKIP] = []
    return result


def junit_cases(path):
    tree = ET.parse(path).getroot()
    cases = list(tree.iter("testcase"))
    require(cases, f"empty JUnit report: {path}")
    require(
        not list(tree.iter("failure")) and not list(tree.iter("error")),
        f"JUnit contains failures: {path}",
    )
    return cases


def test_evidence(directory, row, expected_nodeids=None):
    name = row["file"]
    candidates = list(directory.glob(f"*-{Path(name).stem}.collection.json"))
    require(len(candidates) == 1, f"missing/ambiguous collection manifest: {name}")
    manifest_path = candidates[0]
    manifest = read(manifest_path)
    junit = manifest_path.with_name(manifest_path.name.replace(".collection.json", ".xml"))
    log = manifest_path.with_name(manifest_path.name.replace(".collection.json", ".log"))
    require(log.is_file() and log.stat().st_size > 0, f"missing test log: {name}")
    require(
        row["command"][8] == str(manifest_path)
        and row["command"][10] == str(junit)
        and row["log"] == str(log),
        f"worker output paths differ from evidence: {name}",
    )
    with log.open() as stream:
        require(
            stream.readline().rstrip("\n") == "Command: " + json.dumps(row["command"]),
            f"logged worker command differs from result: {name}",
        )
    require(
        not row["timed_out"] and row["returncode"] in {0, 5}, f"unsuccessful test process: {name}"
    )
    require(manifest.get("exitstatus") == row["returncode"], f"exit status mismatch: {name}")
    require(not manifest.get("deselected"), f"deselected tests: {name}")
    nodeids = manifest["nodeids"]
    require(len(set(nodeids)) == len(nodeids), f"duplicate collected IDs: {name}")
    require(all(node.startswith(name + "::") for node in nodeids), f"foreign test IDs: {name}")
    if expected_nodeids is not None:
        require(
            set(nodeids) == set(expected_nodeids),
            f"collection differs from final reference: {name}",
        )
    skipped_collection = manifest["collection_reports"]
    if skipped_collection:
        require(
            name == OPTIONAL_SKIP and not nodeids and row["returncode"] == 5,
            f"unexpected collection skip: {name}",
        )
        require(
            len(skipped_collection) == 1
            and skipped_collection[0]["outcome"] == "skipped"
            and skipped_collection[0]["nodeid"] == name,
            f"invalid collection skip: {name}",
        )
        require(
            "could not import 'ott': No module named 'ott'" in skipped_collection[0]["reason"],
            "optional-backend skip lacks reason",
        )
    else:
        require(nodeids and row["returncode"] == 0, f"empty or unsuccessful collection: {name}")
    reports = {}
    for item in manifest["runtime_reports"]:
        require(item["nodeid"] in nodeids, f"uncollected runtime test: {name}")
        require(
            math.isfinite(item["seconds"]) and item["seconds"] >= 0,
            f"invalid execution duration: {name}",
        )
        phases = reports.setdefault(item["nodeid"], {})
        require(item["when"] not in phases, f"duplicate execution phase: {name}")
        phases[item["when"]] = item["outcome"]
    require(set(reports) == set(nodeids), f"unexecuted collected test: {name}")
    runtime_skips = set()
    for nodeid, phases in reports.items():
        require(phases.get("teardown") == "passed", f"teardown incomplete: {nodeid}")
        require(
            (
                set(phases) == {"setup", "call", "teardown"}
                and phases.get("setup") == "passed"
                and phases.get("call") in {"passed", "skipped"}
            )
            or (set(phases) == {"setup", "teardown"} and phases.get("setup") == "skipped"),
            f"incomplete test execution: {nodeid}",
        )
        if "skipped" in phases.values():
            runtime_skips.add(nodeid)
    expected_cases = Counter()
    nodeid_by_case = {}
    for nodeid in nodeids:
        parts = nodeid.split("::")
        classname = parts[0][:-3].replace("/", ".") + "".join("." + part for part in parts[1:-1])
        expected_cases[classname, parts[-1]] += 1
        nodeid_by_case[classname, parts[-1]] = nodeid
    if skipped_collection:
        expected_cases["", name[:-3].replace("/", ".")] += 1
        nodeid_by_case["", name[:-3].replace("/", ".")] = name
    cases = junit_cases(junit)
    require(
        Counter((case.get("classname", ""), case.get("name")) for case in cases) == expected_cases,
        f"JUnit identities differ from collection: {name}",
    )
    skips = [
        {
            "file": name,
            "nodeid": nodeid_by_case[case.get("classname", ""), case.get("name")],
            "test": case.get("name"),
            "reason": case.find("skipped").get("message", ""),
            "detail": case.find("skipped").text,
        }
        for case in cases
        if case.find("skipped") is not None
    ]
    require(
        {skip["nodeid"] for skip in skips}
        == runtime_skips | ({name} if skipped_collection else set()),
        f"JUnit skip identities differ from execution: {name}",
    )
    counts = {
        "collected": len(nodeids),
        "junit_cases": len(cases),
        "passed": len(cases) - len(skips),
        "skipped": len(skips),
    }
    for key, value in counts.items():
        require(row.get(key, 0) == value, f"stored {key} count mismatch: {name}")
    return counts, skips


def coverage_evidence(directory, installed, expected_package):
    data = read(directory / "coverage.json")
    require(data["meta"]["branch_coverage"] is True, "coverage lacks branches")
    files = {}
    for path, item in data["files"].items():
        relative = str(Path(path).relative_to(installed))
        require(relative not in files, "duplicate coverage module")
        for key in ("executed_lines", "missing_lines", "excluded_lines"):
            values = item[key]
            require(
                len(values) == len(set(values))
                and all(type(value) is int and value > 0 for value in values),
                f"invalid or duplicate coverage lines: {relative}/{key}",
            )
        for key in ("executed_branches", "missing_branches"):
            values = item[key]
            require(
                all(
                    len(value) == 2 and all(type(part) is int for part in value) for value in values
                )
                and len(values) == len(set(map(tuple, values))),
                f"invalid or duplicate coverage branches: {relative}/{key}",
            )
        lines = set(item["executed_lines"]) - set(item["excluded_lines"])
        missing = set(item["missing_lines"])
        branches = set(map(tuple, item["executed_branches"]))
        missing_branches = set(map(tuple, item["missing_branches"]))
        require(
            not lines & missing
            and not branches & missing_branches
            and not missing & set(item["excluded_lines"]),
            f"overlapping coverage evidence: {relative}",
        )
        counts = {
            "covered_lines": len(lines),
            "num_statements": len(lines) + len(missing),
            "covered_branches": len(branches),
            "num_branches": len(branches) + len(missing_branches),
        }
        require(
            all(item["summary"][key] == value for key, value in counts.items()),
            f"coverage summary differs from line/branch evidence: {relative}",
        )
        files[relative] = counts
    expected = {name for name in expected_package if name.endswith(".py")} - EXCLUDED_EXAMPLES
    require(
        set(files) == expected and len(files) == 64,
        "coverage omits core modules or includes unexpected modules",
    )
    learning = {name: counts for name, counts in files.items() if name.startswith("learning/")}
    require(len(learning) == 21, "learning coverage must include all 21 modules")
    result = {}
    for name, rows, threshold in [("global", files, 85), ("learning", learning, 95)]:
        combined = sum((Counter(row) for row in rows.values()), Counter())
        totals = {
            key: combined[key]
            for key in ("covered_lines", "num_statements", "covered_branches", "num_branches")
        }
        numerator = totals["covered_lines"] + totals["covered_branches"]
        denominator = totals["num_statements"] + totals["num_branches"]
        require(denominator > 0, f"empty {name} coverage")
        require(100 * numerator >= threshold * denominator, f"{name} coverage below {threshold}%")
        result[name] = {
            **totals,
            "percent": 100 * numerator / denominator,
            "threshold": threshold,
            "module_count": len(rows),
        }
    require(
        all(
            data["totals"][key] == value
            for key, value in result["global"].items()
            if key in {"covered_lines", "num_statements", "covered_branches", "num_branches"}
        ),
        "coverage totals mismatch",
    )
    require(
        math.isclose(
            data["totals"]["percent_covered"], result["global"]["percent"], rel_tol=0, abs_tol=1e-10
        ),
        "coverage percentage mismatch",
    )
    result["excluded_examples"] = sorted(EXCLUDED_EXAMPLES)
    return result


def validate_environment(name, current, matrix_directory, expected_collected):
    require(expected_collected == EXPECTED_COLLECTED, "final suite must collect exactly 1147 tests")
    directory = matrix_directory / name
    path = directory / "results.json"
    if not path.exists():
        return {
            "environment": name,
            "status": "pending",
            "reason": "run has not produced results",
            "evidence": str(path.relative_to(ROOT)),
        }
    summary = read(path)
    if summary["status"] == "running":
        return {
            "environment": name,
            "status": "pending",
            "completed_files": len(summary["files"]),
            "active_file": summary.get("active_file"),
            "evidence": str(path.relative_to(ROOT)),
        }
    require(
        summary["status"] == "passed" and summary["source_unchanged"],
        f"selected full run failed: {name}: {summary.get('error')}",
    )
    require(
        math.isfinite(summary["seconds"]) and summary["seconds"] >= 0,
        f"invalid full-suite duration: {name}",
    )
    before, after = (
        read(directory / "source_hashes_before.json"),
        read(directory / "source_hashes_after.json"),
    )
    require(before == after, f"source changed during {name}")
    differences = {
        key for key in before.keys() | current.keys() if before.get(key) != current.get(key)
    }
    require(
        not differences,
        f"unexpected source differences for {name}: {sorted(differences)}",
    )
    installed = Path(summary["installed_package"])
    require(
        ROOT / ".tox" / name in installed.parents and "site-packages" in installed.parts,
        f"wrong installed path: {name}",
    )
    expected_package = {
        key.removeprefix("geojax/"): value
        for key, value in current.items()
        if key.startswith("geojax/")
    }
    manifest = read(directory / "installed_package_hashes.json")
    require(
        manifest == expected_package and len(manifest) == 67,
        f"installed-package hash manifest mismatch: {name}",
    )
    require(
        set(manifest)
        == {str(path.relative_to(installed)) for path in installed.rglob("*.py")} | {"py.typed"}
        and all(
            (installed / key).is_file() and sha(installed / key) == value
            for key, value in manifest.items()
        ),
        f"installed package has changed: {name}",
    )
    dependencies = dependency_evidence(name, summary, installed, expected_package)
    executable = str(ROOT / ".tox" / name / "bin/python")
    require(summary["executable"] == executable, f"wrong test interpreter: {name}")
    work = Path(summary["working_directory"]).resolve()
    require(
        work != ROOT and ROOT not in work.parents, f"working directory was inside checkout: {name}"
    )
    files = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "tests").rglob("test_*.py"))
    require(
        len(files) == 41
        and summary["test_files"] == files
        and [row["file"] for row in summary["files"]] == files,
        f"incomplete full suite: {name}",
    )
    reference = expected_collection()
    require(set(reference) == set(files), "final collection reference must cover all 41 files")
    totals, skips = Counter(), []
    for row in summary["files"]:
        require(
            math.isfinite(row["seconds"]) and row["seconds"] >= 0,
            f"invalid file duration: {name}/{row['file']}",
        )
        command = row["command"]
        require(
            command[:3] == [executable, "-I", str(ROOT / "scripts/run_test_suite.py")]
            and command[3:7] == ["--repository", str(ROOT), "--worker", str(ROOT / row["file"])]
            and len(command) == 11
            and command[7] == "--manifest"
            and command[9] == "--junit",
            f"unexpected worker command: {name}/{row['file']}",
        )
        counts, reasons = test_evidence(directory, row, reference[row["file"]])
        totals.update(counts)
        skips.extend(reasons)
    require(dict(totals) == summary["totals"], f"aggregate test counts differ: {name}")
    require(
        totals["collected"] == expected_collected
        and totals["junit_cases"] == expected_collected + 1
        and totals["passed"] == expected_collected - (not name.endswith("float64")),
        f"unexpected collected, executed, or JUnit count: {name}",
    )
    require(
        totals["skipped"] == (1 if name.endswith("float64") else 2),
        f"unexpected skip count: {name}",
    )
    expected_skips = {OPTIONAL_SKIP} | ({PRECISION_SKIP} if name.endswith("float32") else set())
    require({row["nodeid"] for row in skips} == expected_skips, f"unexpected skipped test: {name}")
    for skip in skips:
        if skip["nodeid"] == OPTIONAL_SKIP:
            require(
                skip["reason"] == "collection skipped"
                and "could not import 'ott': No module named 'ott'" in skip["detail"],
                f"unexpected optional-backend skip reason: {name}",
            )
        else:
            require(
                skip["reason"] == "float64 is disabled in this test matrix entry",
                f"unexpected precision skip reason: {name}",
            )
    require(
        [row["name"] for row in summary["checks"]]
        == ["coverage-combine", "coverage-json", "coverage-global", "coverage-learning"],
        f"missing coverage gates: {name}",
    )
    for check in summary["checks"]:
        require(
            check["returncode"] == 0 and not check["timed_out"], f"coverage process failed: {name}"
        )
        require((directory / f"{check['name']}.log").is_file(), f"missing coverage log: {name}")
    require(
        "--fail-under=85" in summary["checks"][2]["command"]
        and "--fail-under=95" in summary["checks"][3]["command"]
        and "--include=*/geojax/learning/*" in summary["checks"][3]["command"],
        "coverage thresholds were changed",
    )
    coverage = coverage_evidence(directory, installed, expected_package)
    factor = name.split("-")[0][2:]
    python_version = f"{factor[0]}.{factor[1:]}"
    require(summary["python"].startswith(python_version + "."), f"wrong Python version: {name}")
    require(
        summary["precision_x64"] == ("1" if name.endswith("float64") else "0"),
        f"wrong precision: {name}",
    )
    return {
        "environment": name,
        "status": "passed",
        "evidence": str(path.relative_to(ROOT)),
        "python": summary["python"],
        "versions": summary["versions"],
        "dependency_verification": dependencies,
        "precision_x64": summary["precision_x64"],
        "counts": dict(totals),
        "test_files": 41,
        "coverage": coverage,
        "seconds": summary["seconds"],
        "skips": skips,
        "installed_hash_count": 67,
        "runner_sha256": before["scripts/run_test_suite.py"],
        "collection_reference_sha256": sha(BASE / "final/collection.log"),
        "provenance": "complete final-source installed-wheel run",
        "source_unchanged_during_run": True,
        "differences_from_current": sorted(differences),
        "source_manifest_sha256": sha(directory / "source_hashes_before.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "final_matrix_verification.json")
    parser.add_argument("--matrix", type=Path, default=BASE / "matrix_final_a")
    parser.add_argument("--second-batch", type=Path, default=BASE / "matrix_final_b")
    parser.add_argument("--pilot", type=Path, default=BASE / "matrix_final_pilot")
    parser.add_argument("--expected-collected", type=int, required=True)
    arguments = parser.parse_args()
    require(
        arguments.expected_collected == EXPECTED_COLLECTED,
        "final suite must collect exactly 1147 tests",
    )
    report = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "environments": [],
        "errors": [],
    }
    try:
        config = configparser.ConfigParser()
        config.read(ROOT / "tox.ini")
        require(
            set(config["tox"]["env_list"].split()) == ENVIRONMENTS, "tox environment matrix changed"
        )
        current = current_hashes()
        for name in sorted(ENVIRONMENTS):
            try:
                directory = (
                    arguments.pilot
                    if name == "py311-min-float32"
                    else arguments.matrix
                    if name in FIRST_BATCH
                    else arguments.second_batch
                )
                report["environments"].append(
                    validate_environment(
                        name, current, directory.resolve(), arguments.expected_collected
                    )
                )
            except Exception as error:
                report["environments"].append(
                    {"environment": name, "status": "failed", "error": str(error)}
                )
                report["errors"].append(f"{name}: {error}")
        report["status"] = (
            "failed"
            if report["errors"]
            else "pending"
            if any(row["status"] == "pending" for row in report["environments"])
            else "passed"
        )
        report["verified_environments"] = sum(
            row["status"] == "passed" for row in report["environments"]
        )
        report["expected_environments"] = 10
        report["current_runner_sha256"] = current["scripts/run_test_suite.py"]
        report["note"] = (
            "Only complete runs against byte-identical final library, tests, and runner source are selected. Earlier source snapshots, interrupted runs, and failed runs remain historical evidence."
        )
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append(str(error))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(arguments.output)
    print(
        f"{report['status']}: {report.get('verified_environments', 0)}/10 environments verified; {arguments.output}"
    )
    for error in report["errors"]:
        print(error, file=sys.stderr)
    return {"passed": 0, "failed": 1, "pending": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
