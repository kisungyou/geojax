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
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "audit/2026-09-25/validation"
CORRECTION = BASE / "runner_checks/watchdog_correction"
RETAINED = {"py311-min-float64", "py311-stable-float32", "py314-stable-float32"}
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
CHANGED_CONTROLS = {
    "scripts/run_test_suite.py",
    "tests/test_test_suite_runner.py",
    ".github/workflows/ci.yml",
}
EXCLUDED_EXAMPLES = {"examples/largest_eigenvector.py", "examples/spd_frechet_mean.py"}


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


def junit_cases(path):
    tree = ET.parse(path).getroot()
    cases = list(tree.iter("testcase"))
    require(cases, f"empty JUnit report: {path}")
    require(
        not list(tree.iter("failure")) and not list(tree.iter("error")),
        f"JUnit contains failures: {path}",
    )
    return cases


def test_evidence(directory, row):
    name = row["file"]
    candidates = list(directory.glob(f"*-{Path(name).stem}.collection.json"))
    require(len(candidates) == 1, f"missing/ambiguous collection manifest: {name}")
    manifest_path = candidates[0]
    manifest = read(manifest_path)
    junit = manifest_path.with_name(manifest_path.name.replace(".collection.json", ".xml"))
    log = manifest_path.with_name(manifest_path.name.replace(".collection.json", ".log"))
    require(log.is_file() and log.stat().st_size > 0, f"missing test log: {name}")
    require(
        not row["timed_out"] and row["returncode"] in {0, 5}, f"unsuccessful test process: {name}"
    )
    require(manifest.get("exitstatus") == row["returncode"], f"exit status mismatch: {name}")
    require(not manifest.get("deselected"), f"deselected tests: {name}")
    nodeids = manifest["nodeids"]
    require(len(set(nodeids)) == len(nodeids), f"duplicate collected IDs: {name}")
    require(all(node.startswith(name + "::") for node in nodeids), f"foreign test IDs: {name}")
    skipped_collection = manifest["collection_reports"]
    if skipped_collection:
        require(
            name == "tests/test_optional_transport.py" and not nodeids and row["returncode"] == 5,
            f"unexpected collection skip: {name}",
        )
        require(
            len(skipped_collection) == 1 and skipped_collection[0]["outcome"] == "skipped",
            f"invalid collection skip: {name}",
        )
        require(
            "could not import" in skipped_collection[0]["reason"]
            and "ott" in skipped_collection[0]["reason"],
            "optional-backend skip lacks reason",
        )
    else:
        require(nodeids and row["returncode"] == 0, f"empty or unsuccessful collection: {name}")
    reports = {}
    for item in manifest["runtime_reports"]:
        require(item["nodeid"] in nodeids, f"uncollected runtime test: {name}")
        phases = reports.setdefault(item["nodeid"], {})
        require(item["when"] not in phases, f"duplicate execution phase: {name}")
        phases[item["when"]] = item["outcome"]
    require(set(reports) == set(nodeids), f"unexecuted collected test: {name}")
    runtime_skips = set()
    for nodeid, phases in reports.items():
        require(phases.get("teardown") == "passed", f"teardown incomplete: {nodeid}")
        require(
            (phases.get("setup") == "passed" and phases.get("call") in {"passed", "skipped"})
            or (phases.get("setup") == "skipped" and "call" not in phases),
            f"incomplete test execution: {nodeid}",
        )
        if "skipped" in phases.values():
            runtime_skips.add(nodeid)
    expected_cases = Counter()
    for nodeid in nodeids:
        parts = nodeid.split("::")
        classname = parts[0][:-3].replace("/", ".") + "".join("." + part for part in parts[1:-1])
        expected_cases[classname, parts[-1]] += 1
    if skipped_collection:
        expected_cases["", name[:-3].replace("/", ".")] += 1
    cases = junit_cases(junit)
    require(
        Counter((case.get("classname", ""), case.get("name")) for case in cases) == expected_cases,
        f"JUnit identities differ from collection: {name}",
    )
    skips = [
        {
            "file": name,
            "test": case.get("name"),
            "reason": case.find("skipped").get("message", ""),
            "detail": case.find("skipped").text,
        }
        for case in cases
        if case.find("skipped") is not None
    ]
    require(
        len(skips) == len(runtime_skips) + len(skipped_collection),
        f"skip count differs from execution: {name}",
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
        lines = set(item["executed_lines"]) - set(item["excluded_lines"])
        missing = set(item["missing_lines"])
        branches = set(map(tuple, item["executed_branches"]))
        missing_branches = set(map(tuple, item["missing_branches"]))
        require(
            not lines & missing and not branches & missing_branches,
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
        totals = dict(sum((Counter(row) for row in rows.values()), Counter()))
        numerator = totals["covered_lines"] + totals["covered_branches"]
        denominator = totals["num_statements"] + totals["num_branches"]
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
        math.isclose(data["totals"]["percent_covered"], result["global"]["percent"], abs_tol=1e-10),
        "coverage percentage mismatch",
    )
    result["excluded_examples"] = sorted(EXCLUDED_EXAMPLES)
    return result


def validate_supplements(current):
    summary = read(CORRECTION / "corrected_harness_results.json")
    require(
        summary["status"] == "passed" and summary["source_unchanged"],
        "supplementary harness checks incomplete",
    )
    require(
        all(current[name] == value for name, value in summary["source_hashes"].items()),
        "supplementary harness checks used stale source",
    )
    require(
        {row["environment"] for row in summary["runs"]} == RETAINED and len(summary["runs"]) == 3,
        "supplementary environment mismatch",
    )
    result = {}
    for row in summary["runs"]:
        name = row["environment"]
        require(
            row["returncode"] == 0
            and "-I" in row["command"]
            and "pythonpath=" in row["command"]
            and "faulthandler_timeout=0" in row["command"],
            f"supplementary isolation missing: {name}",
        )
        cases = junit_cases(CORRECTION / f"{name}.xml")
        require(
            len(cases) == 15 and not any(case.find("skipped") is not None for case in cases),
            f"supplementary tests did not all pass: {name}",
        )
        require(
            Counter(case.get("classname") for case in cases)
            == {"tests.test_test_suite_runner": 9, "tests.test_release_contract": 6}
            and len({(case.get("classname"), case.get("name")) for case in cases}) == 15,
            f"unexpected or duplicated supplementary tests: {name}",
        )
        expected_prefix = ROOT / ".tox" / name
        require(
            expected_prefix in Path(row["metadata"]["geojax"]).parents,
            f"supplementary import outside environment: {name}",
        )
        require(
            ROOT not in Path(row["working_directory"]).resolve().parents,
            f"supplementary test cwd inside checkout: {name}",
        )
        result[name] = {
            "passed": 15,
            "seconds": row["seconds"],
            "evidence": str(CORRECTION.relative_to(ROOT) / f"{name}.xml"),
            "reported_separately_from_full_suite": True,
        }
    return result


def validate_environment(name, current, transition, supplements):
    retained = name in RETAINED
    directory = BASE / ("matrix" if retained else "matrix_after_watchdog_fix") / name
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
    before, after = (
        read(directory / "source_hashes_before.json"),
        read(directory / "source_hashes_after.json"),
    )
    require(before == after, f"source changed during {name}")
    differences = {
        key for key in before.keys() | current.keys() if before.get(key) != current.get(key)
    }
    require(
        differences == (CHANGED_CONTROLS if retained else set()),
        f"unexpected source differences for {name}: {sorted(differences)}",
    )
    if retained:
        for key in CHANGED_CONTROLS:
            require(
                before[key] == transition["changed_controls"][key]["old_sha256"]
                and current[key] == transition["changed_controls"][key]["new_sha256"],
                f"control migration hash mismatch: {key}",
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
        all(
            (installed / key).is_file() and sha(installed / key) == value
            for key, value in manifest.items()
        ),
        f"installed package has changed: {name}",
    )
    files = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "tests").rglob("test_*.py"))
    require(
        len(files) == 41
        and summary["test_files"] == files
        and [row["file"] for row in summary["files"]] == files,
        f"incomplete full suite: {name}",
    )
    totals, skips = Counter(), []
    for row in summary["files"]:
        counts, reasons = test_evidence(directory, row)
        totals.update(counts)
        skips.extend(reasons)
    require(dict(totals) == summary["totals"], f"aggregate test counts differ: {name}")
    expected_collected = 1124 if retained else 1126
    require(totals["collected"] == expected_collected, f"unexpected collected count: {name}")
    require(
        totals["skipped"] == (1 if name.endswith("float64") else 2),
        f"unexpected skip count: {name}",
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
        and "--fail-under=95" in summary["checks"][3]["command"],
        "coverage thresholds were changed",
    )
    coverage = coverage_evidence(directory, installed, expected_package)
    factor = name.split("-")[0][2:]
    python_version = f"{factor[0]}.{factor[1:]}"
    expected_jax = (
        "0.6.0" if "-min-" in name else "0.10.2" if name.startswith("py311") else "0.11.0"
    )
    expected_numpy = "1.26.4" if "-min-" in name else "2.4.4"
    require(summary["python"].startswith(python_version + "."), f"wrong Python version: {name}")
    require(
        summary["versions"]["jax"] == summary["versions"]["jaxlib"] == expected_jax
        and summary["versions"]["numpy"] == expected_numpy,
        f"wrong dependency versions: {name}",
    )
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
        "precision_x64": summary["precision_x64"],
        "counts": dict(totals),
        "test_files": 41,
        "coverage": coverage,
        "seconds": summary["seconds"],
        "skips": skips,
        "installed_hash_count": 67,
        "runner_sha256": before["scripts/run_test_suite.py"],
        "provenance": "retained complete run before diagnostic correction"
        if retained
        else "complete run after diagnostic correction",
        "source_unchanged_during_run": True,
        "differences_from_current": sorted(differences),
        "supplementary_harness_checks": supplements.get(name),
        "source_manifest_sha256": sha(directory / "source_hashes_before.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "final_matrix_verification.json")
    arguments = parser.parse_args()
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
        transition = read(CORRECTION / "transition.json")
        require(
            set(transition["changed_controls"]) == CHANGED_CONTROLS,
            "unexpected diagnostic patch scope",
        )
        require(
            transition["all_geojax_python_unchanged"]
            and transition["all_numerical_tests_unchanged"],
            "transition changed package or numerical tests",
        )
        old_tests, new_tests = (
            read(CORRECTION / "source_tests_before.json"),
            read(CORRECTION / "source_tests_after.json"),
        )
        require(
            {
                key
                for key in old_tests.keys() | new_tests.keys()
                if old_tests.get(key) != new_tests.get(key)
            }
            == {"tests/test_test_suite_runner.py"},
            "transition altered numerical source/tests",
        )
        require(
            all(current[key] == value for key, value in new_tests.items()),
            "package/tests changed since diagnostic correction",
        )
        supplements = validate_supplements(current)
        completion = read(BASE / "matrix/batch_completion.json")
        require(
            set(completion["retained_complete_environments"]) == RETAINED,
            "historical retained selection changed",
        )
        require(
            completion["runs"]["py311-min-float32"]["status"] == "failed",
            "historical watchdog failure was relabeled",
        )
        for name in sorted(ENVIRONMENTS):
            try:
                report["environments"].append(
                    validate_environment(name, current, transition, supplements)
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
        report["supplementary_harness_checks"] = supplements
        report["note"] = (
            "Full-suite counts and the 15 supplementary harness checks per retained environment are reported separately. The historical failed minimum-float32 run and stale dispatcher summary are never selected as passing evidence."
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
