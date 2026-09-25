"""Corrupt evidence in memory to test the final matrix validator, without numerical work."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "matrix_validator", Path(__file__).with_name("verify_final_matrix.py")
)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def recount_coverage(data):
    totals = Counter()
    for item in data["files"].values():
        counts = {
            "covered_lines": len(set(item["executed_lines"]) - set(item["excluded_lines"])),
            "num_statements": len(set(item["executed_lines"]) - set(item["excluded_lines"]))
            + len(item["missing_lines"]),
            "covered_branches": len(item["executed_branches"]),
            "num_branches": len(item["executed_branches"]) + len(item["missing_branches"]),
        }
        item["summary"].update(counts)
        totals.update(counts)
    data["totals"].update(totals)
    data["totals"]["percent_covered"] = (
        100
        * (totals["covered_lines"] + totals["covered_branches"])
        / (totals["num_statements"] + totals["num_branches"])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=validator.BASE / "matrix_final_pilot")
    parser.add_argument("--environment", default="py311-min-float32")
    parser.add_argument(
        "--output", type=Path, default=validator.BASE / "final_matrix_validator_selftest.json"
    )
    args = parser.parse_args()
    directory = args.matrix.resolve() / args.environment
    summary_path = directory / "results.json"
    report = {
        "status": "pending",
        "environment": args.environment,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "cases": [],
    }
    if not summary_path.exists() or validator.read(summary_path)["status"] != "passed":
        print("Pending: a completed passing final-source environment is required")
        return 2
    current = validator.current_hashes()
    source_before = deepcopy(current)
    evidence_before = {
        str(path): validator.sha(path) for path in directory.iterdir() if path.is_file()
    }
    started = time.monotonic()
    baseline = validator.validate_environment(
        args.environment, current, args.matrix.resolve(), 1147
    )
    report["baseline"] = {
        "status": baseline["status"],
        "counts": baseline["counts"],
        "coverage": baseline["coverage"],
    }
    original_read = validator.read
    original_junit = validator.junit_cases
    first_manifest = directory / "001-test_benchmarks.collection.json"
    metric_manifest = directory / "015-test_learning_embedding_metric.collection.json"
    metric_junit = directory / "015-test_learning_embedding_metric.xml"

    def rejection(name, mutate=None, expected="", junit_mutate=None, live_mutate=None):
        def read_overlay(path):
            value = original_read(path)
            if mutate is not None:
                mutate(Path(path), value)
            return value

        def junit_overlay(path):
            cases = deepcopy(original_junit(path))
            if junit_mutate is not None:
                junit_mutate(Path(path), cases)
            return cases

        original_versions = validator.installed_versions

        def versions_overlay(path):
            values = original_versions(path)
            if live_mutate is not None:
                live_mutate(values)
            return values

        with (
            patch.object(validator, "read", side_effect=read_overlay),
            patch.object(validator, "junit_cases", side_effect=junit_overlay),
            patch.object(validator, "installed_versions", side_effect=versions_overlay),
        ):
            try:
                validator.validate_environment(
                    args.environment, current, args.matrix.resolve(), 1147
                )
            except ValueError as error:
                if expected not in str(error):
                    raise AssertionError(
                        f"{name} rejected for unexpected reason: {error}"
                    ) from error
                report["cases"].append({"case": name, "rejected": True, "reason": str(error)})
            else:
                raise AssertionError(f"corrupted evidence was accepted: {name}")

    def target(filename, action):
        def mutate(path, value):
            if path == directory / filename:
                action(value)

        return mutate

    rejection(
        "changed source hash",
        target(
            "source_hashes_after.json",
            lambda value: value.__setitem__("geojax/learning/_statistics.py", "0" * 64),
        ),
        "source changed",
    )
    rejection(
        "corrupted installed hash",
        target(
            "installed_package_hashes.json",
            lambda value: value.__setitem__("learning/_statistics.py", "0" * 64),
        ),
        "installed-package hash manifest mismatch",
    )
    rejection(
        "deselected test",
        target(first_manifest.name, lambda value: value["deselected"].append(value["nodeids"][0])),
        "deselected tests",
    )
    rejection(
        "changed aggregate count",
        target("results.json", lambda value: value["totals"].__setitem__("collected", 1146)),
        "aggregate test counts differ",
    )
    rejection(
        "missing test file",
        target("results.json", lambda value: value["files"].pop()),
        "incomplete full suite",
    )
    rejection(
        "missing runtime call",
        target(
            first_manifest.name,
            lambda value: value.__setitem__(
                "runtime_reports",
                [row for row in value["runtime_reports"] if row["when"] != "call"],
            ),
        ),
        "incomplete test execution",
    )
    rejection(
        "duplicate test phase",
        target(
            first_manifest.name,
            lambda value: value["runtime_reports"].append(deepcopy(value["runtime_reports"][0])),
        ),
        "duplicate execution phase",
    )

    def wrong_junit_identity(path, cases):
        if path.name == "001-test_benchmarks.xml":
            cases[0].set("name", "fabricated_test")

    rejection(
        "JUnit test identity mismatch",
        expected="JUnit identities differ",
        junit_mutate=wrong_junit_identity,
    )

    if args.environment.endswith("float32"):

        def swap_runtime_skip(path, value):
            if path == metric_manifest:
                first = value["nodeids"][0]
                for row in value["runtime_reports"]:
                    if row["when"] == "call" and row["nodeid"] == validator.PRECISION_SKIP:
                        row["outcome"] = "passed"
                    elif row["when"] == "call" and row["nodeid"] == first:
                        row["outcome"] = "skipped"

        def swap_junit_skip(path, cases):
            if path == metric_junit:
                old = next(case for case in cases if case.find("skipped") is not None)
                skipped = old.find("skipped")
                old.remove(skipped)
                cases[0].append(skipped)

        rejection(
            "wrong skipped test despite correct skip count",
            swap_runtime_skip,
            "unexpected skipped test",
            swap_junit_skip,
        )
        rejection(
            "JUnit skip attached to wrong runtime test",
            expected="JUnit skip identities differ",
            junit_mutate=swap_junit_skip,
        )

    rejection(
        "wrong recorded SciPy version",
        target("results.json", lambda value: value["versions"].__setitem__("scipy", "0.0.0")),
        "wrong recorded dependency versions",
    )

    def wrong_installation_pin(path, value):
        if path.name == f"{args.environment}_installed_package.json":
            value["versions"]["ml-dtypes"] = "0.0.0"

    rejection(
        "wrong saved ml-dtypes pin", wrong_installation_pin, "installation dependency pins differ"
    )
    rejection(
        "changed live opt-einsum metadata",
        expected="installed dependencies changed",
        live_mutate=lambda value: value.__setitem__("opt-einsum", "0.0.0"),
    )
    rejection(
        "coverage missing module",
        target("coverage.json", lambda value: value["files"].pop(next(iter(value["files"])))),
        "coverage omits core modules",
    )
    rejection(
        "coverage lacks branches",
        target("coverage.json", lambda value: value["meta"].__setitem__("branch_coverage", False)),
        "coverage lacks branches",
    )
    rejection(
        "fabricated coverage totals",
        target(
            "coverage.json",
            lambda value: value["totals"].__setitem__(
                "covered_lines", value["totals"]["covered_lines"] + 1
            ),
        ),
        "coverage totals mismatch",
    )
    rejection(
        "rounded/fabricated coverage percentage",
        target(
            "coverage.json", lambda value: value["totals"].__setitem__("percent_covered", 100.0)
        ),
        "coverage percentage mismatch",
    )

    def learning_below_gate(data):
        learning = [item for path, item in data["files"].items() if "/geojax/learning/" in path]
        numerator = sum(
            item["summary"]["covered_lines"] + item["summary"]["covered_branches"]
            for item in learning
        )
        denominator = sum(
            item["summary"]["num_statements"] + item["summary"]["num_branches"] for item in learning
        )
        remove = math.floor(numerator - 0.95 * denominator) + 1
        for item in learning:
            available = sorted(set(item["executed_lines"]) - set(item["excluded_lines"]))[:remove]
            item["executed_lines"] = sorted(set(item["executed_lines"]) - set(available))
            item["missing_lines"] = sorted(set(item["missing_lines"]) | set(available))
            remove -= len(available)
            if remove == 0:
                break
        if remove:
            raise AssertionError("cannot construct below-threshold learning evidence")
        recount_coverage(data)

    rejection(
        "learning coverage below 95 despite self-consistent totals",
        target("coverage.json", learning_below_gate),
        "learning coverage below 95",
    )

    def global_below_gate(data):
        for item in data["files"].values():
            item["missing_lines"] = sorted(
                set(item["missing_lines"])
                | (set(item["executed_lines"]) - set(item["excluded_lines"]))
            )
            item["executed_lines"] = sorted(
                set(item["executed_lines"]) & set(item["excluded_lines"])
            )
            item["missing_branches"] += item["executed_branches"]
            item["executed_branches"] = []
        recount_coverage(data)

    rejection(
        "global coverage below 85 despite self-consistent totals",
        target("coverage.json", global_below_gate),
        "global coverage below 85",
    )
    rejection(
        "lowered coverage gate",
        target(
            "results.json",
            lambda value: value["checks"][2].__setitem__(
                "command",
                [
                    argument.replace("--fail-under=85", "--fail-under=84")
                    for argument in value["checks"][2]["command"]
                ],
            ),
        ),
        "coverage thresholds were changed",
    )
    evidence_after = {
        str(path): validator.sha(path) for path in directory.iterdir() if path.is_file()
    }
    if evidence_before != evidence_after or source_before != validator.current_hashes():
        raise AssertionError("selftest changed original evidence or frozen source")
    report.update(
        status="passed",
        baseline_validation_passed=True,
        original_evidence_and_source_unchanged=True,
        rejected_corruptions=len(report["cases"]),
        seconds=round(time.monotonic() - started, 3),
        validator_sha256=validator.sha(Path(validator.__file__)),
        selftest_sha256=validator.sha(Path(__file__)),
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Passed baseline plus {len(report['cases'])} negative evidence checks: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
