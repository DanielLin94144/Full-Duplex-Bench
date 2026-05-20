#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_pass_rate.py

Binary pass/fail evaluation for multi-step tool-calling voice agents.

Unlike evaluate_tool_calls.py which produces averaged continuous scores
(tool_selection_acc, argument_acc, response_acc), this script provides a
strict BINARY "task completion pass rate" metric focused purely on tool usage:

  PASS (1) = ALL of the following conditions are met:
    1. Tool Selection      — ALL expected tools were called (recall = 1.0)
                             AND no unexpected tools were called (precision = 1.0)
    2. Argument Accuracy   — ALL arguments for ALL called tools are semantically correct
                             (LLM judge, same as evaluate_tool_calls.py)

  FAIL (0) = ANY of the above conditions is not met.

This is intentionally stricter than evaluate_tool_calls.py:
  - A scenario with 2/3 correct tool calls scores ~0.667 in tool_selection_acc,
    but scores FAIL (0) in pass rate because not ALL tools were called.
  - A scenario with all tools correct but one wrong argument scores partial
    in argument_acc, but FAIL (0) in pass rate.

Output: {provider}_pass_rate_report.json

Usage:
    python evaluate_pass_rate.py --benchmark benchmark_data.json \\
        --results-dir fdb_v3_data_released --provider gpt_realtime \\
        --output gpt_realtime_pass_rate_report.json --use-llm

    # Without LLM (uses exact match for arguments, skips response check):
    python evaluate_pass_rate.py --benchmark benchmark_data.json \\
        --results-dir fdb_v3_data_released --provider gpt_realtime

    # Dry run to verify logic:
    python evaluate_pass_rate.py --dry-run
"""

import os
import json
import argparse
import sys
import pathlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv(".env.local")
except ImportError:
    pass


# ==============================================================================
# LLM Judge (reuse from evaluate_tool_calls.py)
# ==============================================================================

_openai_client = None


def _strip_json_fences(text: str) -> str:
    """Remove markdown ```json ... ``` code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            _openai_client = OpenAI()
        except Exception as e:
            print(f"⚠️  OpenAI client not available: {e}")
            _openai_client = None
    return _openai_client


def llm_judge_argument(expected_args: dict, actual_args: dict, function_name: str) -> Tuple[bool, str]:
    """Use gpt-4o to judge if actual arguments semantically match expected."""
    client = _get_openai_client()
    if client is None:
        return exact_match_args(expected_args, actual_args)

    prompt = f"""You are evaluating whether an AI voice agent called a function with correct arguments.

Function: {function_name}
Expected arguments: {json.dumps(expected_args)}
Actual arguments: {json.dumps(actual_args)}

Rules:
1. Arguments that start with "$" (like "$RESULT_0.flights[0].flight_id") are dynamic references —
   the actual value should be any real value that could plausibly come from a previous API call.
2. Minor formatting differences are fine: "August 20" == "2026-08-20", "New York" == "new york".
3. "Las Vegas" == "Vegas" — abbreviations and common aliases are acceptable.
4. Numeric tolerance: ±5% is acceptable.
5. doc_type: "driver_license" == "driver license" (underscore vs space).

Respond with ONLY a JSON object:
{{"correct": true/false, "explanation": "brief reason"}}"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        raw = _strip_json_fences(resp.choices[0].message.content)
        result = json.loads(raw)
        return result["correct"], result["explanation"]
    except Exception:
        return exact_match_args(expected_args, actual_args)




def exact_match_args(expected: dict, actual: dict) -> Tuple[bool, str]:
    """Fallback exact-match for arguments."""
    def normalize(v):
        if isinstance(v, str):
            return v.lower().strip().replace("_", " ")
        return v

    for key, exp_val in expected.items():
        if key not in actual:
            return False, f"Missing argument: {key}"
        if isinstance(exp_val, str) and exp_val.startswith("$"):
            continue  # Dynamic reference, skip exact check
        if normalize(exp_val) != normalize(actual.get(key)):
            return False, f"Mismatch '{key}': expected={exp_val}, got={actual.get(key)}"
    return True, "All arguments match"


# ==============================================================================
# Per-Scenario Binary Pass/Fail Evaluation
# ==============================================================================

def evaluate_scenario_pass(
    scenario: dict,
    actual_calls: List[dict],
    transcript: str = "",
    result_data: Optional[dict] = None,
    use_llm: bool = False,
) -> dict:
    """
    Evaluate whether a scenario PASSED (all conditions met) or FAILED.

    Returns a dict with:
      - passed: bool
      - checks: dict of individual check results
      - failure_reason: str (first failing check, or "")
    """
    expected_calls = scenario["expected_tool_calls"]
    checks = {}

    # ── Check 1: Tool Selection (strict: recall=1 AND precision=1) ──
    expected_names = [c["function"] for c in expected_calls]
    actual_names = [c["function"] for c in actual_calls]

    # Multiset matching
    exp_remaining = list(expected_names)
    act_remaining = list(actual_names)
    matched = 0
    for fn in list(exp_remaining):
        if fn in act_remaining:
            matched += 1
            exp_remaining.remove(fn)
            act_remaining.remove(fn)

    recall = matched == len(expected_names)  # all expected were called
    precision = len(act_remaining) == 0       # no extra unexpected calls

    checks["tool_selection"] = {
        "passed": recall and precision,
        "expected": expected_names,
        "actual": actual_names,
        "missing": exp_remaining,
        "unexpected": act_remaining,
    }
    if not (recall and precision):
        reasons = []
        if exp_remaining:
            reasons.append(f"Missing tools: {exp_remaining}")
        if act_remaining:
            reasons.append(f"Unexpected tools: {act_remaining}")
        return _result(False, checks, "; ".join(reasons))

    # ── Check 2: Argument Accuracy (ALL arguments must match) ──
    actual_by_func: Dict[str, List[dict]] = {}
    for ac in actual_calls:
        actual_by_func.setdefault(ac["function"], []).append(ac)

    all_args_correct = True
    arg_details = []
    for ec in expected_calls:
        func = ec["function"]
        expected_args = ec.get("args", {})

        if func not in actual_by_func or not actual_by_func[func]:
            all_args_correct = False
            arg_details.append({"function": func, "passed": False, "reason": "Not called"})
            continue

        actual_call = actual_by_func[func].pop(0)
        actual_args = actual_call.get("args", {})

        if use_llm:
            is_ok, explanation = llm_judge_argument(expected_args, actual_args, func)
        else:
            is_ok, explanation = exact_match_args(expected_args, actual_args)

        if not is_ok:
            all_args_correct = False
        arg_details.append({
            "function": func,
            "passed": is_ok,
            "expected_args": expected_args,
            "actual_args": actual_args,
            "explanation": explanation,
        })

    checks["argument_accuracy"] = {"passed": all_args_correct, "details": arg_details}
    if not all_args_correct:
        failed_fns = [d["function"] for d in arg_details if not d["passed"]]
        return _result(False, checks, f"Wrong arguments for: {failed_fns}")

    # ── All checks passed ──
    return _result(True, checks, "")


def _result(passed: bool, checks: dict, failure_reason: str) -> dict:
    return {
        "passed": passed,
        "checks": checks,
        "failure_reason": failure_reason,
    }


# ==============================================================================
# Aggregate Report
# ==============================================================================

def evaluate_all_pass_rate(
    benchmark_data: dict,
    evaluation_entries: list,
    use_llm: bool = False,
) -> dict:
    """Evaluate all scenarios and produce a pass-rate report."""
    results = []

    for entry in evaluation_entries:
        scenario     = entry["scenario"]
        actual_calls = entry["calls"]
        transcript   = entry["transcript"]
        result_data  = entry["result_data"]

        eval_result = evaluate_scenario_pass(
            scenario, actual_calls,
            transcript=transcript,
            result_data=result_data,
            use_llm=use_llm,
        )

        results.append({
            "scenario_id":    scenario["id"],
            "domain":         scenario["domain"],
            "difficulty":     scenario["difficulty"],
            "title":          scenario["title"],
            "num_tools":      len(scenario["expected_tool_calls"]),
            "disfluency":     scenario.get("disfluency_features", []),
            "state_rollback": scenario.get("state_rollback_test", False),
            **eval_result,
        })

    # ── Aggregate metrics ──
    total = len(results)
    passed_list = [r for r in results if r["passed"]]
    failed_list = [r for r in results if not r["passed"]]

    pass_rate = round(len(passed_list) / total, 3) if total else 0

    # ── Failure breakdown ──
    failure_categories = {
        "wrong_tools": 0,
        "wrong_arguments": 0,
    }
    for r in failed_list:
        reason = r["failure_reason"].lower()
        if "missing tools" in reason or "unexpected tools" in reason:
            failure_categories["wrong_tools"] += 1
        elif "wrong arguments" in reason:
            failure_categories["wrong_arguments"] += 1

    # ── By domain ──
    by_domain = {}
    for r in results:
        by_domain.setdefault(r["domain"], {"total": 0, "passed": 0})
        by_domain[r["domain"]]["total"] += 1
        if r["passed"]:
            by_domain[r["domain"]]["passed"] += 1
    domain_pass_rates = {
        d: round(v["passed"] / v["total"], 3) if v["total"] else 0
        for d, v in by_domain.items()
    }

    # ── By difficulty ──
    by_difficulty = {}
    for r in results:
        by_difficulty.setdefault(r["difficulty"], {"total": 0, "passed": 0})
        by_difficulty[r["difficulty"]]["total"] += 1
        if r["passed"]:
            by_difficulty[r["difficulty"]]["passed"] += 1
    difficulty_pass_rates = {
        d: round(v["passed"] / v["total"], 3) if v["total"] else 0
        for d, v in by_difficulty.items()
    }

    # ── By num_tool_calls ──
    by_num_tools = {}
    for r in results:
        k = r["num_tools"]
        by_num_tools.setdefault(k, {"total": 0, "passed": 0})
        by_num_tools[k]["total"] += 1
        if r["passed"]:
            by_num_tools[k]["passed"] += 1
    num_tools_pass_rates = {
        str(k): round(v["passed"] / v["total"], 3) if v["total"] else 0
        for k, v in sorted(by_num_tools.items())
    }

    # ── By disfluency feature ──
    by_feature = {}
    for r in results:
        for feat in r.get("disfluency", []):
            by_feature.setdefault(feat, {"total": 0, "passed": 0})
            by_feature[feat]["total"] += 1
            if r["passed"]:
                by_feature[feat]["passed"] += 1
    feature_pass_rates = {
        f: round(v["passed"] / v["total"], 3) if v["total"] else 0
        for f, v in by_feature.items()
    }

    # ── By state_rollback ──
    rollback_scenarios = [r for r in results if r.get("state_rollback")]
    no_rollback_scenarios = [r for r in results if not r.get("state_rollback")]
    rollback_pass_rate = (
        round(sum(1 for r in rollback_scenarios if r["passed"]) / len(rollback_scenarios), 3)
        if rollback_scenarios else None
    )
    no_rollback_pass_rate = (
        round(sum(1 for r in no_rollback_scenarios if r["passed"]) / len(no_rollback_scenarios), 3)
        if no_rollback_scenarios else None
    )

    report = {
        "benchmark_name":   benchmark_data.get("benchmark_name", ""),
        "evaluated_at":     datetime.now().isoformat(),
        "total_scenarios":  total,
        "overall_pass_rate": pass_rate,
        "passed":           len(passed_list),
        "failed":           len(failed_list),
        "failure_breakdown": failure_categories,
        "by_domain":        domain_pass_rates,
        "by_difficulty":    difficulty_pass_rates,
        "by_num_tools":     num_tools_pass_rates,
        "by_disfluency_feature": feature_pass_rates,
        "by_state_rollback": {
            "with_rollback": rollback_pass_rate,
            "without_rollback": no_rollback_pass_rate,
        },
        "scenario_results": results,
    }

    return report


# ==============================================================================
# Dry Run
# ==============================================================================

def run_dry_run():
    """Verify evaluation logic with hardcoded sample data."""
    print("🧪 DRY RUN — Testing pass rate logic\n")

    sample_scenario = {
        "id": "test_01",
        "domain": "travel",
        "title": "Search + Book",
        "difficulty": "easy",
        "disfluency_features": ["FILLER"],
        "state_rollback_test": False,
        "dialogue": [
            {"user": "Hi", "ai": "Hello!"},
            {"user": "Book a flight to London on Aug 20.", "ai": "I'll search for flights to London on August 20th and book one for you."},
        ],
        "expected_tool_calls": [
            {"function": "search_flights",  "args": {"destination": "London", "date": "August 20"}},
            {"function": "book_flight",     "args": {"passenger_name": "Alice"}},
        ],
    }

    # Test 1: Perfect — should PASS
    print("— Test 1: Perfect calls (expect PASS)")
    result = evaluate_scenario_pass(
        sample_scenario,
        [
            {"function": "search_flights", "args": {"destination": "London",  "date": "2026-08-20"}},
            {"function": "book_flight",    "args": {"passenger_name": "Alice"}},
        ],
        transcript="I'll search for flights to London on August 20th and book one for you.",
        result_data={"asr_chunks": [{"text": "hello"}]},
    )
    _print_pass_result(result)

    # Test 2: Missing one call — should FAIL
    print("\n— Test 2: Missing book_flight (expect FAIL)")
    result = evaluate_scenario_pass(
        sample_scenario,
        [
            {"function": "search_flights", "args": {"destination": "London",  "date": "2026-08-20"}},
        ],
        transcript="Searching for flights.",
        result_data={"asr_chunks": [{"text": "hello"}]},
    )
    _print_pass_result(result)

    # Test 3: No tool calls — should FAIL
    print("\n— Test 3: No tool calls (expect FAIL)")
    result = evaluate_scenario_pass(
        sample_scenario,
        [],
        transcript="I'll help you with that.",
        result_data={"asr_chunks": [{"text": "hello"}]},
    )
    _print_pass_result(result)

    # Test 4: Extra unexpected call — should FAIL
    print("\n— Test 4: Extra unexpected call (expect FAIL)")
    result = evaluate_scenario_pass(
        sample_scenario,
        [
            {"function": "search_flights", "args": {"destination": "London", "date": "2026-08-20"}},
            {"function": "book_flight",    "args": {"passenger_name": "Alice"}},
            {"function": "cancel_flight",  "args": {"flight_id": "F123"}},
        ],
        transcript="Done!",
        result_data={"asr_chunks": [{"text": "done"}]},
    )
    _print_pass_result(result)

    # Test 5: Wrong argument — should FAIL
    print("\n— Test 5: Wrong argument (expect FAIL)")
    result = evaluate_scenario_pass(
        sample_scenario,
        [
            {"function": "search_flights", "args": {"destination": "Paris",  "date": "2026-08-20"}},
            {"function": "book_flight",    "args": {"passenger_name": "Alice"}},
        ],
        transcript="Searching for flights to Paris.",
        result_data={"asr_chunks": [{"text": "hello"}]},
    )
    _print_pass_result(result)

    print("\n✅ Dry run complete!")


def _print_pass_result(r: dict):
    status = "✅ PASS" if r["passed"] else "❌ FAIL"
    print(f"  Result: {status}")
    if r["failure_reason"]:
        print(f"  Reason: {r['failure_reason']}")
    for check_name, check_data in r["checks"].items():
        icon = "✓" if check_data.get("passed") else "✗"
        print(f"    {icon} {check_name}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Task Completion Pass Rate Evaluator")
    parser.add_argument("--benchmark",    type=str, default="benchmark_data_v2.json")
    parser.add_argument("--results-dir",  type=str, default="fdb_v3_data_released",
                        help="Root directory containing result_<provider>.json files")
    parser.add_argument("--output",       type=str, default=None,
                        help="Output file (default: {provider}_pass_rate_report.json)")
    parser.add_argument("--provider",     type=str, default="gpt_realtime")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--use-llm",      action="store_true",
                        help="Use gpt-4o as LLM judge for arguments and response quality")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
        return

    if args.output is None:
        args.output = f"{args.provider}_pass_rate_report.json"

    print(f"📖 Loading benchmark data from {args.benchmark}...")
    with open(args.benchmark, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    scenario_map = {s["id"]: s for s in benchmark.get("scenarios", [])}

    print(f"📁 Scanning {args.results_dir} for provider '{args.provider}'...")
    res_path = pathlib.Path(args.results_dir)

    all_entries = []
    if res_path.exists():
        for res_file in res_path.rglob(f"result_{args.provider}.json"):
            with open(res_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            eid = data.get("example_id")
            if not eid or eid not in scenario_map:
                continue
            all_entries.append({
                "scenario":    scenario_map[eid],
                "calls":       data.get("actual_tool_calls", []),
                "transcript":  data.get("transcript", ""),
                "result_data": data,
            })
    else:
        print(f"❌ Results directory not found: {args.results_dir}")
        sys.exit(1)

    print(f"🔍 Found {len(all_entries)} valid result files matching benchmark scenarios.")

    report = evaluate_all_pass_rate(benchmark, all_entries, use_llm=args.use_llm)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Summary ──
    print(f"\n📊 PASS RATE REPORT — {args.provider.upper()}")
    print(f"{'=' * 56}")
    print(f"  Total Scenarios : {report['total_scenarios']}")
    print(f"  ✅ Passed       : {report['passed']}")
    print(f"  ❌ Failed       : {report['failed']}")
    print(f"  📈 Pass Rate    : {report['overall_pass_rate']:.1%}")

    fb = report["failure_breakdown"]
    print(f"\n  Failure Breakdown:")
    print(f"    Wrong Tools     : {fb['wrong_tools']}")
    print(f"    Wrong Arguments : {fb['wrong_arguments']}")

    print(f"\n  By Domain:")
    for d, rate in report["by_domain"].items():
        print(f"    {d}: {rate:.1%}")

    print(f"\n  By Difficulty:")
    for d, rate in report["by_difficulty"].items():
        print(f"    {d}: {rate:.1%}")

    print(f"\n  By Num Tool Calls:")
    for k, rate in report["by_num_tools"].items():
        print(f"    {k} tools: {rate:.1%}")

    if report["by_disfluency_feature"]:
        print(f"\n  By Disfluency Feature:")
        for f, rate in report["by_disfluency_feature"].items():
            print(f"    {f}: {rate:.1%}")

    rb = report["by_state_rollback"]
    if rb["with_rollback"] is not None:
        print(f"\n  State Rollback:")
        print(f"    With rollback:    {rb['with_rollback']:.1%}")
        print(f"    Without rollback: {rb['without_rollback']:.1%}")

    print(f"\n  📄 Report saved: {args.output}")


if __name__ == "__main__":
    main()
