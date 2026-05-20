#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_tool_calls.py

Multi-metric evaluation engine for multi-step tool calling.
Scores agents on four clearly separated metrics:

  1. tool_selection_acc  — F1 of recall & precision over expected vs actual tool calls
  2. argument_acc        — semantic correctness of arguments (LLM judge, gpt-4o)
  3. response_qual       — spoken response quality (LLM judge, gpt-4o)
  4. latency_metrics     — agent response latency + interruption rate

Latency is derived from time-aligned fields in the result JSON:
  user_speech_end_rel    : when user finished speaking (relative seconds)
  audio_agent_speech_start : when agent first spoke (relative seconds)
  latency = audio_agent_speech_start - user_speech_end_rel
  < 0  => interruption (model spoke before user finished)

Usage:
    python evaluate_tool_calls.py --benchmark benchmark_data.json \
        --results-dir fdb_v3_data_released --provider gemini2_5 \
        --output gemini2_5_evaluation_report.json --use-llm
    python evaluate_tool_calls.py --dry-run
"""

import os
import json
import argparse
import sys
import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv(".env.local")
except ImportError:
    pass


# ==============================================================================
# LLM Judge (gpt-4o)
# ==============================================================================

_openai_client = None


def _strip_json_fences(text: str) -> str:
    """Remove markdown ```json ... ``` code fences that gpt-4o sometimes adds."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop opening fence
        lines = lines[1:]
        # drop closing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _get_openai_client():
    """Lazy-init OpenAI client."""
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
    """
    Use gpt-4o to judge if actual function arguments semantically match expected.
    Returns (is_correct: bool, explanation: str).
    """
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


def llm_judge_response(expected_intent: str, actual_transcript: str) -> Tuple[float, str]:
    """
    Use gpt-4o to judge if the agent's spoken response matches the expected intent.
    Returns (score: float 0.0 or 1.0, explanation: str).
    """
    if not actual_transcript or not actual_transcript.strip():
        return 0.0, "No transcript available."

    client = _get_openai_client()
    if client is None:
        return 0.0, "OpenAI client unavailable."

    prompt = f"""You are evaluating whether an AI voice agent successfully completed the user's requested task.

Expected Task/Action: "{expected_intent}"
Actual Agent Spoken Response: "{actual_transcript}"

Evaluation criteria:
1. Did the agent perform the CORRECT actions (right tools, right parameters)?
2. Did the response indicate the task was completed or is being handled?
3. It is FINE if the agent provides MORE detail than expected (e.g., giving specific results, prices, confirmation numbers). Providing additional helpful information is NOT a penalty.
4. It is INCORRECT if the agent says it cannot perform the action, lacks tools, or refuses.
5. It is INCORRECT if the agent performs the WRONG action (e.g., wrong destination, wrong document type).
6. Partial delivery of multi-step tasks (e.g., completes 2 of 3 required steps) should be scored 0.

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
        is_correct = result.get("correct", False)
        return (1.0 if is_correct else 0.0), result.get("explanation", "")
    except Exception as e:
        return 0.0, f"LLM parsing error: {str(e)}"


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
# Metric 1: Tool Selection Accuracy (F1)
# ==============================================================================

def evaluate_tool_selection(expected_calls: List[dict], actual_calls: List[dict]) -> dict:
    """
    Score: F1 of recall and precision over tool names.

    Recall    = matched_expected / total_expected
    Precision = matched_expected / total_actual  (penalises extra calls)
    F1        = harmonic mean of recall and precision

    Handles duplicates via multiset matching.
    """
    expected_names = [c["function"] for c in expected_calls]
    actual_names   = [c["function"] for c in actual_calls]

    # Multiset intersection
    exp_remaining = list(expected_names)
    act_remaining = list(actual_names)
    matched = 0
    for fn in list(exp_remaining):
        if fn in act_remaining:
            matched += 1
            exp_remaining.remove(fn)
            act_remaining.remove(fn)

    total_expected = len(expected_names)
    total_actual   = len(actual_names)

    recall    = matched / total_expected if total_expected > 0 else 1.0
    precision = matched / total_actual   if total_actual   > 0 else 1.0

    if recall + precision > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0.0

    return {
        "score":            round(f1, 3),
        "recall":           round(recall, 3),
        "precision":        round(precision, 3),
        "matched":          matched,
        "total_expected":   total_expected,
        "total_actual":     total_actual,
        "unmatched_expected": exp_remaining,
        "unexpected_calls":   act_remaining,
    }


# ==============================================================================
# Metric 2: Argument Accuracy
# ==============================================================================

def evaluate_argument_accuracy(
    expected_calls: List[dict],
    actual_calls: List[dict],
    use_llm: bool = False,
) -> dict:
    """
    Score: For each expected call that was actually called, judge argument correctness.
    Calls that were NOT called at all score 0 but are noted separately.
    """
    if not expected_calls:
        return {"score": 1.0, "note": "No expected calls", "details": []}

    # Build map: function -> list of actual calls (FIFO for duplicates)
    actual_by_func: Dict[str, List[dict]] = {}
    for ac in actual_calls:
        actual_by_func.setdefault(ac["function"], []).append(ac)

    call_scores = []
    for ec in expected_calls:
        func = ec["function"]
        expected_args = ec.get("args", {})

        if func not in actual_by_func or not actual_by_func[func]:
            call_scores.append({
                "function": func,
                "score":    0.0,
                "reason":   "Function not called",
            })
            continue

        actual_call = actual_by_func[func].pop(0)
        actual_args = actual_call.get("args", {})

        if use_llm:
            is_ok, explanation = llm_judge_argument(expected_args, actual_args, func)
        else:
            is_ok, explanation = exact_match_args(expected_args, actual_args)

        call_scores.append({
            "function":     func,
            "score":        1.0 if is_ok else 0.0,
            "expected_args": expected_args,
            "actual_args":   actual_args,
            "explanation":   explanation,
        })

    avg = sum(c["score"] for c in call_scores) / len(call_scores) if call_scores else 0.0
    return {"score": round(avg, 3), "details": call_scores}


# ==============================================================================
# Metric 3: Response Quality
# ==============================================================================

def evaluate_response_quality(scenario: dict, transcript: str, use_llm: bool = False) -> dict:
    """
    Score: Does the agent's spoken response match the expected intent?
    Expected intent is taken from the last 'ai' turn in the scenario dialogue.
    """
    expected_intent = ""
    dialogue = scenario.get("dialogue", [])
    if dialogue:
        expected_intent = dialogue[-1].get("ai", "")

    if not use_llm:
        # Without LLM, we cannot evaluate semantics — skip gracefully
        return {"score": None, "explanation": "LLM judge disabled; response accuracy skipped."}

    score, explanation = llm_judge_response(expected_intent, transcript)
    return {"score": score, "explanation": explanation}


# ==============================================================================
# Metric 4: Latency & Interruption
# ==============================================================================

def evaluate_latency(result_data: Optional[dict]) -> dict:
    """
    Compute per-sample latency from the result JSON.

    Fields used:
      user_speech_end_rel      : float (seconds, relative to audio start)
      audio_agent_speech_start : float (seconds, relative to audio start)

    latency = audio_agent_speech_start - user_speech_end_rel
    < 0  => model started speaking BEFORE user finished => interruption
    """
    if result_data is None:
        return {"available": False}

    user_end   = result_data.get("user_speech_end_rel")
    agent_start = result_data.get("audio_agent_speech_start")

    if user_end is None or agent_start is None:
        return {"available": False, "reason": "Missing timing fields"}

    latency = round(agent_start - user_end, 3)
    is_interruption = latency < 0

    return {
        "available":             True,
        "user_speech_end_rel":   user_end,
        "audio_agent_speech_start": agent_start,
        "agent_response_latency_s": latency,
        "is_interruption":       is_interruption,
    }


# ==============================================================================
# Per-Scenario Evaluation
# ==============================================================================

def evaluate_scenario(
    scenario: dict,
    actual_calls: List[dict],
    transcript: str = "",
    result_data: Optional[dict] = None,
    use_llm: bool = False,
) -> dict:
    """Run all four metrics on a single scenario."""
    expected_calls = scenario["expected_tool_calls"]

    # Turn-taking success: did the agent produce any speech output?
    asr_chunks = []
    if result_data:
        asr_chunks = result_data.get("asr_chunks", [])
    turn_take_success = bool(transcript.strip()) or bool(asr_chunks)

    tool_sel  = evaluate_tool_selection(expected_calls, actual_calls)
    arg_acc   = evaluate_argument_accuracy(expected_calls, actual_calls, use_llm=use_llm)
    resp_qual = evaluate_response_quality(scenario, transcript, use_llm=use_llm)
    lat       = evaluate_latency(result_data)

    return {
        "scenario_id":          scenario["id"],
        "domain":               scenario["domain"],
        "difficulty":           scenario["difficulty"],
        "title":                scenario["title"],
        "turn_take_success":    turn_take_success,
        "metrics": {
            "tool_selection_acc": tool_sel,
            "argument_acc":       arg_acc,
            "response_qual":      resp_qual,
        },
        "latency":              lat,
        "expected_calls_count": len(expected_calls),
        "actual_calls_count":   len(actual_calls),
    }


# ==============================================================================
# Aggregate Report
# ==============================================================================

def evaluate_all_v2(benchmark_data: dict, evaluation_entries: list, use_llm: bool = False) -> dict:
    """Evaluate all scenarios and produce an aggregate report."""
    results   = []

    for entry in evaluation_entries:
        scenario     = entry["scenario"]
        actual_calls = entry["calls"]
        transcript   = entry["transcript"]
        result_data  = entry["result_data"]

        result = evaluate_scenario(
            scenario, actual_calls, transcript=transcript,
            result_data=result_data, use_llm=use_llm,
        )
        results.append(result)

    # ---- Turn-taking success ----
    turn_taken = [r for r in results if r["turn_take_success"]]
    turn_take_count = len(turn_taken)
    turn_take_rate = round(turn_take_count / len(results), 3) if results else None

    # ---- Aggregate per-metric (on turn-taken samples only) ----
    sel_scores  = [r["metrics"]["tool_selection_acc"]["score"] for r in turn_taken]
    arg_scores  = [r["metrics"]["argument_acc"]["score"] for r in turn_taken]
    resp_scores = [
        r["metrics"]["response_qual"]["score"]
        for r in turn_taken
        if r["metrics"]["response_qual"].get("score") is not None
    ]

    # All-samples scores (for reference)
    sel_scores_all  = [r["metrics"]["tool_selection_acc"]["score"] for r in results]
    arg_scores_all  = [r["metrics"]["argument_acc"]["score"] for r in results]

    def _avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else None

    # ---- Latency aggregation (turn-taken only) ----
    latency_samples    = [r["latency"] for r in turn_taken if r["latency"].get("available")]
    interruptions      = [s for s in latency_samples if s["is_interruption"]]
    non_interruptions  = [s for s in latency_samples if not s["is_interruption"]]
    latency_values     = [s["agent_response_latency_s"] for s in non_interruptions]

    def _std(lst):
        if len(lst) < 2:
            return None
        avg = sum(lst) / len(lst)
        variance = sum((x - avg) ** 2 for x in lst) / (len(lst) - 1)
        return round(math.sqrt(variance), 3)

    latency_report = {
        "total_samples":       len(latency_samples),
        "interruption_count":  len(interruptions),
        "interruption_rate":   round(len(interruptions) / len(latency_samples), 3) if latency_samples else None,
        "avg_response_latency_s": _avg(latency_values),
        "std_response_latency_s": _std(latency_values),
        "min_latency_s":       round(min(latency_values), 3) if latency_values else None,
        "max_latency_s":       round(max(latency_values), 3) if latency_values else None,
        "note":                "avg/std/min/max exclude interruption samples; computed on turn-taken samples only",
    }

    # ---- By domain / difficulty (per-metric breakdown) ----
    domain_metrics_all = {}
    difficulty_metrics_all = {}
    for r in results:
        d = r["domain"]
        domain_metrics_all.setdefault(d, {"tool_selection_acc": [], "argument_acc": [], "response_qual": []})
        domain_metrics_all[d]["tool_selection_acc"].append(r["metrics"]["tool_selection_acc"]["score"])
        domain_metrics_all[d]["argument_acc"].append(r["metrics"]["argument_acc"]["score"])
        rq = r["metrics"]["response_qual"].get("score")
        if rq is not None:
            domain_metrics_all[d]["response_qual"].append(rq)

        diff = r["difficulty"]
        difficulty_metrics_all.setdefault(diff, {"tool_selection_acc": [], "argument_acc": [], "response_qual": []})
        difficulty_metrics_all[diff]["tool_selection_acc"].append(r["metrics"]["tool_selection_acc"]["score"])
        difficulty_metrics_all[diff]["argument_acc"].append(r["metrics"]["argument_acc"]["score"])
        if rq is not None:
            difficulty_metrics_all[diff]["response_qual"].append(rq)

    domain_metrics_tt = {}
    difficulty_metrics_tt = {}
    for r in turn_taken:
        d = r["domain"]
        domain_metrics_tt.setdefault(d, {"tool_selection_acc": [], "argument_acc": [], "response_qual": []})
        domain_metrics_tt[d]["tool_selection_acc"].append(r["metrics"]["tool_selection_acc"]["score"])
        domain_metrics_tt[d]["argument_acc"].append(r["metrics"]["argument_acc"]["score"])
        rq = r["metrics"]["response_qual"].get("score")
        if rq is not None:
            domain_metrics_tt[d]["response_qual"].append(rq)

        diff = r["difficulty"]
        difficulty_metrics_tt.setdefault(diff, {"tool_selection_acc": [], "argument_acc": [], "response_qual": []})
        difficulty_metrics_tt[diff]["tool_selection_acc"].append(r["metrics"]["tool_selection_acc"]["score"])
        difficulty_metrics_tt[diff]["argument_acc"].append(r["metrics"]["argument_acc"]["score"])
        if rq is not None:
            difficulty_metrics_tt[diff]["response_qual"].append(rq)

    def _metric_summary(metrics_dict):
        return {
            k: {m: _avg(scores) for m, scores in v.items() if scores}
            for k, v in metrics_dict.items()
        }

    report = {
        "benchmark_name":  benchmark_data["benchmark_name"],
        "evaluated_at":    datetime.now().isoformat(),
        "total_scenarios": len(results),
        "turn_taking": {
            "total":           len(results),
            "turn_taken":      turn_take_count,
            "no_response":     len(results) - turn_take_count,
            "turn_take_rate":  turn_take_rate,
        },
        "by_metric": {
            "tool_selection_acc":     _avg(sel_scores),
            "argument_acc":           _avg(arg_scores),
            "response_qual":          _avg(resp_scores) if use_llm else None,
            "tool_selection_acc_all": _avg(sel_scores_all),
            "argument_acc_all":       _avg(arg_scores_all),
            "note":                   "*_all includes no-response samples (scored 0); default metrics are turn-taken only",
        },
        "latency": latency_report,
        "by_domain": _metric_summary(domain_metrics_all),
        "by_difficulty": _metric_summary(difficulty_metrics_all),
        "by_domain_turn_taken": _metric_summary(domain_metrics_tt),
        "by_difficulty_turn_taken": _metric_summary(difficulty_metrics_tt),
        "scenario_results": results,
    }

    return report


# ==============================================================================
# Dry Run
# ==============================================================================

def run_dry_run():
    """Verify evaluation logic with hardcoded sample data."""
    print("🧪 DRY RUN — Testing evaluation logic\n")

    sample_scenario = {
        "id": "test_01",
        "domain": "travel",
        "title": "Search + Book",
        "difficulty": "easy",
        "dialogue": [
            {"user": "Hi", "ai": "Hello!"},
            {"user": "Book a flight to London on Aug 20.", "ai": "I'll search for flights to London on August 20th and book one for you."},
        ],
        "expected_tool_calls": [
            {"function": "search_flights",  "args": {"destination": "London", "date": "August 20"}},
            {"function": "book_flight",     "args": {"passenger_name": "Alice"}},
        ],
    }

    # Test 1: Perfect
    print("— Test 1: Perfect calls")
    result = evaluate_scenario(
        sample_scenario,
        [
            {"function": "search_flights", "args": {"destination": "London",  "date": "2026-08-20"}},
            {"function": "book_flight",    "args": {"passenger_name": "Alice"}},
        ],
        transcript="I'll search for flights to London on August 20th and book one for you.",
        result_data={"user_speech_end_rel": 10.0, "audio_agent_speech_start": 12.5},
    )
    _print_result(result)

    # Test 2: Missing one call
    print("\n— Test 2: Missing book_flight")
    result = evaluate_scenario(
        sample_scenario,
        [{"function": "search_flights", "args": {"destination": "London", "date": "2026-08-20"}}],
        transcript="I found some flights to London.",
        result_data={"user_speech_end_rel": 10.0, "audio_agent_speech_start": 11.0},
    )
    _print_result(result)

    # Test 3: Interruption (agent started before user finished)
    print("\n— Test 3: Interruption (agent_start < user_end)")
    result = evaluate_scenario(
        sample_scenario,
        [{"function": "search_flights", "args": {"destination": "London", "date": "2026-08-20"}}],
        transcript="Looking up flights now.",
        result_data={"user_speech_end_rel": 10.0, "audio_agent_speech_start": 8.0},
    )
    _print_result(result)

    # Test 4: Extra unexpected call
    print("\n— Test 4: Extra unexpected call")
    result = evaluate_scenario(
        sample_scenario,
        [
            {"function": "search_flights",    "args": {"destination": "London", "date": "2026-08-20"}},
            {"function": "book_flight",       "args": {"passenger_name": "Alice"}},
            {"function": "update_identity_doc", "args": {"doc_type": "passport"}},
        ],
        transcript="Done!",
        result_data=None,
    )
    _print_result(result)

    print("\n✅ Dry run complete!")


def _print_result(r: dict):
    m = r["metrics"]
    lat = r["latency"]
    print(f"    tool_selection_acc: {m['tool_selection_acc']['score']}  (recall={m['tool_selection_acc']['recall']}, precision={m['tool_selection_acc']['precision']})")
    print(f"    argument_acc:       {m['argument_acc']['score']}")
    if m['response_qual'].get('score') is not None:
        print(f"    response_qual:      {m['response_qual']['score']}")
    else:
        print(f"    response_qual:      (LLM disabled)")
    if lat.get("available"):
        tag = " ⚡ INTERRUPTION" if lat["is_interruption"] else ""
        print(f"    latency:            {lat['agent_response_latency_s']}s{tag}")
    else:
        print(f"    latency:            N/A")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Step Tool Call Evaluator")
    parser.add_argument("--benchmark",    type=str, default="benchmark_data_v2.json")
    parser.add_argument("--results-dir",  type=str, default="fdb_v3_data_released",
                        help="Root directory containing result_<provider>.json files")
    parser.add_argument("--output",       type=str, default="evaluation_report.json")
    parser.add_argument("--provider",     type=str, default="gpt_realtime")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--use-llm",      action="store_true",
                        help="Use gpt-4o as LLM judge for argument and response accuracy")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
        return

    print(f"📖 Loading benchmark data from {args.benchmark}...")
    with open(args.benchmark, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    
    # Store dynamic lookup for scenarios
    scenario_map = {s["id"]: s for s in benchmark.get("scenarios", [])}

    print(f"📁 Scanning {args.results_dir} for provider '{args.provider}'...")
    import pathlib
    res_path = pathlib.Path(args.results_dir)

    all_scenarios_to_evaluate = []

    if res_path.exists():
        for res_file in res_path.rglob(f"result_{args.provider}.json"):
            with open(res_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            eid = data.get("example_id")
            if not eid or eid not in scenario_map:
                continue
            
            all_scenarios_to_evaluate.append({
                "scenario":    scenario_map[eid],
                "calls":       data.get("actual_tool_calls", []),
                "transcript":  data.get("transcript", ""),
                "result_data": data,
            })
    else:
        print(f"❌ Results directory not found: {args.results_dir}")
        sys.exit(1)

    print(f"🔍 Found {len(all_scenarios_to_evaluate)} valid result files matching benchmark scenarios.")

    # Modified evaluate_all to take the list directly
    report = evaluate_all_v2(benchmark, all_scenarios_to_evaluate, use_llm=args.use_llm)


    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ---- Summary ----
    print(f"\n📊 EVALUATION REPORT")
    print(f"{'=' * 56}")
    tt = report["turn_taking"]
    print(f"  Scenarios evaluated : {report['total_scenarios']}")
    print(f"  Turn-Take Success   : {tt['turn_taken']}/{tt['total']} ({tt['turn_take_rate']:.1%})")
    if tt['no_response'] > 0:
        print(f"  No Response (silent): {tt['no_response']}")
    print(f"\n  By Metric (turn-taken only, N={tt['turn_taken']}):")
    bm = report["by_metric"]
    print(f"    Tool Selection Acc : {bm['tool_selection_acc']:.1%}")
    print(f"    Argument Acc       : {bm['argument_acc']:.1%}")
    if bm['response_qual'] is not None:
        print(f"    Response Qual      : {bm['response_qual']:.1%}")
    else:
        print(f"    Response Qual      : (LLM judge disabled, use --use-llm)")
    lat = report["latency"]
    print(f"\n  Latency (N={lat['total_samples']}, turn-taken only):")
    if lat['avg_response_latency_s'] is not None:
        std_str = f" ± {lat['std_response_latency_s']:.2f}s" if lat.get('std_response_latency_s') else ""
        print(f"    Avg latency        : {lat['avg_response_latency_s']:.2f}s{std_str}  (excl. interruptions)")
        print(f"    Min / Max          : {lat['min_latency_s']:.2f}s / {lat['max_latency_s']:.2f}s")
    if lat['interruption_rate'] is not None:
        print(f"    Interruptions      : {lat['interruption_count']} / {lat['total_samples']}  ({lat['interruption_rate']:.1%})")
    print(f"\n  By Domain (all samples):")
    for d, metrics in report["by_domain"].items():
        parts = [f"{m}={v:.1%}" for m, v in metrics.items()]
        print(f"    {d}: {', '.join(parts)}")
    print(f"\n  By Difficulty (all samples):")
    for d, metrics in report["by_difficulty"].items():
        parts = [f"{m}={v:.1%}" for m, v in metrics.items()]
        print(f"    {d}: {', '.join(parts)}")
    print(f"\n  📄 Full report saved: {args.output}")


if __name__ == "__main__":
    main()
