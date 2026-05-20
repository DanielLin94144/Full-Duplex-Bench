#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_tool_latency.py

Post-evaluation latency analysis for multi-step tool benchmark results.
Extracts 3 fine-grained latency metrics from each result_*.json:

  1. First Response Latency
     = (first ASR chunk timestamp) - user_speech_end_rel
     How long user waits until the agent says ANYTHING (including filler).

  2. Tool Call Latency
     = actual_tool_calls[0].timestamp_start - user_speech_end_rel
     How fast the model decided to call the tool.

  3. Task Completion Latency
     = (start of key-information sentence) - user_speech_end_rel
     When the user actually gets the factual answer / confirmation.
     Uses gpt-4o to identify which ASR chunk starts the real answer vs filler.

Usage:
    python analyze_tool_latency.py --results-dir fdb_v3_data_released --provider gemini2_5
    python analyze_tool_latency.py --results-dir fdb_v3_data_released --provider gpt_realtime --skip-existing
"""

import os
import sys
import json
import math
import glob
import argparse
from pathlib import Path
from typing import Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env.local"))
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    print("❌ Missing openai. Install: pip install openai")
    sys.exit(1)


# ==============================================================================
# gpt-4o Judge: Identify Key Information Sentence
# ==============================================================================

SYSTEM_PROMPT = """You are an expert audio transcript analyst for a voice AI assistant evaluation.

You will receive:
1. USER_SPEECH_END_REL: timestamp (seconds) when the user finished speaking.
2. ASR_CHUNKS: list of the AI agent's spoken words with [start, end] timestamps.
3. TOOL_CALLS: list of tool calls the agent made (with timestamps).

The typical flow after a user query is:
  [User finishes] → (silence) → [Filler sentence] → (silence during tool execution) → [Key information response]

Your task: Identify and separate the agent's speech into:

1. **filler_sentence**: Conversational filler like "Let me check that for you", "Sure, I'll look that up", "One moment please". 
   - If the agent immediately starts with the factual answer, this is "" (empty).
   
2. **key_info_sentence**: The part of the response containing the ACTUAL factual information or task confirmation.
   - E.g., "I found flights to London starting at $450" or "Your passport has been updated with number E772211"
   - This is the sentence the user is actually waiting for.

3. **key_info_start_time**: The timestamp (from asr_chunks) when the key information sentence begins.

Output ONLY a valid JSON object:
{
  "filler_sentence": "string or empty",
  "filler_start_time": float or null,
  "filler_end_time": float or null,
  "key_info_sentence": "string",
  "key_info_start_time": float,
  "key_info_end_time": float
}

Do not include markdown formatting, code fences, or extra text. Output ONLY raw JSON."""


def identify_key_info(user_end: float, asr_chunks: list, tool_calls: list, client: OpenAI) -> dict:
    """Use gpt-4o to identify the key information sentence from agent speech."""
    user_prompt = (
        f"USER_SPEECH_END_REL: {user_end}\n\n"
        f"ASR_CHUNKS:\n{json.dumps(asr_chunks, indent=2, ensure_ascii=False)}\n\n"
        f"TOOL_CALLS:\n{json.dumps(tool_calls, indent=2, ensure_ascii=False)}"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        max_tokens=500,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    return json.loads(raw)


# ==============================================================================
# Latency Calculation
# ==============================================================================

def compute_latencies(result_data: dict, client: Optional[OpenAI] = None) -> Optional[dict]:
    """
    Compute 3 latency metrics from a single result JSON.
    Returns None if required fields are missing.
    """
    user_end = result_data.get("user_speech_end_rel")
    asr_chunks = result_data.get("asr_chunks", [])
    tool_calls = result_data.get("actual_tool_calls", [])

    if user_end is None:
        return None

    # Turn-taking success: did the agent produce any speech?
    turn_take_success = bool(asr_chunks)

    metrics = {
        "user_speech_end_rel": user_end,
        "turn_take_success": turn_take_success,
    }

    # --- Metric 1: First Response Latency ---
    if asr_chunks:
        first_chunk_start = asr_chunks[0]["timestamp"][0]
        first_response_latency = round(first_chunk_start - user_end, 3)
        metrics["first_response_latency_s"] = first_response_latency
        metrics["agent_first_word_time"] = first_chunk_start
    else:
        metrics["first_response_latency_s"] = None
        metrics["note_first_response"] = "No ASR chunks available"

    # --- Metric 2: Tool Call Latency ---
    if tool_calls:
        first_tool_start = tool_calls[0].get("timestamp_start")
        if first_tool_start is not None:
            tool_call_latency = round(first_tool_start - user_end, 3)
            metrics["tool_call_latency_s"] = tool_call_latency
            metrics["first_tool_call_time"] = first_tool_start
            metrics["first_tool_name"] = tool_calls[0].get("function", "unknown")
        else:
            metrics["tool_call_latency_s"] = None
            metrics["note_tool_call"] = "Tool call has no timestamp"
    else:
        metrics["tool_call_latency_s"] = None
        metrics["note_tool_call"] = "No tool calls recorded"

    # --- Metric 3: Task Completion Latency (requires LLM) ---
    if client and asr_chunks:
        try:
            llm_result = identify_key_info(user_end, asr_chunks, tool_calls, client)
            metrics["filler_sentence"] = llm_result.get("filler_sentence", "")
            metrics["filler_start_time"] = llm_result.get("filler_start_time")
            metrics["filler_end_time"] = llm_result.get("filler_end_time")
            metrics["key_info_sentence"] = llm_result.get("key_info_sentence", "")

            key_start = llm_result.get("key_info_start_time")
            if key_start is not None:
                task_completion_latency = round(key_start - user_end, 3)
                metrics["task_completion_latency_s"] = task_completion_latency
                metrics["key_info_start_time"] = key_start
                metrics["key_info_end_time"] = llm_result.get("key_info_end_time")
            else:
                metrics["task_completion_latency_s"] = None
                metrics["note_task_completion"] = "Could not identify key info start"
        except Exception as e:
            metrics["task_completion_latency_s"] = None
            metrics["note_task_completion"] = f"LLM error: {str(e)}"
    else:
        metrics["task_completion_latency_s"] = None
        if not asr_chunks:
            metrics["note_task_completion"] = "No ASR chunks"
        else:
            metrics["note_task_completion"] = "LLM client not available"

    return metrics


# ==============================================================================
# Aggregate Statistics
# ==============================================================================

def aggregate_stats(all_metrics: list) -> dict:
    """Compute aggregate statistics across all analyzed scenarios."""
    def _stats(values):
        values = [v for v in values if v is not None]
        if not values:
            return {"count": 0}
        avg = sum(values) / len(values)
        if len(values) >= 2:
            std = math.sqrt(sum((x - avg) ** 2 for x in values) / (len(values) - 1))
        else:
            std = 0.0
        return {
            "count": len(values),
            "mean": round(avg, 3),
            "std": round(std, 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
            "median": round(sorted(values)[len(values) // 2], 3),
        }

    # Turn-taking
    turn_taken = [m for m in all_metrics if m.get("turn_take_success")]
    no_response = [m for m in all_metrics if not m.get("turn_take_success")]

    # Compute latency metrics on non-interrupted turn-taken samples only
    valid_latencies = [m for m in turn_taken if m.get("first_response_latency_s") is not None and m["first_response_latency_s"] >= 0]
    first_resp = [m["first_response_latency_s"] for m in valid_latencies]
    tool_call = [m["tool_call_latency_s"] for m in valid_latencies]
    task_comp = [m["task_completion_latency_s"] for m in valid_latencies]

    # Count fillers (on valid latencies only)
    has_filler = sum(1 for m in valid_latencies if m.get("filler_sentence"))
    no_filler = sum(1 for m in valid_latencies if m.get("filler_sentence") == "")

    return {
        "total_analyzed": len(all_metrics),
        "turn_taking": {
            "total": len(all_metrics),
            "turn_taken": len(turn_taken),
            "no_response": len(no_response),
            "turn_take_rate": round(len(turn_taken) / len(all_metrics), 3) if all_metrics else None,
        },
        "first_response_latency": _stats(first_resp),
        "tool_call_latency": _stats(tool_call),
        "task_completion_latency": _stats(task_comp),
        "filler_stats": {
            "with_filler": has_filler,
            "without_filler": no_filler,
        },
        "note": "All latency metrics computed on non-interrupted turn-taken samples only",
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Post-evaluation latency analysis")
    parser.add_argument("--results-dir", type=str, default="fdb_v3_data_released",
                        help="Root directory containing result_<provider>.json files")
    parser.add_argument("--provider", type=str, default="gemini2_5",
                        help="Provider to analyze (gemini2_5, gpt_realtime, grok, etc.)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files that already have latency_tool_analysis.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Path for aggregate report (default: <provider>_latency_report.json)")
    args = parser.parse_args()

    output_path = args.output or f"{args.provider}_latency_report.json"
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    results_dir = Path(args.results_dir).resolve()
    result_files = sorted(results_dir.rglob(f"result_{args.provider}.json"))

    print(f"📁 Found {len(result_files)} result files for provider '{args.provider}'")

    all_metrics = []
    success = 0
    skipped = 0
    failed = 0

    for f_path in result_files:
        example_dir = f_path.parent
        analysis_path = example_dir / f"latency_tool_analysis_{args.provider}.json"

        if args.skip_existing and analysis_path.exists():
            # Load existing analysis
            with open(analysis_path) as f:
                existing = json.load(f)
            all_metrics.append(existing)
            skipped += 1
            continue

        with open(f_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                failed += 1
                continue

        example_id = data.get("example_id", f_path.parent.name)
        print(f"  🔍 {example_id}...", end=" ", flush=True)

        try:
            metrics = compute_latencies(data, client=client)
            if metrics is None:
                print("⚠️ missing fields")
                failed += 1
                continue

            metrics["example_id"] = example_id
            metrics["provider"] = args.provider

            # Save per-file analysis
            with open(analysis_path, "w", encoding="utf-8") as out_f:
                json.dump(metrics, out_f, indent=2, ensure_ascii=False)

            all_metrics.append(metrics)
            success += 1

            # Quick summary line
            fr = metrics.get("first_response_latency_s")
            tc = metrics.get("tool_call_latency_s")
            tk = metrics.get("task_completion_latency_s")
            filler = "✓" if metrics.get("filler_sentence") else "✗"
            print(f"FR={fr}s  TC={tc}s  TK={tk}s  filler={filler}")

        except Exception as e:
            print(f"❌ {e}")
            failed += 1

    # --- Aggregate ---
    print(f"\n{'=' * 60}")
    print(f"✅ Analyzed: {success}, Skipped (existing): {skipped}, Failed: {failed}")

    if all_metrics:
        stats = aggregate_stats(all_metrics)
        stats["provider"] = args.provider

        # Print summary
        print(f"\n📊 LATENCY REPORT — {args.provider.upper()}")
        print(f"{'=' * 56}")

        tt = stats["turn_taking"]
        print(f"  Turn-Take Success: {tt['turn_taken']}/{tt['total']} ({tt['turn_take_rate']:.1%})")
        if tt['no_response'] > 0:
            print(f"  No Response (silent): {tt['no_response']}")
        print(f"  (Latency metrics below exclude interruptions)\n")

        for metric_name, label in [
            ("first_response_latency", "First Response"),
            ("tool_call_latency", "Tool Call"),
            ("task_completion_latency", "Task Completion"),
        ]:
            s = stats[metric_name]
            if s["count"] > 0:
                print(f"  {label} (N={s['count']}):")
                print(f"    Mean: {s['mean']:.2f}s ± {s['std']:.2f}s")
                print(f"    Min / Median / Max: {s['min']:.2f}s / {s['median']:.2f}s / {s['max']:.2f}s")
            else:
                print(f"  {label}: No data")

        fs = stats["filler_stats"]
        total_filler = fs["with_filler"] + fs["without_filler"]
        if total_filler > 0:
            print(f"\n  Filler Usage: {fs['with_filler']}/{total_filler} "
                  f"({fs['with_filler']/total_filler:.0%}) use filler sentences")

        # Save aggregate report
        report = {
            "aggregate": stats,
            "per_scenario": all_metrics,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  📄 Report saved: {output_path}")


if __name__ == "__main__":
    main()
