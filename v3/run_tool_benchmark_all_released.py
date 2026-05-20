#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tool_benchmark_all_released.py

Batch evaluation pipeline for FDB-v3 using the released flat data layout.

Released layout (fdb_v3_data_released):
    {example_id}_{speaker_id}/
        input.wav
        metadata.json

Usage:
    python run_tool_benchmark_all_released.py --provider gpt_realtime
    python run_tool_benchmark_all_released.py --provider gemini2_5 --root_dir fdb_v3_data_released --force
"""

import os
import sys
import re
import argparse
import time
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_tool_benchmark import (
    load_asr_model,
    process_single,
    NpEncoder
)

# Benchmark data for the released version
DATA_JSON_PATH = PROJECT_ROOT / "benchmark_data_v2.json"

# Regex: everything up to the last _<24-hex-chars> is the example_id
_FOLDER_RE = re.compile(r"^(.+)_([0-9a-f]{24})$")


def load_data_released():
    """Load scenario data from benchmark_data_v2.json."""
    with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict) and "scenarios" in data:
            data = data["scenarios"]
    return {item["id"]: item for item in data}


def discover_inputs_released(root_dir):
    """Discover all input.wav files in the released flat layout.

    Folder name format: {example_id}_{speaker_id}
    e.g. ecommerce_01_65e8cf8f4c7424fa062e54a3

    Returns list of (speaker_id, example_id, input_path).
    """
    root_dir = Path(root_dir)
    inputs = []
    if not root_dir.exists():
        print(f"❌ Root directory not found: {root_dir}")
        return inputs

    for folder in sorted(root_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        m = _FOLDER_RE.match(folder.name)
        if not m:
            continue
        example_id = m.group(1)
        speaker_id = m.group(2)
        input_path = folder / "input.wav"
        if input_path.exists():
            inputs.append((speaker_id, example_id, input_path))

        # Pick up per-folder metadata.json so this script also works on
        # datasets whose ids are not in benchmark_data_v2.json.
        meta_path = folder / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    _PER_FOLDER_DATA[example_id] = json.load(f)
            except Exception as e:
                print(f"  ⚠️  Failed to load {meta_path}: {e}")

    return inputs


_PER_FOLDER_DATA = {}


def main():
    parser = argparse.ArgumentParser(description="Batch FDB-v3 evaluation (released data layout)")
    parser.add_argument("--root_dir", type=str, default="fdb_v3_data_released",
                        help="Root directory containing released example folders")
    parser.add_argument("--provider", type=str, default=os.getenv("LK_PROVIDER", "gpt_realtime"),
                        help="Model provider (gpt_realtime, grok, gemini2_5, gemini3_1, ultravox, cascaded)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    parser.add_argument("--asr-only", action="store_true", help="Skip inference, only run ASR")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        print(f"❌ Root directory not found: {root_dir}")
        sys.exit(1)

    print(f"🚀 Batch evaluation started (released layout)")
    print(f"🤖 Provider: {args.provider}")
    print(f"📁 Root Dir: {root_dir}")

    # 1. Load Data
    try:
        data = load_data_released()
    except FileNotFoundError:
        data = {}
    print(f"📋 Loaded {len(data)} scenarios from benchmark_data_v2.json")

    # 2. Load ASR Model once
    print(f"🔊 Loading ASR model...")
    asr_model = load_asr_model()
    print(f"✅ ASR model loaded")

    # 3. Discover Inputs (also picks up per-folder metadata.json)
    inputs = discover_inputs_released(root_dir)
    if _PER_FOLDER_DATA:
        data = {**data, **_PER_FOLDER_DATA}
        print(f"📋 Merged {len(_PER_FOLDER_DATA)} per-folder metadata.json entries → {len(data)} total")
    print(f"📂 Found {len(inputs)} input.wav files")

    if not inputs:
        print("Done (no files to process).")
        return

    # 4. Process all
    stats = {
        "total": 0,
        "success": 0,
        "error": 0,
        "skipped": 0,
        "latency_sum": 0.0,
        "perceived_latency_sum": 0.0,
        "latency_count": 0
    }

    start_time = time.time()

    for speaker_id, example_id, input_path in inputs:
        stats["total"] += 1
        print(f"\n[{stats['total']}/{len(inputs)}] Processing Speaker={speaker_id[:8]}... Example={example_id}...")

        try:
            result = process_single(
                speaker_id, example_id, input_path,
                args.provider, data, asr_model,
                asr_only=args.asr_only, force=args.force
            )

            if result is None:
                stats["skipped"] += 1
            elif result.get("status") == "completed":
                stats["success"] += 1
                lat = result.get("latency", {}).get("first_speech_s")
                if lat:
                    stats["latency_sum"] += lat
                    stats["latency_count"] += 1

                per_lat = result.get("perceived_total_latency")
                if per_lat:
                    stats["perceived_latency_sum"] += per_lat
            else:
                stats["error"] += 1

        except Exception as e:
            print(f"  ❌ Unhandled error in process_single: {e}")
            stats["error"] += 1

    duration = time.time() - start_time
    print(f"\n{'-'*60}")
    print(f"🏁 Batch Evaluation Complete")
    print(f"⏱️  Total Duration: {duration/60:.2f} minutes")
    print(f"📊 Summary:")
    print(f"   - Total:   {stats['total']}")
    print(f"   - Success: {stats['success']}")
    print(f"   - Error:   {stats['error']}")
    print(f"   - Skipped: {stats['skipped']}")

    if stats["latency_count"] > 0:
        avg_lat = stats["latency_sum"] / stats["latency_count"]
        print(f"   - Avg First-Speech Latency: {avg_lat:.2f}s")

    if stats["success"] > 0 and stats["perceived_latency_sum"] > 0:
        avg_per_lat = stats["perceived_latency_sum"] / stats["success"]
        print(f"   - Avg Perceived Latency:    {avg_per_lat:.2f}s")
    print(f"{'-'*60}")



if __name__ == "__main__":
    main()
