#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tool_benchmark.py

Unified evaluation pipeline for FDB-v3.
For each input.wav:
  1. Start a LiveKit agent (using livekit_inference.py)
  2. Stream input.wav into the room
  3. Record model's audio response → output_{provider}.wav
  4. Run ASR on the output
  5. Measure response latency

Usage:
  conda activate fdb

  # Evaluate all examples
  python run_tool_benchmark.py --provider gpt_realtime

  # Single participant
  python run_tool_benchmark.py --provider gpt_realtime --pid 5d8b69704a2eb200174c772a

  # Single example
  python run_tool_benchmark.py --provider gpt_realtime --example 14

  # Skip inference, only re-run ASR + evaluation on existing output files
  python run_tool_benchmark.py --provider gpt_realtime --asr-only
"""

import os
import sys
import json
import time
import wave
import asyncio
import argparse
import datetime
import numpy as np
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(".env.local")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
COLLECTED_AUDIO_DIR = PROJECT_ROOT / "fdb_v3_data_released"
DATA_JSON_PATH = PROJECT_ROOT / "benchmark_data_v2.json"
ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"

SEARCH_CATEGORIES = {"fast_search"}
SAMPLE_RATE = 24000  # LiveKit default sample rate
CHANNELS = 1


def load_data():
    """Load scenario data from benchmark_data.json."""
    with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict) and "scenarios" in data:
            data = data["scenarios"]
    return {item["id"]: item for item in data}


def discover_inputs(root_dir=None):
    """Discover all input.wav files.
    Returns list of (pid, example_id, input_path).
    Supports both legacy nested layout and flat released layout.
    """
    if root_dir is None:
        root_dir = COLLECTED_AUDIO_DIR
    else:
        root_dir = Path(root_dir)

    inputs = []
    if not root_dir.exists():
        print(f"❌ Audio directory not found: {root_dir}")
        return inputs

    # Try legacy nested layout first: {pid_dir}/example_{example_id}/input.wav
    for pid_dir in sorted(root_dir.iterdir()):
        if not pid_dir.is_dir() or pid_dir.name.startswith("."):
            continue
        pid = pid_dir.name
        for example_dir in sorted(pid_dir.iterdir()):
            if not example_dir.is_dir() or not example_dir.name.startswith("example_"):
                continue
            example_id = example_dir.name.replace("example_", "")
            input_path = example_dir / "input.wav"
            if input_path.exists():
                inputs.append((pid, example_id, input_path))

    # If nothing found, try released flat layout: {example_id}_{speaker_id}/input.wav
    if not inputs:
        import re
        folder_re = re.compile(r"^(.+)_([0-9a-f]{24})$")
        for folder in sorted(root_dir.iterdir()):
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            m = folder_re.match(folder.name)
            if not m:
                continue
            example_id = m.group(1)
            speaker_id = m.group(2)
            input_path = folder / "input.wav"
            if input_path.exists():
                inputs.append((speaker_id, example_id, input_path))

    return inputs


# ==============================================================================
# ASR
# ==============================================================================

def load_asr_model():
    """Load NeMo ASR model."""
    print("🔊 Loading ASR model...")
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)
    if hasattr(model, 'cuda'):
        model = model.cuda()
    print("✅ ASR model loaded")
    return model


def run_asr(asr_model, audio_path):
    """Run ASR on an audio file and return transcript."""
    try:
        outputs = asr_model.transcribe([str(audio_path)], timestamps=True)
        if not outputs:
            return {"text": "", "chunks": []}

        result = outputs[0]
        chunks = []
        text = ""

        if hasattr(result, "timestamp") and "word" in result.timestamp:
            for w in result.timestamp["word"]:
                text += w["word"] + " "
                chunks.append({
                    "text": w["word"],
                    "timestamp": [w["start"], w["end"]],
                })
        else:
            if hasattr(result, 'text'):
                text = result.text
            elif isinstance(result, str):
                text = result

        return {"text": text.strip(), "chunks": chunks}
    except Exception as e:
        print(f"  ❌ ASR error: {e}")
        return {"text": "", "chunks": [], "error": str(e)}


# ==============================================================================
# Latency Measurement
# ==============================================================================

def measure_latency_from_audio(input_path, output_path, silence_threshold_db=-40):
    """Measure response latency:
    Time from end of input audio to first non-silence in output audio.
    Returns latency in seconds.
    """
    try:
        from pydub import AudioSegment

        input_audio = AudioSegment.from_file(str(input_path))
        output_audio = AudioSegment.from_file(str(output_path))

        input_duration_s = len(input_audio) / 1000.0
        output_duration_s = len(output_audio) / 1000.0

        # Find first non-silent chunk in output (check every 50ms)
        chunk_ms = 50
        first_speech_ms = None
        for i in range(0, len(output_audio), chunk_ms):
            chunk = output_audio[i:i + chunk_ms]
            if chunk.dBFS > silence_threshold_db:
                first_speech_ms = i
                break

        if first_speech_ms is not None:
            first_speech_s = first_speech_ms / 1000.0
        else:
            first_speech_s = output_duration_s  # No speech detected

        return {
            "input_duration_s": round(input_duration_s, 3),
            "output_duration_s": round(output_duration_s, 3),
            "first_speech_s": round(first_speech_s, 3),
        }
    except Exception as e:
        return {"error": str(e)}


# ==============================================================================
# LiveKit Inference
# ==============================================================================

def run_livekit_inference(input_path, output_path, provider):
    """
    Stream input audio into a LiveKit room and record the agent's response
    by calling livekit_inference.py as a subprocess.
    """
    import uuid
    import subprocess

    room_name = f"eval-{uuid.uuid4().hex[:8]}"
    print(f"  🔗 Streaming via livekit_inference.py into room: {room_name}")

    client_script = PROJECT_ROOT / "livekit_inference.py"
    
    try:
        # Run the client script as a subprocess
        result = subprocess.run(
            [
                sys.executable, str(client_script),
                "-i", str(input_path),
                "-o", str(output_path),
                "--room", room_name
            ],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"  ✅ livekit_inference.py finished successfully.")
        
        # Parse STREAM_START_TIME
        stream_start_time = None
        for line in result.stdout.splitlines():
            if line.startswith("STREAM_START_TIME: "):
                try:
                    stream_start_time = float(line[19:])
                except:
                    pass
                break
        
        return room_name, stream_start_time
    except subprocess.CalledProcessError as e:
        print(f"  ❌ livekit_inference.py failed with exit code {e.returncode}")
        return None, None
    except Exception as e:
        print(f"  ❌ livekit_inference.py execution error: {e}")
        return None, None


# ==============================================================================
# Search Verification
# ==============================================================================

def verify_search_answer(item, transcript):
    """Verify a fast_search answer using evaluate_model_answers.py."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from evaluate_model_answers import evaluate_single
        return evaluate_single(item, transcript)
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ==============================================================================
# Main Pipeline
# ==============================================================================

def process_single(pid, example_id, input_path, provider, data, asr_model,
                   asr_only=False, force=False):
    """Process a single example through the full pipeline."""
    item = data.get(example_id)
    if item is None:
        print(f"  ⚠️  Example {example_id} not found in data — skipping")
        return None

    category = item.get("domain", item.get("category", "unknown"))
    example_dir = input_path.parent
    output_path = example_dir / f"output_{provider}.wav"
    result_path = example_dir / f"result_{provider}.json"

    # Check if already evaluated
    if result_path.exists() and not force:
        print(f"  ⏭️  Already evaluated — skipping (use --force to re-run)")
        return None

    result = {
        "pid": pid,
        "example_id": example_id,
        "category": category,
        "title": item["title"],
        "provider": provider,
        "evaluated_at": datetime.datetime.now().isoformat(),
    }

    # Step 1: Run inference (unless --asr-only)
    if not asr_only:
        if output_path.exists() and not force:
            print(f"  📦 Output already exists: {output_path.name}")
        else:
            print(f"  🚀 Running LiveKit inference with provider={provider}...")
            inference_start = time.time()
            try:
                room_name, stream_start_time = run_livekit_inference(input_path, output_path, provider)
                inference_time = time.time() - inference_start
                result["inference_time_s"] = round(inference_time, 2)
                if not room_name:
                    result["status"] = "inference_failed"
                    with open(result_path, "w") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False, cls=NpEncoder)
                    return result
                result["room_name"] = room_name
                result["stream_start_time"] = stream_start_time
            except Exception as e:
                print(f"  ❌ Inference error: {e}")
                import traceback
                traceback.print_exc()
                result["status"] = "inference_error"
                result["error"] = str(e)
                with open(result_path, "w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                return result

    # Step 2: Check output exists
    if not output_path.exists():
        print(f"  ⚠️  No output file found: {output_path.name}")
        result["status"] = "no_output"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return result

    # Step 3: Run ASR on input audio to find speech end
    print(f"  🗣️  Running ASR on input audio...")
    mono_path = Path(input_path).parent / "input_mono.wav"
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-i", str(input_path), "-ac", "1", str(mono_path)], capture_output=True)
    input_asr = run_asr(asr_model, str(mono_path))
    result["input_transcript"] = input_asr["text"]
    result["input_asr_chunks"] = input_asr["chunks"]
    
    # Find the end of speech for the first turn
    user_speech_end_rel = 0
    if input_asr["chunks"]:
        # Find the end of the first turn (look for gap > 2s)
        for i in range(len(input_asr["chunks"]) - 1):
            curr_end = input_asr["chunks"][i]["timestamp"][1]
            next_start = input_asr["chunks"][i+1]["timestamp"][0]
            if next_start - curr_end > 2.0:
                user_speech_end_rel = curr_end
                break
        else:
            # No large gap found, take the end of the last word
            user_speech_end_rel = input_asr["chunks"][-1]["timestamp"][1]
    
    result["user_speech_end_rel"] = user_speech_end_rel

    # Step 4: Measure latency from audio
    print(f"  ⏱️  Measuring latency...")
    latency = measure_latency_from_audio(input_path, output_path)
    result["latency"] = latency
    
    # Step 4.5: Calculate absolute total latency
    stream_start_time = result.get("stream_start_time")
    if stream_start_time and user_speech_end_rel:
        user_done_unix = stream_start_time + user_speech_end_rel
        result["user_done_unix"] = user_done_unix
    else:
        user_done_unix = None

    # Step 4.6: Extract granular search latency from agent logs if available
    room_name = result.get("room_name")
    if room_name:
        try:
            hb_path = Path("/tmp/agent_heartbeat.log")
            if hb_path.exists():
                with open(hb_path, "r") as f:
                    for line in f:
                        if line.startswith("LATENCY_TRACK_JSON: "):
                            try:
                                metrics = json.loads(line[20:])
                                if metrics.get("room") == room_name:
                                    # Always keep the one with actual execution if possible
                                    existing = result.get("search_latency_breakdown", {})
                                    if metrics.get("execution", 0) > existing.get("execution", 0) or "total" not in existing:
                                        result["search_latency_breakdown"] = metrics
                                        
                                        # Calculate absolute end-to-end latency if possible
                                        agent_start_at = metrics.get("agent_start_at")
                                        if agent_start_at and user_done_unix:
                                            abs_latency = agent_start_at - user_done_unix
                                            result["absolute_total_latency"] = round(abs_latency, 3)
                            except:
                                pass
        except Exception as e:
            print(f"  ⚠️  Failed to extract search latency: {e}")

    # Step 4: Run ASR on output
    print(f"  🗣️  Running ASR on model output...")
    asr_result = run_asr(asr_model, str(output_path))
    result["transcript"] = asr_result["text"]
    result["asr_chunks"] = asr_result.get("chunks", [])
    print(f"  📝 Transcript: {asr_result['text'][:100]}...")

    # Step 4.7: Calculate perceived total latency (gap in audio)
    user_speech_end_rel = result.get("user_speech_end_rel", 0)
    if asr_result["chunks"] and user_speech_end_rel:
        agent_speech_start_rel = asr_result["chunks"][0]["timestamp"][0]
        perceived_total_latency = agent_speech_start_rel - user_speech_end_rel
        result["perceived_total_latency"] = round(perceived_total_latency, 3)
        result["audio_agent_speech_start"] = agent_speech_start_rel
        print(f"  ⏱️  Perceived Latency: {perceived_total_latency:.2f}s (Agent Start: {agent_speech_start_rel}s - User End: {user_speech_end_rel}s)")

    # Step 5: For search items, verify correctness
    if category in SEARCH_CATEGORIES:
        print(f"  🔍 Verifying search answer...")
        verification = verify_search_answer(item, asr_result["text"])
        result["search_verification"] = verification
        print(f"  📊 Verification: {verification.get('status', 'unknown')}")

    # Step 6: Extract actual tool calls from telemetry
    actual_tool_calls = []
    stream_start_time = result.get("stream_start_time", 0)
    if room_name:
        try:
            telemetry_path = Path("/tmp/agent_tool_calls.log")
            if telemetry_path.exists():
                with open(telemetry_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            t_data = json.loads(line)
                            if t_data.get("room") == room_name:
                                call_data = t_data.get("call")
                                if stream_start_time:
                                    if "timestamp_start" in call_data:
                                        call_data["timestamp_start"] = round(call_data["timestamp_start"] - stream_start_time, 2)
                                    if "timestamp_end" in call_data:
                                        call_data["timestamp_end"] = round(call_data["timestamp_end"] - stream_start_time, 2)
                                    # Fallback for old single timestamp logic
                                    if "timestamp" in call_data:
                                        call_data["timestamp"] = round(call_data["timestamp"] - stream_start_time, 2)
                                actual_tool_calls.append(call_data)
        except Exception as e:
            print(f"  ⚠️  Failed to extract tool calls from telemetry: {e}")
            
    result["actual_tool_calls"] = actual_tool_calls
    result["status"] = "completed"

    # Save result
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, cls=NpEncoder)
    print(f"  💾 Saved: {result_path.name}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Unified FDB-v3 evaluation pipeline")
    parser.add_argument("--provider", type=str, default=None,
                        help="Model provider (gpt_realtime, grok, gemini2_5, etc.). Default: from .env.local")
    parser.add_argument("--pid", type=str, help="Process only this participant ID")
    parser.add_argument("--example", type=str, help="Process only this example ID")
    parser.add_argument("--asr-only", action="store_true",
                        help="Skip inference, only re-run ASR + evaluation on existing outputs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    args = parser.parse_args()

    # Determine provider
    provider = args.provider or os.getenv("LK_PROVIDER", "gpt_realtime")
    print(f"🤖 Provider: {provider}")

    # Load data
    data = load_data()
    print(f"📋 Loaded {len(data)} scenarios")

    # Clear heartbeat log at start of run
    hb_path = Path("/tmp/agent_heartbeat.log")
    if hb_path.exists() and not args.asr_only:
        try:
            hb_path.unlink()
            print(f"  🧹 Cleared agent heartbeat log.")
        except:
            pass

    # Discover inputs
    inputs = discover_inputs()
    print(f"📁 Found {len(inputs)} input.wav files")

    if not inputs:
        print("❌ No input.wav files found. Check that fdb_v3_data_released/ exists.")
        sys.exit(1)

    # Filter
    if args.pid:
        inputs = [(p, e, i) for p, e, i in inputs if p == args.pid]
        print(f"   Filtered to PID={args.pid}: {len(inputs)}")
    if args.example:
        inputs = [(p, e, i) for p, e, i in inputs if e == args.example]
        print(f"   Filtered to example={args.example}: {len(inputs)}")

    # Load ASR model
    asr_model = load_asr_model()

    # Process all
    results = []
    success = 0
    failed = 0
    skipped = 0

    for pid, example_id, input_path in inputs:
        item = data.get(example_id)
        category = item.get("domain", item.get("category", "unknown")) if item else "unknown"
        pid_short = pid[:8]
        print(f"\n{'='*60}")
        print(f"📂 PID={pid_short}... / example_{example_id} [{category}]")

        result = process_single(
            pid, example_id, input_path, provider, data, asr_model,
            asr_only=args.asr_only, force=args.force
        )

        if result is None:
            skipped += 1
        elif result.get("status") == "completed":
            results.append(result)
            success += 1
        else:
            results.append(result)
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Evaluation Summary ({provider})")
    print(f"   Completed: {success}, Failed: {failed}, Skipped: {skipped}")

    if results:
        # Calculate aggregate stats
        completed = [r for r in results if r.get("status") == "completed"]
        if completed:
            latencies = [
                r["latency"]["first_speech_s"]
                for r in completed
                if "latency" in r and "first_speech_s" in r["latency"]
            ]
            if latencies:
                print(f"   Avg first-speech latency: {np.mean(latencies):.2f}s")
                print(f"   Median first-speech latency: {np.median(latencies):.2f}s")

            # Search verification summary
            search_results = [r for r in completed if "search_verification" in r]
            if search_results:
                correct = sum(1 for r in search_results
                              if r["search_verification"].get("status") == "correct")
                total = len(search_results)
                print(f"   Search accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

        # Save aggregate results
        summary_path = COLLECTED_AUDIO_DIR / f"evaluation_summary_{provider}.json"
        with open(summary_path, "w") as f:
            json.dump({
                "provider": provider,
                "evaluated_at": datetime.datetime.now().isoformat(),
                "total": len(results),
                "completed": success,
                "failed": failed,
                "skipped": skipped,
                "results": results,
            }, f, indent=2, ensure_ascii=False, cls=NpEncoder)
        print(f"   📄 Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
