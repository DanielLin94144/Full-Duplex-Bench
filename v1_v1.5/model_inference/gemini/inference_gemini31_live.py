#!/usr/bin/env python3
"""
Gemini 3.1 Flash Live Preview Inference Script

This is a thin layer on top of `inference_gemini25_native.py`. The 2.5 file
already implements:
  - the audio constants (SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE, CHUNK_SIZE,
    REC_TICK_MS),
  - the SynchronizedRecorder class,
  - the resample_to_16k / load_audio_chunks helpers.

We import those as-is and only re-implement the parts that genuinely differ
for the 3.1 Live Preview model:

  - MODEL = "gemini-3.1-flash-live-preview"
  - genai.Client(vertexai=False, ...) — needed because the public API key
    path is the only one that supports gemini-3.1-flash-live-preview today,
    and we have to override GOOGLE_GENAI_USE_VERTEXAI if it's set.
  - --thinking-level CLI flag, mapped to the 3.x ThinkingConfig key
    `thinking_level` (the 2.5 file doesn't have this knob).
  - Minimum-config policy: only response_modalities + thinking_config are
    sent, so the API's documented defaults govern VAD/system-instruction
    behaviour. (2.5's CONFIG includes an opinionated realtime_input_config;
    that's deliberately not reused here.)
  - --prefix support so the v1.5 clean pass writes clean_output.wav rather
    than overwriting output.wav.

Usage:
    # v1.0 / v1.5 overlap pass
    python inference_gemini31_live.py --base-dir /path/to/v1_5 --task user_interruption --overwrite
    # v1.5 clean pass
    python inference_gemini31_live.py --base-dir /path/to/v1_5 --task user_interruption --prefix clean_ --overwrite
"""
import argparse
import asyncio
import os
import time
import traceback
from glob import glob
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from google import genai

# Shared building blocks (do not modify the 2.5 file).
from inference_gemini25_native import (
    SynchronizedRecorder,
    resample_to_16k,
    load_audio_chunks,
    SEND_SAMPLE_RATE,
    RECEIVE_SAMPLE_RATE,
    CHUNK_SIZE,
)

load_dotenv()

MODEL = "gemini-3.1-flash-live-preview"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment. Check .env file.")


def make_client() -> genai.Client:
    # vertexai=False forces the public Gemini API key endpoint regardless of
    # GOOGLE_GENAI_USE_VERTEXAI; gemini-3.1-flash-live-preview is currently
    # only reachable that way.
    return genai.Client(vertexai=False, api_key=GEMINI_API_KEY)


def build_config(thinking_level: str | None) -> dict:
    # Minimum-config so we measure the model out-of-the-box. Per the
    # gemini-3.1-flash-live-preview model card, the API default for
    # thinking_level is "minimal" (chosen to optimize for lowest latency);
    # we pass it through explicitly when --thinking-level is set.
    # See: https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-live-preview
    cfg: dict = {"response_modalities": ["AUDIO"]}
    if thinking_level is not None:
        cfg["thinking_config"] = {"thinking_level": thinking_level}
    return cfg


async def run_session(
    client: genai.Client,
    config: dict,
    session_id: int,
    chunks: List[bytes],
    start_idx: int,
    recorder: SynchronizedRecorder,
    session_start_time: float,
) -> int:
    chunk_duration = CHUNK_SIZE / SEND_SAMPLE_RATE
    session_time_offset = start_idx * chunk_duration

    print(f"[DEBUG][Session {session_id}] Start from chunk {start_idx} (t={session_time_offset:.2f}s)")

    session_done = False
    idx = start_idx
    total = len(chunks)
    audio_received = 0

    async with client.aio.live.connect(model=MODEL, config=config) as sess:
        print(f"[DEBUG][Session {session_id}] Connected to Live API")

        async def sender():
            nonlocal idx, session_done
            while idx < total and not session_done:
                await sess.send_realtime_input(
                    audio={"data": chunks[idx], "mime_type": "audio/pcm"}
                )
                idx += 1
                await asyncio.sleep(chunk_duration)

            if not session_done:
                print(f"[DEBUG][Session {session_id}] Sender finished sending all chunks")
                try:
                    await sess.send_realtime_input(audio_stream_end=True)
                except Exception:
                    pass

        async def receiver():
            nonlocal session_done, audio_received
            print(f"[DEBUG][Session {session_id}] Receiver started")
            async for resp in sess.receive():
                sc = resp.server_content
                if not sc:
                    continue

                if getattr(sc, "interrupted", False):
                    current_time = time.time() - session_start_time
                    print(f"[DEBUG][Session {session_id}] *** INTERRUPTED at t={current_time:.2f}s, chunk {idx} ***")
                    recorder.interrupt()
                    session_done = True
                    return

                if sc.model_turn:
                    for part in sc.model_turn.parts:
                        if part.inline_data and isinstance(part.inline_data.data, bytes):
                            await recorder.add(part.inline_data.data)
                            audio_received += len(part.inline_data.data)

                        if getattr(part, "generation_complete", False):
                            session_done = True
                            return

                if getattr(sc, "turn_complete", False) or getattr(sc, "generation_complete", False):
                    audio_sec = audio_received / 2 / RECEIVE_SAMPLE_RATE
                    print(f"[DEBUG][Session {session_id}] Turn complete (audio: {audio_sec:.2f}s)")
                    session_done = True
                    return
            print(f"[DEBUG][Session {session_id}] Receiver loop ended naturally")

        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())

        await sender_task

        try:
            await asyncio.wait_for(receiver_task, timeout=5.0)
        except asyncio.TimeoutError:
            print(f"[DEBUG][Session {session_id}] Timeout")
            receiver_task.cancel()

    return idx


async def process_single_file(input_wav: str, output_wav: str, config: dict, overwrite: bool = True) -> bool:
    input_path = Path(input_wav)
    if not input_path.exists():
        print(f"[ERROR] {input_wav} not found.")
        return False

    if os.path.exists(output_wav) and not overwrite:
        return True

    wav16k_path, duration = resample_to_16k(input_path)
    chunks = load_audio_chunks(wav16k_path)
    total_chunks = len(chunks)
    print(f"[INFO] Loaded {total_chunks} chunks, duration: {duration:.2f}s")

    recorder = SynchronizedRecorder(RECEIVE_SAMPLE_RATE, duration, output_wav)
    recorder_task = asyncio.create_task(recorder.run())

    client = make_client()

    session_start_time = time.time()

    chunk_idx = 0
    session_id = 1

    while chunk_idx < total_chunks:
        try:
            new_idx = await run_session(
                client, config, session_id, chunks, chunk_idx, recorder, session_start_time
            )
        except Exception as e:
            print(f"[ERROR] Session {session_id}: {e}")
            traceback.print_exc()
            break

        if new_idx == chunk_idx:
            chunk_idx += 1
        else:
            chunk_idx = new_idx

        session_id += 1

    recorder.stop()
    await recorder_task

    if os.path.exists(wav16k_path):
        os.remove(wav16k_path)

    print(f"[INFO] Saved {output_wav}")
    return True


async def batch_process(args):
    base_dir = os.path.expanduser(args.base_dir)
    pattern = os.path.join(base_dir, args.task or "*", "*", f"{args.prefix}input.wav")
    files = sorted(glob(pattern))
    print(f"Found {len(files)} files.")

    config = build_config(args.thinking_level)
    sem = asyncio.Semaphore(max(1, args.concurrency))
    total = len(files)

    async def process_one(i: int, f: str) -> bool:
        out_wav = os.path.join(os.path.dirname(f), f"{args.prefix}output.wav")
        if os.path.exists(out_wav) and not args.overwrite:
            print(f"[{i+1}/{total}] Skip (exists): {f}")
            return True
        async with sem:
            print(f"\n[{i+1}/{total}] START {f}")
            try:
                ok = await process_single_file(f, out_wav, config, args.overwrite)
                print(f"[{i+1}/{total}] {'DONE' if ok else 'FAIL'} {f}")
                return ok
            except Exception as e:
                print(f"[{i+1}/{total}] FAIL {f}: {e}")
                return False

    results = await asyncio.gather(*[process_one(i, f) for i, f in enumerate(files)])
    print(f"\nDone. {sum(1 for r in results if r)}/{total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument(
        "--prefix",
        default="",
        help="Filename prefix on input/output (use 'clean_' for v1.5 clean pass).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of files to process in parallel. Default: 1 (sequential).",
    )
    parser.add_argument(
        "--thinking-level",
        default="minimal",
        choices=["minimal", "low", "medium", "high", "none"],
        help=(
            "Gemini 3.1 thinking level. Default 'minimal' matches the API "
            "default documented on the gemini-3.1-flash-live-preview model "
            "card ('The default is `minimal` to optimize for lowest latency'). "
            "Use 'none' to omit thinking_config entirely and let any future "
            "server-side default apply unchanged."
        ),
    )
    args = parser.parse_args()
    if args.thinking_level == "none":
        args.thinking_level = None
    asyncio.run(batch_process(args))
