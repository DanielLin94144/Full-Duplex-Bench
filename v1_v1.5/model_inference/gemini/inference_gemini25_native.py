#!/usr/bin/env python3
"""
Gemini 2.5 Flash Native Audio Inference Script

Key Features:
1. TIME-SYNCHRONIZED recording - output aligned with input timeline
2. MULTI-SESSION for interruption handling (supports barge-in recovery)

Usage:
    python inference_gemini25_native.py --base-dir /path/to/data --task synthetic_user_interruption --overwrite
"""
import os
import asyncio
import time
import math
import traceback
import argparse
import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from scipy import signal as ss
from dotenv import load_dotenv

from google import genai
from google.genai import types
from glob import glob

# Load environment variables
load_dotenv()

# ===== Audio Config =====
SEND_SAMPLE_RATE = 16000   # Input to Gemini (16kHz)
RECEIVE_SAMPLE_RATE = 24000  # Output from Gemini (24kHz)
CHUNK_SIZE = 1024  # Standard chunk size
REC_TICK_MS = 10   # Recorder tick interval

# ===== Model Config =====
# MODEL = "gemini-2.0-flash-live-001"  # DEPRECATED
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment. Check .env file.")

# See: https://ai.google.dev/api/live for all options
CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": "You are a helpful and friendly AI assistant.",
    "realtime_input_config": {
        "automatic_activity_detection": {
            "disabled": False,
            # LOW sensitivity = less likely to detect speech (reduce false positives)
            "start_of_speech_sensitivity": "START_SENSITIVITY_HIGH",
            "end_of_speech_sensitivity": "END_SENSITIVITY_HIGH",
            # Require longer speech before committing start-of-speech (default ~40ms for HIGH)
            "prefix_padding_ms": 40,
            # Require longer silence before ending speech (default ~300ms for HIGH)
            "silence_duration_ms": 300,
        },
        # Optional: Set to "NO_INTERRUPTION" to completely disable user interruptions
        "activity_handling": "START_OF_ACTIVITY_INTERRUPTS",
    },
}


class SynchronizedRecorder:
    """
    Time-synchronized audio recorder.
    Uses wall-clock timing to ensure output matches input timeline.
    """
    
    def __init__(self, out_sr: int, target_sec: float, outfile: str):
        self.out_sr = out_sr
        self.target_samples = int(round(target_sec * out_sr))
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.outfile = outfile
        
        # Timing - smaller tick for more responsive interrupt handling
        self.tick_samples = out_sr * REC_TICK_MS // 1000
        self._silence = np.zeros(self.tick_samples, dtype=np.int16)
        
        # State
        self.count = 0
        self.muted = False
        self.running = True
        self.start_time = None
        
        # Stats
        self.audio_bytes_written = 0
        self.silence_samples_written = 0
    
    async def add(self, pcm: bytes):
        """Add audio data. Un-mutes if muted."""
        if self.muted:
            self.muted = False
        await self.queue.put(pcm)
    
    def interrupt(self):
        """Immediately stop speaking - clear queue and mute."""
        cleared = 0
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                cleared += 1
            except asyncio.QueueEmpty:
                break
        self.muted = True
        print(f"[DEBUG] Recorder: INTERRUPTED! Cleared {cleared} chunks, writing silence")
    
    def stop(self):
        self.running = False
    
    async def run(self):
        """
        Main loop - keeps running until stop() is called.
        Writes audio when available, silence when muted/empty.
        """
        print("[DEBUG] Recorder: Started")
        
        os.makedirs(os.path.dirname(self.outfile), exist_ok=True)
        wf = wave.open(self.outfile, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self.out_sr)
        
        try:
            # Keep running until stop() is called AND we've written enough samples
            while self.running or self.count < self.target_samples:
                # Stop if we've written enough AND stop was requested
                if not self.running and self.count >= self.target_samples:
                    break
                
                # If NOT muted and have audio → write audio
                if not self.muted and not self.queue.empty():
                    pcm = await self.queue.get()
                    smp = np.frombuffer(pcm, dtype=np.int16)
                    
                    # Write audio and track it
                    if self.count + len(smp) <= self.target_samples:
                        wf.writeframes(pcm)
                        self.count += len(smp)
                        self.audio_bytes_written += len(pcm)  # Track actual bytes
                    elif self.count < self.target_samples:
                        # Write partial
                        n = self.target_samples - self.count
                        wf.writeframes(smp[:n].tobytes())
                        self.audio_bytes_written += n * 2
                        self.count = self.target_samples
                else:
                    # Write silence (but only if we haven't reached target yet)
                    if self.count >= self.target_samples:
                        # Wait a bit before checking again
                        await asyncio.sleep(0.01)
                        continue
                    
                    remain = self.target_samples - self.count
                    n = min(self.tick_samples, remain)
                    smp = self._silence[:n]
                    
                    wf.writeframes(smp.tobytes())
                    self.count += n
                    self.silence_samples_written += n
                
                # IMPORTANT: Simulate real-time pacing!
                # If we wrote audio, sleep for its duration
                if not self.muted and 'smp' in locals() and len(smp) > 0:
                    duration_sec = len(smp) / self.out_sr
                    await asyncio.sleep(duration_sec)
                # If we wrote silence (muted/empty), sleep for one tick
                elif self.muted and self.queue.empty():
                    await asyncio.sleep(self.tick_samples / self.out_sr)
            
            # Fill any remaining silence if needed
            if self.count < self.target_samples:
                remaining = self.target_samples - self.count
                wf.writeframes(np.zeros(remaining, dtype=np.int16).tobytes())
                self.silence_samples_written += remaining
                self.count = self.target_samples
                
        finally:
            wf.close()
        
        audio_sec = self.audio_bytes_written / 2 / self.out_sr
        silence_sec = self.silence_samples_written / self.out_sr
        print(f"[DEBUG] Recorder: Done. Audio: {audio_sec:.2f}s, Silence: {silence_sec:.2f}s")


def resample_to_16k(input_path: Path) -> Tuple[Path, float]:
    """Resample to 16kHz mono."""
    data, sr = sf.read(input_path, always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    duration = len(data) / sr

    if sr != SEND_SAMPLE_RATE:
        g = math.gcd(int(sr), SEND_SAMPLE_RATE)
        data = ss.resample_poly(data, SEND_SAMPLE_RATE // g, int(sr) // g)

    # REMOVED: Max Normalization
    # This was amplifying background noise and triggering VAD!
    # data = (data / max_val * 32767).astype(np.int16)
    
    # Just clip and convert to int16 to preserve original volume
    data = np.clip(data * 32767, -32768, 32767).astype(np.int16)

    out_path = input_path.with_name(f"{input_path.stem}_16k_mono.wav")
    sf.write(out_path, data, SEND_SAMPLE_RATE, subtype="PCM_16")
    return out_path, duration


def load_audio_chunks(wav16k_path: Path) -> List[bytes]:
    """Split into chunks."""
    data, _ = sf.read(wav16k_path, dtype="int16")
    pad = (-len(data)) % CHUNK_SIZE
    if pad:
        data = np.pad(data, (0, pad))

    return [data[i:i+CHUNK_SIZE].tobytes() for i in range(0, len(data), CHUNK_SIZE)]


async def run_session(
    client: genai.Client,
    session_id: int,
    chunks: List[bytes],
    start_idx: int,
    recorder: SynchronizedRecorder,
    session_start_time: float,
) -> int:
    """
    Run one session.
    """
    chunk_duration = CHUNK_SIZE / SEND_SAMPLE_RATE
    session_time_offset = start_idx * chunk_duration
    
    print(f"[DEBUG][Session {session_id}] Start from chunk {start_idx} (t={session_time_offset:.2f}s)")
    
    session_done = False
    idx = start_idx
    total = len(chunks)
    audio_received = 0
    was_interrupted = False
    
    async with client.aio.live.connect(model=MODEL, config=CONFIG) as sess:
        print(f"[DEBUG][Session {session_id}] Connected to Live API")
        
        async def sender():
            nonlocal idx, session_done
            while idx < total and not session_done:
                chunk = chunks[idx]
                
                # REMOVED: Client-side Noise Gate
                
                await sess.send_realtime_input(
                    audio={"data": chunk, "mime_type": "audio/pcm"}
                )
                idx += 1
                # Real-time pacing
                await asyncio.sleep(chunk_duration)
            
            if not session_done:
                print(f"[DEBUG][Session {session_id}] Sender finished sending all chunks")
                try:
                    await sess.send_realtime_input(audio_stream_end=True)
                except Exception:
                    pass
        
        async def receiver():
            nonlocal session_done, audio_received, was_interrupted
            print(f"[DEBUG][Session {session_id}] Receiver started")
            async for resp in sess.receive():
                sc = resp.server_content
                if not sc:
                    continue
                
                # INTERRUPT
                if getattr(sc, "interrupted", False):
                    current_time = time.time() - session_start_time
                    print(f"[DEBUG][Session {session_id}] *** INTERRUPTED at t={current_time:.2f}s, chunk {idx} ***")
                    recorder.interrupt()
                    was_interrupted = True
                    session_done = True
                    return
                
                # Model Audio
                if sc.model_turn:
                    for part in sc.model_turn.parts:
                        if part.inline_data and isinstance(part.inline_data.data, bytes):
                            await recorder.add(part.inline_data.data)
                            audio_received += len(part.inline_data.data)
                        
                        if getattr(part, "generation_complete", False):
                            session_done = True
                            return
                
                # Turn Complete
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


async def process_single_file(input_wav: str, output_wav: str, overwrite: bool = True) -> bool:
    """Process file with time-synchronized multi-session approach."""
    input_path = Path(input_wav)
    if not input_path.exists():
        print(f"[ERROR] {input_wav} not found.")
        return False

    if os.path.exists(output_wav) and not overwrite:
        return True

    # Prepare audio
    wav16k_path, duration = resample_to_16k(input_path)
    chunks = load_audio_chunks(wav16k_path)
    total_chunks = len(chunks)
    print(f"[INFO] Loaded {total_chunks} chunks, duration: {duration:.2f}s")

    # Create recorder
    recorder = SynchronizedRecorder(RECEIVE_SAMPLE_RATE, duration, output_wav)
    recorder_task = asyncio.create_task(recorder.run())

    # Create client
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Track overall start time for synchronization
    session_start_time = time.time()

    # Multi-session loop
    chunk_idx = 0
    session_id = 1
    
    while chunk_idx < total_chunks:
        try:
            new_idx = await run_session(
                client, session_id, chunks, chunk_idx, recorder, session_start_time
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
    
    # Finish recording
    recorder.stop()
    await recorder_task

    # Cleanup
    if os.path.exists(wav16k_path):
        os.remove(wav16k_path)

    print(f"[INFO] Saved {output_wav}")
    return True


async def batch_process(args):
    """Process all files."""
    base_dir = os.path.expanduser(args.base_dir)
    pattern = os.path.join(base_dir, args.task or "*", "*", f"{args.prefix}input.wav")
    files = sorted(glob(pattern))
    print(f"Found {len(files)} files.")

    success = 0
    for i, f in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {f}")
        # out_dir = os.path.join(os.path.dirname(f), "gemini25_native")
        # os.makedirs(out_dir, exist_ok=True)
        out_wav = os.path.join(os.path.dirname(f), "output.wav")

        if os.path.exists(out_wav) and not args.overwrite:
            print("Skip")
            success += 1
            continue

        try:
            if await process_single_file(f, out_wav, args.overwrite):
                success += 1
        except Exception as e:
            print(f"Failed: {e}")

    print(f"\nDone. {success}/{len(files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--overwrite", action="store_true")
    asyncio.run(batch_process(parser.parse_args()))
