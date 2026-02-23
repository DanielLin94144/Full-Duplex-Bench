from __future__ import annotations

import argparse
import asyncio
import json
import queue
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import socketio
import torch
import torchaudio.functional as AF
from glob import glob

### Configuration ###
root_dir_path = "YOUR_ROOT_DIRECTORY_PATH"
tasks = [
    "YOUR_TASK_NAME",
]
prefix = ""  # "" or "clean_": the prefix for input wav files
overwrite = True  # Whether to overwrite existing output files
#####################

all_wav_files = []
for task in tasks:
    root_dir = f"{root_dir_path}/{task}/"
    root_file_dir = f"{root_dir}/*/{prefix}input.wav"
    wav_files = sorted(glob(root_file_dir))
    all_wav_files.extend(wav_files)

FRAME_MS = 30
SEND_SR = 16_000
RECV_SR = 24_000
TX_SAMP = int(SEND_SR * FRAME_MS / 1000)
RX_SAMP = int(RECV_SR * FRAME_MS / 1000)
RX_BYTES = RX_SAMP * 2


def _mono(sig: np.ndarray) -> np.ndarray:
    return sig if sig.ndim == 1 else sig.mean(axis=1)


def _resample(sig: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return sig
    wav = torch.from_numpy(sig.astype(np.float32) / 32768).unsqueeze(0)
    wav_rs = AF.resample(wav, orig_sr, target_sr)
    return (wav_rs.squeeze().numpy() * 32768).astype(np.int16)


def _chunk(sig: np.ndarray, frame_len: int) -> List[np.ndarray]:
    pad = (-len(sig)) % frame_len
    if pad:
        sig = np.concatenate([sig, np.zeros(pad, dtype=sig.dtype)])
    return [sig[i : i + frame_len] for i in range(0, len(sig), frame_len)]


def _compact_json(obj):
    return json.dumps(obj, separators=(",", ":"))


class FreezeOmniClient:
    def __init__(self, server_ip: str, inp: Path, out: Path):
        self.server_ip = server_ip
        self.inp = inp
        self.out = out
        self.audio_q = queue.Queue()
        self.pending = bytearray()
        self.muted = False  # true after stop_tts until next audio

        self.sio = socketio.Client(
            ssl_verify=False,
            reconnection=True,
            reconnection_attempts=0,
            reconnection_delay=2,
            reconnection_delay_max=30,
            randomization_factor=0.2,
        )

        self.sio.on("connect", self._on_connect)
        self.sio.on("disconnect", self._on_disconnect)
        self.sio.on("audio", self._on_audio)
        self.sio.on("stop_tts", self._on_stop_tts)
        self.sio.on("too_many_users", self._on_too_many)

    # ---------------- Socket.IO callbacks ----------------
    def _on_connect(self):
        print("[SIO] ✅ Connected", flush=True)
        asyncio.run(self._stream())

    def _on_disconnect(self):
        print("[SIO] 🔌 Disconnected", flush=True)

    def _on_audio(self, data: bytes):
        self.audio_q.put(data)
        self.muted = False  # new audio resumes output

    def _on_stop_tts(self):
        print("[SIO] ⏹️  stop_tts → mute", flush=True)
        self.pending.clear()  # discard any buffered TTS
        self.muted = True

    def _on_too_many(self, *_, **__):
        print("[SIO] ❌ Too many users", file=sys.stderr)
        self.sio.disconnect()

    # ---------------- Main coroutine ----------------
    async def _stream(self):
        wav, sr = sf.read(self.inp, dtype="int16")
        wav = _mono(wav)
        wav = _resample(wav, sr, SEND_SR)
        tx_frames = _chunk(wav, TX_SAMP)
        total_frames = len(tx_frames)
        frames_written = 0

        with sf.SoundFile(
            self.out, "w", samplerate=RECV_SR, channels=1, subtype="PCM_16"
        ) as fout:
            self.sio.emit("recording-started")
            frame_dur = FRAME_MS / 1000.0

            for frame in tx_frames:
                self.sio.emit(
                    "audio",
                    _compact_json(
                        {"audio": list(frame.tobytes()), "sample_rate": SEND_SR}
                    ),
                )

                while not self.audio_q.empty():
                    self.pending.extend(self.audio_q.get())

                # decide chunk to write
                if self.muted:
                    chunk = b""  # force silence
                else:
                    chunk = self.pending[:RX_BYTES]
                    self.pending = self.pending[RX_BYTES:]

                if len(chunk) < RX_BYTES:
                    chunk += b"\x00" * (RX_BYTES - len(chunk))
                fout.write(np.frombuffer(chunk, dtype=np.int16))
                frames_written += 1

                await asyncio.sleep(frame_dur)

            self.sio.emit("recording-stopped")
            flush_until = time.time() + 1.0
            while time.time() < flush_until and frames_written < total_frames:
                while not self.audio_q.empty():
                    self.pending.extend(self.audio_q.get())
                chunk = b"" if self.muted else self.pending[:RX_BYTES]
                self.pending = self.pending[RX_BYTES:]
                if len(chunk) < RX_BYTES:
                    chunk += b"\x00" * (RX_BYTES - len(chunk))
                fout.write(np.frombuffer(chunk, dtype=np.int16))
                frames_written += 1
                await asyncio.sleep(frame_dur)

            while frames_written < total_frames:
                fout.write(np.zeros(RX_SAMP, dtype=np.int16))
                frames_written += 1

        self.sio.disconnect()
        print(
            f"[DONE] input len = {len(wav) / SEND_SR:.2f}s | output len = {sf.info(self.out).duration:.2f}s"
        )

    # ---------------- Public API ----------------
    def run(self):
        url = f"https://{self.server_ip}"
        try:
            self.sio.connect(url, transports=["websocket"], wait_timeout=10)
            self.sio.wait()
            if self.sio.connected:
                # disconnect after processing
                self.sio.disconnect()

        except KeyboardInterrupt:
            self.sio.disconnect()
        except Exception as e:
            print(f"[ERR] {e}", file=sys.stderr)
            self.sio.disconnect()


# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Freeze-Omni streaming client with instant stop_tts mute"
    )
    ap.add_argument("--server_ip", required=True)
    args = ap.parse_args()

    for inp in all_wav_files:
        args.input = Path(inp)
        args.output = Path(inp.replace("input.wav", "output.wav"))
        if not overwrite and args.output.exists():
            print(f"[SKIP] {args.output} already exists, skipping...")
            continue
        print(f"[RUN] {args.input} → {args.output}")

        FreezeOmniClient(args.server_ip, args.input, args.output).run()


if __name__ == "__main__":
    main()
