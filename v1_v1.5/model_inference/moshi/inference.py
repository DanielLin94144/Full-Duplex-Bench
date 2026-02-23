#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import sphn
import torch
import torchaudio.functional as AF
import websockets
import websockets.exceptions as wsex


### Configuration ###
root_dir_path = "YOUR_ROOT_DIRECTORY_PATH"
tasks = [
    "YOUR_TASK_NAME",
]
prefix = ""  # "" or "clean_": the prefix for input wav files
overwrite = True  # Whether to overwrite existing output files
#####################


SEND_SR = 24_000
FRAME_SMP = 1_920
SKIP_FRAMES = 1
FRAME_SEC = FRAME_SMP / SEND_SR


def _patch_sphn():
    if not hasattr(sphn.OpusStreamWriter, "read_bytes"):
        for alt in ("get_bytes", "flush_bytes", "read_data"):
            if hasattr(sphn.OpusStreamWriter, alt):
                setattr(
                    sphn.OpusStreamWriter,
                    "read_bytes",
                    getattr(sphn.OpusStreamWriter, alt),
                )
                break
        else:
            setattr(sphn.OpusStreamWriter, "read_bytes", lambda self: b"")
    if not hasattr(sphn.OpusStreamReader, "read_pcm"):
        for alt in ("get_pcm", "receive_pcm", "read_float"):
            if hasattr(sphn.OpusStreamReader, alt):
                setattr(
                    sphn.OpusStreamReader,
                    "read_pcm",
                    getattr(sphn.OpusStreamReader, alt),
                )
                break
        else:
            setattr(
                sphn.OpusStreamReader, "read_pcm", lambda self: np.empty(0, np.float32)
            )


_patch_sphn()


def _mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1)


def _resample(x: np.ndarray, sr: int, tgt: int) -> np.ndarray:
    if sr == tgt:
        return x
    y = torch.from_numpy(x.astype(np.float32) / 32768).unsqueeze(0)
    y = AF.resample(y, sr, tgt)[0].numpy()
    return (y * 32768).astype(np.int16)


def _chunk(sig: np.ndarray) -> List[np.ndarray]:
    pad = (-len(sig)) % FRAME_SMP
    if pad:
        sig = np.concatenate([sig, np.zeros(pad, sig.dtype)])
    return [sig[i : i + FRAME_SMP] for i in range(0, len(sig), FRAME_SMP)]


class MoshiFileClient:
    def __init__(self, ws_url: str, inp: Path, out: Path):
        self.url, self.inp, self.out = ws_url, inp, out

        sig16, sr = sf.read(inp, dtype="int16")
        self.sig24 = _resample(_mono(sig16), sr, SEND_SR)
        self.max_samples = len(self.sig24)  # output 目標長度

        self.writer = sphn.OpusStreamWriter(SEND_SR)
        self.reader = sphn.OpusStreamReader(SEND_SR)

    # ---------------- sender ----------------
    async def _send(self, ws):
        for frame in _chunk(self.sig24):
            pkt0 = self.writer.append_pcm(frame.astype(np.float32) / 32768)
            if isinstance(pkt0, (bytes, bytearray)):
                await ws.send(b"\x01" + pkt0)
            queued = self.writer.read_bytes()
            if queued:
                await ws.send(b"\x01" + queued)
            await asyncio.sleep(FRAME_SEC)

        queued = self.writer.read_bytes()
        if queued:
            await ws.send(b"\x01" + queued)
        await asyncio.sleep(0.5)
        await ws.close()

    # ---------------- receiver ----------------
    async def _recv(self, ws):
        samples_written = 0
        first_pcm_seen = False

        with sf.SoundFile(
            self.out, "w", samplerate=SEND_SR, channels=1, subtype="PCM_16"
        ) as fout:
            try:
                async for msg in ws:
                    if not msg or msg[0] not in (1, 2):
                        continue
                    kind, payload = msg[0], msg[1:]

                    if kind == 1:  # audio bytes
                        self.reader.append_bytes(payload)
                        while True:
                            pcm = self.reader.read_pcm()
                            if pcm.size == 0:
                                break

                            if not first_pcm_seen:
                                pad = min(SKIP_FRAMES * FRAME_SMP, self.max_samples)
                                fout.write(np.zeros(pad, dtype=np.int16))
                                samples_written += pad
                                first_pcm_seen = True

                            remain = self.max_samples - samples_written
                            if remain <= 0:
                                continue
                            n_write = min(pcm.size, remain)
                            fout.write((pcm[:n_write] * 32768).astype(np.int16))
                            samples_written += n_write

                    else:
                        print("[TEXT]", payload.decode(errors="ignore"))

            except wsex.ConnectionClosedError as e:
                print("[WARN] recv closed:", e)

            if samples_written < self.max_samples:
                fout.write(np.zeros(self.max_samples - samples_written, dtype=np.int16))

    async def _run(self):
        async with websockets.connect(self.url, max_size=None) as ws:
            try:
                first = await asyncio.wait_for(ws.recv(), timeout=1.0)
                if not (isinstance(first, (bytes, bytearray)) and first[:1] == b"\x00"):
                    ws._put_message(first)
            except Exception:
                pass

            await asyncio.gather(self._send(ws), self._recv(ws))
        print("[DONE]", self.inp)

    def run(self):
        try:
            asyncio.run(self._run())
        except wsex.ConnectionClosedError as e:
            print("[WARN] closed:", e)


def _ws_url(addr: str) -> str:
    if "://" in addr:
        proto, rest = addr.split("://", 1)
        proto = "ws" if proto in {"http", "ws"} else "wss"
        return f"{proto}://{rest.rstrip('/')}/api/chat"
    if ":" not in addr:
        addr += ":8998"
    return f"ws://{addr}/api/chat"


def _input_files() -> List[Path]:
    files: List[Path] = []
    for t in tasks:
        pattern = root_dir_path / f"{t}/*/{prefix}input.wav"
        files += [Path(p) for p in sorted(glob(pattern))]
    return files


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("moshi_batch_client")
    ap.add_argument("--server_ip", required=True, help="host[:port] or http(s):// URL")
    args = ap.parse_args()

    url = _ws_url(args.server_ip)
    for inp in _input_files():
        out = inp.with_name(inp.name.replace("input.wav", "output.wav"))
        if not overwrite and out.exists():
            print("[SKIP]", out)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        print("[RUN]", inp)
        MoshiFileClient(url, inp, out).run()


if __name__ == "__main__":
    main()
