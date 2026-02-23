#!/usr/bin/env python
# coding: utf-8
"""
-----------------------------------------------------------------------
Usage:
    python inference.py --input input.wav --output output.wav \
        --region us-east-1 --model amazon.nova-sonic-v1:0 
"""

from glob import glob
import os
import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from rx.subject import Subject
from rx import operators as ops
from rx.scheduler.eventloop import AsyncIOScheduler


### Configuration ###
# AWS credentials are loaded from .env file
# Required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

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

region = "us-east-1"
model = "amazon.nova-sonic-v1:0"


INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
CHUNK_SIZE = 512
FRAME_DUR = CHUNK_SIZE / INPUT_SAMPLE_RATE
OUT_SAMPLES_PER_FRAME = int(FRAME_DUR * OUTPUT_SAMPLE_RATE)


def _mono(sig: np.ndarray) -> np.ndarray:
    return sig if sig.ndim == 1 else sig.mean(axis=1)


def _resample(sig: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return sig
    wav_f32 = torch.from_numpy(sig.astype(np.float32) / 32768.0).unsqueeze(0)
    wav_rs = AF.resample(wav_f32, orig_sr, target_sr)
    return (wav_rs.squeeze().numpy() * 32768).astype(np.int16)


def _chunk(sig: np.ndarray, frame_len: int) -> List[np.ndarray]:
    pad = (-len(sig)) % frame_len
    if pad:
        sig = np.concatenate([sig, np.zeros(pad, dtype=sig.dtype)])
    return [sig[i : i + frame_len] for i in range(0, len(sig), frame_len)]


from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.models import (
    InvokeModelWithBidirectionalStreamInputChunk,
    BidirectionalInputPayloadPart,
)
from aws_sdk_bedrock_runtime.config import (
    Config,
    HTTPAuthSchemeResolver,
    SigV4AuthScheme,
)
from smithy_aws_core.credentials_resolvers.environment import (
    EnvironmentCredentialsResolver,
)

DEBUG = False


def dprint(msg):
    if DEBUG:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


class BedrockStreamManager:
    START_SESSION_EVENT = '{"event":{"sessionStart":{"inferenceConfiguration":{"maxTokens":1024,"topP":0.9,"temperature":0.7}}}}'
    PROMPT_END_EVENT = '{"event":{"promptEnd":{"promptName":"%s"}}}'
    SESSION_END_EVENT = '{"event":{"sessionEnd":{}}}'

    START_PROMPT_EVENT = '{"event":{"promptStart":{"promptName":"%s","textOutputConfiguration":{"mediaType":"text/plain"},"audioOutputConfiguration":{"mediaType":"audio/lpcm","sampleRateHertz":24000,"sampleSizeBits":16,"channelCount":1,"voiceId":"matthew","encoding":"base64","audioType":"SPEECH"},"toolUseOutputConfiguration":{"mediaType":"application/json"},"toolConfiguration":{"tools":[]}}}}'
    CONTENT_START_EVENT = '{"event":{"contentStart":{"promptName":"%s","contentName":"%s","type":"AUDIO","interactive":true,"role":"USER","audioInputConfiguration":{"mediaType":"audio/lpcm","sampleRateHertz":16000,"sampleSizeBits":16,"channelCount":1,"audioType":"SPEECH","encoding":"base64"}}}}'
    AUDIO_EVENT = (
        '{"event":{"audioInput":{"promptName":"%s","contentName":"%s","content":"%s"}}}'
    )
    CONTENT_END_EVENT = (
        '{"event":{"contentEnd":{"promptName":"%s","contentName":"%s"}}}'
    )

    ####
    SYS_START = (
        '{"event":{"contentStart":{"promptName":"%s","contentName":"sys",'
        '"role":"SYSTEM","type":"TEXT","interactive":true,'
        '"textInputConfiguration":{"mediaType":"text/plain"}}}}'
    )
    SYS_TEXT = (
        '{"event":{"textInput":{"promptName":"%s","contentName":"sys",'
        '"content":"You are a helpful assistant."}}}'
    )
    SYS_END = '{"event":{"contentEnd":{"promptName":"%s","contentName":"sys"}}}'
    ####

    def __init__(self, model_id: str, region: str):
        self.model_id = model_id
        self.region = region
        self.prompt_name = "p"
        self.content_idx = 0
        self.content_name = f"c{self.content_idx}"

        cfg = Config(
            endpoint_uri=f"https://bedrock-runtime.{region}.amazonaws.com",
            region=region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()},
        )
        self.client = BedrockRuntimeClient(config=cfg)
        self.scheduler = AsyncIOScheduler(asyncio.get_event_loop())

        self.audio_out_q = asyncio.Queue()
        self.is_active = False
        self.barge_in: bool = False
        self.interrupted = False

    def next_content(self) -> None:
        self.content_idx += 1
        self.content_name = f"c{self.content_idx}"
        self.interrupted = False

    async def start(self):
        self.stream_rsp = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self.is_active = True

        await self._send_event(self.START_SESSION_EVENT)
        await self._send_event(self.START_PROMPT_EVENT % self.prompt_name)
        await self._send_event(self.SYS_START % self.prompt_name)
        await self._send_event(self.SYS_TEXT % self.prompt_name)
        await self._send_event(self.SYS_END % self.prompt_name)
        # listener
        self.listener_task = asyncio.create_task(self._listen_loop())

    async def begin_audio(self):
        await self._send_event(
            self.CONTENT_START_EVENT % (self.prompt_name, self.content_name)
        )

    def send_chunk(self, pcm_bytes: bytes):
        if not self.is_active:
            return
        b64 = base64.b64encode(pcm_bytes).decode("utf-8")
        asyncio.get_event_loop().create_task(
            self._send_event(
                self.AUDIO_EVENT % (self.prompt_name, self.content_name, b64)
            )
        )

    async def end_audio(self):
        await self._send_event(
            self.CONTENT_END_EVENT % (self.prompt_name, self.content_name)
        )

    async def close(self, *, timeout: float = 10.0):
        if not self.is_active:
            return

        if not self.interrupted:
            await self._send_event(self.PROMPT_END_EVENT % self.prompt_name)
            await self._send_event(self.SESSION_END_EVENT)

        await self.stream_rsp.input_stream.close()
        self.is_active = False

        if hasattr(self, "listener_task") and not self.listener_task.done():
            try:
                await asyncio.wait_for(self.listener_task, timeout=timeout)
            except:
                print("error: listener task did not finish in time, canceling...")
                pass

    async def _send_event(self, event_json: str):
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )
        await self.stream_rsp.input_stream.send(chunk)
        dprint(f"⇢ {event_json[:60]}...")

    async def _listen_loop(self):

        try:
            while self.is_active:
                output = await self.stream_rsp.await_output()
                res = await output[1].receive()
                if not res.value or not res.value.bytes_:
                    continue

                msg_raw = res.value.bytes_.decode()
                dprint("⇠ " + msg_raw[:120] + ("..." if len(msg_raw) > 120 else ""))

                msg = json.loads(msg_raw)

                if "audioOutput" in msg.get("event", {}):
                    pcm = base64.b64decode(msg["event"]["audioOutput"]["content"])
                    await self.audio_out_q.put(pcm)

                if "textOutput" in msg.get("event", {}):
                    txt = msg["event"]["textOutput"]["content"]
                    if '"interrupted"' in txt:
                        print(">>> INTERRUPTED FLAG:", txt)

                        self.barge_in = True
                        self.interrupted = True

                        while not self.audio_out_q.empty():
                            try:
                                self.audio_out_q.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        continue

                if "sessionEnd" in msg.get("event", {}):
                    self.is_active = False
        finally:
            await self.audio_out_q.put(None)


min_flush_default = int(0.005 * OUTPUT_SAMPLE_RATE)  # 5 ms
min_flush = min_flush_default


async def stream_wav(input_path: Path, output_path: Path, region: str, model: str):
    wav, sr = sf.read(str(input_path), dtype="int16")
    wav = _mono(wav)
    wav = _resample(wav, sr, INPUT_SAMPLE_RATE)
    orig_samples = len(wav)
    frames = _chunk(wav, CHUNK_SIZE)

    mgr = BedrockStreamManager(model_id=model, region=region)
    await mgr.start()
    await mgr.begin_audio()

    async def writer_task(t0: float):
        expected = int(orig_samples * OUTPUT_SAMPLE_RATE / INPUT_SAMPLE_RATE)
        total_out = 0

        pcm_buffer, buffer_samples = [], 0
        min_flush = min_flush_default  # 200 ms
        recent_voice = t0
        voice_gap = 0.5
        silence_10ms = np.zeros(int(0.01 * OUTPUT_SAMPLE_RATE), np.int16)

        with sf.SoundFile(
            str(output_path),
            "w",
            samplerate=OUTPUT_SAMPLE_RATE,
            channels=1,
            subtype="PCM_16",
        ) as fout:
            while total_out < expected:

                try:
                    data = await asyncio.wait_for(
                        mgr.audio_out_q.get(), timeout=FRAME_DUR
                    )
                except asyncio.TimeoutError:
                    data = None

                now = time.time()

                if mgr.barge_in:
                    pcm_buffer, buffer_samples = [], 0

                    mgr.barge_in = False
                    recent_voice = now
                    min_flush = 0
                    continue

                if data:
                    pcm = np.frombuffer(data, dtype=np.int16)
                    if pcm.size:
                        recent_voice = now
                        pcm_buffer.append(pcm)
                        buffer_samples += pcm.size
                else:
                    if now - recent_voice > voice_gap:
                        pcm_buffer.append(silence_10ms)
                        buffer_samples += silence_10ms.size

                if buffer_samples >= max(min_flush, 1):
                    merged = np.concatenate(pcm_buffer)
                    pcm_buffer, buffer_samples = [], 0
                    min_flush = min_flush_default

                    should_be = int((now - t0) * OUTPUT_SAMPLE_RATE)

                    if total_out < should_be:
                        pad = np.zeros(should_be - total_out, np.int16)
                        fout.write(pad)
                        total_out += pad.size

                    remain = expected - total_out
                    if merged.size > remain:
                        merged = merged[:remain]
                    fout.write(merged)
                    total_out += merged.size

        if pcm_buffer and total_out < expected:
            merged = np.concatenate(pcm_buffer)
            should_be = int((time.time() - t0) * OUTPUT_SAMPLE_RATE)
            if total_out < should_be:
                pad = np.zeros(
                    min(should_be - total_out, expected - total_out), np.int16
                )
                fout.write(pad)
                total_out += pad.size
            fout.write(merged[: expected - total_out])

    t0 = time.time()
    wt = asyncio.create_task(writer_task(t0))

    for f in frames:

        if mgr.barge_in:

            await mgr.end_audio()

            mgr.next_content()

            await mgr.begin_audio()
            mgr.barge_in = False

        mgr.send_chunk(f.tobytes())
        await asyncio.sleep(FRAME_DUR)

    await mgr.end_audio()
    await wt
    await mgr.close()

    print(f"[DONE] {input_path.name} → {output_path.name}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():

    for inp in all_wav_files:
        input = Path(inp)
        output = Path(inp.replace("input.wav", "output.wav"))
        if not overwrite and output.exists():
            print(f"[SKIP] {output} already exists, skipping...")
            continue
        print(f"[RUN] {input} → {output}")

        asyncio.run(stream_wav(input, output, region=region, model=model))


if __name__ == "__main__":
    main()
