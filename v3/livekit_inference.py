#!/usr/bin/env python3
"""
Headless LiveKit client: stream a local WAV file into a LiveKit room and
capture the agent's audio response – no browser or web UI required.

This script acts as a "user" participant. It joins the same LiveKit room that
the agent (lk_agent_tool.py) is listening on, publishes audio from a WAV file,
and saves the agent's spoken response to an output WAV file.

Usage:
    python livekit_inference.py -i input.wav -o response.wav [--room ROOM_NAME]

Requirements:
    pip install "livekit[crypto]~=1.0" python-dotenv numpy

Environment variables (in .env.local):
    LIVEKIT_URL          – wss://your-project.livekit.cloud
    LIVEKIT_API_KEY      – your LiveKit API key
    LIVEKIT_API_SECRET   – your LiveKit API secret
"""

import argparse
import asyncio
import logging
import os
import subprocess
import wave

import numpy as np

from dotenv import load_dotenv
from livekit import api, rtc

load_dotenv(".env.local")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("livekit_inference")

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

SAMPLE_WIDTH = 2  # 16-bit PCM = 2 bytes per sample


def read_wav_pcm16(path: str, target_rate: int = 48000) -> tuple[bytes, int, int]:
    """
    Read any audio file and return (pcm_bytes, sample_rate, channels).
    Converts to mono PCM-16 at *target_rate* Hz via ffmpeg if needed.
    """
    try:
        with wave.open(path, "rb") as wf:
            rate = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            if ch == 1 and sw == 2 and rate == target_rate:
                return wf.readframes(wf.getnframes()), rate, 1
    except wave.Error:
        pass

    log.info("Converting '%s' to %d Hz mono 16-bit PCM via ffmpeg …", path, target_rate)
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", path,
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ar", str(target_rate),
                "-ac", "1",
                "pipe:1",
            ],
            capture_output=True,
            check=True,
        )
        return result.stdout, target_rate, 1
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is required. Install with: brew install ffmpeg")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg conversion failed: {e.stderr.decode()}")


def write_wav(path: str, pcm_data: bytes, sample_rate: int, channels: int = 1):
    """Write raw PCM-16 bytes to a WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


# ---------------------------------------------------------------------------
# Main client logic
# ---------------------------------------------------------------------------

# LiveKit WebRTC uses 48 kHz by default for audio tracks.
PUBLISH_SAMPLE_RATE = 48000
CAPTURE_SAMPLE_RATE = 24000  # sample rate we request when receiving agent audio
CHUNK_DURATION_MS = 20  # 20 ms chunks (standard for WebRTC)


async def run(input_wav: str, output_wav: str, room_name: str):
    url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not all([url, api_key, api_secret]):
        raise RuntimeError(
            "Missing environment variables. Set LIVEKIT_URL, LIVEKIT_API_KEY, "
            "and LIVEKIT_API_SECRET in .env.local"
        )

    # ── Read input audio ──────────────────────────────────────────────
    pcm_data, src_rate, src_channels = read_wav_pcm16(input_wav, PUBLISH_SAMPLE_RATE)
    total_samples = len(pcm_data) // SAMPLE_WIDTH
    duration_sec = total_samples / src_rate
    log.info(
        "Input: %s  (%.2fs, %d Hz, mono 16-bit, %d bytes)",
        input_wav, duration_sec, src_rate, len(pcm_data),
    )

    # ── Generate access token ─────────────────────────────────────────
    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity("wav-file-user")
        .with_name("WAV File User")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
            )
        )
        .to_jwt()
    )

    # ── Pre-allocate output buffer (same length as input) ───────────
    # The output WAV will be exactly the same duration as the input WAV.
    # Agent audio is written into this buffer at the time offset it
    # arrives (relative to when we start streaming the input).
    target_samples = int(duration_sec * CAPTURE_SAMPLE_RATE)
    output_buf = np.zeros(target_samples, dtype=np.int16)
    write_pos = 0  # next sample position to write into output_buf
    recording_started = asyncio.Event()
    recording_stop = asyncio.Event()

    # ── Connect to room ───────────────────────────────────────────────
    room = rtc.Room()

    receiving_task: asyncio.Task | None = None

    async def _receive_agent_audio(track: rtc.Track):
        """Collect agent audio frames into output_buf in real-time."""
        nonlocal write_pos
        stream = rtc.AudioStream(
            track,
            sample_rate=CAPTURE_SAMPLE_RATE,
            num_channels=1,
        )
        log.info("Receiving agent audio …")

        # Wait until we actually start streaming input so the clocks align
        await recording_started.wait()

        try:
            while not recording_stop.is_set():
                try:
                    # Timeout after 0.5s to check recording_stop
                    frame_event = await asyncio.wait_for(stream.__anext__(), timeout=0.5)
                    frame = frame_event.frame
                    samples = np.frombuffer(bytes(frame.data), dtype=np.int16)
                    n = len(samples)
                    remaining = target_samples - write_pos
                    if remaining <= 0:
                        break
                    to_write = min(n, remaining)
                    output_buf[write_pos : write_pos + to_write] = samples[:to_write]
                    write_pos += to_write
                except asyncio.TimeoutError:
                    continue
                except StopAsyncIteration:
                    break
        except Exception as e:
            log.warning("Audio receive error: %s", e)
        finally:
            log.info("Closing audio stream...")
            await stream.aclose()
            log.info(
                "Recording stopped at %.2fs of %.2fs.",
                write_pos / CAPTURE_SAMPLE_RATE,
                duration_sec,
            )

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            log.info(
                "Subscribed to audio track from '%s' (%s)",
                participant.identity, participant.name,
            )
            nonlocal receiving_task
            receiving_task = asyncio.create_task(_receive_agent_audio(track))

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        log.info("Participant joined: %s (%s)", participant.identity, participant.name)

    @room.on("disconnected")
    def on_disconnected():
        log.info("Disconnected from room.")

    log.info("Connecting to room '%s' at %s …", room_name, url)
    await room.connect(
        url,
        token,
        options=rtc.RoomOptions(auto_subscribe=True),
    )
    log.info("Connected to room '%s'. Waiting for agent to join …", room.name)

    # ── Publish audio from WAV ────────────────────────────────────────
    source = rtc.AudioSource(src_rate, src_channels)
    local_track = rtc.LocalAudioTrack.create_audio_track("wav-input", source)

    pub_opts = rtc.TrackPublishOptions()
    pub_opts.source = rtc.TrackSource.SOURCE_MICROPHONE
    await room.local_participant.publish_track(local_track, pub_opts)
    log.info("Published audio track. Streaming WAV …")

    # Give a moment for the agent to connect and subscribe
    await asyncio.sleep(2)

    # ── Stream input & record output in parallel ──────────────────────
    samples_per_chunk = src_rate * CHUNK_DURATION_MS // 1000
    chunk_bytes = samples_per_chunk * src_channels * SAMPLE_WIDTH

    # Signal recording start — the receive task starts writing from here
    import time
    stream_start_time = time.time()
    print(f"STREAM_START_TIME: {stream_start_time}")
    recording_started.set()
    log.info("Recording started (%.2fs window).", duration_sec)

    offset = 0
    chunks_sent = 0

    while offset < len(pcm_data):
        end = min(offset + chunk_bytes, len(pcm_data))
        chunk = pcm_data[offset:end]
        num_samples = len(chunk) // (SAMPLE_WIDTH * src_channels)

        frame = rtc.AudioFrame(
            data=chunk,
            sample_rate=src_rate,
            num_channels=src_channels,
            samples_per_channel=num_samples,
        )
        await source.capture_frame(frame)
        offset = end
        chunks_sent += 1
        await asyncio.sleep(CHUNK_DURATION_MS / 1000)

    log.info("Finished streaming %d chunks (%.2fs of audio).", chunks_sent, duration_sec)

    # Send trailing silence so VAD detects end-of-speech
    silence_duration_ms = 1500
    silence_chunks = silence_duration_ms // CHUNK_DURATION_MS
    silence_data = b"\x00" * chunk_bytes
    for _ in range(silence_chunks):
        frame = rtc.AudioFrame(
            data=silence_data,
            sample_rate=src_rate,
            num_channels=src_channels,
            samples_per_channel=samples_per_chunk,
        )
        await source.capture_frame(frame)
        await asyncio.sleep(CHUNK_DURATION_MS / 1000)

    # ── Wait until exactly input-duration has elapsed since start ─────
    elapsed = time.time() - stream_start_time
    remaining_wait = duration_sec - elapsed
    if remaining_wait > 0:
        log.info("Waiting %.2fs for recording window to finish …", remaining_wait)
        await asyncio.sleep(remaining_wait)

    recording_stop.set()
    log.info("Recording window closed.")

    # Give the receive task a moment to finish its current frame
    if receiving_task and not receiving_task.done():
        await asyncio.sleep(0.1)
        if not receiving_task.done():
            receiving_task.cancel()
            try:
                await receiving_task
            except asyncio.CancelledError:
                pass

    # ── Save output (exact same duration as input) ────────────────────
    out_bytes = output_buf.tobytes()
    write_wav(output_wav, out_bytes, CAPTURE_SAMPLE_RATE, 1)
    actual_speech = np.count_nonzero(output_buf) / CAPTURE_SAMPLE_RATE
    log.info(
        "Saved: %s  (%.2fs, %d Hz, %d bytes, ~%.2fs of non-silence)",
        output_wav, duration_sec, CAPTURE_SAMPLE_RATE, len(out_bytes), actual_speech,
    )

    # ── Cleanup ───────────────────────────────────────────────────────
    log.info("Disconnecting from room...")
    await room.disconnect()
    log.info("Disconnected. Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stream a local WAV file to a LiveKit room and capture the agent's "
            "audio response. Requires lk_agent_tool.py running in a separate terminal."
        ),
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to the input WAV file to send.",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Path for the output WAV file (agent's response).",
    )
    parser.add_argument(
        "--room", default="test-room",
        help="LiveKit room name (default: test-room).",
    )
    args = parser.parse_args()
    asyncio.run(run(args.input, args.output, args.room))


if __name__ == "__main__":
    main()