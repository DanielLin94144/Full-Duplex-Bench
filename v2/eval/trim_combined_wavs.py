#!/usr/bin/env python3

"""
Recursively find all files named 'A.wav', 'B.wav', or 'combined.wav' under a root
directory and trim them to a target duration (default: 120 seconds).

Behavior:
- Prefers using ffmpeg (if available) for broad WAV compatibility and speed.
- Falls back to Python's wave module for uncompressed PCM WAVs when ffmpeg is
  not available.
- Skips files already at or below the target duration when duration can be
  determined.
- Modifies files in-place safely via a temporary output then atomic replace.

Usage:
  python3 trim_combined_wavs.py /path/to/root [--seconds 120] [--dry-run]
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import typing as t
import wave


def is_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def find_combined_wav_files(root_dir: str, case_insensitive: bool = False) -> t.List[str]:
    target_names = ["A.wav", "B.wav", "combined.wav"]
    results: t.List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if case_insensitive:
                if filename.lower() in [name.lower() for name in target_names]:
                    results.append(os.path.join(dirpath, filename))
            else:
                if filename in target_names:
                    results.append(os.path.join(dirpath, filename))
    return results


def probe_duration_seconds(path: str) -> t.Optional[float]:
    """Return duration in seconds if determinable; otherwise None.

    Prefers ffprobe if available. Falls back to wave for PCM WAVs.
    """
    if is_ffmpeg_available():
        try:
            # Probe container-level duration for reliability across formats
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            text = (result.stdout or "").strip()
            if text:
                return float(text)
        except Exception:
            pass

    # Fallback: wave module for PCM
    try:
        with contextlib.closing(wave.open(path, "rb")) as wav:
            # The wave module only supports uncompressed PCM
            # If compressed, getsampwidth/getframerate may still work but reading could fail.
            frames = wav.getnframes()
            framerate = wav.getframerate()
            if framerate > 0:
                return frames / float(framerate)
    except Exception:
        return None
    return None


def trim_with_ffmpeg(input_path: str, output_path: str, seconds: int) -> None:
    """Use ffmpeg to trim to the first `seconds` seconds without re-encoding."""
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-t",
        str(seconds),
        "-c",
        "copy",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def trim_with_wave_module(input_path: str, output_path: str, seconds: int) -> None:
    """Trim PCM WAV to the first `seconds` seconds using the stdlib wave module."""
    with contextlib.closing(wave.open(input_path, "rb")) as in_wav:
        params = in_wav.getparams()  # nchannels, sampwidth, framerate, nframes, comptype, compname
        if params.comptype not in ("NONE", ""):
            raise RuntimeError("Input WAV is compressed; stdlib wave cannot process it")

        target_frames = int(seconds * params.framerate)
        frames_to_keep = min(params.nframes, target_frames)

        with contextlib.closing(wave.open(output_path, "wb")) as out_wav:
            out_wav.setparams(params)
            # Stream frames in chunks to avoid large memory spikes
            remaining = frames_to_keep
            chunk_frames = 16384
            while remaining > 0:
                count = min(chunk_frames, remaining)
                data = in_wav.readframes(count)
                if not data:
                    break
                out_wav.writeframes(data)
                remaining -= count


def process_file_inplace(path: str, seconds: int, dry_run: bool = False) -> str:
    """Process one file in place, returning a short status string.

    Returns one of: 'skipped', 'trimmed', 'failed'.
    """
    duration = probe_duration_seconds(path)
    if duration is not None and duration <= seconds + 1e-6:
        print(f"Skipped (already <= {seconds}s): {path}")
        return "skipped"

    if dry_run:
        print(f"Would trim to {seconds}s: {path}")
        return "skipped"

    directory = os.path.dirname(path) or "."
    tmp_file = None
    try:
        fd, tmp_file = tempfile.mkstemp(prefix=".trim_tmp_", suffix=".wav", dir=directory)
        os.close(fd)  # Will reopen via libraries

        if is_ffmpeg_available():
            trim_with_ffmpeg(path, tmp_file, seconds)
        else:
            trim_with_wave_module(path, tmp_file, seconds)

        # Atomic replace on success
        os.replace(tmp_file, path)
        print(f"Trimmed: {path}")
        return "trimmed"
    except Exception as exc:
        print(f"Failed: {path} -> {exc}")
        return "failed"
    finally:
        # Clean up tmp file if it still exists
        if tmp_file and os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except OSError:
                pass


def parse_args(argv: t.Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find 'A.wav', 'B.wav', and 'combined.wav' files under ROOT and"
            " trim them to a target duration. Uses ffmpeg if available, else falls back"
            " to stdlib wave for PCM WAVs."
        )
    )
    parser.add_argument("root", help="Root directory to search under")
    parser.add_argument(
        "--seconds",
        type=int,
        default=120,
        help="Target duration in seconds (default: 120)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying any files",
    )
    parser.add_argument(
        "--case-insensitive",
        action="store_true",
        help="Match filenames case-insensitively",
    )
    return parser.parse_args(argv)


def main(argv: t.Sequence[str] | None = None) -> int:
    ns = parse_args(sys.argv[1:] if argv is None else argv)
    root = os.path.abspath(ns.root)

    if not os.path.exists(root):
        print(f"Root path does not exist: {root}", file=sys.stderr)
        return 2
    if not os.path.isdir(root):
        print(f"Root path is not a directory: {root}", file=sys.stderr)
        return 2

    paths = find_combined_wav_files(root, case_insensitive=ns.case_insensitive)
    if not paths:
        print("No 'A.wav', 'B.wav', or 'combined.wav' files found.")
        return 0

    print(
        f"Found {len(paths)} file(s) (A.wav, B.wav, or combined.wav) under {root}."
        f" Target: {ns.seconds}s. {'DRY RUN' if ns.dry_run else ''}"
    )

    counts = {"trimmed": 0, "skipped": 0, "failed": 0}
    for path in paths:
        status = process_file_inplace(path, seconds=ns.seconds, dry_run=ns.dry_run)
        counts[status] = counts.get(status, 0) + 1

    print(
        f"Done. Trimmed: {counts['trimmed']}, Skipped: {counts['skipped']}, Failed: {counts['failed']}"
    )
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


