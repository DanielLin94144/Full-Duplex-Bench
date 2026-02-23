#!/usr/bin/env python3
"""
==============================================

Writes a **`latency_intervals.json`** file inside every folder that
contains `input.wav` + `output.wav`:
```json
{
  "latency_stop_list": [[3.486, 4.834], ...],
  "latency_resp_list": [[4.834, 5.200], ...]
}
```
"""

from __future__ import annotations
import argparse, json, importlib
from pathlib import Path
from typing import List, Tuple
import torch, torchaudio

# ---------------- Parameters ----------------
SR = 16_000
USER_MERGE_GAP = 0.6  # merge user pauses ≤300 ms
MODEL_MERGE_GAP = 0.5  # merge model splits ≤50 ms
AUDIO_EXT = "wav"
OUT_FILENAME = "latency_intervals.json"

# ---------------- Silero‑VAD ---------------
vad_module = importlib.import_module("silero_vad")
if hasattr(vad_module, "VoiceActivityDetector"):
    from silero_vad import VoiceActivityDetector

    _VAD = VoiceActivityDetector(sample_rate=SR)

    def _vad_ts(wav):
        return [(t["start"], t["end"]) for t in _VAD.get_speech_ts(wav)]

else:
    from silero_vad import get_speech_timestamps

    _M, _ = torch.hub.load(
        "snakers4/silero-vad", model="silero_vad", trust_repo=True, onnx=False
    )

    def _vad_ts(w):
        return [
            (t["start"], t["end"])
            for t in get_speech_timestamps(w, _M, sampling_rate=SR)
        ]


# -------------- Helpers --------------------
def load_wav(p: Path):
    wav, sr = torchaudio.load(p)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.squeeze(0)


def _merge(seg: List[Tuple[float, float]], gap_thr: float):
    if not seg:
        return []
    seg = sorted(seg)
    merged = [seg[0]]
    for s, e in seg[1:]:
        ps, pe = merged[-1]
        if s - pe <= gap_thr:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def seg_sec(wav, gap_thr):
    return _merge([(s / SR, e / SR) for s, e in _vad_ts(wav)], gap_thr)


# -------------- Interval utilities ---------


def overlaps(user, model):
    raw = []
    i = j = 0
    while i < len(user) and j < len(model):
        u_s, u_e = user[i]
        m_s, m_e = model[j]
        s = max(u_s, m_s)
        e = min(u_e, m_e)
        if e > s:
            raw.append((s, e))
        if u_e < m_e:
            i += 1
        else:
            j += 1
    # keep shortest interval per rounded end‑time (ms precision)
    best = {}
    for s, e in raw:
        key = int(round(e * 1000))
        if key not in best or (e - s) < (best[key][1] - best[key][0]):
            best[key] = (s, e)
    return [
        [round(s, 3), round(e, 3)] for s, e in sorted(best.values(), key=lambda x: x[1])
    ]


def response_gaps(user, model):
    model_starts = [s for s, _ in model]
    tmp = {}
    for u_s, u_e in user:
        nxt = next((s for s in model_starts if s > u_e), None)
        if nxt is None:
            continue
        key = int(round(nxt * 1000))
        candidate = [round(u_e, 3), round(nxt, 3)]
        # keep shorter gap (later user_end)
        if key not in tmp or candidate[0] > tmp[key][0]:
            tmp[key] = candidate
    return [iv for _, iv in sorted(tmp.items(), key=lambda kv: kv[1][1])]


# -------------- Folder processing ----------


def process_folder(fld: Path):
    user_seg = seg_sec(load_wav(fld / f"input.{AUDIO_EXT}"), USER_MERGE_GAP)
    model_seg = seg_sec(load_wav(fld / f"output.{AUDIO_EXT}"), MODEL_MERGE_GAP)
    data = {
        "latency_stop_list": overlaps(user_seg, model_seg),
        "latency_resp_list": response_gaps(user_seg, model_seg),
    }
    print(data)
    with open(fld / OUT_FILENAME, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    print("✔", fld.name, "→", OUT_FILENAME)


# -------------- CLI ------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Save overlap & response gaps JSON in each folder"
    )
    ap.add_argument("--root_dir", required=True)
    args = ap.parse_args()

    root_dir = Path(args.root_dir)
    for fld in sorted(root_dir.iterdir()):
        if (fld / f"input.{AUDIO_EXT}").exists() and (
            fld / f"output.{AUDIO_EXT}"
        ).exists():
            process_folder(fld)


if __name__ == "__main__":
    main()
