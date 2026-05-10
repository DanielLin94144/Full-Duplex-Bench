import os
import sys
import json
import math
import argparse
import tempfile
from collections import deque
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
import utmosv2

# Optional deps (lazy)
_parselmouth = None
_librosa = None


def _lazy_import_parselmouth():
    global _parselmouth
    if _parselmouth is not None:
        return _parselmouth
    try:
        import parselmouth  # praat-parselmouth

        _parselmouth = parselmouth
    except Exception:
        _parselmouth = False
    return _parselmouth


def _lazy_import_librosa():
    global _librosa
    if _librosa is not None:
        return _librosa
    try:
        import librosa

        _librosa = librosa
    except Exception:
        _librosa = False
    return _librosa


# -------------------------------------------------------------
# Global model singletons
# -------------------------------------------------------------
_subjective_model = None  # not currently used
_objective_model = None
_utmosv2_model = None
_silero_model = None
_silero_utils = None  # tuple of silero functions


def _load_models():
    global _subjective_model, _objective_model, _utmosv2_model
    if _subjective_model is None:
        _subjective_model = SQUIM_SUBJECTIVE.get_model()
    if _objective_model is None:
        _objective_model = SQUIM_OBJECTIVE.get_model()
    if _utmosv2_model is None:
        if torch.cuda.is_available():
            _utmosv2_model = utmosv2.create_model(pretrained=True, device="cuda")
        else:
            _utmosv2_model = utmosv2.create_model(pretrained=True)


def _load_silero(force_reload: bool = False):
    """Load Silero VAD model + utils via torch.hub (cached)."""
    global _silero_model, _silero_utils
    if _silero_model is not None and _silero_utils is not None and not force_reload:
        return _silero_model, _silero_utils
    _silero_model, _silero_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=force_reload,
        onnx=False,
    )
    return _silero_model, _silero_utils


# =============================================================
# Constants for sudden cutoff
# =============================================================
FRAME_MS = 30  # per-frame length (ms)
CUTOFF_DB = -18  # drop threshold dB between adjacent frames
MARGIN_DB = 6  # prev frame must be >= noise_floor + margin to count
HIST_FRAMES = 30
EPS = 1e-10


def _rms_db(block: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(block**2)) + EPS)


def detect_sudden_cutoffs(
    waveform: torch.Tensor,
    sr: int,
    frame_ms: int = FRAME_MS,
    cutoff_db: float = CUTOFF_DB,
    margin_db: float = MARGIN_DB,
    hist_frames: int = HIST_FRAMES,
) -> List[float]:
    """Return list of times (s) where a sudden level drop is detected."""
    if waveform.ndim == 2:
        wav = waveform.mean(dim=0).cpu().numpy()
    else:
        wav = waveform.squeeze().cpu().numpy()

    hop = int(sr * frame_ms / 1_000)
    if hop <= 0:
        return []

    history = deque(maxlen=hist_frames)
    prev_db = _rms_db(wav[:hop]) if len(wav) >= hop else _rms_db(wav)
    times_s: List[float] = []

    for i in range(1, max(1, len(wav) // hop)):
        frame = wav[i * hop : (i + 1) * hop]
        if frame.size == 0:
            break
        cur_db = _rms_db(frame)
        history.append(cur_db)

        diff_db = cur_db - prev_db
        noise_floor = np.percentile(history, 10) if history else cur_db

        if diff_db <= cutoff_db and prev_db >= noise_floor + margin_db:
            times_s.append(i * frame_ms / 1_000)

        prev_db = cur_db

    return times_s


# =============================================================
# Audio helpers
# =============================================================


def _load_audio(path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    return wav, sr


def _mix_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) == 1:
        return wav
    return wav.mean(dim=0, keepdim=True)


def _slice_wave(
    wav: torch.Tensor, sr: int, start_s: float, end_s: float
) -> torch.Tensor:
    n = wav.shape[-1]
    s_idx = max(0, min(n, int(round(start_s * sr))))
    e_idx = max(0, min(n, int(round(end_s * sr))))
    if e_idx <= s_idx:
        return wav[..., :0]
    return wav[..., s_idx:e_idx]


# =============================================================
# Silero VAD wrapper (returns trimmed_wav, speech_ts list)
# =============================================================


def _apply_silero_vad(
    waveform: torch.Tensor,
    sr: int,
    threshold: float = 0.5,
    min_speech_ms: int = 60,
    min_silence_ms: int = 50,
    window_size_samples: Optional[int] = None,
    collapse: str = "concat",
) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
    model, utils = _load_silero()
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    target_sr = 16000
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        wf_mono = waveform.mean(dim=0, keepdim=True)
    else:
        wf_mono = waveform
    if sr != target_sr:
        wf_mono = torchaudio.functional.resample(wf_mono, sr, target_sr)
        sr_vad = target_sr
    else:
        sr_vad = sr

    wav_1d = wf_mono.squeeze(0).cpu()

    params = {
        "threshold": threshold,
        "min_speech_duration_ms": min_speech_ms,
        "min_silence_duration_ms": min_silence_ms,
    }
    if window_size_samples is not None:
        params["window_size_samples"] = int(window_size_samples)

    speech_ts = get_speech_timestamps(
        wav_1d,
        model,
        sampling_rate=sr_vad,
        **params,
    )

    if not speech_ts:
        dur_s = waveform.shape[-1] / sr
        return waveform, [(0.0, dur_s)]

    segs = [(seg["start"] / sr_vad, seg["end"] / sr_vad) for seg in speech_ts]

    if collapse == "trim_edges":
        s_s, e_s = segs[0][0], segs[-1][1]
        trimmed = _slice_wave(waveform, sr, s_s, e_s)
        return trimmed, [(s_s, e_s)]

    # concat speech-only
    parts = [_slice_wave(waveform, sr, s_s, e_s) for (s_s, e_s) in segs]
    trimmed = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
    # still return original speech-ts (in original timeline) for reference
    return trimmed, segs


# =============================================================
# Torchaudio VAD wrapper (compat)
# =============================================================


def _apply_torchaudio_vad(
    waveform: torch.Tensor, sr: int, vad_kwargs: Optional[dict] = None
) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
    vad_kwargs = vad_kwargs or {}
    try:
        trimmed = torchaudio.functional.vad(waveform, sr, **vad_kwargs)
        if trimmed.numel() == 0:
            dur_s = waveform.shape[-1] / sr
            return waveform, [(0.0, dur_s)]
        dur_s = trimmed.shape[-1] / sr
        return trimmed, [(0.0, dur_s)]
    except Exception:
        dur_s = waveform.shape[-1] / sr
        return waveform, [(0.0, dur_s)]


# =============================================================
# Trim dispatch (ALWAYS returns (trimmed_wav, speech_ts))
# =============================================================


def _trim_dispatch(
    wav: torch.Tensor,
    sr: int,
    config: Dict[str, Any],
) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
    mode = config.get("trim_mode", "torchaudio")
    if mode == "none":
        dur_s = wav.shape[-1] / sr
        return wav, [(0.0, dur_s)]
    if mode == "torchaudio":
        return _apply_torchaudio_vad(_mix_mono(wav), sr, config.get("vad_kwargs", {}))
    if mode == "silero":
        sv = config.get("silero_vad", {})
        return _apply_silero_vad(
            wav,
            sr,
            threshold=sv.get("threshold", 0.5),
            min_speech_ms=sv.get("min_speech_ms", 60),
            min_silence_ms=sv.get("min_silence_ms", 50),
            window_size_samples=sv.get("window_size_samples", None),
            collapse=sv.get("collapse", "concat"),
        )
    # fallback
    dur_s = wav.shape[-1] / sr
    return wav, [(0.0, dur_s)]


# =============================================================
# Metric helpers (SQuIM, UTMOS)
# =============================================================


def _run_squim_objective(wav: torch.Tensor) -> Dict[str, float]:
    _load_models()
    stoi, pesq, si_sdr = _objective_model(wav)
    return {
        "stoi": float(stoi.item()),
        "pesq": float(pesq.item()),
        "si_sdr": float(si_sdr.item()),
    }


def _run_utmosv2(wav: torch.Tensor, sr: int) -> float:
    _load_models()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
        torchaudio.save(tmpf.name, wav, sr)
        mos = _utmosv2_model.predict(input_path=tmpf.name)
    try:
        os.unlink(tmpf.name)
    except OSError:
        pass
    return float(mos)


# =============================================================
# Chunk utilities / WPM
# =============================================================


def _norm_ts(ch: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    ts = ch.get("timestamp")
    if ts is None:
        ts = ch.get("timestamps")
    if not ts or len(ts) < 2:
        return None
    return float(ts[0]), float(ts[1])


def _choose_split_time_word_aligned(
    distractor_end: float, chunks: List[Dict[str, Any]]
) -> float:
    if not chunks:
        return distractor_end
    normed = []
    for ch in chunks:
        ts_pair = _norm_ts(ch)
        if ts_pair is None:
            continue
        normed.append(ts_pair)
    if not normed:
        return distractor_end
    normed.sort(key=lambda x: x[0])
    last_end = normed[-1][1]
    for s, e in normed:
        if distractor_end < s:
            return s
        if s <= distractor_end < e:
            return s
    return last_end


def _partition_chunks_word_aligned(
    chunks: List[Dict[str, Any]], split_t: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pre, post = [], []
    for ch in chunks:
        ts_pair = _norm_ts(ch)
        if ts_pair is None:
            continue
        s, e = ts_pair
        if e <= split_t:
            pre.append(ch)
        else:
            post.append(ch)
    return pre, post


def _speech_stats(chunks: List[Dict[str, Any]]) -> Tuple[float, int]:
    speech = 0.0
    n = 0

    if len(chunks) == 0:
        return speech, n

    start_speech, _ = _norm_ts(chunks[0])
    _, end_speech = _norm_ts(chunks[-1])

    if end_speech - start_speech >= 0:
        speech = end_speech - start_speech

    n = len(chunks)

    return speech, n


def _wpm_speech_only(chunks: List[Dict[str, Any]]) -> float:
    speech_s, n = _speech_stats(chunks)
    if speech_s <= 0:
        return 0.0
    return float(n / (speech_s / 60.0))


# =============================================================
# Robust Pitch helpers (Parselmouth → Librosa → Torchaudio → 0)
# =============================================================


def _pitch_parselmouth(
    wav_np: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
    time_step: float,
):
    pm = _lazy_import_parselmouth()
    if not pm:
        return None, None
    try:
        snd = pm.Sound(wav_np, sampling_frequency=sr)
        pitch = snd.to_pitch(time_step=time_step, pitch_floor=fmin, pitch_ceiling=fmax)
        f0 = pitch.selected_array["frequency"]  # Hz
        times = pitch.xs()
        return f0.astype(np.float32), times.astype(np.float32)
    except Exception:
        return None, None


def _pitch_librosa(
    wav_np: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
    frame_time: float,
):
    lb = _lazy_import_librosa()
    if not lb:
        return None, None
    try:
        hop_length = max(1, int(round(sr * frame_time)))
        f0, voiced_flag, voiced_probs = lb.pyin(
            wav_np.astype(np.float32),
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length,
        )
        times = lb.times_like(f0, sr=sr, hop_length=hop_length)
        return f0.astype(np.float32), times.astype(np.float32)
    except Exception:
        return None, None


def _pitch_torchaudio(
    wav_t: torch.Tensor,
    sr: int,
    frame_time: float,
    fmin: float,
    fmax: float,
):
    try:
        if wav_t.ndim == 2:
            wav_t = wav_t.mean(dim=0)
        if wav_t.ndim == 1:
            wav_t = wav_t.unsqueeze(0)
        f0 = (
            torchaudio.functional.detect_pitch_frequency(
                wav_t,
                sample_rate=sr,
                frame_time=frame_time,
                freq_low=fmin,
                freq_high=fmax,
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )
        hop = int(round(sr * frame_time))
        n = f0.shape[0]
        times = np.arange(n, dtype=np.float32) * (hop / sr)
        return f0.astype(np.float32), times
    except Exception:
        return None, None


def _compute_pitch_stats_robust(
    wav: torch.Tensor,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 600.0,
    frame_time: float = 0.01,
    voiced_floor_hz: float | None = None,
):
    """
    Robust multi-backend pitch stats (Praat -> librosa.pyin -> torchaudio YIN).
    Returns (mean_Hz, std_Hz) over **voiced frames only**.
    """
    if wav.ndim == 2:
        x = wav.mean(dim=0).cpu().numpy()
    else:
        x = wav.squeeze().cpu().numpy()

    f0, _ = _pitch_parselmouth(x, sr, fmin, fmax, frame_time)
    if f0 is None:
        f0, _ = _pitch_librosa(x, sr, fmin, fmax, frame_time)
    if f0 is None:
        f0, _ = _pitch_torchaudio(torch.from_numpy(x), sr, frame_time, fmin, fmax)

    if f0 is None:
        return 0.0, 0.0

    vf = ~np.isnan(f0)
    vf &= f0 > 0
    if voiced_floor_hz is not None:
        vf &= f0 >= voiced_floor_hz
    vf &= f0 <= fmax

    voiced = f0[vf]
    if voiced.size == 0:
        return 0.0, 0.0
    if voiced.size == 1:
        return float(voiced[0]), 0.0

    return float(np.mean(voiced)), float(np.std(voiced, ddof=1))


# =============================================================
# Robust Intensity helpers (Parselmouth → Librosa → Manual)
# =============================================================


def _intensity_parselmouth(
    wav_np: np.ndarray,
    sr: int,
    time_step: float,
):
    pm = _lazy_import_parselmouth()
    if not pm:
        return None
    try:
        snd = pm.Sound(wav_np, sampling_frequency=sr)
        intensity = snd.to_intensity(time_step=time_step)
        vals = intensity.values.T.flatten()
        vals = vals[vals > -200]  # Praat silence floor
        if vals.size == 0:
            return None
        return vals.astype(np.float32)
    except Exception:
        return None


def _intensity_librosa(
    wav_np: np.ndarray,
    sr: int,
    frame_time: float,
):
    lb = _lazy_import_librosa()
    if not lb:
        return None
    try:
        hop = max(1, int(round(sr * frame_time)))
        frame_length = hop
        rms = lb.feature.rms(
            y=wav_np.astype(np.float32),
            frame_length=frame_length,
            hop_length=hop,
            center=False,
        ).squeeze(0)
        rms = np.maximum(rms, 1e-10)
        db = 20.0 * np.log10(rms)
        return db.astype(np.float32)
    except Exception:
        return None


def _intensity_manual(
    wav_np: np.ndarray,
    sr: int,
    frame_time: float,
):
    frame_len = max(1, int(round(sr * frame_time)))
    hop = frame_len
    n = wav_np.shape[0]
    vals = []
    for s in range(0, n, hop):
        e = min(n, s + frame_len)
        frame = wav_np[s:e]
        if frame.size == 0:
            continue
        rms = np.sqrt(np.mean(frame**2) + 1e-10)
        vals.append(20.0 * np.log10(rms))
    if not vals:
        return None
    return np.array(vals, dtype=np.float32)


def _compute_intensity_stats_robust(
    wav: torch.Tensor,
    sr: int,
    frame_time: float = 0.01,
):
    if wav.ndim == 2:
        x = wav.mean(dim=0).cpu().numpy()
    else:
        x = wav.squeeze().cpu().numpy()

    vals = _intensity_parselmouth(x, sr, frame_time)
    if vals is None:
        vals = _intensity_librosa(x, sr, frame_time)
    if vals is None:
        vals = _intensity_manual(x, sr, frame_time)

    if vals is None or vals.size == 0:
        return 0.0, 0.0
    if vals.size == 1:
        return float(vals[0]), 0.0

    return float(np.mean(vals)), float(np.std(vals, ddof=1))


# =============================================================
# Segment evaluation (core)
# =============================================================


def _eval_segment(
    wav: torch.Tensor,
    sr: int,
    chunks: Optional[List[Dict[str, Any]]],
    config: Dict[str, Any],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if wav is None or wav.numel() == 0:
        return out

    # trim silence
    wav_t, speech_ts = _trim_dispatch(wav, sr, config)
    if wav_t.numel() == 0:
        return out

    # SQuIM
    if config.get("squim", False):
        try:
            out.update(_run_squim_objective(_mix_mono(wav_t)))
        except Exception as e:
            print(f"[WARN] SQuIM failed: {e}")
            out.update(
                {"stoi": float("nan"), "pesq": float("nan"), "si_sdr": float("nan")}
            )

    # UTMOS
    if config.get("utmosv2", False):
        try:
            out["utmosv2"] = _run_utmosv2(_mix_mono(wav_t), sr)
        except Exception as e:
            print(f"[WARN] UTMOSv2 failed: {e}")
            out["utmosv2"] = float("nan")

    # WPM
    if config.get("speaking_rate", False) and chunks is not None:
        out["wpm"] = _wpm_speech_only(chunks)
        speech_s, _ = _speech_stats(chunks)
        out["speech_dur_s"] = float(speech_s)

    # Sudden cutoff
    if config.get("sudden_cutoff", False):
        cut_times = detect_sudden_cutoffs(_mix_mono(wav_t), sr)
        out["cutoff_count"] = float(len(cut_times))

    # Pitch / Intensity (robust, no-NaN)
    if config.get("pitch", False):
        pp = config.get("pitch_params", {})
        mp, sp = _compute_pitch_stats_robust(
            wav_t,
            sr,
            fmin=pp.get("freq_low", 50.0),
            fmax=pp.get("freq_high", 600.0),
            frame_time=pp.get("frame_time", 0.01),
            voiced_floor_hz=pp.get("voiced_floor_hz", None),
        )
        out["mean_pitch"] = mp
        out["std_pitch"] = sp

    if config.get("intensity", False):
        ip = config.get("intensity_params", {})
        mi, si = _compute_intensity_stats_robust(
            wav_t,
            sr,
            frame_time=ip.get("frame_time", 0.01),
        )
        out["mean_intensity"] = mi
        out["std_intensity"] = si

    return out


# =============================================================
# File-level evaluation (split aware, word aligned)
# =============================================================


def _safe_dict(val):
    """保證回傳 dict；遇到 None / 非 dict 統一轉成 {}。"""
    return val if isinstance(val, dict) else {}


def eval_general_split(
    config: Dict[str, Any],
    wav_path: str,
    output_json_path: str,
    metadata_path: str,
    sr: int = 16000,
) -> Dict[str, Any]:

    prev_json_path = os.path.join(os.path.dirname(wav_path), "general_split.json")
    existing: Dict[str, Any] = {}
    if os.path.isfile(prev_json_path):
        with open(prev_json_path) as f:
            try:
                with open(prev_json_path) as f:
                    loaded = json.load(f)
                    existing = _safe_dict(loaded)

            except Exception:
                existing = {}

    waveform, sr = _load_audio(wav_path, target_sr=sr)
    total_samps = waveform.shape[-1]
    total_dur_s = total_samps / sr

    # metadata
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    distractor_end = float(meta["timestamps"][1])

    # chunks
    with open(output_json_path, "r") as f:
        out_json = json.load(f)
    chunks = out_json.get("chunks", [])

    # choose split
    split_t = _choose_split_time_word_aligned(distractor_end, chunks)
    split_t = max(0.0, min(total_dur_s, split_t))

    # audio split
    pre_wav = _slice_wave(waveform, sr, 0.0, split_t)
    post_wav = _slice_wave(waveform, sr, split_t, total_dur_s)

    # chunk split
    pre_chunks, post_chunks = _partition_chunks_word_aligned(chunks, split_t)

    existing_pre = _safe_dict(existing.get("pre"))
    existing_post = _safe_dict(existing.get("post"))

    pre_conf = _prune_config(existing_pre, config)
    post_conf = _prune_config(existing_post, config)

    pre_out = _eval_segment(pre_wav, sr, pre_chunks, pre_conf)
    post_out = _eval_segment(post_wav, sr, post_chunks, post_conf)

    merged_pre = {**_safe_dict(existing.get("pre")), **pre_out}
    merged_post = {**_safe_dict(existing.get("post")), **post_out}

    clean_out = _safe_dict(existing.get("clean"))

    clean_wav = wav_path.replace("output.wav", "clean_output.wav")
    clean_js = output_json_path.replace("output.json", "clean_output.json")
    if os.path.isfile(clean_wav) and os.path.isfile(clean_js):
        with open(clean_js) as f:
            clean_chunks = json.load(f).get("chunks", [])
            wav_c, _ = _load_audio(clean_wav, target_sr=sr)
            clean_conf = _prune_config(clean_out, config)
            new_clean = _eval_segment(wav_c, sr, clean_chunks, clean_conf)

        clean_out = {**clean_out, **new_clean}

    result = {
        "pre": merged_pre,
        "post": merged_post,
        "split_t": float(split_t),
        "distractor_end": float(distractor_end),
        "pre_dur_s": float(split_t),
        "post_dur_s": float(max(0.0, total_dur_s - split_t)),
    }
    if clean_out:
        result["clean"] = clean_out

    print("Evaluated split results:")
    print(existing)
    print("---")
    print(result)
    return result


# =============================================================
# Robust aggregation helpers
# =============================================================


def _robust_filter_vals(vals: List[float], agg_cfg: Dict[str, Any]) -> List[float]:
    """Apply outlier filtering per agg_cfg; return filtered list (copy)."""
    mode = (agg_cfg or {}).get("mode", "none")
    vals = [float(v) for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return []
    if mode in (None, "none"):
        return vals
    min_n = agg_cfg.get("min_n", 3)
    if len(vals) < min_n:
        return vals

    if mode == "iqr":
        k = float(agg_cfg.get("iqr_k", 1.5))
        q1 = np.percentile(vals, 25)
        q3 = np.percentile(vals, 75)
        iqr = q3 - q1
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        return [v for v in vals if lo <= v <= hi]

    if mode == "mad":
        k = float(agg_cfg.get("mad_k", 3.5))
        med = np.median(vals)
        mad = np.median(np.abs(np.array(vals) - med))
        if mad <= 0:
            return vals
        # scaled MAD ~ std
        dev = np.abs(np.array(vals) - med) / (mad * 1.4826)
        return [v for v, d in zip(vals, dev) if d <= k]

    if mode == "zscore":
        zt = float(agg_cfg.get("z_thresh", 3.0))
        mu = np.mean(vals)
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        if sd <= 0:
            return vals
        return [v for v in vals if abs((v - mu) / sd) <= zt]

    if mode == "winsor":
        lo_p, hi_p = agg_cfg.get("winsor_limits", (0.05, 0.05))
        lo = np.percentile(vals, lo_p * 100.0)
        hi = np.percentile(vals, 100.0 - hi_p * 100.0)
        return [min(max(v, lo), hi) for v in vals]

    if mode == "trim":
        p = float(agg_cfg.get("trim_prop", 0.05))
        lo = np.percentile(vals, p * 100.0)
        hi = np.percentile(vals, 100.0 - p * 100.0)
        return [v for v in vals if lo <= v <= hi]

    return vals


def _aggregate_results(
    results: List[Dict[str, Any]], config: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    agg_cfg = config.get("agg", {})

    def _agg_side(side: str) -> Dict[str, float]:
        keys = set()
        for r in results:
            keys.update(r.get(side, {}).keys())
        out = {}
        for k in keys:
            vals = []
            for r in results:
                v = r.get(side, {}).get(k)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if math.isnan(fv):
                    continue
                vals.append(fv)
            clean = _robust_filter_vals(vals, agg_cfg)
            if clean:
                out[k] = float(sum(clean) / len(clean))
        return out

    return {"pre": _agg_side("pre"), "post": _agg_side("post")}


# =============================================================
# Directory evaluation helper
# =============================================================


def _collect_example_roots(data_dir: str) -> List[str]:
    roots = []
    for root, dirs, files in os.walk(data_dir):
        if {"output.json", "metadata.json", "output.wav"}.issubset(files):
            roots.append(root)
    return roots


_METRIC_KEYS = {
    "squim": ["stoi", "pesq", "si_sdr"],
    "utmosv2": ["utmosv2"],
    "speaking_rate": ["wpm", "speech_dur_s"],
    "sudden_cutoff": ["cutoff_count"],
    "pitch": ["mean_pitch", "std_pitch"],
    "intensity": ["mean_intensity", "std_intensity"],
}


def _metric_complete(seg: dict, keys: list[str]) -> bool:
    for k in keys:
        v = seg.get(k)
        if v is None:
            return False
        if isinstance(v, float) and math.isnan(v):
            return False
    return True


def _prune_config(seg: dict, conf: dict) -> dict:
    new_conf = dict(conf)
    for m, keys in _METRIC_KEYS.items():
        if m == "speaking_rate":
            print("[INFO] WPM will always be recalculated.")
            continue
        if conf.get(m, False) and _metric_complete(seg, keys):
            new_conf[m] = False
    return new_conf


def eval_general_all_split(
    config: Dict[str, Any],
    data_dir: str,
    aggregate: bool = False,
) -> Any:
    paths = _collect_example_roots(data_dir)
    results = []
    for p in paths:
        wav_path = os.path.join(p, "output.wav")
        out_json_path = os.path.join(p, "output.json")
        meta_path = os.path.join(p, "metadata.json")
        print(f"Evaluating {p} ...")
        out = eval_general_split(config, wav_path, out_json_path, meta_path)
        # save per-example
        out_file = os.path.join(p, "general_split.json")
        with open(out_file, "w") as f:
            json.dump(out, f, indent=2)
        results.append({"path": p, **out})

    if not results:
        return [] if not aggregate else {"pre": {}, "post": {}}

    if not aggregate:
        return results

    # aggregated
    return _aggregate_results(results, config)


# =============================================================
# CLI
# =============================================================


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Eval outputs split at distractor_end (word-aligned + robust metrics)"
    )
    ap.add_argument(
        "--wav",
        type=str,
        default=None,
        help="Single wav to score (use with --output_json & --meta)",
    )
    ap.add_argument("--output_json", type=str, default=None, help="Path to output.json")
    ap.add_argument("--meta", type=str, default=None, help="Path to metadata.json")
    ap.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Batch mode: directory with subfolders",
    )
    ap.add_argument(
        "--aggregate",
        action="store_true",
        help="Average results across files in batch mode",
    )

    # metric toggles
    ap.add_argument("--squim", action="store_true")
    ap.add_argument("--utmosv2", action="store_true")
    ap.add_argument("--speaking_rate", action="store_true")
    ap.add_argument("--sudden_cutoff", action="store_true")
    ap.add_argument("--pitch", action="store_true")
    ap.add_argument("--intensity", action="store_true")

    # pitch/intensity params
    ap.add_argument("--pitch_frame_time", type=float, default=0.01)
    ap.add_argument("--pitch_fmin", type=float, default=50.0)
    ap.add_argument("--pitch_fmax", type=float, default=600.0)
    ap.add_argument("--intensity_frame_time", type=float, default=0.01)

    # trim mode flags
    ap.add_argument(
        "--trim_mode",
        type=str,
        default="torchaudio",
        choices=["none", "torchaudio", "silero"],
        help="Silence trim strategy",
    )
    ap.add_argument("--silero_threshold", type=float, default=0.5)
    ap.add_argument("--silero_min_speech_ms", type=int, default=60)
    ap.add_argument("--silero_min_silence_ms", type=int, default=50)
    ap.add_argument("--silero_window_size", type=int, default=-1, help="-1=default")
    ap.add_argument(
        "--silero_collapse",
        type=str,
        default="concat",
        choices=["trim_edges", "concat"],
    )

    # aggregation / outlier
    ap.add_argument(
        "--agg_mode",
        type=str,
        default="none",
        choices=["none", "iqr", "mad", "zscore", "winsor", "trim"],
        help="Outlier filter mode for aggregation",
    )
    ap.add_argument("--agg_iqr_k", type=float, default=1.5)
    ap.add_argument("--agg_mad_k", type=float, default=3.5)
    ap.add_argument("--agg_z_thresh", type=float, default=3.0)
    ap.add_argument("--agg_winsor_lo", type=float, default=0.05)
    ap.add_argument("--agg_winsor_hi", type=float, default=0.05)
    ap.add_argument("--agg_trim_prop", type=float, default=0.05)
    ap.add_argument("--agg_min_n", type=int, default=3)

    return ap.parse_args()


def _args_to_config(args) -> Dict[str, Any]:
    return {
        "squim": args.squim,
        "utmosv2": args.utmosv2,
        "speaking_rate": args.speaking_rate,
        "sudden_cutoff": args.sudden_cutoff,
        "pitch": args.pitch,
        "intensity": args.intensity,
        "pitch_params": {
            "frame_time": args.pitch_frame_time,
            "freq_low": args.pitch_fmin,
            "freq_high": args.pitch_fmax,
        },
        "intensity_params": {
            "frame_time": args.intensity_frame_time,
        },
        "trim_mode": args.trim_mode,
        "vad_kwargs": {},  # for torchaudio
        "silero_vad": {
            "threshold": args.silero_threshold,
            "min_speech_ms": args.silero_min_speech_ms,
            "min_silence_ms": args.silero_min_silence_ms,
            "window_size_samples": (
                None if args.silero_window_size < 0 else args.silero_window_size
            ),
            "collapse": args.silero_collapse,
        },
        "agg": {
            "mode": args.agg_mode,
            "iqr_k": args.agg_iqr_k,
            "mad_k": args.agg_mad_k,
            "z_thresh": args.agg_z_thresh,
            "winsor_limits": (args.agg_winsor_lo, args.agg_winsor_hi),
            "trim_prop": args.agg_trim_prop,
            "min_n": args.agg_min_n,
        },
    }


def main():
    args = _parse_args()
    config = _args_to_config(args)

    if args.data_dir:
        res = eval_general_all_split(config, args.data_dir, aggregate=args.aggregate)
        print(json.dumps(res, indent=2))
        return

    if not (args.wav and args.output_json and args.meta):
        raise SystemExit(
            "Single-file mode requires --wav, --output_json, --meta OR use --data_dir"
        )

    res = eval_general_split(config, args.wav, args.output_json, args.meta)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
