#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Load {root_dir_dir}/*/general_split.json files, extract pre/post metrics,
and run paired t-tests with optional outlier trimming.

Example:
    python paired_from_json.py \
        --root_dir /path/to/root_dir_dir \
        --metrics utmosv2 wpm mean_pitch std_pitch mean_intensity std_intensity \
        --outlier_rule mad \
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Sequence, Literal, Optional, Dict, Any, List
from scipy import stats


# ---------------------------------------------------------------------
# Load JSON rows
# ---------------------------------------------------------------------
def load_general_split_files(
    root_dir: Path,
    require_behaviour: str = "C_RESPOND",
) -> pd.DataFrame:
    """
    Scan one directory level below root_dir for general_split.json,
    and keep the sample only if content_tag.json contains `require_behaviour`.
    """
    rows = []
    for sub in sorted(root_dir.glob("*/general_split.json")):

        tag_path = sub.with_name("content_tag.json")
        try:
            tags = json.loads(tag_path.read_text()) if tag_path.exists() else {}
            behaviours = tags.get("behaviour") or tags.get("behavior") or []
        except Exception as e:
            print(f"[WARN] Could not parse {tag_path}: {e}; skip.")
            continue

        if require_behaviour not in behaviours:
            continue

        try:
            data = json.loads(sub.read_text())
        except Exception as e:
            print(f"[WARN] Could not read {sub}: {e}")
            continue
        if not data:
            print(f"[WARN] Empty data in {sub}; skipping.")
            continue

        file_id = sub.parent.name
        pre = data.get("pre", {})
        post = data.get("post", {})
        clean = data.get("clean", {})

        row = {"file_id": file_id}
        for k in ("split_t", "distractor_end", "pre_dur_s", "post_dur_s"):
            row[k] = data.get(k, np.nan)

        all_keys = set(pre) | set(post) | set(clean)
        for k in all_keys:
            row[f"pre_{k}"] = pre.get(k, np.nan)
            row[f"post_{k}"] = post.get(k, np.nan)
            row[f"clean_{k}"] = clean.get(k, np.nan)

        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No C_RESPOND samples found under {root_dir}; "
            f"check content_tag.json or behaviour filter."
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Mask invalid / sentinel values
# ---------------------------------------------------------------------
def mask_invalid(series: pd.Series, metric: str) -> pd.Series:
    """
    Replace common placeholder values with NaN.
    Adjust the heuristics as needed for your data.
    """
    s = series.copy()

    # Common sentinels seen in your examples:
    # 0.0 for rates/pitch/stds; -200.0 for intensity
    if metric in {"wpm", "speech_dur_s", "mean_pitch", "std_pitch", "std_intensity"}:
        s = s.mask(s == 0.0, np.nan)

    if metric in {"mean_intensity"}:
        # treat unrealistic very low dBFS as missing sentinel
        s = s.mask(s <= -190.0, np.nan)

    return s


# ---------------------------------------------------------------------
# Core paired t-test with optional outlier trimming
# ---------------------------------------------------------------------
def paired_compare(
    before: Sequence[float],
    after: Sequence[float],
    *,
    measure_name: str = "Score",
    alpha: float = 0.05,
    outlier_rule: Literal["mad", "iqr", "none"] = "mad",
    outlier_on: Literal["diff", "both"] = "diff",
    outlier_k: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Returns dict of paired t-test results (no Wilcoxon).
    """
    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    if b.shape != a.shape:
        raise ValueError("before and after must have the same length.")

    # remove NaN pairs
    mask = ~np.isnan(b) & ~np.isnan(a)
    b0, a0 = b[mask], a[mask]
    n_raw = b.size
    n_valid = b0.size
    if n_valid < 2:
        raise ValueError("Fewer than 2 valid pairs; cannot analyze.")

    # Outlier detection -------------------------------------------------
    if outlier_rule not in ("mad", "iqr", "none"):
        raise ValueError("outlier_rule must be 'mad', 'iqr', or 'none'.")

    if outlier_rule == "none":
        clean_mask = np.ones(n_valid, dtype=bool)
    else:
        if outlier_on == "diff":
            vals = a0 - b0
        elif outlier_on == "both":
            vals = np.concatenate([b0, a0])
        else:
            raise ValueError("outlier_on must be 'diff' or 'both'.")

        if outlier_rule == "mad":
            if outlier_k is None:
                outlier_k = 3.5
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            scale = 1.4826 * mad if mad > 0 else 0.0
            if scale == 0:
                clean_mask = np.ones(n_valid, dtype=bool)
            else:
                lo = med - outlier_k * scale
                hi = med + outlier_k * scale
                if outlier_on == "diff":
                    diff_vals = a0 - b0
                    clean_mask = (diff_vals >= lo) & (diff_vals <= hi)
                else:
                    clean_mask = (b0 >= lo) & (b0 <= hi) & (a0 >= lo) & (a0 <= hi)

        elif outlier_rule == "iqr":
            if outlier_k is None:
                outlier_k = 1.5
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lo = q1 - outlier_k * iqr
            hi = q3 + outlier_k * iqr
            if outlier_on == "diff":
                diff_vals = a0 - b0
                clean_mask = (diff_vals >= lo) & (diff_vals <= hi)
            else:
                clean_mask = (b0 >= lo) & (b0 <= hi) & (a0 >= lo) & (a0 <= hi)

    # Apply outlier mask
    b1, a1 = b0[clean_mask], a0[clean_mask]
    n_used = b1.size
    n_dropped = n_valid - n_used
    if n_used < 2:
        raise ValueError("All data flagged as outliers; relax trimming.")

    # Compute stats -----------------------------------------------------
    d = a1 - b1
    mean_b = b1.mean()
    mean_a = a1.mean()
    mean_d = d.mean()
    sd_d = d.std(ddof=1)
    se_d = sd_d / np.sqrt(n_used)

    t_stat, p_t = stats.ttest_rel(a1, b1)
    tcrit = stats.t.ppf(1 - alpha / 2, df=n_used - 1)
    ci_lo = mean_d - tcrit * se_d
    ci_hi = mean_d + tcrit * se_d
    d_z = mean_d / sd_d if sd_d > 0 else np.nan

    conclusion = format_conclusion(
        measure_name=measure_name,
        mean_b=mean_b,
        mean_a=mean_a,
        mean_d=mean_d,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        t_stat=t_stat,
        df=n_used - 1,
        p_t=p_t,
        d_z=d_z,
        n_used=n_used,
        n_dropped=n_dropped,
        outlier_rule=outlier_rule,
    )

    return {
        "measure": measure_name,
        "n_raw": n_raw,
        "n_valid": n_valid,
        "n_used": n_used,
        "n_dropped": n_dropped,
        "mean_before": mean_b,
        "mean_after": mean_a,
        "mean_diff": mean_d,
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
        "t": t_stat,
        "df": n_used - 1,
        "p_t": p_t,
        "cohen_dz": d_z,
        "conclusion": conclusion,
    }


def _format_p(p: float) -> str:
    if p is None or np.isnan(p):
        return "--"
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}"


def format_conclusion(
    *,
    measure_name: str,
    mean_b: float,
    mean_a: float,
    mean_d: float,
    ci_lo: float,
    ci_hi: float,
    t_stat: float,
    df: int,
    p_t: float,
    d_z: float,
    n_used: int,
    n_dropped: int,
    outlier_rule: str,
) -> str:
    # direction tag
    if mean_d > 0:
        dir_en = "increased"
    elif mean_d < 0:
        dir_en = "decreased"
    else:
        dir_en = "did not change"

    # significance tag
    sig = p_t < 0.05
    ci_cross0 = not (ci_lo > 0 or ci_hi < 0)

    if sig:
        sig_phrase = "a statistically reliable"
    else:
        sig_phrase = "no statistically reliable"
    base = (
        f"{measure_name}: {sig_phrase} change was "
        f"observed; After {dir_en} relative to Before "
        f"(M_before={mean_b:.2f}, M_after={mean_a:.2f}, "
        f"Δ={mean_d:.2f}, 95% CI [{ci_lo:.2f}, {ci_hi:.2f}]; "
        f"paired t({df})={t_stat:.2f}, p={_format_p(p_t)}, d_z={d_z:.2f})."
    )
    if n_dropped > 0:
        base += f" Outlier trimming ({outlier_rule}) removed {n_dropped}/{n_used + n_dropped} pairs."
    return base


# ---------------------------------------------------------------------
# Batch runner across metrics
# ---------------------------------------------------------------------
def run_batch_tests(
    df: pd.DataFrame,
    metrics: List[str],
    *,
    alpha: float = 0.05,
    outlier_rule: str = "mad",
    outlier_on: str = "diff",
    outlier_k: Optional[float] = None,
) -> pd.DataFrame:

    pair_defs = [("pre", "post"), ("clean", "post")]

    results = []
    for m in metrics:
        for p1, p2 in pair_defs:
            col1 = f"{p1}_{m}"
            col2 = f"{p2}_{m}"
            if col1 not in df.columns or col2 not in df.columns:
                print(f"[WARN] Columns {col1}/{col2} missing; skip.")
                continue

            a = mask_invalid(df[col1], m).to_numpy()
            b = mask_invalid(df[col2], m).to_numpy()

            try:
                res = paired_compare(
                    a,
                    b,
                    measure_name=f"{p1}_vs_{p2}:{m}",
                    alpha=alpha,
                    outlier_rule=outlier_rule,
                    outlier_on=outlier_on,
                    outlier_k=outlier_k,
                )
                res["pair"] = f"{p1}_vs_{p2}"
                res["metric"] = m
                results.append(res)
            except Exception as e:
                print(f"[WARN] {p1} vs {p2} '{m}' skipped: {e}")

    if not results:
        raise RuntimeError("No metrics successfully analyzed.")

    import statsmodels.stats.multitest as smm

    pvals = [r["p_t"] for r in results]
    _, p_adj, _, _ = smm.multipletests(pvals, method="holm")
    for r, p_corr in zip(results, p_adj):
        r["p_adj"] = p_corr

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Paired pre/post t-tests from general_split.json files."
    )
    p.add_argument(
        "--root_dir",
        required=True,
        type=str,
        help="Root directory containing */general_split.json.",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metrics (e.g., utmosv2 wpm mean_pitch). Default: all metrics in first file.",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="Alpha for CI.")
    p.add_argument("--outlier_rule", choices=["mad", "iqr", "none"], default="mad")
    p.add_argument("--outlier_on", choices=["diff", "both"], default="diff")
    p.add_argument("--outlier_k", type=float, default=None)
    p.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV path. Default: pair_t_<root_dirname>.csv",
    )
    return p.parse_args()


def main_cli():
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()

    df = load_general_split_files(root_dir)
    wpm_cols = [c for c in df.columns if c.endswith("wpm")]
    df = df[(df[wpm_cols] != 0).all(axis=1)]

    # Decide metrics
    if args.metrics is None:
        metrics = [c[4:] for c in df.columns if c.startswith("pre_")]
    else:
        metrics = args.metrics

    res_df = run_batch_tests(
        df=df,
        metrics=metrics,
        alpha=args.alpha,
        outlier_rule=args.outlier_rule,
        outlier_on=args.outlier_on,
        outlier_k=args.outlier_k,
    )

    # Output name
    if args.out_csv is None:
        out = f"pair_t_{root_dir.name}.txt"
    else:
        out = Path(args.out_csv).expanduser().resolve()

    # Save conclusion to txt file
    with open(out, "w", encoding="utf-8") as f:
        for _, row in res_df.iterrows():
            f.write(row["conclusion"] + "\n")

    print(f"\n[RESULTS] Saved summary to {out}\n")

    # Print human-readable conclusions
    for _, row in res_df.iterrows():
        print(row["conclusion"])


if __name__ == "__main__":
    main_cli()
