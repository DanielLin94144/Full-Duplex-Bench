import json
from pathlib import Path
from collections import defaultdict
import csv
from typing import List, Tuple, Dict, Any, Optional


def overlap_duration(
    a_start: float, a_end: float, b_start: float, b_end: float
) -> float:
    """Return the overlap duration between [a_start, a_end] and [b_start, b_end]."""
    s = max(a_start, b_start)
    e = min(a_end, b_end)
    return max(0.0, e - s)


def summarize_judge_scores(
    base_dir: str = ".",
    examiner_prefix: str = "Examiner_",
    processed_suffix: str = "_processed.json",
    turn_key: str = "Turn-taking event and score",
    task_key: str = "Task-specific score",
    output_csv_overall: str = "judge_summary.csv",
    output_csv_binned: str = "judge_summary_binned.csv",
    # Bin configuration (defaults to 120 secs; set to None to dynamically scale by max bin end)
    fixed_bins: Optional[List[Tuple[float, float]]] = [
        (i, i + 15.0) for i in range(0, 120, 15)
    ],
):
    base = Path(base_dir)

    # Overall structural data
    overall = defaultdict(
        lambda: {
            "files": 0,
            "events": 0,
            "tt_sum": 0.0,  # sum of all TT scores across events
            "if_sum": 0.0,  # sum of all IF scores across events
            "task_sum": 0.0,  # sum of task-specific scores across files
        }
    )

    # Binned time-weighted tracking
    # Structure: {(setup, subtask): {
    #   "bins": [(start, end), ...],
    #   "bin_tt_sum": [float...],   # Accumulated (TT_score * overlap_seconds)
    #   "bin_if_sum": [float...],   # Accumulated (IF_score * overlap_seconds)
    #   "bin_dur":    [float...]    # Accumulated overlap_seconds
    # }}
    binned: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # If fixed_bins is None, track the max end timestamp for dynamic bins
    if fixed_bins is None:
        max_end_tracker = defaultdict(float)  # {(setup, subtask): max_end}
    else:
        max_end_tracker = None  # not used

    # First pass: Aggregate overall statistics
    for split_dir in base.iterdir():
        if not split_dir.is_dir() or (examiner_prefix and not split_dir.name.startswith(examiner_prefix)):
            continue

        json_files = list(split_dir.rglob(f"*{processed_suffix}"))
        if not json_files:
            continue

        for jf in json_files:
            # We want key = (Split, TaskID)
            # jf.name is like "Daily.ordering.001_processed.json"
            # Getting the base name before _processed
            base_filename = jf.name.replace(processed_suffix, "")
            key = (split_dir.name, base_filename)
            try:
                with jf.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to read {jf}: {e}")
                continue

            events = data.get(turn_key, [])
            ev_count = 0
            tt_local_sum = 0.0
            if_local_sum = 0.0

            for item in events:
                # item: [[start, end], tt_score, if_score]
                if (
                    isinstance(item, list)
                    and len(item) >= 3
                    and isinstance(item[0], list)
                    and len(item[0]) >= 2
                ):
                    try:
                        start = float(item[0][0])
                        end = float(item[0][1])
                        tt_score = float(item[1])
                        if_score = float(item[2])
                    except Exception:
                        continue

                    if end <= start:
                        continue

                    ev_count += 1
                    tt_local_sum += tt_score
                    if_local_sum += if_score

                    # Track max end for dynamic bins
                    if max_end_tracker is not None:
                        if end > max_end_tracker[key]:
                            max_end_tracker[key] = end

            overall[key]["tt_sum"] += tt_local_sum
            overall[key]["if_sum"] += if_local_sum
            overall[key]["events"] += ev_count

            task_score = data.get(task_key, None)
            if isinstance(task_score, (int, float)):
                overall[key]["task_sum"] += float(task_score)
                overall[key]["files"] += 1
            else:
                # Ignore if task score is missing
                pass

    # Helper to build bins dynamically (15s intervals up to max elapsed)
    def build_bins(up_to: float, step: float = 15.0) -> List[Tuple[float, float]]:
        if up_to <= 0:
            up_to = step
        num = int((up_to + step - 1e-9) // step)  # ceil-like
        return [(i * step, (i + 1) * step) for i in range(num)]

    # Initialize binned struture
    for key in overall.keys():
        if fixed_bins is None:
            bins = build_bins(max_end_tracker.get(key, 0.0), 15.0)
        else:
            bins = fixed_bins[:]
        binned[key] = {
            "bins": bins,
            "bin_tt_sum": [0.0 for _ in bins],
            "bin_if_sum": [0.0 for _ in bins],
            "bin_dur": [0.0 for _ in bins],
        }

    # Second pass: Compute binned overlaps and scores
    for split_dir in base.iterdir():
        if not split_dir.is_dir() or (examiner_prefix and not split_dir.name.startswith(examiner_prefix)):
            continue

        json_files = list(split_dir.glob(f"*{processed_suffix}")) # Changed rglob to glob
        if not json_files:
            continue

        for jf in json_files:
            base_filename = jf.name.replace(processed_suffix, "")
            key = (split_dir.name, base_filename)
            if key not in binned:
                continue

            bins = binned[key]["bins"]

            try:
                with jf.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to read {jf}: {e}")
                continue

            events = data.get(turn_key, [])
            for item in events:
                if (
                    isinstance(item, list)
                    and len(item) >= 3
                    and isinstance(item[0], list)
                    and len(item[0]) >= 2
                    and isinstance(item[1], (int, float))
                    and isinstance(item[2], (int, float))
                ):
                    try:
                        start = float(item[0][0])
                        end = float(item[0][1])
                        tt_score = float(item[1])
                        if_score = float(item[2])
                    except Exception:
                        continue

                    if end <= start:
                        continue

                    # Calculate overlaps for every bin
                    for i, (b_start, b_end) in enumerate(bins):
                        ov = overlap_duration(start, end, b_start, b_end)
                        if ov > 0.0:
                            binned[key]["bin_tt_sum"][i] += tt_score * ov
                            binned[key]["bin_if_sum"][i] += if_score * ov
                            binned[key]["bin_dur"][i] += ov

    # === Output Overall CSV ===
    overall_header = [
        "setup",
        "subtask",
        "n_files",
        "n_events",
        "avg_turn_taking",
        "avg_instruction_following",
        "avg_task_specific",
    ]
    overall_rows = []
    for (setup, subtask), ctr in sorted(overall.items()):
        n_files = ctr["files"]
        n_events = ctr["events"]
        avg_tt = (ctr["tt_sum"] / n_events) if n_events > 0 else None
        avg_if = (ctr["if_sum"] / n_events) if n_events > 0 else None
        avg_task = (ctr["task_sum"] / n_files) if n_files > 0 else None

        overall_rows.append(
            [
                setup,
                subtask,
                n_files,
                n_events,
                None if avg_tt is None else round(avg_tt, 3),
                None if avg_if is None else round(avg_if, 3),
                None if avg_task is None else round(avg_task, 3),
            ]
        )

    if overall_rows:
        with open(output_csv_overall, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(overall_header)
            writer.writerows(overall_rows)
        print(f"Exported: {output_csv_overall}")
    else:
        print("No processed json files found matching conditions.")

    # === Output Binned CSV ===
    # Formats the CSV dynamically based on the bins array
    bin_pairs = []
    assert fixed_bins is not None, "If using dynamic bins, adapt the output logic to match."
    for b_start, b_end in fixed_bins:
        label = f"{int(b_start)}-{int(b_end)}"
        bin_pairs.append((f"TT[{label}]", f"IF[{label}]"))

    binned_header = ["setup", "subtask", "n_files", "n_events"] + [
        h for pair in bin_pairs for h in pair
    ]
    binned_rows = []

    for key in sorted(binned.keys()):
        setup, subtask = key
        ctr = overall.get(key, {"files": 0, "events": 0})
        row = [setup, subtask, ctr["files"], ctr["events"]]

        bins = binned[key]["bins"]
        tt_sum = binned[key]["bin_tt_sum"]
        if_sum = binned[key]["bin_if_sum"]
        dur = binned[key]["bin_dur"]

        # Create a lookup table to match the local bins with global fixed bins
        bin_map = {bins[i]: i for i in range(len(bins))}

        for fb in fixed_bins:
            if fb in bin_map and dur[bin_map[fb]] > 0:
                i = bin_map[fb]
                tt_avg = tt_sum[i] / dur[i]
                if_avg = if_sum[i] / dur[i]
                row.extend([round(tt_avg, 3), round(if_avg, 3)])
            else:
                row.extend([None, None])

        binned_rows.append(row)

    with open(output_csv_binned, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(binned_header)
        writer.writerows(binned_rows)
    print(f"Exported: {output_csv_binned}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Score results")
    parser.add_argument("--root_dir", default="../eval_results", help="Directory containing eval results")
    parser.add_argument("--prefix", default="", help="Directory prefix to evaluate")
    args = parser.parse_args()

    summarize_judge_scores(
        base_dir=args.root_dir,
        examiner_prefix=args.prefix,
        fixed_bins=[(i, i + 15.0) for i in range(0, 120, 15)],
    )
