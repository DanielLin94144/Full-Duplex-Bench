import json
from pathlib import Path
from collections import defaultdict
import csv


def summarize_judge_scores(
    base_dir: str = ".",
    examiner_prefix: str = "Examiner_",
    processed_suffix: str = "_processed.json",
    turn_key: str = "Turn-taking event and score",
    task_key: str = "Task-specific score",
    output_csv: str = "judge_summary.csv",
):
    base = Path(base_dir)

    # Structure: {(split, task_id): counters}
    agg = defaultdict(
        lambda: {
            "files": 0,
            "events": 0,
            "tt_sum": 0.0,  # sum of all turn-taking scores across all events
            "if_sum": 0.0,  # sum of all instruction-following scores across all events
            "task_sum": 0.0,  # sum of task-specific scores across files
        }
    )

    # Discover processed JSON files based on FDB-v2 flat structure
    for split_dir in base.iterdir():
        if not split_dir.is_dir() or (examiner_prefix and not split_dir.name.startswith(examiner_prefix)):
            continue

        json_files = list(split_dir.rglob(f"*{processed_suffix}"))
        if not json_files:
            continue

        for jf in json_files:
            base_filename = jf.name.replace(processed_suffix, "")
            key = (split_dir.name, base_filename)
            try:
                with jf.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to read {jf}: {e}")
                continue

            # Get turn-taking events
            events = data.get(turn_key, [])
            # events format: [ [start, end], tt_score, if_score ]
            tt_local_sum = 0.0
            if_local_sum = 0.0
            ev_count = 0

            for item in events:
                if (
                    isinstance(item, list)
                    and len(item) >= 3
                    and isinstance(item[1], (int, float))
                    and isinstance(item[2], (int, float))
                ):
                    tt_local_sum += float(item[1])
                    if_local_sum += float(item[2])
                    ev_count += 1

            agg[key]["tt_sum"] += tt_local_sum
            agg[key]["if_sum"] += if_local_sum
            agg[key]["events"] += ev_count

            # Task-specific score averaged by file count
            task_score = data.get(task_key, None)
            if isinstance(task_score, (int, float)):
                agg[key]["task_sum"] += float(task_score)
                agg[key]["files"] += 1
            else:
                # If task score is missing, ignore but keep turn-taking records
                print(
                    f"[WARN] {jf} is missing a valid '{task_key}', only computing turn-taking stats."
                )
                pass

    # Output summary CSV
    rows = []
    header = [
        "setup",
        "subtask",
        "n_files",
        "n_events",
        "avg_turn_taking",
        "avg_instruction_following",
        "avg_task_specific",
    ]

    for (setup, subtask), ctr in sorted(agg.items()):
        n_files = ctr["files"]
        n_events = ctr["events"]
        avg_tt = (ctr["tt_sum"] / n_events) if n_events > 0 else None
        avg_if = (ctr["if_sum"] / n_events) if n_events > 0 else None
        avg_task = (ctr["task_sum"] / n_files) if n_files > 0 else None

        rows.append(
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

    # Print to console
    if rows:
        colw = [max(len(h), 12) for h in header]

        def fmt_cell(x, i):
            s = "" if x is None else str(x)
            return s.ljust(colw[i])

        print(" | ".join(h.ljust(colw[i]) for i, h in enumerate(header)))
        print("-+-".join("-" * w for w in colw))
        for row in rows:
            print(" | ".join(fmt_cell(row[i], i) for i in range(len(header))))
    else:
        print("No json files found matching conditions.")

    # Save to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"\nExported: {output_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Score results (No Binning)")
    parser.add_argument("--root_dir", default="../eval_results", help="Directory containing eval results")
    parser.add_argument("--prefix", default="", help="Directory prefix to evaluate")
    args = parser.parse_args()

    summarize_judge_scores(
        base_dir=args.root_dir,
        examiner_prefix=args.prefix
    )
