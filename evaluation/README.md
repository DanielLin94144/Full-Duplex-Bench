# Full-Duplex-Bench Evaluation Scripts

This folder contains evaluation scripts for **Full-Duplex-Bench** tasks, covering both **v1.0** and **v1.5**. Common utilities (e.g., `get_transcript/asr.py`, `evaluate.py`) are shared, while v1.5 adds additional analyses and scripts.

---

## 1) Common Setup (v1.0 & v1.5)

### A. Prepare ASR-aligned transcripts
Before any evaluation, generate word-level time-aligned transcripts:
```bash
python get_transcript/asr.py
```
This creates `output.json` in each sample directory, aligned to the corresponding `output.wav`.

### B. Dataset layout
Each evaluation run expects a **root directory** (`--root_dir`) that contains the audio and aligned transcripts.

**v1.0 expected contents**
```
{root_dir}/
  ├── output.wav       # model response audio
  ├── output.json      # ASR-aligned transcript for output.wav
  └── (optional) metadata / annotations
```

**v1.5 expected contents (adds clean non-overlap references)**
```
{root_dir}/
  ├── output.wav         # model response (may include overlap scenario)
  ├── output.json        # ASR-aligned transcript for output.wav
  ├── clean_input.wav    # clean user input speech (no overlap)
  ├── clean_input.json   # ASR-aligned transcript for clean_input.wav
  ├── clean_output.wav   # model response to the clean input
  ├── clean_output.json  # ASR-aligned transcript for clean_output.wav
  └── (optional) metadata / annotations
```
The **clean files** serve as a non-overlap baseline for adaptation analyses in v1.5.

### C. Shared CLI entry
Both versions use the same base entry for task-level evaluation:
```bash
python evaluate.py --task {TASK_NAME} --root_dir {DIRECTORY_PATH}
```

---

## 2) v1.0 Tasks

Supported tasks:
- `backchannel`
- `pause_handling`
- `smooth_turn_taking`
- `user_interruption`

**Example**
```bash
python evaluate.py --task pause_handling   --root_dir /path/to/data-full-duplex-bench/v1.0/pause_handling_example
```

---

## 3) v1.5 Tasks

v1.5 extends the benchmark with behavior analysis, speech feature adaptation, statistical testing, and timing metrics.

### 3.1 Behavior evaluation (categorical)
Evaluate interruption/behavior categories.
```bash
python evaluate.py --task behavior --root_dir /path/to/data-full-duplex-bench/v1.5/{folder_name}
```
Results are saved in `{folder_name}_behavior.log`.

### 3.2 Speech feature adaptation
Extract speech features **before vs. after** a target event, leveraging both overlapped and clean references where applicable.
```bash
python evaluate.py --task general_before_after  --root_dir /path/to/data-full-duplex-bench/v1.5/{folder_name}
```
Typical metrics produced include: `utmosv2`, `wpm`, `mean_pitch`, `std_pitch`, `mean_intensity`, `std_intensity`

### 3.3 Timing metrics (stop & response latency)
Compute timing-related metrics such as stop latency and response latency.
```bash
python get_timing.py  --root_dir /path/to/data-full-duplex-bench/v1.5/{folder_name}
```
The latency of each sample is computed and saved in `/path/to/data-full-duplex-bench/v1.5/{folder_name}/latency_intervals.json`.

### 3.4 Statistical significance (paired t-test)
Run paired t-tests on metrics with “pre-overlap vs. post-overlap” and "non-overlap vs. post-overlap" setups.
```bash
python significance_test.py   --root_dir /path/to/data-full-duplex-bench/v1.5/{folder_name}  --metrics utmosv2 wpm mean_pitch std_pitch mean_intensity std_intensity
```
**Dependency:** requires feature summary files produced by `--task general_before_after` in the same `--root_dir`. Results are saved in `pair_t_{folder_name}.txt`.

---

### 3.5 Run Order & Dependencies

1. **ASR alignment** (`get_transcript/asr.py`) is required by all tasks to produce `output.json` (and `clean_*.json` for non-overlap audio).
2. Run **Behavior** (`--task behavior`) to analyze the behavior categories.
3. **Feature extraction** (`evaluate.py --task general_before_after`) must be completed **before** running `significance_test.py`.


