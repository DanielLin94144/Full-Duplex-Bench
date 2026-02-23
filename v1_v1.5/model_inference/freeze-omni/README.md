# Freeze-Omni Inference Scripts

This folder provides example inference scripts for [Freeze-Omni](https://arxiv.org/abs/2411.00774).

For installation instructions, please refer to the official [Freeze-Omni repo](https://github.com/VITA-MLLM/Freeze-Omni).

---

## Full-Duplex-Bench v1.0 Inference

The following scripts demonstrate how to run model inference for each corresponding task in **Full-Duplex-Bench v1.0**:

- `backchannel.py`
- `pause_handling.py`
- `smooth_turn_taking.py`
- `user_interruption.py`

Each script directly loads the evaluation samples and runs the model for the target task.

---

## Full-Duplex-Bench v1.5 Inference

For **v1.5**, inference is handled through a **client–server pipeline** using `inference.py`.

### Step 1. Start the Demo Server

First, go into [Freeze-Omni repo](https://github.com/VITA-MLLM/Freeze-Omni). Then, start the official demo server provided by [Freeze-Omni's script](https://github.com/VITA-MLLM/Freeze-Omni/blob/main/scripts/run_demo_server.sh).

### Step 2. Configure `inference.py`

Edit the following section at the top of `inference.py`:

```python
root_dir_path = "YOUR_ROOT_DIRECTORY_PATH"
tasks = [
    "YOUR_TASK_NAME",
]
prefix = ""  # "" or "clean_": the prefix for input wav files
overwrite = True  # Whether to overwrite existing output files
```

- **`root_dir_path`**: base directory of Full-Duplex-Bench v1.5 (e.g., `data-full-duplex-bench/v1.5/`).  
- **`tasks`**: list of tasks to evaluate.  
- **`prefix`**:  
  - `""` → raw input (with overlaps)  
  - `"clean_"` → cleaned non-overlap reference files  
- **`overwrite`**: whether existing outputs should be replaced.  

### Step 3. Run Inference

Once configured, run:

```bash
python inference.py
```

The script will generate output files for each evaluated task.