# Moshi Inference Scripts

This folder provides example inference scripts for [Moshi](https://arxiv.org/abs/2410.00037).

For installation instructions, please refer to the official [Moshi repo](https://github.com/kyutai-labs/moshi).

---

## Full-Duplex-Bench v1.5 Inference

For **v1.5**, inference is handled through a **client–server pipeline** using `inference.py`.

### Step 1. Start the Moshi Server

First, go into the [Moshi repo](https://github.com/kyutai-labs/moshi).  
Then, start the Moshi server using the official script:

```bash
python -m moshi.moshi.server
```

(See [server.py](https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/server.py) for details.)

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
