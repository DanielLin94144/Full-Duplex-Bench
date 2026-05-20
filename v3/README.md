# FDB-v3: Multi-Step Tool Calling Benchmark for Voice Agents

This repository provides the inference and evaluation pipeline for the **FDB-v3 Multi-Step Tool Calling Benchmark**. The benchmark evaluates voice-based AI agents on their ability to correctly invoke tool calls (with correct arguments) and respond naturally, using human-recorded audio with natural disfluencies.

> **Note:** The FDB-v3 benchmark data and codebase were developed by **National Taiwan University (NTU)**. **NVIDIA** contributed through a collaborative research discussion and advisory role.

## Repository Structure

```
v3/
├── README.md
├── run_tool_benchmark_all_released.py   # Main inference pipeline (batch)
├── run_tool_benchmark.py                # Core inference functions (ASR, LiveKit, latency)
├── livekit_inference.py                 # Headless LiveKit client for streaming audio
├── lk_agent_tool.py                     # LiveKit voice agent (native realtime models)
├── cascaded_agent.py                    # Cascaded agent (STT + LLM + TTS pipeline)
├── mock_apis.py                         # Mock API backends for 12 tools
├── latency_injector.py                  # Configurable API latency simulation
├── benchmark_data_v2.json               # Benchmark scenario definitions (79 scenarios)
├── evaluate_tool_calls.py               # Evaluation: tool accuracy (F1 scores)
├── evaluate_pass_rate.py                # Evaluation: binary pass/fail rate
├── analyze_tool_latency.py              # Evaluation: fine-grained latency analysis
├── run_all_evaluations_released.sh      # Run all evaluation steps for a provider
└── run_agent.sh                         # Helper script to start the agent
```

## Data

📥 **Download the benchmark data from [Google Drive](https://drive.google.com/file/d/1SO_4MTazWQ_jvCx0dtmpQ-t40bdd07yz/view?usp=sharing).**

After downloading, extract and place the `fdb_v3_data_released/` folder inside the `v3/` directory so the layout looks like:

```
v3/
    fdb_v3_data_released/
        {example_id}_{speaker_id}/
            input.wav           # Human-recorded spoken user query (48 kHz)
            metadata.json       # Scenario definition and expected tool calls
```

- **100 examples**, 79 unique scenarios, 12 speakers
- **4 domains**: ecommerce_support, finance_billing, housing_location, travel_identity
- **3 difficulty levels**: easy (1 tool call), medium (2), hard (3)

## Prerequisites

### 1. Python Environment

```bash
conda create -n fdb python=3.10
conda activate fdb
```

### 2. Install Dependencies

```bash
# LiveKit agents framework (pick the providers you need)
pip install "livekit-agents[openai,google,xai]~=1.3" \
            "livekit-plugins-ultravox" \
            python-dotenv

# For cascaded agent (Silero VAD + OpenAI Whisper STT + OpenAI gpt-4o LLM + OpenAI TTS)
pip install "livekit-plugins-silero" "livekit-plugins-openai"

# LiveKit client SDK (for streaming audio)
pip install "livekit[crypto]~=1.0" numpy

# ASR model for transcription
pip install nemo_toolkit[asr]

# Audio processing
pip install pydub ffmpeg-python

# For LLM-based evaluation (gpt-4o judge)
pip install openai
```

### 3. External Tools

- **ffmpeg**: Required for audio conversion. Install via `apt install ffmpeg` or `brew install ffmpeg`.

### 4. Environment Variables

Create a `.env.local` file in the `v3/` directory:

```bash
# LiveKit Cloud credentials (sign up at https://cloud.livekit.io)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret

# Provider API keys (set the ones you need)
OPENAI_API_KEY=sk-...          # GPT Realtime / cascaded agent / evaluation LLM judge
XAI_API_KEY=...                # Grok
GOOGLE_API_KEY=...             # Gemini
ULTRAVOX_API_KEY=...           # Ultravox
```

> **Note:** A [LiveKit Cloud](https://cloud.livekit.io) account (free tier available) is required to run inference. The evaluation scripts (Step 3 below) do **not** require LiveKit.

## Running Inference

Inference streams each audio sample through a LiveKit room where a voice agent is running, captures the agent's response, runs ASR, and records tool calls.

### Step 1: Start the Voice Agent

In a **separate terminal**, start the agent server. Choose one of the following:

**Option A: Native Realtime Model Agent** (GPT Realtime, Gemini, Grok, Ultravox)

```bash
cd v3

# Set provider via environment variable
LK_PROVIDER=gpt_realtime python lk_agent_tool.py start

# Or for other providers:
# LK_PROVIDER=gemini2_5 python lk_agent_tool.py start
# LK_PROVIDER=gemini3_1 python lk_agent_tool.py start
# LK_PROVIDER=grok python lk_agent_tool.py start
# LK_PROVIDER=ultravox python lk_agent_tool.py start
```

**Option B: Cascaded Agent** (Silero VAD + OpenAI Whisper STT + gpt-4o LLM + OpenAI TTS)

```bash
cd v3
python cascaded_agent.py start
```

### Step 2: Run Batch Inference

In a **different terminal** (while the agent is running):

```bash
cd v3

# Run inference for a specific provider
python run_tool_benchmark_all_released.py --provider gpt_realtime

# With custom data directory
python run_tool_benchmark_all_released.py --provider gemini2_5 --root_dir fdb_v3_data_released

# Overwrite existing results
python run_tool_benchmark_all_released.py --provider gpt_realtime --force

# Skip inference, only run ASR on existing outputs
python run_tool_benchmark_all_released.py --provider gpt_realtime --asr-only
```

This will process all 100 audio samples and save `result_{provider}.json` files in each example folder under `fdb_v3_data_released/`.

### Supported Providers

| Provider | `--provider` value | Agent Script | Model |
|---|---|---|---|
| GPT Realtime | `gpt_realtime` | `lk_agent_tool.py` | gpt-realtime-1.5 |
| Gemini 2.5 | `gemini2_5` | `lk_agent_tool.py` | gemini-2.5-flash-native-audio |
| Gemini 3.1 | `gemini3_1` | `lk_agent_tool.py` | gemini-3.1-flash-live |
| Grok | `grok` | `lk_agent_tool.py` | Grok Voice Agent |
| Ultravox | `ultravox` | `lk_agent_tool.py` | Ultravox Realtime |
| Cascaded (STT+LLM+TTS) | `cascaded` | `cascaded_agent.py` | Whisper + gpt-4o + OpenAI TTS |

## Running Evaluation

After inference is complete (i.e., `result_{provider}.json` files exist in each example folder), run the evaluation pipeline:

### Option A: Run All Evaluations at Once

```bash
cd v3

# Edit the PROVIDERS array in the script to choose which providers to evaluate
bash run_all_evaluations_released.sh
```

By default this runs three evaluation steps per provider:
1. **Tool accuracy** (`evaluate_tool_calls.py`) — F1 of tool selection, argument accuracy, response quality
2. **Pass rate** (`evaluate_pass_rate.py`) — strict binary pass/fail
3. **Latency analysis** (`analyze_tool_latency.py`) — first response, tool call, and task completion latency

### Option B: Run Individual Evaluation Steps

```bash
cd v3

# 1. Tool accuracy (continuous scores)
python evaluate_tool_calls.py \
    --benchmark benchmark_data_v2.json \
    --results-dir fdb_v3_data_released \
    --provider gpt_realtime \
    --output gpt_realtime_evaluation_report.json \
    --use-llm

# 2. Binary pass/fail rate
python evaluate_pass_rate.py \
    --benchmark benchmark_data_v2.json \
    --results-dir fdb_v3_data_released \
    --provider gpt_realtime \
    --output gpt_realtime_pass_rate_report.json \
    --use-llm

# 3. Fine-grained latency analysis
python analyze_tool_latency.py \
    --results-dir fdb_v3_data_released \
    --provider gpt_realtime
```

**Note**: The `--use-llm` flag uses gpt-4o as an LLM judge for semantic argument matching and response quality evaluation. Without it, argument matching falls back to exact string comparison and response accuracy is skipped.

### Evaluation Outputs

Each evaluation step produces a JSON report in `v3/`:

| Report | Content |
|---|---|
| `{provider}_evaluation_report.json` | Per-scenario tool selection F1, argument accuracy, response accuracy, latency |
| `{provider}_pass_rate_report.json` | Binary pass/fail per scenario, breakdowns by domain/difficulty/disfluency |
| `{provider}_latency_report.json` | First response, tool call, and task completion latency statistics |

## Evaluation Metrics

### Tool Selection Accuracy (F1)
- **Recall**: fraction of expected tools that were actually called
- **Precision**: fraction of actual calls that were expected (penalizes extra calls)
- **F1**: harmonic mean of recall and precision

### Argument Accuracy
- Semantic correctness of function arguments (via gpt-4o judge with `--use-llm`, or exact match)
- Handles dynamic references (`$RESULT_0.field`), formatting differences, and common aliases

### Response Accuracy
- Whether the agent's spoken response matches the expected task completion (gpt-4o judge)
- Only evaluated when `--use-llm` is enabled

### Pass Rate (Strict Binary)
- **PASS**: ALL expected tools called with correct arguments (no missing, no extra)
- **FAIL**: any missing tool, extra tool, or wrong argument

### Latency Metrics
1. **First Response Latency**: time from user speech end to agent's first word
2. **Tool Call Latency**: time from user speech end to first tool invocation
3. **Task Completion Latency**: time from user speech end to the key information in agent's response (identified by gpt-4o)

## Available Tools (12 APIs)

The agent has access to 12 mock API tools across 4 domains:

| Domain | Tools |
|---|---|
| Travel & Identity | `search_flights`, `book_flight`, `update_identity_doc` |
| Finance & Billing | `get_card_benefits`, `get_exchange_rate`, `modify_autopay` |
| Housing & Location | `search_apartments`, `calculate_commute`, `update_search_filter` |
| E-Commerce | `track_order`, `search_products`, `add_to_cart` |

## Quick Start Example

Make sure you have completed the [Prerequisites](#prerequisites) (conda env, dependencies, `.env.local`) and downloaded the [benchmark data](#data) before running.

```bash
# Terminal 1: Start the agent (from the repo root)
cd v3
LK_PROVIDER=gpt_realtime python lk_agent_tool.py start

# Terminal 2: Run inference (from the repo root)
cd v3
python run_tool_benchmark_all_released.py --provider gpt_realtime

# Terminal 2: Run evaluation (after inference completes, stop the agent in Terminal 1)
python evaluate_tool_calls.py \
    --benchmark benchmark_data_v2.json \
    --results-dir fdb_v3_data_released \
    --provider gpt_realtime \
    --output gpt_realtime_evaluation_report.json \
    --use-llm

python evaluate_pass_rate.py \
    --benchmark benchmark_data_v2.json \
    --results-dir fdb_v3_data_released \
    --provider gpt_realtime \
    --output gpt_realtime_pass_rate_report.json \
    --use-llm

python analyze_tool_latency.py \
    --results-dir fdb_v3_data_released \
    --provider gpt_realtime
```

## 📖 Citation

If you found this research helpful, please consider citing our work:

```bibtex
@article{lin2025fdb_v1,
  title={Full-duplex-bench: A benchmark to evaluate full-duplex spoken dialogue models on turn-taking capabilities},
  author={Lin, Guan-Ting and Lian, Jiachen and Li, Tingle and Wang, Qirui and Anumanchipalli, Gopala and Liu, Alexander H and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2503.04721},
  year={2025}
}

@article{lin2025fdb_v15,
  title={Full-Duplex-Bench v1. 5: Evaluating Overlap Handling for Full-Duplex Speech Models},
  author={Lin, Guan-Ting and Kuan, Shih-Yun Shan and Wang, Qirui and Lian, Jiachen and Li, Tingle and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2507.23159},
  year={2025}
}

@article{lin2026fdb_v2,
  title={Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner},
  author={Lin, Guan-Ting and Kuan, Shih-Yun Shan and Shi, Jiatong and Chang, Kai-Wei and Arora, Siddhant and Watanabe, Shinji and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2510.07838},
  year={2026}
}

@article{lin2026fdb_v3,
  title={Full-Duplex-Bench-v3: Benchmarking Tool Use for Full-Duplex Voice Agents Under Real-World Disfluency},
  author={Lin, Guan-Ting and Chen, Chen and Chen, Zhehuai and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2604.04847},
  year={2026}
}
```

---
*For questions, please feel free to submit an issue or contact Guan-Ting Lin (daniel094144@gmail.com).*

