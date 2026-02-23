# Full-Duplex-Bench v1 & v1.5: A Benchmark for Evaluating Turn-Taking and Overlap Handling in Full-Duplex Spoken Dialogue Models
> v1.0 Authors: [Guan-Ting Lin](https://daniellin94144.github.io/), [Jiachen Lian*](https://jlian2.github.io/), [Tingle Li*](https://tinglok.netlify.app/), [Qirui Wang*](https://www.linkedin.com/in/qrw-160509207/), [Gopala Anumanchipalli](https://www2.eecs.berkeley.edu/Faculty/Homepages/gopala.html), [Alexander H. Liu](https://alexander-h-liu.github.io/), [Hung-yi Lee](https://speech.ee.ntu.edu.tw/~hylee/index.html)

> v1.5 Authors: [Guan-Ting Lin](https://daniellin94144.github.io/), Shih-Yun Shan Kuan, [Qirui Wang](https://www.linkedin.com/in/qrw-160509207/), [Jiachen Lian*](https://jlian2.github.io/), [Tingle Li](https://tinglok.netlify.app/), [Hung-yi Lee](https://speech.ee.ntu.edu.tw/~hylee/index.html)

## TL;DR
Benchmark for full-duplex spoken dialogue models — v1.0 evaluates turn-taking, v1.5 adds overlap handling with richer metrics.

## Highlights 💡
### Full-Duplex-Bench v1.0
- Provides an open and standardized benchmark to assess interactive behaviors systematically.
- Evaluates four key turn-taking dimensions: Pause Handling, Backchanneling, Smooth Turn-Taking, and User Interruption Management.
- Leverages automatic metrics for reproducible evaluation across models.
<div align="center"><img src="https://github.com/user-attachments/assets/70b6525c-61ee-4c48-a1fb-59dc6dfe85cc" width="80%"/></div>
<div align="center"><img src="https://github.com/user-attachments/assets/e936d330-1105-42fc-b5c6-d7ee8f40d27c" width="60%"/></div>

### Full-Duplex-Bench v1.5
- Extends the benchmark with four simulated overlap scenarios: user interruption, listener backchannel, side conversation, and ambient speech.
- Supports both open-sourced and commercial models.
- Introduces a comprehensive metric suite — categorical dialogue behaviors, stop and response latency, prosodic adaptation, and perceived speech quality — customizable to application needs.
<div align="center"><img src="https://github.com/user-attachments/assets/969853c2-885f-40f1-bf7b-0c4da0e2fab4" width="75%"/></div>
<div align="center"><img src="https://github.com/user-attachments/assets/b0f43c6e-18a5-4ca1-bceb-0ae285a8782d" width="60%"/></div>


## 📊 Evaluation Results 

### Full-Duplex-Bench (v1.0)
<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2" style="text-align:center">Pause Handling</th>
      <th colspan="3" style="text-align:center">Backchannel</th>
      <th colspan="2" style="text-align:center">Smooth Turn Taking</th>
      <th colspan="3" style="text-align:center">User Interruption</th>
    </tr>
    <tr>
      <th>Synthetic TOR ↓</th><th>Candor TOR ↓</th>
      <th>TOR ↓</th><th>Freq ↑</th><th>JSD ↓</th>
      <th>Candor TOR ↑</th><th>Latency ↓</th>
      <th>TOR ↑</th><th>GPT-4o ↑</th><th>Latency ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>PersonaPlex</b></td>
      <td>0.584</td><td>0.662</td>
      <td>0.327</td><td><b>0.025</b></td><td><b>0.649</b></td>
      <td><b>0.992</b></td><td><b>0.070</b></td>
      <td><b>1.000</b></td><td>4.210</td><td>0.400</td>
    </tr>
    <tr>
      <td><b>dGSLM</b></td>
      <td>0.934</td><td>0.935</td>
      <td>0.691</td><td>0.015</td><td>0.934</td>
      <td>0.975</td><td>0.352</td>
      <td>0.917</td><td>0.201</td><td>2.531</td>
    </tr>
    <tr>
      <td><b>Moshi</b></td>
      <td>0.985</td><td>0.980</td>
      <td>1.000</td><td>0.001</td><td>0.957</td>
      <td>0.941</td><td>0.265</td>
      <td><b>1.000</b></td><td>0.765</td><td><b>0.257</b></td>
    </tr>
    <tr>
      <td><b>Freeze-Omni</b></td>
      <td>0.642</td><td>0.481</td>
      <td><b>0.636</b></td><td>0.001</td><td>0.997</td>
      <td>0.336</td><td>0.953</td>
      <td>0.867</td><td>3.615</td><td>1.409</td>
    </tr>
    <tr>
      <td><i>Gemini Live 2.0 (deprecated)</i></td>
      <td><i>0.255</i></td><td><i>0.310</i></td>
      <td><i>0.091</i></td><td><i>0.012</i></td><td><i>0.896</i></td>
      <td><i>0.655</i></td><td><i>1.301</i></td>
      <td><i>0.891</i></td><td><i>3.376</i></td><td><i>1.183</i></td>
    </tr>
    <tr>
      <td><b>GPT-Realtime</b></td>
      <td><b>0.010</b></td><td><b>0.120</b></td>
      <td><b>0.000</b></td><td>0.007</td><td>0.980</td>
      <td><b>1.000</b></td><td>1.470</td>
      <td>0.970</td><td>3.850</td><td>1.500</td>
    </tr>
  </tbody>
</table>

- **TOR**: Turn-Over Rate (↓: lower is better for Pause/Backchannel, ↑ for Smooth Turn/User Interruption)
- **Freq**: Frequency of backchannels (↑ better)
- **JSD**: Jensen-Shannon Divergence (↓ better)
- **Latency**: Response latency (↓ better)
- **GPT-4o**: GPT-4o-assessed contextual relevance (↑ better)

## Getting Started 🏁
### Installation
```bash
conda create -n full-duplex-bench python=3.10
conda activate full-duplex-bench
pip install -r requirements.txt
```

### Configuration
Create a `.env` file in the project root with your API keys:
```bash
cp .env.example .env
# Edit .env with your actual API keys
```

Required environment variables (see `.env.example` for template):
- `OPENAI_API_KEY` - For GPT-Realtime evaluation and models
- `GEMINI_API_KEY` - For Gemini models
- `HF_TOKEN` - For PersonaPlex and other HuggingFace models
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` - For Nova Sonic

### Step-by-step Instruction
#### 1. Model Inference
The goal of model inference is to let the model generate the time-synchronous `output.wav` given the audio stream of user speech (`input.wav`). You can use your own model to generate the output speech for evaluation.

We provide inference scripts under `v1_v1.5/model_inference/` for different models:
- **Gemini 2.5 Native Audio**: `v1_v1.5/model_inference/gemini/inference_gemini25_native.py`
- **Gemini 2.0 (NOT SUPPORTED)**: originally in `v1_v1.5/model_inference/gemini/inference.py`, but it is not supported anymore due to the API changes.
- **PersonaPlex** (NVIDIA): See [official repo](https://github.com/NVIDIA/personaplex#offline-evaluation)
- **Moshi**: `v1_v1.5/model_inference/moshi/inference.py`
- **Nova Sonic**: `v1_v1.5/model_inference/sonic/inference.py`
- **Freeze-Omni**: `v1_v1.5/model_inference/freeze-omni/`
- **GPT-Realtime**: `v1_v1.5/model_inference/gpt-realtime/`

Example usage for Gemini 2.5:
```bash
python v1_v1.5/model_inference/gemini/inference_gemini25_native.py \
    --base-dir /path/to/data \
    --task backchannel \
    --overwrite
```

#### 2. Prepare for Evaluation with time-aligned transcription
Under `v1_v1.5/get_transcript` folder, you can find `asr.py` to obtain the time-aligned transcription for the model generated audio. For more details please see the readme in the folder.

#### 3. Running Evaluations
Under `v1_v1.5/evaluation` folder, please see the readme file in the folder for detailed instruction to run the evaluation for each tasks.

## Citation 📖
If you have any questions, please feel free to submit an issue or contact Guan-Ting Lin (daniel094144@gmail.com)

If you found this research helpful, please consider citing our work:

```
@article{lin2025full_v1,
  title={Full-duplex-bench: A benchmark to evaluate full-duplex spoken dialogue models on turn-taking capabilities},
  author={Lin, Guan-Ting and Lian, Jiachen and Li, Tingle and Wang, Qirui and Anumanchipalli, Gopala and Liu, Alexander H and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2503.04721},
  year={2025}
}

@article{lin2025full_v15,
  title={Full-Duplex-Bench v1. 5: Evaluating Overlap Handling for Full-Duplex Speech Models},
  author={Lin, Guan-Ting and Kuan, Shih-Yun Shan and Wang, Qirui and Lian, Jiachen and Li, Tingle and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2507.23159},
  year={2025}
}
```


