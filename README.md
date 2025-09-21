# Full-Duplex-Bench v1 & v1.5: A Benchmark for Evaluating Turn-Taking and Overlap Handling in Full-Duplex Spoken Dialogue Models
> v1.0 Authors: [Guan-Ting Lin](https://daniellin94144.github.io/), [Jiachen Lian*](https://jlian2.github.io/), [Tingle Li*](https://tinglok.netlify.app/), [Qirui Wang*](https://www.linkedin.com/in/qrw-160509207/), [Gopala Anumanchipalli](https://www2.eecs.berkeley.edu/Faculty/Homepages/gopala.html), [Alexander H. Liu](https://alexander-h-liu.github.io/), [Hung-yi Lee](https://speech.ee.ntu.edu.tw/~hylee/index.html)

> v1.5 Authors: [Guan-Ting Lin](https://daniellin94144.github.io/), Shih-Yun Shan Kuan, [Qirui Wang](https://www.linkedin.com/in/qrw-160509207/), [Jiachen Lian*](https://jlian2.github.io/), [Tingle Li](https://tinglok.netlify.app/), [Hung-yi Lee](https://speech.ee.ntu.edu.tw/~hylee/index.html)

## TL;DR
Benchmark for full-duplex spoken dialogue models — v1.0 evaluates turn-taking, v1.5 adds overlap handling with richer metrics.

[![arXiv](https://img.shields.io/badge/arXiv-2409.06666-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2503.04721)
[![arXiv](https://img.shields.io/badge/arXiv-2409.06666-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2507.23159)
[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github)](https://github.com/DanielLin94144/Full-Duplex-Bench)

## News 🔥
- **(2025/8/22) v1.5 Server-client Model inference Code Release**: Added server-client inference scripts under [`model_inference/`](./model_inference).
- **(2025/8/15) v1.5 Data Release**: Added v1.5 dataset with overlap scenarios and metadata annotations under [`dataset/`](./dataset).
- **(2025/8/14) v1.5 Evaluation Code Release**: Added support for overlap handling with new metrics in Full-Duplex-Bench v1.5 under [`evaluation/`](./evaluation).
- **(2025/6/05) Paper & ASR Model Update**: Replaced the ASR model with nvidia/parakeet-tdt-0.6b-v2, which offers more reliable time-aligned transcriptions for evaluation purposes. The paper has been updated accordingly to reflect this change.
- **(2025/4/30) Dataset Released:** see under the [`dataset/`](./dataset) folder.
- **(2025/4/30) Evaluation Code Released:** see under the [`evaluation/`](./evaluation) folder.
> Stay tuned for upcoming releases!

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


## Repository Structure 📂

This repository is organized into three main components. Please refer to the respective folders for details:

- [`dataset/`](./dataset): Dataset release and detailed description of v1.0 and v1.5 benchmark data.  
- [`evaluation/`](./evaluation): Evaluation code for running benchmark tasks and metrics.  
- [`model_inference/`](./model_inference): Server–client inference setup for running full-duplex models in a streaming manner.  

Each subfolder contains its own README with more detailed instructions.

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
      <td><b>dGSLM</b></td>
      <td>0.934</td><td>0.935</td>
      <td>0.691</td><td><b>0.015</b></td><td><b>0.934</b></td>
      <td><b>0.975</b></td><td>0.352</td>
      <td>0.917</td><td>0.201</td><td>2.531</td>
    </tr>
    <tr>
      <td><b>Moshi</b></td>
      <td>0.985</td><td>0.980</td>
      <td>1.000</td><td>0.001</td><td>0.957</td>
      <td>0.941</td><td><b>0.265</b></td>
      <td><b>1.000</b></td><td>0.765</td><td><b>0.257</b></td>
    </tr>
    <tr>
      <td><b>Freeze-Omni</b></td>
      <td><b>0.642</b></td><td><b>0.481</b></td>
      <td><b>0.636</b></td><td>0.001</td><td>0.997</td>
      <td>0.336</td><td>0.953</td>
      <td>0.867</td><td><b>3.615</b></td><td>1.409</td>
    </tr>
    <tr>
      <td><i>Gemini Live</i></td>
      <td><i>0.255</i></td><td><i>0.310</i></td>
      <td><i>0.091</i></td><td><i>0.012</i></td><td><i>0.896</i></td>
      <td><i>0.655</i></td><td><i>1.301</i></td>
      <td><i>0.891</i></td><td><i>3.376</i></td><td><i>1.183</i></td>
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
```
conda create -n full-duplex-bench python=3.10
conda activate full-duplex-bench
pip install -r requirements.txt
```

### Step-by-step Instruction
#### 1. Model Inference
The goal of model inference is to let the model generate the time-synchronous `output.wav` given the audio stream of user speech (`input.wav`). You can use you own model to generate the output speech for evaluation.

We will provide the example inference code of Freeze-omni under `model_inference/freeze-omni` for different tasks. 
##### ⚠️ Issue
We have observed the same issue and suspect it is due to recent internal changes in **Gemini**.  
We are investigating and will share updates once a solution is found.

#### 2. Prepare for Evaluation with time-aligned transcription
Under `get_transcript` folder, you can find `asr.py` to obtain the time-aligned transcription for the model generated audio. For more details please see the readme in the folder.

#### 3. Running Evaluations
Under `evaluation` folder, please see the readme file in the folder for detailed instruction to run the evaluation for each tasks.

## Citation 📖
If you have any questions, please feel free to submit an issue or contact Guan-Ting Lin (daniel094144@gmail.com)

If you found this research helpful, please consider citing our work:

```
@article{lin2025full,
  title={Full-duplex-bench: A benchmark to evaluate full-duplex spoken dialogue models on turn-taking capabilities},
  author={Lin, Guan-Ting and Lian, Jiachen and Li, Tingle and Wang, Qirui and Anumanchipalli, Gopala and Liu, Alexander H and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2503.04721},
  year={2025}
}

@article{lin2025full,
  title={Full-Duplex-Bench v1. 5: Evaluating Overlap Handling for Full-Duplex Speech Models},
  author={Lin, Guan-Ting and Kuan, Shih-Yun Shan and Wang, Qirui and Lian, Jiachen and Li, Tingle and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2507.23159},
  year={2025}
}
```

