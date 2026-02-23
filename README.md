
# Full-Duplex-Bench: A Benchmark Suite for Evaluating Full-Duplex Spoken Dialogue Models and Systems

<div align="center">
  <img src="./fdb-logo.png" width="70%" alt="FDB Logo" />
</div>

<div align="center">

[![arXiv v1.0](https://img.shields.io/badge/v1.0_arXiv-2503.04721-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2503.04721)
[![arXiv v1.5](https://img.shields.io/badge/v1.5_arXiv-2507.23159-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2507.23159)
[![arXiv v2.0](https://img.shields.io/badge/v2.0_arXiv-2510.07838-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2510.07838)
[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github)](https://github.com/DanielLin94144/Full-Duplex-Bench)

</div>

Welcome to **Full-Duplex-Bench**, with v1.0, v1.5, and v2.0, a comprehensive framework designed to evaluate the conversational and turn-taking capabilities of spoken language models.
## News 🔥
- **(2026/2/23) Full-Duplex-Bench v2 Framework Release**: Introduced the V2 architecture with a real-time WebRTC orchestrator and automated AI examiner in the [`v2/`](./v2) folder!
- **(2026/2/21) Codebase Update for New Models and Bug Fixes**: Add Gemini 2.5 Native Audio & PersonaPlex, and update the codebase.
- **(2025/8/22) v1.5 Server-client Model inference Code Release**: Added server-client inference scripts under [`v1_v1.5/model_inference/`](./v1_v1.5/model_inference).
- **(2025/8/15) v1.5 Data Release**: Added v1.5 dataset with overlap scenarios and metadata annotations under [`v1_v1.5/dataset/`](./v1_v1.5/dataset).
- **(2025/8/14) v1.5 Evaluation Code Release**: Added support for overlap handling with new metrics in Full-Duplex-Bench v1.5 under [`v1_v1.5/evaluation/`](./v1_v1.5/evaluation).
- **(2025/6/05) Paper & ASR Model Update**: Replaced the ASR model with nvidia/parakeet-tdt-0.6b-v2, which offers more reliable time-aligned transcriptions for evaluation purposes.


## 🏗️ Repository Architecture
Due to the evolution of evaluation paradigms—from static dataset evaluation to dynamic real-time interaction—this repository is organized into distinct architectures:

### [Full-Duplex-Bench v1 & v1.5 (Static Offline Evaluation)](./v1_v1.5)
**👉 [Dive into v1 & v1.5](./v1_v1.5/README.md)**

The **legacy v1 and v1.5** pipelines evaluate models based on pre-recorded static datasets in an offline, server-client inference manner.
- **Highlights (v1.0)**: Systematically assesses 4 dimensions: Pause Handling, Backchanneling, Smooth Turn-Taking, and User Interruption. ([FDB v1.0 paper](https://arxiv.org/abs/2503.04721))
- **Highlights (v1.5)**: Extends the benchmark with overlap scenarios including listener backchannel, side conversation, and ambient speech. ([FDB v1.5 paper](https://arxiv.org/abs/2507.23159))

### [Full-Duplex-Bench-v2 (Real-Time Dynamic Evaluation)](./v2)
**👉 [Dive into v2](./v2/README.md)** | [Demo Website](https://ericsunkuan.github.io/full-duplex-bench-v2-demo/)

**FDB-v2** is our actively evolving, state-of-the-art framework. It orchestrates **real-time audio conversations** (via WebRTC or WebSocket) between your target model (the Examinee) and an automated AI evaluator (the Examiner). 
- **Highlights**: Dynamic multi-turn tasks, WebRTC Node.js orchestrator, conversational constraints, LLM-as-a-judge automated scoring. ([FDB v2.0 paper](https://arxiv.org/abs/2510.07838))
- **Use Case**: Best for evaluating how well a model converses reactively in a live environment.


## 🧭 Getting Started

Depending on your goal, please navigate to the respective folder:

- **To run offline static evaluations or reproduce results from our v1.0/v1.5 papers:**  
  Navigate to the [`v1_v1.5/` directory](./v1_v1.5) to view datasets, setup offline inference, and compute static metrics.

- **To benchmark a model using the latest real-time automated AI examiner (v2):**  
  Navigate to the [`v2/` directory](./v2) and follow the combined Node.js and Python setup instructions.



## 📖 Citation

If you found this research helpful, please consider citing our work:

```bibtex
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

@article{lin2026full_v2,
  title={Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner},
  author={Lin, Guan-Ting and Kuan, Shih-Yun Shan and Shi, Jiatong and Chang, Kai-Wei and Arora, Siddhant and Watanabe, Shinji and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2510.07838},
  year={2026}
}

```

---
*For questions, please feel free to submit an issue or contact Guan-Ting Lin (daniel094144@gmail.com).*
