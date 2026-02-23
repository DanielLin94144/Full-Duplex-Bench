# Model Inference

This folder provides **streaming inference interaction scripts** for multiple full-duplex spoken dialogue models.  
It is designed to support evaluation on [Full-Duplex-Bench](https://arxiv.org/abs/2503.04721) and [Full-Duplex-Bench v1.5](https://arxiv.org/abs/2507.23159).

---

## Supported Models

The following models are included, each in its own subfolder:

- **Freeze-Omni** — [VITA-MLLM/Freeze-Omni](https://github.com/VITA-MLLM/Freeze-Omni)  
- **Gemini** — [Google DeepMind Gemini](https://deepmind.google/technologies/gemini/)  
- **GPT-Realtime** — [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)  
- **Moshi** — [Kyutai Labs Moshi](https://github.com/kyutai-labs/moshi)  
- **PersonaPlex** — [NVIDIA PersonaPlex](https://github.com/NVIDIA/personaplex)  
- **Sonic** — [Amazon Nova Sonic](https://aws.amazon.com/bedrock/)  

Each subfolder contains:
- Setup instructions  
- Environment requirements  
- Example inference scripts 

Please refer to the `README.md` inside each subfolder for **model-specific instructions**.

---

## Usage

1. Navigate to the desired model subfolder (e.g., `freeze-omni/`, `moshi/`).  
2. Follow the installation and server/client setup described in its README.  
3. Run inference on [Full-Duplex-Bench v1.0](https://arxiv.org/abs/2503.04721) or [v1.5](https://arxiv.org/abs/2507.23159) evaluation data.  

---

## Citation

If you use or reference this code in your work, please cite:

```bibtex
@article{lin2025full,
  title={Full-duplex-bench: A benchmark to evaluate full-duplex spoken dialogue models on turn-taking capabilities},
  author={Lin, Guan-Ting and Lian, Jiachen and Li, Tingle and Wang, Qirui and Anumanchipalli, Gopala and Liu, Alexander H and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2503.04721},
  year={2025}
}

@article{lin2025full,
  title={Full-Duplex-Bench v1.5: Evaluating Overlap Handling for Full-Duplex Speech Models},
  author={Lin, Guan-Ting and Kuan, Shih-Yun Shan and Wang, Qirui and Lian, Jiachen and Li, Tingle and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2507.23159},
  year={2025}
}
```
