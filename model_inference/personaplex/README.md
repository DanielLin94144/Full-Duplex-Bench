# PersonaPlex Model Inference

PersonaPlex is a real-time, full-duplex speech-to-speech conversational model that enables persona control through text-based role prompts and audio-based voice conditioning. It is based on the [Moshi](https://arxiv.org/abs/2410.00037) architecture.

## Offline Evaluation

For offline evaluation on Full-Duplex-Bench, please refer to the **official PersonaPlex repository**:

👉 **[PersonaPlex Offline Evaluation](https://github.com/NVIDIA/personaplex#offline-evaluation)**

The official repo provides a complete offline evaluation script that processes pre-recorded audio files without needing a live server. Follow the [evaluation instructions](https://github.com/NVIDIA/personaplex#offline-evaluation) in the official repo.

## FullDuplexBench Prompts

As specified by PersonaPlex documentation:

| Task | Prompt |
|------|--------|
| `user_interruption` | "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way." |
| `backchannel`, `pause_handling`, `smooth_turn_taking` | "You enjoy having a good conversation." |

## References

- [PersonaPlex GitHub](https://github.com/NVIDIA/personaplex)
- [PersonaPlex Model](https://huggingface.co/nvidia/personaplex-7b-v1)
- Based on [Moshi Architecture](https://arxiv.org/abs/2410.00037)
