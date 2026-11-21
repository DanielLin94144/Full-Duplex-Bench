# Evaluation Data of Full-Duplex-Bench (v1.0 & v1.5)

## Data Access

> [!IMPORTANT]
> **The CANDOR subsets of v1.0 (`candor_pause_handling`, `candor_turn_taking`) are available only from BetterUp's gated Hugging Face repository [`betterup/candor-full-duplex-bench`](https://huggingface.co/datasets/betterup/candor-full-duplex-bench).**
> They are governed by the CANDOR Corpus License. The folder layout and file names are unchanged, so inference and evaluation code runs as before. All other subsets are still in our Google Drive folder.

| Subset(s) | Where to get it |
|---|---|
| v1.0 **CANDOR** (`candor_pause_handling`, `candor_turn_taking`) | BetterUp: [`betterup/candor-full-duplex-bench`](https://huggingface.co/datasets/betterup/candor-full-duplex-bench) (gated; see [below](#downloading-the-candor-subsets)) |
| v1.0 ICC, Synthetic; all of v1.5 | Our [Google Drive folder](https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3?usp=sharing) |

### Downloading the CANDOR subsets
1. Sign in to Hugging Face, open [betterup/candor-full-duplex-bench](https://huggingface.co/datasets/betterup/candor-full-duplex-bench), fill in the short access form and agree to the CANDOR Corpus License. You do not need to apply for the full CANDOR corpus.
2. Download (requires a recent `huggingface_hub`: `pip install -U huggingface_hub`):
   ```bash
   hf auth login
   hf download betterup/candor-full-duplex-bench --repo-type dataset --local-dir ./candor-full-duplex-bench
   ```
3. This creates `candor-full-duplex-bench/candor_pause_handling/` (216 samples) and `candor-full-duplex-bench/candor_turn_taking/` (119 samples). Use these folders as the data / `--root_dir` paths for inference and evaluation, or move them into your `v1_0/` folder next to the other subsets.

If you previously downloaded `candor_pause_handling.zip` or `candor_turn_taking.zip` from our Google Drive, please do not share or re-upload them. CANDOR data may only be redistributed with BetterUp's permission.

---

## Overview of Data

### v1.0
| Dataset   | Task                | # of Samples |
|-----------|---------------------|--------------|
| Candor*   | Pause Handling      | 216          |
| Candor*   | Smooth Turn-Taking  | 119          |
| ICC       | Backchannel         | 55           |
| Synthetic | User Interruption   | 200          |
| Synthetic | Pause Handling      | 137          |

\* CANDOR data, available only from BetterUp under the CANDOR Corpus License (see [Data Access](#data-access)).

### v1.5
| Subset             | Task                  | # of Samples |
|--------------------|-----------------------|--------------|
| user_interruption  | User Interruption     | 200          |
| user_backchannel   | User Backchannel      | 99           |
| talking_to_other   | Talking to Other      | 100          |
| background_speech  | Background Speech     | 100          |

---

## File Descriptions

### v1.0 — Per-sample files
Each sample folder in **v1.0** contains some of the following (task-dependent):

- **input.wav**: User speech input stream (the evaluation audio).
- **{TASK}.json**: Task-specific annotation file (`pause.json`, `turn_taking.json`, or `interrupt.json`).
- **transcription.json** *(if available)*: Text transcript with word-level timestamps.
- **context.wav** *(only Synthetic User Interruption)*: Context audio preceding the interruption.
- **interrupt.wav** *(only Synthetic User Interruption)*: The interruption audio segment.

> **Note:** v1.0 focuses on natural conversations (Candor, ICC) and synthetic control sets with task annotations; it does **not** include paired “clean” vs. “overlap-removed” audio.

---

### v1.5 — Per-sample files
Each sample folder in **v1.5** contains:

- **input.wav**: User audio **with** an overlap event (e.g., interruption, background, or competing talk).
- **clean_input.wav**: The corresponding user audio **without** the overlap event (non-overlap reference).
- **metadata.json**: Minimal text metadata and overlap window.

`metadata.json` example:
```json
{
  "context_text": "What are some good places to visit in the city?",
  "current_turn_text": "Oh, before I forget, could you also suggest some good restaurants nearby?",
  "timestamps": [7.275, 11.467]
}
```
- `context_text`: Prior conversational context.
- `current_turn_text`: The current (focal) user turn that co-occurs with or surrounds the overlap.
- `timestamps`: `[start, end]` (seconds) of the overlap event in `input.wav`.

> **Note:** v1.5 provides controlled **paired** audios to isolate overlap effects during evaluation.

---

## Dataset Structure

### v1.0

```
v1_0/
├── candor_pause_handling/{ID}/   # CANDOR subset (see Data Access)
│   ├── input.wav                 # User speech input stream
│   ├── pause.json                # Annotations of pause events
│   └── transcription.json        # Text and time-aligned transcription
│
├── candor_turn_taking/{ID}/      # CANDOR subset (see Data Access)
│   ├── input.wav                 # User speech input stream
│   └── turn_taking.json          # Annotations of turn-taking events
│
├── icc_backchannel/{ID}/
│   ├── input.wav                 # User speech input stream
│   └── transcription.json        # Text and time-aligned transcription
│
├── synthetic_pause_handling/{ID}/
│   ├── input.wav                 # User speech input stream
│   ├── pause.json                # Annotations of synthetic pause events
│   └── transcription.json        # Text and time-aligned transcription
│
└── synthetic_user_interruption/{ID}/
    ├── input.wav                 # User speech input stream
    ├── context.wav               # Context audio preceding interruption
    ├── interrupt.wav             # Interruption audio segment
    └── interrupt.json            # Annotations of interruption events

```

---

### v1.5
```
v1_5/
├── user_interruption/{ID}/
│   ├── input.wav             # With overlap event
│   ├── clean_input.wav       # Non-overlap reference
│   └── metadata.json         # {context_text, current_turn_text, timestamps}
│
├── user_backchannel/{ID}/
│   ├── input.wav
│   ├── clean_input.wav
│   └── metadata.json
│
├── talking_to_other/{ID}/
│   ├── input.wav
│   ├── clean_input.wav
│   └── metadata.json
│
└── background_speech/{ID}/
    ├── input.wav
    ├── clean_input.wav
    └── metadata.json
```

---

## License
### v1.0 CANDOR subsets
The `candor_pause_handling` and `candor_turn_taking` subsets are clips from the CANDOR Corpus ([Reece et al., 2023](https://www.science.org/doi/full/10.1126/sciadv.adf3197)). This benchmark uses the CANDOR Corpus from BetterUp Labs, BetterUp, Inc.
These subsets are **not** licensed by us. They are governed solely by the **CANDOR Corpus License** and are available only from BetterUp ([`betterup/candor-full-duplex-bench`](https://huggingface.co/datasets/betterup/candor-full-duplex-bench)). Please do not redistribute them.

### v1.0 ICC subset
Selected from [ICC](https://aclanthology.org/2024.findings-emnlp.909/). Please follow the license and terms of use of the original ICC release.

### v1.0 Synthetic subsets & v1.5 data
The synthetic data is generated by us and released under the **MIT License**.



## Citation
If you use this dataset, please cite it accordingly. **If you report results on the CANDOR subsets (Candor pause handling / Candor smooth turn-taking), please also cite the CANDOR paper (`reece2023candor` below).**

```
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

@article{reece2023candor,
  title={The CANDOR corpus: Insights from a large multimodal dataset of naturalistic conversation},
  author={Reece, Andrew and Cooney, Gus and Bull, Peter and Chung, Christine and Dawson, Bryn and Fitzpatrick, Casey and Glazer, Tamara and Knox, Dean and Liebscher, Alex and Marin, Sebastian},
  journal={Science Advances},
  volume={9},
  number={13},
  pages={eadf3197},
  year={2023},
  publisher={American Association for the Advancement of Science},
  doi={10.1126/sciadv.adf3197}
}

@inproceedings{umair-etal-2024-large,
  title = {Large Language Models Know What To Say But Not When To Speak},
  author = {Umair, Muhammad and Sarathy, Vasanth and Ruiter, Jan},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2024},
  month = nov,
  year = {2024},
  address = {Miami, Florida, USA},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/2024.findings-emnlp.909/},
  doi = {10.18653/v1/2024.findings-emnlp.909},
  pages = {15503--15514}
}
```
