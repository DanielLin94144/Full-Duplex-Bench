## Run ASR to Prepare for Evaluation
An ASR script is provided to process the audio outputs. Ensure your data follows the structure:
```
{OUTPUT_WAV_ROOT_PATH}/{ID}/output.wav
```

### Running ASR

Use the following command to run the ASR script:

```bash
python asr.py --root_dir {OUTPUT_WAV_ROOT_PATH} --task {TASK_NAME}
```

- Replace `{OUTPUT_WAV_ROOT_PATH}` with the actual root path containing your output WAV files.
- Replace `{TASK_NAME}` with `default` (full transcription) or `user_interruption` (crops audio after the interruption before ASR).
