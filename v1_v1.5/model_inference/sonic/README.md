# Amazon Nova-Sonic Inference Script

This folder provides an example inference script for [Amazon Nova-Sonic](https://aws.amazon.com/bedrock/) running on **Amazon Bedrock** with **bidirectional audio streaming**. For more details, refer to the [Amazon Nova-Sonic documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/amazon-nova-sonic.html) and [Official Sample Code](https://github.com/aws-samples/amazon-nova-samples/tree/main/speech-to-speech/sample-codes/console-python).

The script `inference.py` demonstrates how to run inference for **Full-Duplex-Bench v1.5** tasks.

---

## Prerequisites

1. **AWS Account with Bedrock access**  
   Ensure you have access to Amazon Bedrock in your AWS account.

2. **AWS Credentials**  
   Export your credentials as environment variables or edit them inside the script:
   ```bash
   export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY"
   export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_KEY"
   export AWS_DEFAULT_REGION="us-east-1"
   ```

   (The script also sets them via `os.environ[...]`.)

3. **Dependencies**  
   Install the required Python packages:
   ```bash
   pyaudio>=0.2.13
   rx>=3.2.0
   smithy-aws-core>=0.0.1
   pytz
   aws_sdk_bedrock_runtime
   ```

---

## Usage

Run inference with:

```bash
python inference.py --input input.wav --output output.wav --region us-east-1 --model amazon.nova-sonic-v1:0
```

---

## Configuration in the Script

At the top of `bedrock_wav_client.py`, configure:

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

---

## How It Works

- The script streams **16kHz mono user audio** (`input.wav`) to the Bedrock model.  
- The model returns **24kHz speech output**, written to `output.wav`.  
- The `BedrockStreamManager` manages:
  - Session start / end events  
  - Audio chunk streaming  
  - Handling **barge-in** (user interruptions)  
  - Writing final outputs  



