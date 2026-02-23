import os
import json
import argparse
from glob import glob
from pathlib import Path

import soundfile as sf
import nemo.collections.asr as nemo_asr
from tqdm import tqdm


def transcribe_audio_file(audio_path, asr_model):
    """
    Transcribe a single audio file and return the result dict.
    Returns a dict with 'text' and 'chunks' keys.
    """
    print(f"Processing: {audio_path}")
    
    # Read the audio file
    waveform, sr = sf.read(audio_path)
    
    # Convert to mono if needed
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    
    # Create temporary file for transcription
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, waveform, sr)
        asr_outputs = asr_model.transcribe([tmp.name], timestamps=True)
    
    # Clean up temp file
    os.unlink(tmp.name)
    
    # Extract word timestamps
    result = asr_outputs[0]
    word_timestamps = result.timestamp["word"]
    
    # Build output dict
    chunks = []
    text = ""
    for w in word_timestamps:
        start_time = w["start"]
        end_time = w["end"]
        word = w["word"]
        
        text += word + " "
        chunks.append({
            "text": word,
            "timestamp": [start_time, end_time],
        })
    
    return {
        "text": text.strip(),
        "chunks": chunks,
    }


def find_audio_pairs(root_dir):
    """
    Find all A.wav and B.wav pairs in the directory hierarchy.
    Returns a list of tuples: (sample_folder_path, task_name, setup, model, sample)
    
    Updated to handle simplified structure: root/Task/Task.sample_id/A.wav
    """
    audio_pairs = []
    
    # Look for task folders (case-insensitive)
    # Expected structure: root/Task/Sample_Folder/A.wav
    task_folders = ["Daily", "Correction", "EntityTracking", "Safety"]
    
    for task in task_folders:
        task_path = os.path.join(root_dir, task)
        if not os.path.exists(task_path):
            # Try lowercase version
            task_lower = task.lower()
            task_path = os.path.join(root_dir, task_lower)
            if not os.path.exists(task_path):
                continue
            task = task_lower
        
        # Look for sample folders directly under task folder
        for sample_folder in os.listdir(task_path):
            sample_path = os.path.join(task_path, sample_folder)
            if not os.path.isdir(sample_path):
                continue
            
            # Check if both A.wav and B.wav exist
            a_wav = find_file_recursive(sample_path, "A.wav")
            b_wav = find_file_recursive(sample_path, "B.wav")
            
            if a_wav and b_wav:
                # Create key name: task.sample
                key = f"{task}.{sample_folder}"
                audio_pairs.append({
                    "key": key,
                    "sample_path": sample_path,
                    "a_wav": a_wav,
                    "b_wav": b_wav,
                    "task": task,
                    "setup": "default",  # No setup in this structure
                    "model": "default",  # No model in this structure
                    "sample": sample_folder
                })
    
    return audio_pairs


def find_file_recursive(directory, filename):
    """
    Recursively find a file in a directory.
    Returns the full path if found, None otherwise.
    """
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


def segment_into_sentences(chunks, time_threshold=1.2):
    """
    Segment word chunks into sentences based on time gaps.
    
    Args:
        chunks: List of dicts with 'text' and 'timestamp' keys
        time_threshold: Time gap (seconds) that indicates sentence boundary
    
    Returns:
        List of sentence strings with their start/end timestamps
    """
    if not chunks:
        return []
    
    sentences = []
    current_sentence = []
    
    for i, chunk in enumerate(chunks):
        current_sentence.append(chunk)
        
        # Check if this is the last chunk or if there's a gap to the next chunk
        if i == len(chunks) - 1:
            # Last chunk - finish the sentence
            sentences.append(current_sentence)
        else:
            # Check time gap to next chunk
            current_end = chunk["timestamp"][1]
            next_start = chunks[i + 1]["timestamp"][0]
            gap = next_start - current_end
            
            if gap > time_threshold:
                # Gap is large enough - finish this sentence
                sentences.append(current_sentence)
                current_sentence = []
    
    # Convert to text with timestamps
    result = []
    for sentence_chunks in sentences:
        text = " ".join([c["text"] for c in sentence_chunks])
        start_time = sentence_chunks[0]["timestamp"][0]
        end_time = sentence_chunks[-1]["timestamp"][1]
        result.append({
            "text": text,
            "start": start_time,
            "end": end_time
        })
    
    return result


def combine_transcripts(a_json_path, b_json_path, max_time=60.0, time_threshold=1.2):
    """
    Combine A.json (Examiner) and B.json (Evaluatee) into conversation format.
    
    Args:
        a_json_path: Path to A.json file
        b_json_path: Path to B.json file
        max_time: Maximum timestamp to include (seconds)
        time_threshold: Time gap for sentence segmentation (seconds)
    
    Returns:
        List of conversation turns with speaker and text
    """
    # Load JSON files
    with open(a_json_path, 'r') as f:
        a_data = json.load(f)
    with open(b_json_path, 'r') as f:
        b_data = json.load(f)
    
    # Filter chunks by max_time
    a_chunks = [c for c in a_data["chunks"] if c["timestamp"][1] <= max_time]
    b_chunks = [c for c in b_data["chunks"] if c["timestamp"][1] <= max_time]
    
    # Segment into sentences
    a_sentences = segment_into_sentences(a_chunks, time_threshold)
    b_sentences = segment_into_sentences(b_chunks, time_threshold)
    
    # Tag with speaker
    a_tagged = [{"speaker": "Examiner", "text": s["text"], "start": s["start"]} 
                for s in a_sentences]
    b_tagged = [{"speaker": "Evaluatee", "text": s["text"], "start": s["start"]} 
                for s in b_sentences]
    
    # Combine and sort by start time
    all_sentences = a_tagged + b_tagged
    all_sentences.sort(key=lambda x: x["start"])
    
    # Remove start time from final output (only needed for sorting)
    result = [{"speaker": s["speaker"], "text": s["text"]} for s in all_sentences]
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch transcribe audio files in hierarchical directory structure"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing task folders (daily, correction, entitytracking, safety)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="transcripts.json",
        help="Output JSON file path (default: transcripts.json)"
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=60.0,
        help="Maximum timestamp to include in transcripts (default: 60.0 seconds)"
    )
    parser.add_argument(
        "--time_threshold",
        type=float,
        default=1.2,
        help="Time gap threshold for sentence segmentation (default: 1.2 seconds)"
    )
    parser.add_argument(
        "--skip_asr",
        action="store_true",
        help="Skip ASR inference and only combine existing JSON files"
    )
    
    args = parser.parse_args()
    
    # Find all audio pairs
    print("Scanning directory structure...")
    audio_pairs = find_audio_pairs(args.root_dir)
    print(f"Found {len(audio_pairs)} audio pairs to process")
    
    if not audio_pairs:
        print("No audio pairs found. Please check the directory structure.")
        return
    
    # Load ASR model if needed
    asr_model = None
    if not args.skip_asr:
        print("Loading ASR model...")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2"
        ).cuda()
        print("ASR model loaded")
    
    # Process each audio pair
    all_transcripts = {}
    
    for pair in tqdm(audio_pairs, desc="Processing audio pairs"):
        key = pair["key"]
        sample_path = pair["sample_path"]
        a_wav = pair["a_wav"]
        b_wav = pair["b_wav"]
        
        # Define JSON output paths
        a_json = os.path.join(sample_path, "A.json")
        b_json = os.path.join(sample_path, "B.json")
        
        # Run ASR inference if needed
        if not args.skip_asr:
            # Transcribe A.wav
            if not os.path.exists(a_json):
                a_result = transcribe_audio_file(a_wav, asr_model)
                with open(a_json, 'w') as f:
                    json.dump(a_result, f, indent=4)
            
            # Transcribe B.wav
            if not os.path.exists(b_json):
                b_result = transcribe_audio_file(b_wav, asr_model)
                with open(b_json, 'w') as f:
                    json.dump(b_result, f, indent=4)
        
        # Combine back into a multi-turn conversation list
        combined_transcript = combine_transcripts(a_json, b_json, max_time=args.max_time, time_threshold=args.time_threshold)
        
        # Write individual combined transcript
        transcript_out = os.path.join(sample_path, "transcripts.json")
        with open(transcript_out, 'w') as f:
            json.dump(combined_transcript, f, indent=2)
            
        all_transcripts[key] = combined_transcript
    
    # Write final output
    print(f"Writing transcripts to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(all_transcripts, f, indent=2)
    
    print(f"Done! Processed {len(all_transcripts)} conversations.")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()

