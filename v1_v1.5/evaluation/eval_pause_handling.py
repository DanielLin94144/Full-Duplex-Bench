import os
import json
import re
import argparse
from tqdm import tqdm

turn_duration_threshold = 1
turn_num_words_threshold = 3


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s\[\]]", "", text)


def eval_pause_handling(data_dir):
    audio_output_files = []

    for folder in os.listdir(data_dir):
        if folder.endswith(".DS_Store"):
            continue
        if folder.endswith(".md"):
            continue

        for file_o in os.listdir(os.path.join(data_dir, folder)):
            if file_o.endswith("output.json"):
                audio_output_files.append(os.path.join(data_dir, folder, file_o))

    take_turn_list = []

    for audio_output_file in tqdm(audio_output_files, desc="evaluate"):

        TOR = None

        # if audio_output_file is not found, raise an error
        if not os.path.exists(audio_output_file):
            raise FileNotFoundError(f"Required file '{audio_output_file}' not found.")

        with open(audio_output_file, "r") as f:
            output_data = json.load(f)
            segments_cw = output_data["chunks"]

        # if no transcription from CrisperWhisper， means model does not take turn
        if len(segments_cw) == 0:
            TOR = 0
        else:
            if segments_cw[-1]["timestamp"][-1] == None:
                duration = (
                    segments_cw[-1]["timestamp"][0] - segments_cw[0]["timestamp"][0]
                )
            else:
                duration = (
                    segments_cw[-1]["timestamp"][-1] - segments_cw[0]["timestamp"][0]
                )
            if duration < turn_duration_threshold:
                if len(segments_cw) <= turn_num_words_threshold:
                    TOR = 0
                else:
                    TOR = 1
            else:
                TOR = 1

        take_turn_list.append(TOR)

    average_take_turn = sum(take_turn_list) / len(take_turn_list)

    print("---------------------------------------------------")
    print("[Result]")
    print("Average take turn: ", average_take_turn)
    print("---------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser")
    parser.add_argument("--root_dir", type=str)
    args = parser.parse_args()

    eval_pause_handling(args.root_dir)
