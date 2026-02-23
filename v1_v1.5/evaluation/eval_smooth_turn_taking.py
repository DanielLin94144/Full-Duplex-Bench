import os
import json
import re
import argparse
from tqdm import tqdm

turn_duration_threshold = 1
turn_num_words_threshold = 3


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s\[\]]", "", text)


def round_to_quarter(number):
    return round(number * 4) / 4


def eval_smooth_turn_taking(data_dir):
    audio_input_files = []
    audio_output_files = []

    for folder in os.listdir(data_dir):
        if folder.endswith(".DS_Store"):
            continue
        if folder.endswith(".md"):
            continue

        for file_o in os.listdir(os.path.join(data_dir, folder)):
            if file_o.endswith("output.json"):
                audio_output_files.append(os.path.join(data_dir, folder, file_o))
                for file_i in os.listdir(os.path.join(data_dir, folder)):
                    if file_i.endswith("turn_taking.json"):
                        audio_input_files.append(os.path.join(data_dir, folder, file_i))

    take_turn_list = []
    latency_list = []

    for audio_input_file, audio_output_file in tqdm(
        zip(audio_input_files, audio_output_files), desc="evaluate"
    ):
        # Get input turn end time
        if not os.path.exists(audio_input_file):
            raise FileNotFoundError(f"Required file '{audio_input_file}' not found.")

        with open(audio_input_file, "r") as f:
            input_turn = json.load(f)
        input_end_time = input_turn[0]["timestamp"][0]

        TOR = None
        latency = None
        if not os.path.exists(audio_output_file):
            raise FileNotFoundError(f"Required file '{audio_output_file}' not found.")

        with open(audio_output_file, "r") as f:
            output_data = json.load(f)
            segments_cw = output_data["chunks"]

        # if no transcription from CrisperWhisper， means model does not take turn
        if len(segments_cw) == 0:
            TOR = 0
        else:
            output_start_time = segments_cw[0]["timestamp"][0]
            duration = segments_cw[-1]["timestamp"][-1] - segments_cw[0]["timestamp"][0]
            if duration < turn_duration_threshold:
                if len(segments_cw) <= turn_num_words_threshold:
                    TOR = 0
                else:
                    TOR = 1
                    latency = output_start_time - input_end_time
            else:
                TOR = 1
                latency = output_start_time - input_end_time

        take_turn_list.append(TOR)
        if TOR == 1:
            if latency < 0:
                latency_list.append(0)
            elif latency >= 0:
                latency_list.append(latency)

        print(audio_output_file)
        print(f"the TOR is {TOR}")
        print(f"the latency is {latency}")

        # check there is no negative latency
    for i in latency_list:
        if i < 0:
            print(i)

    average_take_turn = sum(take_turn_list) / len(take_turn_list)
    average_latency = sum(latency_list) / len(latency_list)

    print("---------------------------------------------------")
    print("[Result]")
    print("Average take turn: ", average_take_turn)
    print("Average latency: ", average_latency)
    print("---------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser")
    parser.add_argument("--root_dir", type=str)
    args = parser.parse_args()

    eval_smooth_turn_taking(args.root_dir)
