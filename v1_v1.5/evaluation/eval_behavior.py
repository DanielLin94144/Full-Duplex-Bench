import json
import os

from openai import OpenAI
import json
import re
from typing import Dict, Any, Union, List


def json_dict_to_compact_text(json_list):
    """
    Convert list of dicts to a compact plain text string with minimal spaces.

    Args:
        json_list (list): A list of dictionaries.

    Returns:
        str: Compact string representation of the list of dicts.
    """
    return json.dumps(json_list, separators=(",", ":"), ensure_ascii=False)


def extract_json(text: str, key: str = "behaviour"):
    decoder = json.JSONDecoder()
    pos = text.find("{")
    while pos != -1:
        try:
            obj, end = decoder.raw_decode(text, pos)
            if key in obj:
                return obj

            pos = text.find("{", end)
        except json.JSONDecodeError:

            pos = text.find("{", pos + 1)
    raise ValueError(f"No JSON object with key '{key}' found.")


def parse_eval(data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:

    if isinstance(data, str):
        data = extract_json(data)
    elif not isinstance(data, dict):
        raise ValueError("Input must be a JSON string or dict.")

    return data


def eval_behavior(system_msg, user_msg, client, overlap=1):
    finished = False
    seed = 1
    while not finished:
        try:
            MODEL_NAME = "gpt-4o-2024-08-06"
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                seed=seed,
            )

            prediction = response.choices[0].message.content
            print("Prediction:", prediction)
            result = parse_eval(prediction)
            finished = True
            return result

        except Exception as e:
            print(f"Error: {e}. Retrying...")
            seed += 1
            # wait for a while before retrying
            import time

            time.sleep(5)
            continue


from collections import Counter


def stats_by_axis(records):
    axes = {"C": Counter()}

    for rec in records:
        for tag in rec.get("behaviour", []):
            prefix = tag[0]
            axes[prefix][tag] += 1

    totals = {ax: sum(cnt.values()) for ax, cnt in axes.items()}
    ratios = {
        ax: {tag: count / totals[ax] for tag, count in cnt.items()}
        for ax, cnt in axes.items()
    }
    return axes, totals, ratios


def read_instruction(task):
    file_path = f"./instruction/{task}.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        instruction_text = f.read()
    return instruction_text


def eval_behavior_all(data_dir, client, task, aggregate=False):
    folders = []
    output_list = []

    # read instruction
    instruction = read_instruction(task)

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path) and not folder.startswith("."):
            folders.append(folder_path)

    for dir_path in folders:

        # first check if tax_cot.json exists
        tax_cot_path = os.path.join(dir_path, "content_tag.json")
        if os.path.exists(tax_cot_path):
            print(f"Found {tax_cot_path}")

        print(f"Evaluating {dir_path}...")
        input_clean_name = "clean_input.json"
        input_noisy_name = "input.json"
        output_clean_name = "clean_output.json"
        output_noisy_name = "output.json"

        input_clean_path = os.path.join(dir_path, input_clean_name)
        input_noisy_path = os.path.join(dir_path, input_noisy_name)
        output_clean_path = os.path.join(dir_path, output_clean_name)
        output_noisy_path = os.path.join(dir_path, output_noisy_name)

        # check if all files exist
        if not (
            os.path.exists(input_clean_path)
            and os.path.exists(input_noisy_path)
            and os.path.exists(output_clean_path)
            and os.path.exists(output_noisy_path)
        ):
            print(f"Skipping {dir_path} due to missing files.")
            continue

        with open(input_clean_path, "r") as f:
            input_clean = json.load(f)
        with open(input_noisy_path, "r") as f:
            input_noisy = json.load(f)
        with open(output_clean_path, "r") as f:
            output_clean = json.load(f)
        with open(output_noisy_path, "r") as f:
            output_noisy = json.load(f)

        overlap = check_overlap(input_noisy["chunks"], output_noisy["chunks"])

        input_clean_text = json_dict_to_compact_text(input_clean)
        input_noisy_text = json_dict_to_compact_text(input_noisy)
        output_clean_text = json_dict_to_compact_text(output_clean)
        output_noisy_text = json_dict_to_compact_text(output_noisy)

        final_input = f"""
        {{
            "input_clean": {input_clean_text},
            "input_noisy": {input_noisy_text},
            "output_clean": {output_clean_text},
            "output_noisy": {output_noisy_text}
        }}
        """

        result = eval_behavior(
            system_msg=instruction, user_msg=final_input, client=client, overlap=overlap
        )

        print(result)
        with open(os.path.join(dir_path, "content_tag.json"), "w") as f:
            json.dump(result, f)

        output_list.append(result)

    counts, totals, ratios = stats_by_axis(output_list)

    fmt_ratios = {
        ax: {k: round(v, 2) for k, v in sorted(ratios[ax].items())} for ax in ["C"]
    }

    return fmt_ratios


def check_overlap(list_a: List[Dict], list_b: List[Dict]) -> int:
    for seg_a in list_a:
        start_a, end_a = seg_a["timestamp"]
        for seg_b in list_b:
            start_b, end_b = seg_b["timestamp"]
            if max(start_a, start_b) < min(end_a, end_b):
                return 1
    return 0
