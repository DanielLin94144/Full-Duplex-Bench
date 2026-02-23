#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

# Map split/class/id in prompts file to on-disk item folder names
# We expect item ids like 'Daily.ordering.001' to map to folders like:
# - Daily/Daily.ordering.001
# - Daily/<variant>/Daily.ordering.001 (we'll support both)
# For Safety: 'Safety.health_physical.001' → Safety/Safety.health_physical.001 or Safety/health_physical/Safety.health_physical.001

INTEREST_KEYS = {
    'system_prompt': 'examiner_system_prompt',
    'task_prompt': 'examiner_task_prompt',
    'staged_reveal': 'staged_reveal',
}

VARIANT_DIRS_BY_SPLIT = {
    'Daily': ['ordering', 'planning', 'reservations', 'scheduling', 'troubleshoot'],
    'EntityTracking': ['entity_tracking'],
    'Correction': ['correction'],
    'Safety': ['adult_minors','agent_safety','copyright','cyber_illicit','financial_legal','health_mental','health_physical','misinfo','privacy','toxicity_bias'],
}


def load_prompts(prompts_file: Path):
    data = json.loads(prompts_file.read_text())
    items = []
    # prompts file appears to have splits.Daily.items array; we'll scan for any dicts with 'id'
    def walk(obj):
        if isinstance(obj, dict):
            if 'id' in obj and 'examiner_system_prompt' in obj and 'examiner_task_prompt' in obj and 'staged_reveal' in obj:
                items.append(obj)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
    walk(data)
    return items


def candidate_item_paths(data_root: Path, item_id: str):
    # Item id format: Split.Class.Number (e.g., Daily.ordering.001) or Safety.category.NNN
    parts = item_id.split('.')
    if len(parts) < 2:
        return []
    split = parts[0]
    item_dir_name = item_id
    # Support two-part ids like Correction.001 and EntityTracking.001 which map to split/item_dir directly
    candidates = []
    # 1) Split/item_dir
    candidates.append(data_root / split / item_dir_name)
    # Also support Split root for EntityTracking flat layout (already covered)
    # 2) Split/variant/item_dir for known variants
    for variant in VARIANT_DIRS_BY_SPLIT.get(split, []):
        candidates.append(data_root / split / variant / item_dir_name)
    return candidates


def write_prompt_jsons(data_root: Path, prompts_file: Path, apply: bool, verbose: bool):
    items = load_prompts(prompts_file)
    written = 0
    missing = 0
    for item in items:
        item_id = item.get('id')
        system_prompt = item.get('examiner_system_prompt')
        task_prompt = item.get('examiner_task_prompt')
        staged_reveal = item.get('staged_reveal')
        out_obj = {
            'system_prompt': system_prompt,
            'task_prompt': task_prompt,
            'staged_reveal': staged_reveal,
        }
        found_path = None
        for candidate in candidate_item_paths(data_root, item_id):
            if candidate.exists():
                found_path = candidate
                break
        if not found_path:
            if verbose:
                print(f"[miss] {item_id} -> no folder found under {data_root}")
            missing += 1
            continue
        out_path = found_path / 'prompt.json'
        if verbose:
            print(f"[write] {out_path}")
        if apply:
            out_path.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False))
            written += 1
    if verbose:
        print(f"Summary: written={written}, missing={missing}, total_items={len(items)}")


def main():
    parser = argparse.ArgumentParser(description='Write prompt.json into each item folder using prompts_staged_150.json')
    parser.add_argument('data_root', type=Path, help='Root folder of dataset (e.g., experiments_gpt_gpt_150_long or Examiner_* folder)')
    parser.add_argument('--prompts', type=Path, default=Path('prompts_staged_150.json'), help='Path to prompts file')
    parser.add_argument('--apply', action='store_true', help='Actually write prompt.json files')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    write_prompt_jsons(args.data_root, args.prompts, apply=args.apply, verbose=args.verbose)


if __name__ == '__main__':
    main()
