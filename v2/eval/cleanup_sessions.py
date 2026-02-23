#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

SESSION_PREFIX = 'session_'
TARGET_BASENAMES = ['combined.wav', 'A.wav', 'B.wav']


def find_session_dirs(base: Path):
    for path, dirs, files in os.walk(base):
        dirname = os.path.basename(path)
        if dirname.startswith(SESSION_PREFIX):
            yield Path(path)


def determine_target_directory(session_dir: Path) -> Path:
    # Move combined.wav from .../<Group>/<Variant>/<Item>/session_xxx/* to .../<Group>/<Item>/
    # That is: drop the 'variant' layer (e.g. correction/, entity_tracking/, ordering/, planning/, ...)
    # Assume structure: base/<Group>/<Variant>/<Item>/session_*/
    # Group is the top-level under ROOT
    parts = session_dir.parts
    # We need to map: [..., Group, Variant, Item, session_xxx] -> [..., Group, Item]
    # Ensure at least 4 trailing parts
    if len(parts) < 4:
        return session_dir.parent  # fallback
    group = parts[-4]
    variant = parts[-3]
    item = parts[-2]
    return session_dir.parents[2].parent / group / item


def process(base: Path, apply: bool = False, verbose: bool = True):
    moved = 0
    deleted = 0
    skipped = 0
    for session_dir in find_session_dirs(base):
        parent = session_dir.parent
        target_dir = determine_target_directory(session_dir)
        if not target_dir.exists():
            if verbose:
                print(f"[mkdir] {target_dir}")
            if apply:
                target_dir.mkdir(parents=True, exist_ok=True)
        
        # Move desired wavs (combined, A, B) to the target directory
        for base_name in TARGET_BASENAMES:
            src = session_dir / base_name
            if src.exists():
                dest = target_dir / base_name
                if verbose:
                    print(f"[move] {src} -> {dest}")
                if apply:
                    if dest.exists():
                        dest.unlink()
                    shutil.move(str(src), str(dest))
                    moved += 1
            else:
                # Track missing combined.wav only
                if base_name == 'combined.wav':
                    if verbose:
                        print(f"[skip] missing {src}")
                    skipped += 1
        # Delete the rest of files under session_dir
        for child in session_dir.iterdir():
            if child.name in TARGET_BASENAMES:
                continue
            if verbose:
                print(f"[delete] {child}")
            if apply:
                if child.is_file() or child.is_symlink():
                    child.unlink(missing_ok=True)
                else:
                    shutil.rmtree(child, ignore_errors=True)
                deleted += 1
        # Attempt to remove the empty session dir
        if verbose:
            print(f"[rmdir] {session_dir}")
        if apply:
            try:
                session_dir.rmdir()
            except OSError:
                pass
    if verbose:
        print(f"Summary: moved={moved}, deleted={deleted}, skipped_missing_combined={skipped}")


def main():
    parser = argparse.ArgumentParser(description='Move combined.wav out of session_* and delete the rest.')
    parser.add_argument('root', type=Path, help='Root experiments folder (e.g., experiments_gpt_gpt_150_long)')
    parser.add_argument('--apply', action='store_true', help='Actually perform changes')
    args = parser.parse_args()
    process(args.root, apply=args.apply, verbose=True)


if __name__ == '__main__':
    main()
