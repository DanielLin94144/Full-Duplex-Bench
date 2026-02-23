#!/usr/bin/env python3
"""
Single-item evaluator using Gemini 2.5 Pro.

Loads:
- System prompt and evaluation rubrics from eval_prompts_v2.json
- Task-specific prompt for subsets: Correction, EntityTracking, Safety
- Dataset staged_reveal (T1–T4) for a given task id from staged prompts files
- Transcripts for Channel A (Examiner) and Channel B (Examinee) from A.json / B.json

Builds a consolidated prompt and calls the Gemini API, with retries/backoff.

Usage example:
  python eval_single_item.py \
    --subset Safety \
    --task-id Safety.privacy.026 \
    --transcript-a /absolute/path/to/.../A.json \
    --transcript-b /absolute/path/to/.../B.json \
    --api-key $GEMINI_API_KEY

Notes:
- API key may be supplied via --api-key or the GEMINI_API_KEY environment variable
- Model defaults to gemini-2.5-pro
- Output is saved as JSON or text depending on the response
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class Transcript:
    text: str
    chunks: List[Dict[str, Any]]


# ---------------------------
# File loading helpers
# ---------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_eval_prompts(eval_prompts_path: Path) -> Dict[str, Any]:
    data = load_json(eval_prompts_path)
    required_top_level_keys = [
        "system_prompt",
        "turn_taking_fluency_prompt",
        "multi_turn_instruction_following_prompt",
        "task_specific_prompts",
        "output_format_prompt",
    ]
    for key in required_top_level_keys:
        if key not in data:
            raise KeyError(f"Missing '{key}' in {eval_prompts_path}")
    return data


def normalize_subset_name(subset: str) -> str:
    # Accept variants and normalize to canonical ids used in eval prompts
    s = subset.strip().lower().replace(" ", "")
    if s in {"daily"}:
        return "daily"
    if s in {"correction", "corrections"}:
        return "correction"
    if s in {"entitytracking", "entity_tracking"}:
        return "entity_tracking"
    if s in {"safety"}:
        return "safety"
    raise ValueError(
        "Unknown subset. Use one of: Daily, Correction, EntityTracking, Safety"
    )


def get_task_specific_prompt(eval_prompts: Dict[str, Any], subset_norm: str) -> Optional[str]:
    if subset_norm == "daily":
        return None
    target_id = subset_norm  # matches ids in eval_prompts_v2.json
    items: List[Dict[str, Any]] = eval_prompts.get("task_specific_prompts", [])
    for item in items:
        if item.get("id") == target_id:
            return item.get("prompt", "").strip() or None
    raise KeyError(
        f"task_specific_prompts does not contain id='{target_id}'."
    )


def find_task_staged_reveal_from_file(staged_path: Path, task_id: str) -> Optional[Dict[str, str]]:
    """Return staged_reveal dict for the given task_id if found in this file.

    Handles structures like:
    { "splits": { "Safety": { "tasks": [ {"id": ..., "staged_reveal": {...}} ] } } }
    or
    { "tasks": [ ... ] }
    """
    try:
        data = load_json(staged_path)
    except json.JSONDecodeError:
        return None

    def iter_tasks(obj: Any):
        if isinstance(obj, dict):
            if "tasks" in obj and isinstance(obj["tasks"], list):
                for t in obj["tasks"]:
                    yield t
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    yield from iter_tasks(v)
        elif isinstance(obj, list):
            for it in obj:
                yield from iter_tasks(it)

    for task in iter_tasks(data):
        if isinstance(task, dict) and task.get("id") == task_id:
            sr = task.get("staged_reveal")
            if isinstance(sr, dict):
                # Ensure keys T1..T4 present (some files may have different capitalization)
                normalized = {}
                for k in ["T1", "T2", "T3", "T4"]:
                    if k in sr and isinstance(sr[k], str):
                        normalized[k] = sr[k]
                return normalized or None
    return None


def find_task_staged_reveal(task_id: str, staged_files: List[Path]) -> Dict[str, str]:
    for p in staged_files:
        result = find_task_staged_reveal_from_file(p, task_id)
        if result is not None:
            return result
    raise FileNotFoundError(
        f"Task id '{task_id}' not found in staged files: {[str(p) for p in staged_files]}"
    )


def load_transcript(path: Path) -> Transcript:
    data = load_json(path)
    text = data.get("text") or ""
    chunks = data.get("chunks") or []
    if not isinstance(chunks, list):
        chunks = []
    return Transcript(text=text, chunks=chunks)


def format_chunks_as_lines(chunks: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for ch in chunks:
        ts = ch.get("timestamp")
        content = ch.get("text", "")
        if (
            isinstance(ts, list)
            and len(ts) == 2
            and all(isinstance(x, (int, float)) for x in ts)
        ):
            start, end = ts
            lines.append(f"[{start:.2f}, {end:.2f}]: {content}")
        else:
            # Fallback: no timestamps
            if content:
                lines.append(content)
    return "\n".join(lines)


# ---------------------------
# Prompt construction
# ---------------------------

def build_full_prompt(
    eval_prompts: Dict[str, Any],
    subset_norm: str,
    task_specific_prompt: Optional[str],
    staged_reveal: Dict[str, str],
    transcript_a: Transcript,
    transcript_b: Transcript,
) -> str:
    system_prompt = eval_prompts["system_prompt"].strip()
    tt_prompt = eval_prompts["turn_taking_fluency_prompt"].strip()
    mt_prompt = eval_prompts["multi_turn_instruction_following_prompt"].strip()
    output_format_prompt = eval_prompts["output_format_prompt"].strip()

    dataset_prompt_lines: List[str] = [
        "Dataset staged goals (T1–T4):",
        f"T1: {staged_reveal.get('T1', '')}",
        f"T2: {staged_reveal.get('T2', '')}",
        f"T3: {staged_reveal.get('T3', '')}",
        f"T4: {staged_reveal.get('T4', '')}",
    ]
    dataset_prompt = "\n".join(dataset_prompt_lines)

    transcript_section = []
    transcript_section.append("Transcript summary:")
    transcript_section.append("Channel A (Examiner) summary:")
    transcript_section.append(transcript_a.text)
    transcript_section.append("")
    transcript_section.append("Channel B (Examinee) summary:")
    transcript_section.append(transcript_b.text)
    transcript_section.append("")
    transcript_section.append("Channel B (Examinee) segments with timestamps:")
    transcript_section.append(format_chunks_as_lines(transcript_b.chunks))
    transcript_text = "\n".join(transcript_section)

    sections: List[str] = []
    sections.append("SYSTEM INSTRUCTION:\n" + system_prompt)
    sections.append("TURN-TAKING FLUENCY RUBRIC:\n" + tt_prompt)
    sections.append("MULTI-TURN INSTRUCTION-FOLLOWING RUBRIC:\n" + mt_prompt)
    if task_specific_prompt:
        sections.append("TASK-SPECIFIC RUBRIC:\n" + task_specific_prompt)
    sections.append("OUTPUT FORMAT:\n" + output_format_prompt)
    sections.append(dataset_prompt)
    sections.append(transcript_text)

    # Guidance for brevity and consistency
    sections.append(
        "Constraints:\n"
        "- Keep each evaluation output about ~100 words.\n"
        "- Follow the JSON shape exactly as in OUTPUT FORMAT.\n"
        "- Use time ranges from the provided Channel B timestamps.\n"
    )

    return "\n\n".join(s for s in sections if s)


# ---------------------------
# Gemini API call with retries
# ---------------------------

def generate_with_gemini(
    api_key: str,
    model: str,
    api_version: str,
    prompt_text: str,
    max_output_tokens: int = 20000,
    temperature: float = 0.2,
    max_retries: int = 5,
) -> Dict[str, Any]:
    base = "https://generativelanguage.googleapis.com"
    # support v1beta for preview models
    # if api_version not in ("v1", "v1beta"):
    #     api_version = "v1"
    url = f"{base}/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    payload: Dict[str, Any] = {
        "contents": [
            {
                "parts": [{"text": prompt_text}],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "temperature": temperature,
        },
    }

    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
        except requests.RequestException as e:
            if attempt >= max_retries:
                raise RuntimeError(f"HTTP request failed after {attempt} attempts: {e}")
            sleep_s = min(60, 2 ** attempt + random.random())
            time.sleep(sleep_s)
            continue

        if resp.status_code == 200:
            try:
                return resp.json()
            except json.JSONDecodeError as e:
                if attempt >= max_retries:
                    raise RuntimeError(f"Invalid JSON response after {attempt} attempts: {e}")
                sleep_s = min(60, 2 ** attempt + random.random())
                time.sleep(sleep_s)
                continue

        if resp.status_code in (429, 500, 502, 503, 504):
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Gemini API error {resp.status_code} after {attempt} attempts: {resp.text}"
                )
            retry_after = resp.headers.get("Retry-After")
            try:
                sleep_s = float(retry_after) if retry_after else None
            except ValueError:
                sleep_s = None
            if sleep_s is None:
                sleep_s = min(60, 2 ** attempt + random.random())
            time.sleep(sleep_s)
            continue

        # Non-retryable
        raise RuntimeError(
            f"Gemini API returned {resp.status_code}: {resp.text}"
        )


def extract_text_from_response(resp: Dict[str, Any]) -> str:
    try:
        cands = resp.get("candidates") or []
        if not cands:
            return ""
        content = cands[0].get("content") or {}
        parts = content.get("parts") or []
        if not parts:
            return ""
        text = parts[0].get("text") or ""
        return text
    except Exception:
        return ""


# ---------------------------
# JSON sanitization helpers
# ---------------------------

def strip_markdown_fences(text: str) -> str:
    """Remove surrounding markdown code fences like ```json ... ``` if present.

    Keeps inner content intact. Returns original text if no fences detected.
    """
    t = text.strip()
    if t.startswith("```"):
        # Drop the first fence line
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1 :]
        # Drop trailing fence if present
        if t.endswith("```"):
            t = t[: -3]
    return t.strip()


def extract_first_balanced_json(text: str) -> Optional[str]:
    """Extract the first top-level balanced JSON object from arbitrary text.

    Scans for the first '{' and returns the shortest string that balances braces.
    Returns None if no balanced object is found.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    return candidate
    return None


def try_parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse: strip fences, extract balanced JSON, fall back to raw loads.

    Returns a dict if successful, otherwise None.
    """
    if not text:
        return None
    stripped = strip_markdown_fences(text)
    candidate = extract_first_balanced_json(stripped)
    if candidate is None:
        candidate = stripped
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


# ---------------------------
# Main
# ---------------------------

def main(argv: Optional[List[str]] = None) -> int:
    # Default model from environment variable or fallback
    default_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-09-2025")
    
    parser = argparse.ArgumentParser(description="Evaluate a single item with Gemini LLM-as-a-judge")
    parser.add_argument("--subset", required=True, help="One of: Daily, Correction, EntityTracking, Safety")
    parser.add_argument("--task-id", required=True, help="Dataset task id, e.g., Safety.privacy.026")
    parser.add_argument("--transcript-a", required=True, help="Absolute path to A.json (Examiner)")
    parser.add_argument("--transcript-b", required=True, help="Absolute path to B.json (Examinee)")
    parser.add_argument("--eval-prompts", default=None, help="Path to eval_prompts_v2.json (defaults to alongside this script)")
    parser.add_argument(
        "--staged-files",
        nargs="*",
        default=None,
        help="Paths to staged prompt JSON files to search for the task (defaults to known prompts_staged_*.json in this directory)",
    )
    parser.add_argument("--model", default=default_model, help=f"Model name (default: {default_model}, can be set via GEMINI_MODEL env var)")
    parser.add_argument("--api-version", default="v1", choices=["v1","v1beta"], help="Google Generative Language API version")
    parser.add_argument("--api-key", default=None, help="Gemini API key. If omitted, reads GEMINI_API_KEY env var")
    parser.add_argument("--out", default=None, help="Output file to write (JSON or .txt). If omitted, prints to stdout")
    args = parser.parse_args(argv)

    subset_norm = normalize_subset_name(args.subset)

    script_dir = Path(__file__).resolve().parent
    eval_prompts_path = (
        Path(args.eval_prompts).resolve()
        if args.eval_prompts
        else script_dir / "eval_prompts.json"
    )
    eval_prompts = load_eval_prompts(eval_prompts_path)

    task_specific_prompt = get_task_specific_prompt(eval_prompts, subset_norm)

    # Default staged files list if not provided
    if args.staged_files:
        staged_files = [Path(p).resolve() for p in args.staged_files]
    else:
        candidates = [
            script_dir / "prompts_staged_200.json",
            script_dir / "prompts_staged_150.json",
            script_dir / "prompts_staged_safety_50.json",
        ]
        staged_files = [p for p in candidates if p.exists()]
        if not staged_files:
            raise FileNotFoundError(
                "No staged files found. Provide --staged-files explicitly."
            )

    staged_reveal = find_task_staged_reveal(args.task_id, staged_files)

    transcript_a_path = Path(args.transcript_a).resolve()
    transcript_b_path = Path(args.transcript_b).resolve()
    transcript_a = load_transcript(transcript_a_path)
    transcript_b = load_transcript(transcript_b_path)

    full_prompt = build_full_prompt(
        eval_prompts=eval_prompts,
        subset_norm=subset_norm,
        task_specific_prompt=task_specific_prompt,
        staged_reveal=staged_reveal,
        transcript_a=transcript_a,
        transcript_b=transcript_b,
    )

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("API key not provided. Use --api-key or set GEMINI_API_KEY env var.")

    response_json = generate_with_gemini(
        api_key=api_key,
        model=args.model,
        api_version=args.api_version,
        prompt_text=full_prompt,
    )

    output_text = extract_text_from_response(response_json)
    parsed_json: Optional[Dict[str, Any]] = try_parse_json_from_text(output_text)

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # If extension is .json and model returned valid JSON string, write as-is
        if out_path.suffix.lower() == ".json":
            # Prefer sanitized parsed JSON; else fall back to raw text
            if parsed_json is not None:
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(parsed_json, f, ensure_ascii=False, indent=2)
            else:
                with out_path.open("w", encoding="utf-8") as f:
                    f.write(output_text)
        else:
            with out_path.open("w", encoding="utf-8") as f:
                f.write(output_text)

        print(f"Wrote evaluation to: {out_path}")
    else:
        print(output_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())


