#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
from typing import List, Tuple, Optional, Dict, Any


# =========================
# Utilities: Time/Number to Seconds
# =========================
def parse_time_token(tok):
    if isinstance(tok, (int, float)):
        return float(tok)
    if isinstance(tok, str):
        s = tok.strip().strip('"').strip("'")
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 2:
                m, sec = parts
                return float(m) * 60.0 + float(sec)
            total = 0.0
            mul = 1.0
            for p in reversed(parts):
                total += float(p) * mul
                mul *= 60.0
            return total
        return float(s)
    raise ValueError(f"Unrecognized time token: {tok}")


def coerce_int(x):
    if x is None:
        return None
    if isinstance(x, str) and x.strip().lower() == "null":
        return None
    try:
        return int(float(x))
    except Exception:
        return None


# =========================
# Normalize Intervals: {a,b} -> [a,b]; {"start_time":x,"end_time":y} -> [x,y]
# =========================
_num = r"[+-]?(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
# {a, b}
_re_brace_pair = re.compile(
    r"""
    \{
        \s*([^{}]+?)\s*
    \}
    """,
    re.VERBOSE | re.DOTALL,
)


def _looks_like_time_or_number(s: str) -> bool:
    s = s.strip().strip('"').strip("'")
    if ":" in s:
        try:
            parts = s.split(":")
            float(parts[-1])
            for p in parts[:-1]:
                float(p)
            return True
        except Exception:
            return False
    try:
        float(s)
        return True
    except Exception:
        return False


# {"start_time": x, "end_time": y} or swapped; value can be a number or string
_re_se = re.compile(
    rf"""\{{\s*['"](?:start_time|start|startTime)['"]\s*:\s*({_num}|"[^"]+")\s*,\s*
            ['"](?:end_time|end|endTime)['"]\s*:\s*({_num}|"[^"]+")\s*\}}""",
    re.VERBOSE,
)
_re_es = re.compile(
    rf"""\{{\s*['"](?:end_time|end|endTime)['"]\s*:\s*({_num}|"[^"]+")\s*,\s*
            ['"](?:start_time|start|startTime)['"]\s*:\s*({_num}|"[^"]+")\s*\}}""",
    re.VERBOSE,
)


def normalize_interval_braces(text: str) -> str:
    def repl(m: re.Match) -> str:
        inner = m.group(1)
        parts = [p.strip() for p in inner.split(",")]
        if (
            len(parts) == 2
            and _looks_like_time_or_number(parts[0])
            and _looks_like_time_or_number(parts[1])
        ):
            return f"[{parts[0]}, {parts[1]}]"
        return m.group(0)

    return _re_brace_pair.sub(repl, text)


def normalize_start_end_objects(text: str) -> str:
    text = _re_se.sub(lambda m: f"[{m.group(1)}, {m.group(2)}]", text)
    text = _re_es.sub(lambda m: f"[{m.group(2)}, {m.group(1)}]", text)
    return text


def normalize_intervals_text(raw: str) -> str:
    s = normalize_start_end_objects(raw)
    s = normalize_interval_braces(s)
    return s


# =========================
# Low-level parsing helpers
# =========================
def find_matching_bracket(
    text: str, start: int, open_char: str = "[", close_char: str = "]"
) -> int:
    assert text[start] == open_char
    depth = 0
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                return i
        elif c == '"' or c == "'":
            q = c
            i += 1
            while i < n and text[i] != q:
                if text[i] == "\\":
                    i += 1
                i += 1
        i += 1
    return -1


def skip_ws_commas(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i] in " \t\r\n,":
        i += 1
    return i


def parse_score_token(text: str, i: int) -> Tuple[Optional[str], int]:
    n = len(text)
    i = skip_ws_commas(text, i)
    if i >= n:
        return None, i
    if text[i] == '"':
        j = i + 1
        while j < n and text[j] != '"':
            if text[j] == "\\":
                j += 1
            j += 1
        if j < n and text[j] == '"':
            token = text[i + 1 : j]
            return token, j + 1
        else:
            return None, i
    j = i
    while j < n and text[j] not in ",[]{}:\n\r\t ":
        j += 1
    tok = text[i:j].strip()
    return tok if tok else None, j


def coerce_score(token: Optional[str]) -> Optional[int]:
    if token is None:
        return None
    t = token.strip()
    if t == "":
        return None
    if t.lower() == "null":
        return None
    if t.startswith('"') and t.endswith('"') and len(t) >= 2:
        t = t[1:-1]
    try:
        return int(float(t))
    except Exception:
        return None


# =========================
# JSON Parsing: Direct load if block is valid JSON
# =========================
def flatten_turn_taking(data: Dict[str, Any]) -> Dict[str, Any]:
    key = "Turn-taking event and score"
    items = data.get(key, [])
    if not isinstance(items, list):
        data[key] = []
        return data
    new_items = []
    for item in items:
        if not isinstance(item, list) or len(item) < 3:
            continue
        s2_raw = item[-1]
        s1_raw = item[-2]
        intervals_part = item[:-2]

        # Collect all [start, end] pairs
        def _collect(o):
            if (
                isinstance(o, list)
                and len(o) == 2
                and all(isinstance(x, (int, float, str)) for x in o)
            ):
                try:
                    return [[parse_time_token(o[0]), parse_time_token(o[1])]]
                except Exception:
                    return []
            out = []
            if isinstance(o, list):
                for e in o:
                    got = _collect(e)
                    if got:
                        out.extend(got)
            return out

        intervals = _collect(intervals_part)
        if intervals:
            s1 = coerce_int(s1_raw)
            s2 = coerce_int(s2_raw)
            for iv in intervals:
                new_items.append([iv, s1, s2])
    data[key] = new_items
    return data


def try_parse_clean_json_block(block: str) -> Optional[Dict[str, Any]]:
    """Normalize block and try to parse as full JSON; if successful return flattened dict."""
    txt = normalize_intervals_text(block)
    try:
        d = json.loads(txt)
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    if ("Turn-taking event and score" not in d) and ("Task-specific score" not in d):
        return None
    d = flatten_turn_taking(d)
    return d


# =========================
# Parse Turn-taking Array (Fallback for broken JSON / colon-separated)
# =========================
KEY_PATTERNS = [
    '"Turn-taking event and score"',
    "'Turn-taking event and score'",
]


def extract_turn_taking_array_region(block: str) -> Tuple[int, int, int]:
    for key in KEY_PATTERNS:
        k = block.find(key)
        if k != -1:
            rest = block[k + len(key) :]
            lb_rel = rest.find("[")
            if lb_rel == -1:
                continue
            lb = k + len(key) + lb_rel
            rb = find_matching_bracket(block, lb, "[", "]")
            if rb != -1:
                return k, lb, rb
    return -1, -1, -1


def parse_intervals_expr(interval_expr: str) -> List[List[float]]:
    expr = normalize_intervals_text(interval_expr)
    try:
        obj = json.loads(expr)
    except Exception:
        expr2 = re.sub(r",\s*]", "]", expr)
        obj = json.loads(expr2)
    intervals: List[List[float]] = []

    def _collect(o: Any):
        if (
            isinstance(o, list)
            and len(o) == 2
            and all(isinstance(x, (int, float, str)) for x in o)
        ):
            try:
                intervals.append([parse_time_token(o[0]), parse_time_token(o[1])])
                return
            except Exception:
                pass
        if isinstance(o, list):
            for e in o:
                _collect(e)

    _collect(obj)
    return intervals


def parse_turn_taking_array(array_text: str) -> List[List[Any]]:
    i = 0
    n = len(array_text)
    out: List[List[Any]] = []
    while True:
        i = skip_ws_commas(array_text, i)
        if i >= n:
            break
        if array_text[i] != "[":
            nx = array_text.find("[", i)
            if nx == -1:
                break
            i = nx
        lb = i
        rb = find_matching_bracket(array_text, lb, "[", "]")
        if rb == -1:
            break
        interval_expr = array_text[lb : rb + 1]
        i = rb + 1
        j = skip_ws_commas(array_text, i)
        if j >= n or array_text[j] != ":":
            i = j
            continue
        j += 1
        s1_tok, j = parse_score_token(array_text, j)
        s2_tok, j = parse_score_token(array_text, j)
        s1 = coerce_score(s1_tok)
        s2 = coerce_score(s2_tok)
        try:
            intervals = parse_intervals_expr(interval_expr)
        except Exception:
            intervals = []
        for iv in intervals:
            out.append([iv, s1, s2])
        i = j
    return out


def parse_task_specific_score(block: str) -> Optional[int]:
    m = re.search(r'"Task-specific score"\s*:\s*("[^"]+"|[^\s,}\]]+)', block)
    if not m:
        m = re.search(r"'Task-specific score'\s*:\s*('[^']+'|[^\s,}\]]+)", block)
    if not m:
        return None
    tok = m.group(1)
    if tok and (tok[0] == tok[-1] == '"' or tok[0] == tok[-1] == "'"):
        tok = tok[1:-1]
    return coerce_int(tok)


# =========================
# Text Block Parsing
# =========================
_FENCED = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def process_one_text(full_text: str) -> Dict[str, Any]:
    blocks: List[str] = []
    for m in _FENCED.finditer(full_text):
        payload = m.group(1)
        if "Turn-taking event and score" in payload:
            blocks.append(payload)
    if not blocks:
        blocks = [full_text]

    merged_events: List[List[Any]] = []
    task_scores: List[Optional[int]] = []

    for block in blocks:
        # 1) Try parsing as clean JSON first
        parsed = try_parse_clean_json_block(block)
        if parsed is not None:
            ev = parsed.get("Turn-taking event and score", [])
            if isinstance(ev, list):
                merged_events.extend(ev)
            ts = parsed.get("Task-specific score", None)
            if ts is not None:
                task_scores.append(ts)
            continue

        # 2) Fallback scanning if JSON parsing fails
        kpos, lb, rb = extract_turn_taking_array_region(block)
        if lb != -1 and rb != -1:
            arr_text = block[lb + 1 : rb]
            events = parse_turn_taking_array(arr_text)
            merged_events.extend(events)
            ts = parse_task_specific_score(block[rb:]) or parse_task_specific_score(
                block
            )
            if ts is not None:
                task_scores.append(ts)

    out: Dict[str, Any] = {"Turn-taking event and score": merged_events}
    if task_scores:
        out["Task-specific score"] = task_scores[-1]
    return out


# =========================
# Directory Traversal
# =========================
def should_process(fname: str) -> bool:
    return (
        fname.endswith(".json")
        and not fname.endswith("_processed.json")
        and not fname.endswith(".raw_api.json")
    )


def process_tree(root_folder: str) -> None:
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if not should_process(fname):
                continue
            fpath = os.path.join(dirpath, fname)
            outpath = os.path.join(dirpath, fname[:-5] + "_processed.json")
            try:
                print(f"Processing {fpath} ...")
                with open(fpath, "r", encoding="utf-8") as f:
                    txt = f.read()
                data = process_one_text(txt)
                with open(outpath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"  ✅ Saved -> {outpath}")
            except Exception as e:
                print(f"  ❌ Failed: {fpath} | {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parse evaluation JSONs.")
    parser.add_argument("--root_dir", default="../eval_results", help="Directory containing eval results")
    args = parser.parse_args()
    
    process_tree(args.root_dir)
    print("All done.")


if __name__ == "__main__":
    main()
