#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/prompts_v2.json [--base-out /absolute/path/to/dir] [--voice-a NAME] [--voice-b NAME] [--duration SECONDS] [--split Daily|Safety|...] [--examiner-mode slow|fast]"
  exit 2
fi
JSON_PATH="$1"; shift || true

# Load .env file if it exists so we can use its defaults (e.g. DEFAULT_DURATION)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -o allexport
  source "$PROJECT_ROOT/.env"
  set +o allexport
elif [[ -f ".env" ]]; then
  set -o allexport
  source ".env"
  set +o allexport
fi

BASE_OUT=""
VOICE_A_OVERRIDE="${VOICE_A:-}"
VOICE_B_OVERRIDE="${VOICE_B:-shimmer}"
DURATION="${DEFAULT_DURATION:-120}"
SPLIT_FILTER=""
EXAMINER_MODE="${EXAMINER_MODE:-slow}"
RUN_LIMIT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapter-b)
      ADAPTER_B_JS="$2"; shift 2 ;;
    --base-out)
      BASE_OUT="$2"; shift 2 ;;
    --voice-a)
      VOICE_A_OVERRIDE="$2"; shift 2 ;;
    --voice-b)
      VOICE_B_OVERRIDE="$2"; shift 2 ;;
    --duration)
      DURATION="$2"; shift 2 ;;
    --limit)
      RUN_LIMIT="$2"; shift 2 ;;
    --split)
      SPLIT_FILTER="$2"; shift 2 ;;
    --examiner-mode)
      if [[ "$2" != "slow" && "$2" != "fast" ]]; then echo "Invalid --examiner-mode: must be 'slow' or 'fast'"; exit 2; fi
      EXAMINER_MODE="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"; exit 2 ;;
  esac
done

# Require jq
if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required. Install with: brew install jq" >&2
  exit 1
fi

if [[ ! -f "$JSON_PATH" ]]; then
  echo "ERROR: JSON file not found: $JSON_PATH" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCHER="$SCRIPT_DIR/single_conversation.sh"
if [[ ! -f "$LAUNCHER" ]]; then
  echo "ERROR: Launcher not found: $LAUNCHER" >&2
  exit 1
fi

# Fixed Role B system prompt (examinee)
ROLE_B_FIXED_PROMPT="${EXAMINEE_SYSTEM_PROMPT:-You are a helpful AI assistant. Always speak in English.}"

# Track active child to support Ctrl+C abort of entire runner
ACTIVE_PID=""
ACTIVE_PGID=""

cleanup_runner() {
  echo "[Runner] Abort requested; shutting down active session…"
  if [[ -n "${ACTIVE_PID:-}" ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
    if [[ -z "${ACTIVE_PGID:-}" ]]; then
      ACTIVE_PGID=$(ps -o pgid= -p "$ACTIVE_PID" | tr -d ' ')
    fi
    if [[ -n "$ACTIVE_PGID" ]]; then
      echo "[Runner] Sending SIGINT to process group -$ACTIVE_PGID"
      kill -INT -"$ACTIVE_PGID" 2>/dev/null || true
    else
      echo "[Runner] Sending SIGINT to PID $ACTIVE_PID"
      kill -INT "$ACTIVE_PID" 2>/dev/null || true
    fi
    GRACE=10
    for (( i=0; i<GRACE; i++ )); do
      if ! kill -0 "$ACTIVE_PID" 2>/dev/null; then break; fi
      sleep 1
    done
    if kill -0 "$ACTIVE_PID" 2>/dev/null; then
      if [[ -n "$ACTIVE_PGID" ]]; then
        echo "[Runner] Sending SIGTERM to group -$ACTIVE_PGID"
        kill -TERM -"$ACTIVE_PGID" 2>/dev/null || true
      else
        echo "[Runner] Sending SIGTERM to PID $ACTIVE_PID"
        kill -TERM "$ACTIVE_PID" 2>/dev/null || true
      fi
      sleep 2
    fi
    if kill -0 "$ACTIVE_PID" 2>/dev/null; then
      if [[ -n "$ACTIVE_PGID" ]]; then
        echo "[Runner] Sending SIGKILL to group -$ACTIVE_PGID"
        kill -KILL -"$ACTIVE_PGID" 2>/dev/null || true
      else
        echo "[Runner] Sending SIGKILL to PID $ACTIVE_PID"
        kill -KILL "$ACTIVE_PID" 2>/dev/null || true
      fi
    fi
    wait "$ACTIVE_PID" 2>/dev/null || true
  fi
  echo "[Runner] Aborted."
  exit 130
}

trap cleanup_runner INT TERM

# Base output default under orchestrator directory if not provided
if [[ -z "$BASE_OUT" ]]; then
  BASE_OUT="$SCRIPT_DIR/outputs"
fi
mkdir -p "$BASE_OUT"

# Build list of tasks from v2 JSON
if [[ -n "$SPLIT_FILTER" ]]; then
  TASKS_JQ=".splits[\"$SPLIT_FILTER\"]?.tasks // []"
else
  # Concatenate all tasks arrays from all splits
  TASKS_JQ="[.splits | to_entries | .[].value.tasks] | add"
fi

num_tasks=$(jq "$TASKS_JQ | length" "$JSON_PATH")
tasks_run=0

for (( ti=0; ti<num_tasks; ti++ )); do
  if [[ -n "$RUN_LIMIT" && "$tasks_run" -ge "$RUN_LIMIT" ]]; then
    echo "Reached limit of $RUN_LIMIT tasks. Stopping."
    break
  fi

  task_jq="$TASKS_JQ | .[$ti]"
  task_id=$(jq -r "$task_jq.id" "$JSON_PATH")
  split=$(jq -r "$task_jq.split" "$JSON_PATH")
  class_id=$(jq -r "$task_jq.class_id" "$JSON_PATH")
  system_prompt=$(jq -r "$task_jq.examiner_system_prompt" "$JSON_PATH")
  system_prompt="${system_prompt} Do not talk over the other speaker."
  task_prompt=$(jq -r "$task_jq.examiner_task_prompt" "$JSON_PATH")

  out_dir="$BASE_OUT/$split/$task_id"
  mkdir -p "$out_dir"

  echo "\n=== Running experiment: $split / $class_id / $task_id (examiner-mode: $EXAMINER_MODE) ==="
  echo "Output: $out_dir"

  # Compose args: Role A = system prompt (PROMPT_A) + task prompt passed as task-prompt-a; Role B fixed
  ARGS=("--record-dir" "$out_dir" "--prompt-a" "$system_prompt" "--task-prompt-a" "$task_prompt" "--prompt-b" "$ROLE_B_FIXED_PROMPT" "--adapter-b" "$ADAPTER_B_JS" "--examiner-mode" "$EXAMINER_MODE")
  if [[ -n "$VOICE_A_OVERRIDE" ]]; then ARGS+=("--voice-a" "$VOICE_A_OVERRIDE"); fi
  if [[ -n "$VOICE_B_OVERRIDE" ]]; then ARGS+=("--voice-b" "$VOICE_B_OVERRIDE"); fi

  if [[ -n "${DURATION}" ]]; then
    USE_GROUP=0
    if command -v setsid >/dev/null 2>&1; then
      USE_GROUP=1
      setsid bash "$LAUNCHER" "${ARGS[@]}" &
    else
      bash "$LAUNCHER" "${ARGS[@]}" &
    fi
    LAUNCH_PID=$!
    ACTIVE_PID=$LAUNCH_PID
    LAUNCH_PGID=$(ps -o pgid= -p "$LAUNCH_PID" | tr -d ' ')
    if [[ -z "$LAUNCH_PGID" ]]; then LAUNCH_PGID=$LAUNCH_PID; fi
    ACTIVE_PGID=$LAUNCH_PGID
    echo "[Runner] Launched PID $LAUNCH_PID (PGID $LAUNCH_PGID, group_mode=$USE_GROUP) for ${DURATION}s"
    sleep 2
    sleep "$DURATION" || true
    if [[ "$USE_GROUP" -eq 1 ]]; then
      echo "[Runner] Sending SIGINT to process group -$LAUNCH_PGID"
      kill -INT -"$LAUNCH_PGID" 2>/dev/null || true
    else
      echo "[Runner] Sending SIGINT to PID $LAUNCH_PID"
      kill -INT "$LAUNCH_PID" 2>/dev/null || true
    fi
    GRACE=20
    for (( i=0; i<GRACE; i++ )); do
      if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then break; fi
      sleep 1
    done
    if kill -0 "$LAUNCH_PID" 2>/dev/null; then
      if [[ "$USE_GROUP" -eq 1 ]]; then
        echo "[Runner] Still running after SIGINT grace; sending SIGTERM to group -$LAUNCH_PGID"
        kill -TERM -"$LAUNCH_PGID" 2>/dev/null || true
      else
        echo "[Runner] Still running after SIGINT grace; sending SIGTERM to PID $LAUNCH_PID"
        kill -TERM "$LAUNCH_PID" 2>/dev/null || true
      fi
      sleep 3
    fi
    if kill -0 "$LAUNCH_PID" 2>/dev/null; then
      if [[ "$USE_GROUP" -eq 1 ]]; then
        echo "[Runner] Still running; sending SIGKILL to group -$LAUNCH_PGID"
        kill -KILL -"$LAUNCH_PGID" 2>/dev/null || true
      else
        echo "[Runner] Still running; sending SIGKILL to PID $LAUNCH_PID"
        kill -KILL "$LAUNCH_PID" 2>/dev/null || true
      fi
    fi
    wait "$LAUNCH_PID" 2>/dev/null || true
    ACTIVE_PID=""; ACTIVE_PGID=""
    echo "[Runner] Completed: $task_id"
  else
    USE_GROUP=0
    if command -v setsid >/dev/null 2>&1; then
      USE_GROUP=1
      setsid bash "$LAUNCHER" "${ARGS[@]}" &
    else
      bash "$LAUNCHER" "${ARGS[@]}" &
    fi
    ACTIVE_PID=$!
    ACTIVE_PGID=$(ps -o pgid= -p "$ACTIVE_PID" | tr -d ' ')
    echo "[Runner] Launched interactive session PID $ACTIVE_PID (PGID ${ACTIVE_PGID:-unknown})"
    wait "$ACTIVE_PID" 2>/dev/null || true
    ACTIVE_PID=""; ACTIVE_PGID=""
  fi

  tasks_run=$((tasks_run + 1))
done

echo "All experiments completed. Base directory: $BASE_OUT"
