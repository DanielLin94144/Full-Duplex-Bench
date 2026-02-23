#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$THIS_DIR/.." && pwd)"

# Load .env file automatically
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -o allexport
  source "$PROJECT_ROOT/.env"
  set +o allexport
elif [[ -f ".env" ]]; then
  set -o allexport
  source ".env"
  set +o allexport
fi

ROOT_DIR=""
EVAL_PY=""
EVAL_PROMPTS=""
STAGED_FILES=""
API_KEY=""
OUT_ROOT=""
NUM="20"
SLEEP_S="0"
MODEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root_dir)
      ROOT_DIR="$2"; shift 2 ;;
    --eval-py)
      EVAL_PY="$2"; shift 2 ;;
    --eval-prompts)
      EVAL_PROMPTS="$2"; shift 2 ;;
    --staged-files)
      STAGED_FILES="$2"; shift 2 ;;
    --api-key)
      API_KEY="$2"; shift 2 ;;
    --out-root)
      OUT_ROOT="$2"; shift 2 ;;
    --num)
      NUM="$2"; shift 2 ;;
    --sleep)
      SLEEP_S="$2"; shift 2 ;;
    --model)
      MODEL="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$ROOT_DIR" || -z "$EVAL_PY" || -z "$EVAL_PROMPTS" || -z "$STAGED_FILES" || -z "$API_KEY" || -z "$OUT_ROOT" ]]; then
  echo "Missing required args. See header for usage." >&2
  exit 1
fi

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Root dir not found: $ROOT_DIR" >&2
  exit 1
fi

if [[ ! -f "$EVAL_PY" ]]; then
  echo "Evaluator not found: $EVAL_PY" >&2
  exit 1
fi

if [[ ! -f "$EVAL_PROMPTS" ]]; then
  echo "eval_prompts_v2.json not found: $EVAL_PROMPTS" >&2
  exit 1
fi

IFS=',' read -r -a STAGED_ARRAY <<< "$STAGED_FILES"
for p in "${STAGED_ARRAY[@]}"; do
  if [[ ! -f "$p" ]]; then
    echo "Staged file not found: $p" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"

TMP_A_LIST=$(mktemp -t rbe_a.XXXXXX)
TMP_TASK_DIRS=$(mktemp -t rbe_dirs.XXXXXX)
TMP_SAMPLED=$(mktemp -t rbe_samp.XXXXXX)
cleanup() {
  rm -f "$TMP_A_LIST" "$TMP_TASK_DIRS" "$TMP_SAMPLED" || true
}
trap cleanup EXIT

# Collect all candidate A.json files
find "$ROOT_DIR" -type f -name 'A.json' > "$TMP_A_LIST"

# Filter to task directories that also have B.json
while IFS= read -r AFILE; do
  TDIR=$(dirname "$AFILE")
  if [[ -f "$TDIR/B.json" ]]; then
    echo "$TDIR" >> "$TMP_TASK_DIRS"
  fi
done < "$TMP_A_LIST"

# Sample N task directories uniformly at random using awk+sort
awk 'BEGIN{srand()} {print rand()"\t"$0}' "$TMP_TASK_DIRS" | sort -k1,1n | cut -f2- | head -n "$NUM" > "$TMP_SAMPLED"

echo "Scanning root: $ROOT_DIR"
echo "Selected $(wc -l < "$TMP_SAMPLED") tasks from $ROOT_DIR"

if [[ ! -s "$TMP_SAMPLED" ]]; then
  echo "No tasks found (no A.json with sibling B.json under root). Exiting."
  exit 0
fi

RC=0
while IFS= read -r TASK_DIR; do
  [[ -z "$TASK_DIR" ]] && continue
  A_JSON="$TASK_DIR/A.json"
  B_JSON="$TASK_DIR/B.json"

  PARENT_DIR=$(dirname "$TASK_DIR")
  TASK_ID=$(basename "$TASK_DIR")

  # Detect subset by scanning path segments upward for known subset names
  SUBSET=""
  IFS='/' read -r -a SEGMENTS <<< "$TASK_DIR"
  for seg in "${SEGMENTS[@]}"; do
    case "$seg" in
      Daily|Correction|EntityTracking|Safety)
        SUBSET="$seg" ;;
    esac
  done
  # Fallback to immediate parent name if not detected
  if [[ -z "$SUBSET" ]]; then
    SUBSET=$(basename "$PARENT_DIR")
  fi

  OUT_FILE="$OUT_ROOT/$TASK_ID.json"
  echo "Running: subset=$SUBSET task-id=$TASK_ID"

  if [[ -n "$MODEL" ]]; then
    python "$EVAL_PY" \
      --subset "$SUBSET" \
      --task-id "$TASK_ID" \
      --transcript-a "$A_JSON" \
      --transcript-b "$B_JSON" \
      --eval-prompts "$EVAL_PROMPTS" \
      --staged-files "${STAGED_ARRAY[@]}" \
      --api-key "$API_KEY" \
      --model "$MODEL" \
      --out "$OUT_FILE" || RC=$?
  else
    python "$EVAL_PY" \
      --subset "$SUBSET" \
      --task-id "$TASK_ID" \
      --transcript-a "$A_JSON" \
      --transcript-b "$B_JSON" \
      --eval-prompts "$EVAL_PROMPTS" \
      --staged-files "${STAGED_ARRAY[@]}" \
      --api-key "$API_KEY" \
      --out "$OUT_FILE" || RC=$?
  fi

  if [[ "$SLEEP_S" != "0" ]]; then
    sleep "$SLEEP_S"
  fi
done < "$TMP_SAMPLED"

exit "$RC"


