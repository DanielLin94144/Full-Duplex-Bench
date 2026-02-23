#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_BATCH="$THIS_DIR/run_batch_eval.sh"
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
MODEL=""
SLEEP_S="0"
OUTPUT_DIR=""

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
    --model)
      MODEL="$2"; shift 2 ;;
    --sleep)
      SLEEP_S="$2"; shift 2 ;;
    --output)
      OUTPUT_DIR="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$ROOT_DIR" || -z "$EVAL_PY" || -z "$EVAL_PROMPTS" || -z "$STAGED_FILES" || -z "$API_KEY" ]]; then
  echo "Missing required args. See header for usage." >&2
  exit 1
fi

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Root dir not found: $ROOT_DIR" >&2
  exit 1
fi

ROOT_NAME="$(basename "$ROOT_DIR")"
if [[ -n "$OUTPUT_DIR" ]]; then
  OUT_BASE="$OUTPUT_DIR"
else
  OUT_BASE="$THIS_DIR/eval/outputs/$ROOT_NAME"
fi
mkdir -p "$OUT_BASE"

declare -a SPLITS=("Correction" "Daily" "EntityTracking" "Safety")

# Prefer alignment/<split> if exists, else <split> at root
for SPLIT in "${SPLITS[@]}"; do
  CAND1="$ROOT_DIR/alignment/$SPLIT"
  CAND2="$ROOT_DIR/$SPLIT"
  SPLIT_ROOT=""
  if [[ -d "$CAND1" ]]; then
    SPLIT_ROOT="$CAND1"
  elif [[ -d "$CAND2" ]]; then
    SPLIT_ROOT="$CAND2"
  else
    echo "Skip split $SPLIT: not found under $ROOT_DIR" >&2
    continue
  fi

  OUT_DIR="$OUT_BASE/$SPLIT"
  mkdir -p "$OUT_DIR"

  echo "Evaluating split=$SPLIT from $SPLIT_ROOT -> $OUT_DIR"
  if [[ -n "$MODEL" ]]; then
    bash "$RUN_BATCH" \
      --root_dir "$SPLIT_ROOT" \
      --eval-py "$EVAL_PY" \
      --eval-prompts "$EVAL_PROMPTS" \
      --staged-files "$STAGED_FILES" \
      --api-key "$API_KEY" \
      --out-root "$OUT_DIR" \
      --num 1000000 \
      --sleep "$SLEEP_S" \
      --model "$MODEL"
  else
    bash "$RUN_BATCH" \
      --root_dir "$SPLIT_ROOT" \
      --eval-py "$EVAL_PY" \
      --eval-prompts "$EVAL_PROMPTS" \
      --staged-files "$STAGED_FILES" \
      --api-key "$API_KEY" \
      --out-root "$OUT_DIR" \
      --num 1000000 \
      --sleep "$SLEEP_S"
  fi
done

echo "All splits completed for $ROOT_NAME. Outputs at: $OUT_BASE"


