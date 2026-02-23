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

# Default values (can be overridden via .env or CLI)
EXPERIMENT_ROOT=""
ASR_OUTPUT=""
EVAL_OUTPUT=""
MAX_TIME="${MAX_ASR_TIME:-120}"
API_KEY="${GEMINI_API_KEY:-}"
MODEL="${GEMINI_MODEL:-gemini-2.5-flash-preview-09-2025}"
SLEEP_S="0"
EVAL_PY="$PROJECT_ROOT/eval/eval_single_item.py"
EVAL_PROMPTS="$PROJECT_ROOT/eval/eval_prompts.json"
STAGED_FILES="$PROJECT_ROOT/prompts_staged_200.json"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root_dir)
      EXPERIMENT_ROOT="$2"
      shift 2
      ;;
    --asr-output)
      ASR_OUTPUT="$2"
      shift 2
      ;;
    --eval-output)
      EVAL_OUTPUT="$2"
      shift 2
      ;;
    --max-time)
      MAX_TIME="$2"
      shift 2
      ;;
    --api-key)
      API_KEY="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --sleep)
      SLEEP_S="$2"
      shift 2
      ;;
    --eval-py)
      EVAL_PY="$2"
      shift 2
      ;;
    --eval-prompts)
      EVAL_PROMPTS="$2"
      shift 2
      ;;
    --staged-files)
      STAGED_FILES="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 --root_dir <experiment_root> [options]"
      echo ""
      echo "Required arguments:"
      echo "  --root_dir <path>      Experiment root directory"
      echo ""
      echo "Optional arguments:"
      echo "  --asr-output <path>    ASR output directory (default: ./experiments_<name>_asr)"
      echo "  --eval-output <path>   Evaluation output directory (default: ./eval_results)"
      echo "  --max-time <seconds>   Max audio processing time (default: 120)"
      echo "  --model <name>         Evaluation model (default: gemini-2.5-flash-preview-09-2025)"
      echo "  --sleep <seconds>      Sleep between API calls (default: 0)"
      echo "  --eval-py <path>       Evaluation script path"
      echo "  --eval-prompts <path>  Evaluation prompts JSON"
      echo "  --staged-files <path>  Staged prompts JSON"
      echo "  -h, --help             Show this help message"
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      # If no flag, treat as experiment root
      EXPERIMENT_ROOT="$1"
      shift
      ;;
  esac
done

# Validate required arguments
if [[ -z "$EXPERIMENT_ROOT" ]]; then
  echo "Error: Experiment root directory is required." >&2
  echo "Usage: $0 --root_dir <experiment_root>" >&2
  echo "   or: $0 <experiment_root>" >&2
  exit 1
fi

if [[ ! -d "$EXPERIMENT_ROOT" ]]; then
  echo "Error: Experiment root directory not found: $EXPERIMENT_ROOT" >&2
  exit 1
fi

# Check for API key in environment if not provided
if [[ -z "$API_KEY" ]]; then
  if [[ -n "${GEMINI_API_KEY:-}" ]]; then
    API_KEY="$GEMINI_API_KEY"
    echo "Using API key from GEMINI_API_KEY environment variable"
  elif [[ -n "${API_KEY_ENV:-}" ]]; then
    API_KEY="$API_KEY_ENV"
    echo "Using API key from API_KEY_ENV environment variable"
  else
    echo "Error: API key is required. Provide via --api-key or set GEMINI_API_KEY in your .env or environment variables." >&2
    exit 1
  fi
fi

# Set default output directories based on experiment name
EXPERIMENT_NAME="$(basename "$EXPERIMENT_ROOT")"
if [[ -z "$ASR_OUTPUT" ]]; then
  ASR_OUTPUT="./experiments_${EXPERIMENT_NAME}_asr"
fi
if [[ -z "$EVAL_OUTPUT" ]]; then
  EVAL_OUTPUT="$PROJECT_ROOT/eval_results"
fi

# Validate required files
if [[ ! -f "$EVAL_PY" ]]; then
  echo "Error: Evaluation script not found: $EVAL_PY" >&2
  exit 1
fi
if [[ ! -f "$EVAL_PROMPTS" ]]; then
  echo "Error: Evaluation prompts not found: $EVAL_PROMPTS" >&2
  exit 1
fi
if [[ ! -f "$STAGED_FILES" ]]; then
  echo "Error: Staged files not found: $STAGED_FILES" >&2
  exit 1
fi

echo "========================================="
echo "Running Evaluation Pipeline"
echo "========================================="
echo "Experiment root:  $EXPERIMENT_ROOT"
echo "Experiment name:  $EXPERIMENT_NAME"
echo "ASR output:       $ASR_OUTPUT"
echo "Eval output:      $EVAL_OUTPUT"
echo "Max time:         ${MAX_TIME}s"
echo "Model:            $MODEL"
echo "Eval script:      $EVAL_PY"
echo "Eval prompts:     $EVAL_PROMPTS"
echo "Staged files:     $STAGED_FILES"
echo "========================================="
echo ""

# Step 1: ASR Batch Processing
echo "[1/2] Running ASR batch transcription..."
echo "Command: python $PROJECT_ROOT/eval/asr_batch.py --root_dir \"$EXPERIMENT_ROOT\" --output \"$ASR_OUTPUT\" --max_time $MAX_TIME"
python "$PROJECT_ROOT/eval/asr_batch.py" \
  --root_dir "$EXPERIMENT_ROOT" \
  --output "$ASR_OUTPUT" \
  --max_time "$MAX_TIME"
echo "✓ ASR transcription completed"
echo ""

# Step 2: Dataset Evaluation
echo "[2/2] Running dataset evaluation..."
echo "Command: bash $THIS_DIR/run_dataset_eval.sh --root_dir \"$EXPERIMENT_ROOT\" --eval-py \"$EVAL_PY\" --eval-prompts \"$EVAL_PROMPTS\" --staged-files \"$STAGED_FILES\" --api-key \"***\" --model \"$MODEL\" --sleep $SLEEP_S --output \"$EVAL_OUTPUT\""
bash "$THIS_DIR/run_dataset_eval.sh" \
  --root_dir "$EXPERIMENT_ROOT" \
  --eval-py "$EVAL_PY" \
  --eval-prompts "$EVAL_PROMPTS" \
  --staged-files "$STAGED_FILES" \
  --api-key "$API_KEY" \
  --model "$MODEL" \
  --sleep "$SLEEP_S" \
  --output "$EVAL_OUTPUT"
echo "✓ Dataset evaluation completed"
echo ""

echo "========================================="
echo "Evaluation Pipeline Completed!"
echo "========================================="
echo "ASR results:      $ASR_OUTPUT"
echo "Eval results:     $EVAL_OUTPUT"
echo "========================================="

