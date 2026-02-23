#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENT_ROOT=""
PROMPTS_FILE="prompts_staged_200.json"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      EXPERIMENT_ROOT="$2"
      shift 2
      ;;
    --prompts)
      PROMPTS_FILE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 --root <experiment_root> [--prompts <prompts_json>]"
      echo "   or: $0 <experiment_root>"
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
  echo "Usage: $0 --root <experiment_root> [--prompts <prompts_json>]" >&2
  echo "   or: $0 <experiment_root>" >&2
  exit 1
fi

if [[ ! -d "$EXPERIMENT_ROOT" ]]; then
  echo "Error: Experiment root directory not found: $EXPERIMENT_ROOT" >&2
  exit 1
fi

if [[ ! -f "$PROMPTS_FILE" ]]; then
  echo "Error: Prompts file not found: $PROMPTS_FILE" >&2
  exit 1
fi

echo "========================================="
echo "Preparing experiment: $(basename "$EXPERIMENT_ROOT")"
echo "Experiment root: $EXPERIMENT_ROOT"
echo "Prompts file: $PROMPTS_FILE"
echo "========================================="
echo ""

# Step 1: Cleanup sessions
echo "[1/3] Running cleanup_sessions.py..."
python "$THIS_DIR/cleanup_sessions.py" "$EXPERIMENT_ROOT" --apply
echo "✓ Cleanup completed"
echo ""

# Step 2: Write prompts
echo "[2/3] Running write_prompts_json.py..."
python "$THIS_DIR/write_prompts_json.py" "$EXPERIMENT_ROOT" --prompts "$PROMPTS_FILE" --apply
echo "✓ Prompts written"
echo ""

# Step 3: Trim combined wavs
echo "[3/3] Running trim_combined_wavs.py..."
python "$THIS_DIR/trim_combined_wavs.py" "$EXPERIMENT_ROOT"
echo "✓ WAV files trimmed"
echo ""

echo "========================================="
echo "All steps completed successfully!"
echo "========================================="

