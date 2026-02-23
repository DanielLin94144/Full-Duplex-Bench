#!/usr/bin/env bash

# Launches a two‑model conversation:
#   Role A: OpenAI GPT Realtime
#   Role B: OpenAI GPT Realtime
#
# Contracts preserved:
#   - Orchestrator signaling on ws://localhost:$SIGNAL_PORT/signal
#   - 48 kHz PCM16 mono, strict 10 ms cadence over WebRTC
#   - Graceful shutdown: ask orchestrator to mix down on SIGINT
#
# Usage:
#   1) Put OPENAI_API_KEY in .env (plus any VOICE/PROMPT/SEED vars you use).
#   2) ./run_gpt4o_gpt4o_conversation_run.sh
#

set -euo pipefail

# ── Load .env (OPENAI_API_KEY, optional SEED_PATH, VOICEs, PROMPTs) ─────────
if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
fi

# ── Config (defaults can be overridden via .env) ─────────────────────────────────────────────
# See .env.example for all available configuration options
# Assign a unique available port between 9000-15000 if not specified to avoid EADDRINUSE collisions across multiple loop runs
while true; do
  RANDOM_PORT=$((9000 + RANDOM % 6000))
  if ! (echo >/dev/tcp/localhost/$RANDOM_PORT) >/dev/null 2>&1; then
    break
  fi
done
SIGNAL_PORT="${SIGNAL_PORT:-$RANDOM_PORT}"
SIGNAL_URL="ws://localhost:${SIGNAL_PORT}/signal"
TOKEN_SERVER_PORT="${TOKEN_SERVER_PORT:-3002}"
TOKEN_SERVER_URL="${OPENAI_TOKEN_SERVER:-http://localhost:${TOKEN_SERVER_PORT}}"
RECORD_DIR="${RECORD_DIR:-outputs}"

# Model configuration
OPENAI_REALTIME_MODEL="${OPENAI_REALTIME_MODEL:-gpt-4o-realtime-preview}"

# Voices/prompts (fall back to sane defaults from .env or hardcoded)
VOICE_A="${VOICE_A:-echo}"
VOICE_B="${VOICE_B:-shimmer}"
PROMPT_A="${PROMPT_A:-}"
PROMPT_B="${PROMPT_B:-${EXAMINEE_SYSTEM_PROMPT:-You are a helpful AI assistant.}}"

# Optional task prompts (can be set via .env or CLI)
TASK_PROMPT="${TASK_PROMPT:-}"
TASK_PROMPT_A="${TASK_PROMPT_A:-}"
TASK_PROMPT_B="${TASK_PROMPT_B:-}"

# Examiner mode: slow (default) or fast
EXAMINER_MODE="${EXAMINER_MODE:-slow}"

# Paths (adjust if needed)
ORCH_JS="orchestrator.js"
ADAPTER_A_JS="adapters/gptRealtime_adapter.js"
ADAPTER_B_JS="adapters/gptRealtime_adapter.js"
TOKEN_SERVER_JS="adapters/openai_token_server.js"

# ── CLI overrides (prompts/voices/ports) ────────────────────────────────────────────────
print_usage() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  --adapter-a PATH          Set adapter for Role A (default: $ADAPTER_A_JS)"
  echo "  --adapter-b PATH          Set adapter for Role B (default: $ADAPTER_B_JS)"
  echo "  --prompt-a 'TEXT'          Set system prompt for Role A (GPT-4o)"
  echo "  --prompt-a-file PATH       Read Role A prompt from file"
  echo "  --prompt-b 'TEXT'          Set system prompt for Role B (GPT-4o)"
  echo "  --prompt-b-file PATH       Read Role B prompt from file"
  echo "  --task-prompt 'TEXT'       Set a global task prompt (applied to both roles)"
  echo "  --task-prompt-file PATH    Read global task prompt from file"
  echo "  --task-prompt-a 'TEXT'     Set task prompt only for Role A"
  echo "  --task-prompt-a-file PATH  Read Role A task prompt from file"
  echo "  --task-prompt-b 'TEXT'     Set task prompt only for Role B"
  echo "  --task-prompt-b-file PATH  Read Role B task prompt from file"
  echo "  --voice-a NAME             Set voice for Role A (default: $VOICE_A)"
  echo "  --voice-b NAME             Set voice for Role B (default: $VOICE_B)"
  echo "  --examiner-mode MODE       GPT examiner VAD mode: slow (default) or fast"
  echo "  --signal-port N            Orchestrator signaling port (default: $SIGNAL_PORT)"
  echo "  --token-server-port N      OpenAI token server port (default: $TOKEN_SERVER_PORT)"
  echo "  --record-dir PATH          Output directory for recordings (default: $RECORD_DIR)"
  echo "  -h, --help                 Show this help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapter-a)
      if [[ -z "${2:-}" ]]; then echo "Missing value for --adapter-a"; exit 2; fi
      ADAPTER_A_JS="$2"; shift 2 ;;
    --adapter-b)
      if [[ -z "${2:-}" ]]; then echo "Missing value for --adapter-b"; exit 2; fi
      ADAPTER_B_JS="$2"; shift 2 ;;
    --prompt-a)
      if [[ -z "${2:-}" ]]; then echo "Missing value for --prompt-a"; exit 2; fi
      PROMPT_A="$2"; shift 2 ;;
    --prompt-a-file)
      if [[ -z "${2:-}" ]]; then echo "Missing path for --prompt-a-file"; exit 2; fi
      PROMPT_A="$(cat "$2")"; shift 2 ;;
    --prompt-b)
      PROMPT_B="${2:-}"; shift 2 ;;
    --prompt-b-file)
      if [[ -n "${2:-}" && -f "$2" ]]; then
        PROMPT_B="$(cat "$2")"
      else
        PROMPT_B=""
      fi
      shift 2 ;;
    --task-prompt)
      if [[ -z "${2:-}" ]]; then echo "Missing value for --task-prompt"; exit 2; fi
      TASK_PROMPT="$2"; shift 2 ;;
    --task-prompt-file)
      if [[ -z "${2:-}" ]]; then echo "Missing path for --task-prompt-file"; exit 2; fi
      TASK_PROMPT="$(cat "$2")"; shift 2 ;;
    --task-prompt-a)
      if [[ -z "${2:-}" ]]; then echo "Missing value for --task-prompt-a"; exit 2; fi
      TASK_PROMPT_A="$2"; shift 2 ;;
    --task-prompt-a-file)
      if [[ -z "${2:-}" ]]; then echo "Missing path for --task-prompt-a-file"; exit 2; fi
      TASK_PROMPT_A="$(cat "$2")"; shift 2 ;;
    --task-prompt-b)
      if [[ -z "${2:-}" ]]; then echo "Missing value for --task-prompt-b"; exit 2; fi
      TASK_PROMPT_B="$2"; shift 2 ;;
    --task-prompt-b-file)
      if [[ -z "${2:-}" ]]; then echo "Missing path for --task-prompt-b-file"; exit 2; fi
      TASK_PROMPT_B="$(cat "$2")"; shift 2 ;;
    --voice-a)
      if [[ -z "${2:-}" ]]; then echo "Missing value for --voice-a"; exit 2; fi
      VOICE_A="$2"; shift 2 ;;
    --voice-b)
      if [[ -z "${2:-}" ]]; then echo "Missing value for --voice-b"; exit 2; fi
      VOICE_B="$2"; shift 2 ;;
    --signal-port)
      if [[ -z "${2:-}" ]]; then echo "Missing value for --signal-port"; exit 2; fi
      SIGNAL_PORT="$2"; SIGNAL_URL="ws://localhost:${SIGNAL_PORT}/signal"; shift 2 ;;
    --token-server-port)
      if [[ -z "${2:-}" ]]; then echo "Missing value for --token-server-port"; exit 2; fi
      TOKEN_SERVER_PORT="$2"; TOKEN_SERVER_URL="http://localhost:${TOKEN_SERVER_PORT}"; shift 2 ;;
    --record-dir)
      if [[ -z "${2:-}" ]]; then echo "Missing path for --record-dir"; exit 2; fi
      RECORD_DIR="$2"; shift 2 ;;
    --examiner-mode)
      if [[ -z "${2:-}" ]]; then echo "Missing value for --examiner-mode"; exit 2; fi
      if [[ "$2" != "slow" && "$2" != "fast" ]]; then echo "Invalid --examiner-mode: must be 'slow' or 'fast'"; exit 2; fi
      EXAMINER_MODE="$2"; shift 2 ;;
    -h|--help)
      print_usage; exit 0 ;;
    *)
      echo "Unknown option: $1"; echo ""; print_usage; exit 2 ;;
  esac
done

# Export recording dir for child processes (orchestrator, adapters)
export RECORD_DIR

# Prepare Role A prefill (USER role) and keep system prompt separate
PREFILL_A=""
# if [[ -n "$TASK_PROMPT_A" ]]; then
#   PREFILL_A="$TASK_PROMPT_A"
# elif [[ -n "$TASK_PROMPT" ]]; then
#   PREFILL_A="$TASK_PROMPT"
# fi

# ── Guardrails ──────────────────────────────────────────────────────────────────────────────
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "[Launcher] ERROR: OPENAI_API_KEY is not set (for GPT‑4o token server)."
  exit 1
fi

# ── Start orchestrator ─────────────────────────────────────────────────────────────────────
echo "🚦 Starting orchestrator on port ${SIGNAL_PORT}…"
echo "[Launcher] Recording directory: ${RECORD_DIR}"
node "$ORCH_JS" --port "$SIGNAL_PORT" &
ORCH_PID=$!
echo "[Launcher] Orchestrator PID: $ORCH_PID"

# optional tiny wait to ensure signaling socket is ready
sleep 0.8

echo "[Launcher] Orchestrator signaling at ${SIGNAL_URL}"

# ── Start OpenAI token server (for ephemeral Realtime tokens) ───────────────────────────────
echo "🔑 Starting OpenAI token server on port ${TOKEN_SERVER_PORT}…"
# Inherit env so server can read OPENAI_API_KEY
node "$TOKEN_SERVER_JS" &
TOKEN_PID=$!
echo "[Launcher] Token server PID: $TOKEN_PID"

# give it a moment to bind
sleep 0.5

COMPOSED_PROMPT_A="$PROMPT_A"
if [[ -n "$TASK_PROMPT_A" ]]; then
  printf -v COMPOSED_PROMPT_A "%s\n\n%s" "$PROMPT_A" "$TASK_PROMPT_A"
elif [[ -n "$TASK_PROMPT" ]]; then
  printf -v COMPOSED_PROMPT_A "%s\n\n%s" "$PROMPT_A" "$TASK_PROMPT"
fi

# ── Launch GPT-4o adapter (Role A) ─────────────────────────────────────────────────────────
echo "🧊 Launching GPT-4o adapter (Role A) with examiner mode: ${EXAMINER_MODE}…"
node "$ADAPTER_A_JS" \
  --role A \
  --signalUrl "$SIGNAL_URL" \
  --tokenServer "$TOKEN_SERVER_URL" \
  --voice "$VOICE_A" \
  --vadMode "$EXAMINER_MODE" \
  --systemPrompt "$COMPOSED_PROMPT_A" &
PID_A=$!
echo "[Launcher] Adapter A PID: $PID_A"

# ── Launch GPT-4o adapter (Role B) ─────────────────────────────────────────────────────────
echo "🧊 Launching adapter (Role B)…"
if [[ "$ADAPTER_B_JS" == "adapters/gptRealtime_adapter.js" ]]; then
  node "$ADAPTER_B_JS" \
    --role B \
    --signalUrl "$SIGNAL_URL" \
    --tokenServer "$TOKEN_SERVER_URL" \
    --voice "$VOICE_B" \
    --vadMode "$EXAMINER_MODE" \
    --systemPrompt "$PROMPT_B" &
else
  node "$ADAPTER_B_JS" \
    --role B \
    --signalUrl "$SIGNAL_URL" \
    --moshiUrl "ws://localhost:8998/api/chat" \
    --tokenServer "$TOKEN_SERVER_URL" &
fi
PID_B=$!
echo "[Launcher] Adapter B PID: $PID_B"

# ── Signal handling & cleanup ──────────────────────────────────────────────────────────────
cleanup() {
  echo -e "\n[Launcher] Shutting down processes…"
  # Ask orchestrator to mix down (SIGINT)
  kill -SIGINT "$ORCH_PID" 2>/dev/null || true
  # Terminate adapters and token server
  kill "$PID_A" "$PID_B" "$TOKEN_PID" 2>/dev/null || true
  # Wait for orchestrator to finish mixdown
  wait "$ORCH_PID" 2>/dev/null || true
  echo "[Launcher] Bye."
}
trap cleanup SIGINT SIGTERM

# Wait until user hits Ctrl+C
wait
