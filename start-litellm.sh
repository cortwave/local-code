#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Start LiteLLM proxy for Qwen3-Coder-Next
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
MAX_MODEL_LEN=16384
VLLM_PORT=8000
LITELLM_PORT=4000
QUANTIZATION="nvfp4"
MASTER_KEY="sk-local"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()   { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[x]${NC} $*" >&2; }
info()  { echo -e "${BLUE}[i]${NC} $*"; }

# ============================================================================
# Parse arguments
# ============================================================================

usage() {
    cat <<EOF
Usage: $(basename "$0") [start|stop] [OPTIONS]

Commands:
  start (default)    Start LiteLLM proxy
  stop               Stop LiteLLM proxy

Options:
  --max-model-len N      Max context length, must match vLLM (default: $MAX_MODEL_LEN)
  --vllm-port N          vLLM backend port (default: $VLLM_PORT)
  --port N               LiteLLM proxy port (default: $LITELLM_PORT)
  --quantization TYPE    Quantization type: fp8 or nvfp4 (default: $QUANTIZATION)
  --master-key KEY       API key for LiteLLM (default: $MASTER_KEY)
  -h, --help             Show this help
EOF
}

COMMAND="start"

while [[ $# -gt 0 ]]; do
    case $1 in
        start)
            COMMAND="start"; shift ;;
        stop)
            COMMAND="stop"; shift ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"; shift 2 ;;
        --vllm-port)
            VLLM_PORT="$2"; shift 2 ;;
        --port)
            LITELLM_PORT="$2"; shift 2 ;;
        --quantization)
            QUANTIZATION="$2"; shift 2 ;;
        --master-key)
            MASTER_KEY="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            error "Unknown option: $1"
            usage; exit 1 ;;
    esac
done

# ============================================================================
# Resolve model from quantization
# ============================================================================

case "$QUANTIZATION" in
    fp8)
        MODEL="Qwen/Qwen3-Coder-Next-FP8" ;;
    nvfp4)
        MODEL="GadflyII/Qwen3-Coder-Next-NVFP4" ;;
    *)
        error "Unknown quantization type: $QUANTIZATION (must be fp8 or nvfp4)"
        exit 1 ;;
esac

# ============================================================================
# Stop command
# ============================================================================

stop_litellm() {
    local pidfile=".litellm.pid"
    if [[ -f "$pidfile" ]]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping LiteLLM (PID $pid)..."
            kill "$pid"
            for i in $(seq 1 10); do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            if kill -0 "$pid" 2>/dev/null; then
                warn "Force killing LiteLLM (PID $pid)..."
                kill -9 "$pid" 2>/dev/null || true
            fi
            log "LiteLLM stopped."
        else
            warn "LiteLLM (PID $pid) is not running."
        fi
        rm -f "$pidfile"
    else
        info "No PID file for LiteLLM."
    fi
}

if [[ "$COMMAND" == "stop" ]]; then
    stop_litellm
    exit 0
fi

# ============================================================================
# Check prerequisites
# ============================================================================

if [[ ! -d ".venv" ]]; then
    error "Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

if [[ -f ".litellm.pid" ]]; then
    pid=$(cat ".litellm.pid")
    if kill -0 "$pid" 2>/dev/null; then
        error "LiteLLM is already running (PID $pid). Run './start-litellm.sh stop' first."
        exit 1
    else
        rm -f ".litellm.pid"
    fi
fi

# ============================================================================
# Generate LiteLLM config
# ============================================================================

log "Generating litellm-config.yaml..."

MAX_INPUT_TOKENS=$((MAX_MODEL_LEN - 8192))
MAX_OUTPUT_TOKENS=8192

cat > litellm-config.yaml <<EOF
model_list:
  - model_name: "claude-*"
    litellm_params:
      model: hosted_vllm/${MODEL}
      api_base: http://localhost:${VLLM_PORT}/v1
      api_key: "not-needed"
    model_info:
      max_tokens: ${MAX_MODEL_LEN}
      max_input_tokens: ${MAX_INPUT_TOKENS}
      max_output_tokens: ${MAX_OUTPUT_TOKENS}

litellm_settings:
  drop_params: true
  request_timeout: 600
  modify_params: true

general_settings:
  master_key: ${MASTER_KEY}
  disable_key_check: false
EOF

log "Config written to litellm-config.yaml"

# ============================================================================
# Start LiteLLM
# ============================================================================

mkdir -p logs

log "Starting LiteLLM proxy..."
info "  Port: $LITELLM_PORT"
info "  vLLM backend: http://localhost:$VLLM_PORT"
info "  Model: $MODEL"

.venv/bin/litellm \
    --config litellm-config.yaml \
    --port "$LITELLM_PORT" \
    > logs/litellm.log 2>&1 &

LITELLM_PID=$!
echo "$LITELLM_PID" > .litellm.pid
log "LiteLLM started (PID $LITELLM_PID). Logs: logs/litellm.log"

# Wait for LiteLLM to be healthy
log "Waiting for LiteLLM to be ready..."

elapsed=0
while [[ $elapsed -lt 60 ]]; do
    if ! kill -0 "$LITELLM_PID" 2>/dev/null; then
        error "LiteLLM process died. Check logs/litellm.log for details."
        rm -f .litellm.pid
        exit 1
    fi
    if curl -sf -H "Authorization: Bearer ${MASTER_KEY}" "http://localhost:${LITELLM_PORT}/health" >/dev/null 2>&1; then
        log "LiteLLM is ready!"
        break
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done

if [[ $elapsed -ge 60 ]]; then
    error "LiteLLM did not become healthy within 60s. Check logs/litellm.log"
    kill "$LITELLM_PID" 2>/dev/null || true
    rm -f .litellm.pid
    exit 1
fi

# ============================================================================
# Done
# ============================================================================

echo ""
log "LiteLLM is running."
echo ""
echo -e "${BLUE}To use with Claude Code, run:${NC}"
echo ""
echo "  export ANTHROPIC_BASE_URL=http://localhost:${LITELLM_PORT}"
echo "  export ANTHROPIC_AUTH_TOKEN=${MASTER_KEY}"
echo "  claude"
