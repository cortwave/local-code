#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Start vLLM server for Qwen3-Coder-Next
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
MAX_MODEL_LEN=16384
TENSOR_PARALLEL_SIZE=1
VLLM_PORT=8000
GPU_MEMORY_UTILIZATION=0.95
QUANTIZATION="nvfp4"
HEALTH_TIMEOUT=600  # 10 minutes (model download can take a while)

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
  start (default)    Start vLLM server
  stop               Stop vLLM server

Options:
  --max-model-len N            Max context length (default: $MAX_MODEL_LEN)
  --tensor-parallel-size N     Number of GPUs for TP (default: $TENSOR_PARALLEL_SIZE)
  --port N                     vLLM serving port (default: $VLLM_PORT)
  --gpu-memory-utilization N   GPU memory fraction (default: $GPU_MEMORY_UTILIZATION)
  --quantization TYPE          Quantization type: fp8 (~80GB) or nvfp4 (~45GB) (default: $QUANTIZATION)
  -h, --help                   Show this help
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
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --port)
            VLLM_PORT="$2"; shift 2 ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --quantization)
            QUANTIZATION="$2"; shift 2 ;;
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

stop_vllm() {
    local pidfile=".vllm.pid"
    if [[ -f "$pidfile" ]]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping vLLM (PID $pid)..."
            kill "$pid"
            for i in $(seq 1 10); do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            if kill -0 "$pid" 2>/dev/null; then
                warn "Force killing vLLM (PID $pid)..."
                kill -9 "$pid" 2>/dev/null || true
            fi
            log "vLLM stopped."
        else
            warn "vLLM (PID $pid) is not running."
        fi
        rm -f "$pidfile"
    else
        info "No PID file for vLLM."
    fi
}

if [[ "$COMMAND" == "stop" ]]; then
    stop_vllm
    exit 0
fi

# ============================================================================
# Check prerequisites
# ============================================================================

if [[ ! -d ".venv" ]]; then
    error "Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

if [[ -f ".vllm.pid" ]]; then
    pid=$(cat ".vllm.pid")
    if kill -0 "$pid" 2>/dev/null; then
        error "vLLM is already running (PID $pid). Run './start-vllm.sh stop' first."
        exit 1
    else
        rm -f ".vllm.pid"
    fi
fi

# ============================================================================
# Start vLLM
# ============================================================================

mkdir -p logs

log "Starting vLLM (model: $MODEL)..."
info "  Port: $VLLM_PORT"
info "  Tensor parallel size: $TENSOR_PARALLEL_SIZE"
info "  Max model length: $MAX_MODEL_LEN"
info "  GPU memory utilization: $GPU_MEMORY_UTILIZATION"
info "  Quantization: $QUANTIZATION"

VLLM_EXTRA_ARGS=()
if [[ "$QUANTIZATION" == "nvfp4" ]]; then
    VLLM_EXTRA_ARGS+=(--kv-cache-dtype fp8)
fi

HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --compilation-config '{"cudagraph_mode":"NONE"}' \
    --attention-backend FLASHINFER \
    "${VLLM_EXTRA_ARGS[@]}" \
    > logs/vllm.log 2>&1 &

VLLM_PID=$!
echo "$VLLM_PID" > .vllm.pid
log "vLLM started (PID $VLLM_PID). Logs: logs/vllm.log"

# Wait for vLLM to be healthy
log "Waiting for vLLM to be ready (this may take a while on first run as the model downloads)..."

elapsed=0
while [[ $elapsed -lt $HEALTH_TIMEOUT ]]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        error "vLLM process died. Check logs/vllm.log for details."
        rm -f .vllm.pid
        exit 1
    fi
    if curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
        log "vLLM is ready!"
        break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    if ((elapsed % 30 == 0)); then
        info "  Still waiting... (${elapsed}s elapsed)"
    fi
done

if [[ $elapsed -ge $HEALTH_TIMEOUT ]]; then
    error "vLLM did not become healthy within ${HEALTH_TIMEOUT}s. Check logs/vllm.log"
    kill "$VLLM_PID" 2>/dev/null || true
    rm -f .vllm.pid
    exit 1
fi
