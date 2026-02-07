#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Local Claude Code Setup
# Sets up Qwen3-Coder-Next-NVFP4 via vLLM + LiteLLM for use with Claude Code
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
MAX_MODEL_LEN=32768
TENSOR_PARALLEL_SIZE=1
VLLM_PORT=8000
LITELLM_PORT=4000
GPU_MEMORY_UTILIZATION=0.95
MODEL="GadflyII/Qwen3-Coder-Next-NVFP4"
MASTER_KEY="sk-local"
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
  start (default)    Start vLLM and LiteLLM services
  stop               Stop all running services

Options:
  --max-model-len N            Max context length (default: $MAX_MODEL_LEN)
  --tensor-parallel-size N     Number of GPUs for TP (default: $TENSOR_PARALLEL_SIZE)
  --vllm-port N                vLLM serving port (default: $VLLM_PORT)
  --litellm-port N             LiteLLM proxy port (default: $LITELLM_PORT)
  --gpu-memory-utilization N   GPU memory fraction (default: $GPU_MEMORY_UTILIZATION)
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
        --vllm-port)
            VLLM_PORT="$2"; shift 2 ;;
        --litellm-port)
            LITELLM_PORT="$2"; shift 2 ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            error "Unknown option: $1"
            usage; exit 1 ;;
    esac
done

# ============================================================================
# Stop command
# ============================================================================

stop_services() {
    log "Stopping services..."

    for service in vllm litellm; do
        local pidfile=".${service}.pid"
        if [[ -f "$pidfile" ]]; then
            local pid
            pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                log "Stopping $service (PID $pid)..."
                kill "$pid"
                # Wait for graceful shutdown
                for i in $(seq 1 10); do
                    if ! kill -0 "$pid" 2>/dev/null; then
                        break
                    fi
                    sleep 1
                done
                # Force kill if still running
                if kill -0 "$pid" 2>/dev/null; then
                    warn "Force killing $service (PID $pid)..."
                    kill -9 "$pid" 2>/dev/null || true
                fi
                log "$service stopped."
            else
                warn "$service (PID $pid) is not running."
            fi
            rm -f "$pidfile"
        else
            info "No PID file for $service."
        fi
    done

    log "All services stopped."
}

if [[ "$COMMAND" == "stop" ]]; then
    stop_services
    exit 0
fi

# ============================================================================
# Preflight checks
# ============================================================================

log "Running preflight checks..."

# Check nvidia-smi
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. CUDA drivers must be installed."
    exit 1
fi

info "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | while read -r line; do
    info "  $line"
done

# Install uv if missing
if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        error "uv installation failed."
        exit 1
    fi
    log "uv installed."
else
    info "uv already installed."
fi
info "uv version: $(uv --version)"

# Install nvtop if missing
if ! command -v nvtop &>/dev/null; then
    log "Installing nvtop..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq nvtop
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y nvtop
    else
        warn "Could not install nvtop: no supported package manager found. Install it manually."
    fi
    if command -v nvtop &>/dev/null; then
        log "nvtop installed."
    fi
else
    info "nvtop already installed."
fi

# Install Claude Code if missing
if ! command -v claude &>/dev/null; then
    log "Installing Claude Code..."
    curl -fsSL https://claude.ai/install.sh | bash
    if ! command -v claude &>/dev/null; then
        error "Claude Code installation failed."
        exit 1
    fi
    log "Claude Code installed."
else
    info "Claude Code already installed."
fi

# Check if services are already running
for service in vllm litellm; do
    pidfile=".${service}.pid"
    if [[ -f "$pidfile" ]]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            error "$service is already running (PID $pid). Run './setup.sh stop' first."
            exit 1
        else
            rm -f "$pidfile"
        fi
    fi
done

# ============================================================================
# Environment setup
# ============================================================================

log "Setting up Python environment..."

if [[ ! -d ".venv" ]]; then
    uv venv .venv
    log "Created virtual environment."
else
    info "Virtual environment already exists."
fi

log "Installing dependencies (this may take a few minutes)..."
uv pip install --python .venv/bin/python --prerelease=allow "vllm>=0.16.0rc1" "litellm[proxy]" hf_transfer
log "Dependencies installed."

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
# Start services
# ============================================================================

mkdir -p logs

# --- Start vLLM ---

log "Starting vLLM (model: $MODEL)..."
info "  Port: $VLLM_PORT"
info "  Tensor parallel size: $TENSOR_PARALLEL_SIZE"
info "  Max model length: $MAX_MODEL_LEN"
info "  GPU memory utilization: $GPU_MEMORY_UTILIZATION"

HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --kv-cache-dtype fp8 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --enforce-eager \
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

# --- Start LiteLLM ---

log "Starting LiteLLM proxy..."
info "  Port: $LITELLM_PORT"

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
log "Setup complete! Both services are running."
echo ""
echo -e "${BLUE}To use with Claude Code, run:${NC}"
echo ""
echo "  export ANTHROPIC_BASE_URL=http://localhost:${LITELLM_PORT}"
echo "  export ANTHROPIC_AUTH_TOKEN=${MASTER_KEY}"
echo "  claude"
echo ""
echo -e "${BLUE}To stop services:${NC}"
echo ""
echo "  ./setup.sh stop"
echo ""
echo -e "${BLUE}Logs:${NC}"
echo "  vLLM:    logs/vllm.log"
echo "  LiteLLM: logs/litellm.log"
