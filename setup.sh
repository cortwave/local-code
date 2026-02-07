#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Local Claude Code Setup
# Installs dependencies for Qwen3-Coder-Next via vLLM + LiteLLM
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

# Install Claude Code
log "Installing Claude Code..."
curl -fsSL https://claude.ai/install.sh | bash

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
uv pip install --python .venv/bin/python "vllm>=0.15,<0.16" "litellm[proxy]" hf_transfer
log "Dependencies installed."

# ============================================================================
# Done
# ============================================================================

echo ""
log "Setup complete! Dependencies installed."
echo ""
echo -e "${BLUE}To start services:${NC}"
echo "  ./start-vllm.sh [OPTIONS]"
echo "  ./start-litellm.sh [OPTIONS]"
echo ""
echo -e "${BLUE}To stop services:${NC}"
echo "  ./start-vllm.sh stop"
echo "  ./start-litellm.sh stop"
