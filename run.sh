#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Run/stop both vLLM and LiteLLM services
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMMAND="${1:-start}"

case "$COMMAND" in
    start)
        shift || true
        "$SCRIPT_DIR/start-vllm.sh" "$@"
        "$SCRIPT_DIR/start-litellm.sh" "$@"
        ;;
    stop)
        "$SCRIPT_DIR/start-litellm.sh" stop
        "$SCRIPT_DIR/start-vllm.sh" stop
        ;;
    *)
        echo "Usage: $(basename "$0") [start|stop] [OPTIONS]"
        echo ""
        echo "Passes all OPTIONS to both start-vllm.sh and start-litellm.sh."
        echo "Run ./start-vllm.sh --help or ./start-litellm.sh --help for details."
        exit 1
        ;;
esac
