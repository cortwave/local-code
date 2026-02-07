# Local Claude Code with Qwen3-Coder-Next-FP8

Run Claude Code against a local [Qwen3-Coder-Next-FP8](https://huggingface.co/Qwen/Qwen3-Coder-Next-FP8) model using vLLM and LiteLLM.

## Architecture

```
Claude Code ──(Anthropic API)──► LiteLLM :4000 ──(OpenAI API)──► vLLM :8000 ──► GPU
```

## Prerequisites

- NVIDIA GPU with CUDA drivers installed (tested on RTX 6000 PRO 96GB)
- [uv](https://docs.astral.sh/uv/) package manager
- `nvidia-smi` available in PATH

## Quick Start

```bash
# Clone and run
git clone <this-repo>
cd local-code

# Start services (installs everything on first run)
./setup.sh

# In your working terminal
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_AUTH_TOKEN=sk-local
claude
```

The first run will download the model (~80GB) and install Python dependencies.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--max-model-len` | 32768 | Max context length in tokens |
| `--tensor-parallel-size` | 1 | Number of GPUs for tensor parallelism |
| `--vllm-port` | 8000 | vLLM serving port |
| `--litellm-port` | 4000 | LiteLLM proxy port |
| `--gpu-memory-utilization` | 0.95 | GPU memory fraction |

Example with custom settings:

```bash
./setup.sh --max-model-len 65536 --tensor-parallel-size 2 --gpu-memory-utilization 0.9
```

## Managing Services

```bash
# Stop all services
./setup.sh stop

# View logs
tail -f logs/vllm.log
tail -f logs/litellm.log
```

## How It Works

1. **vLLM** serves the Qwen3-Coder-Next-FP8 model with an OpenAI-compatible API, with tool calling enabled via `--tool-call-parser qwen3_coder`
2. **LiteLLM** acts as a proxy that translates Anthropic API requests (from Claude Code) into OpenAI API requests (for vLLM)
3. The `claude-*` wildcard in LiteLLM config catches all model names Claude Code sends and routes them to the local model
