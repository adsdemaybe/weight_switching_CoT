#!/bin/bash
# Serve Qwen2.5-7B-Instruct via vLLM on :8000. GB10 has 121GB unified memory.
cd ~/weight_switching_CoT
export PATH="$PWD/.venv-vllm/bin:$HOME/.local/bin:$PATH"   # vLLM shells out to ninja
exec .venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --served-model-name qwen \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.45
