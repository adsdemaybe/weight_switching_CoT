#!/bin/bash
# Train 20 per-language LoRA experts on the GB10 against the 7B base.
cd ~/weight_switching_CoT
export WSC_BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
exec .venv/bin/python -m router.train_experts --steps 60 --n-train 48 "$@"
