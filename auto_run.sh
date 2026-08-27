#!/bin/bash
# Wait until the shared GPU is free of OTHER users' jobs, then run the full
# eval + sweep. Non-contending by design: we never start while another CUDA
# process is resident, because contention on this box has wedged the GPU
# (cudaErrorLaunchFailure) once already.
cd ~/weight_switching_CoT
export WSC_BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"

mypid=$$
echo "AUTO_RUN armed at $(date +%H:%M), waiting for a free GPU..."
while true; do
  # any resident CUDA compute app (there will be none of ours yet)
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c '[0-9]')
  if [ "$procs" -eq 0 ]; then
    break
  fi
  sleep 60
done
echo "GPU_FREE at $(date +%H:%M) — starting run"

echo "START_PIPELINE_$(date +%H:%M)"
.venv/bin/python pipeline.py --per-lang 5 || { echo "PIPELINE_FAILED"; exit 1; }
echo "START_SWEEP_$(date +%H:%M)"
.venv/bin/python optimize.py --per-lang 3 --patience 2 || { echo "SWEEP_FAILED"; exit 1; }
echo "CHAIN_DONE_$(date +%H:%M)"
