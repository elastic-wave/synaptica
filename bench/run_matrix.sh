#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="/media/ubuntu/ssd_drive/projects/synaptica/releases/gguf"  # e.g., /home/ubuntu/jetson-llm-quant-poc/releases
PROMPTS="bench/prompts/quick.txt"
OUT_DIR="bench/results_matrix"
N_PREDICT=128

mkdir -p "$OUT_DIR"

# Define test parameters
QUANTS=("q4km" "q5km" "q8_0")
CONTEXTS=(512 768 1024)
GPU_LAYERS=(0 32 64)

for quant in "${QUANTS[@]}"; do
  for ctx in "${CONTEXTS[@]}"; do
    for gl in "${GPU_LAYERS[@]}"; do
      model_path="${MODEL_ROOT}/tinyllama-${quant}.gguf"
      out_csv="${OUT_DIR}/tinyllama_${quant}_c${ctx}_gl${gl}.csv"

      echo
      echo "=== Running ${quant} | context=${ctx} | gpu-layers=${gl} ==="
      echo "Output → ${out_csv}"

      # kill any old server
      pkill -f llama-server || true
      sleep 2

      # start server in background
      cd /media/ubuntu/ssd_drive/llama.cpp
      ./build/bin/llama-server \
        -m "${model_path}" \
        -c "${ctx}" \
        --gpu-layers "${gl}" \
        --host 0.0.0.0 \
        --port 8080 \
        >"${OUT_DIR}/server_${quant}_c${ctx}_gl${gl}.log" 2>&1 &

      # give server a few seconds to initialize
      sleep 6

      # run benchmark
      cd /media/ubuntu/ssd_drive/projects/synaptica
      python bench/bench_llamacpp.py \
        --prompts "${PROMPTS}" \
        --out "${out_csv}" \
        --n_predict "${N_PREDICT}"

      # stop server
      pkill -f llama-server || true
      sleep 2
    done
  done
done

echo "=== All benchmarks complete ==="
echo "Results saved under ${OUT_DIR}/"
