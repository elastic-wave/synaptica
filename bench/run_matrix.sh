#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="/media/ubuntu/ssd_drive/projects/synaptica/releases/gguf"  # e.g., /home/ubuntu/jetson-llm-quant-poc/releases
PROMPTS="bench/prompts/quick.txt"
OUT_DIR="bench/results_matrix"
N_PREDICT=128
PORT=8080

mkdir -p "$OUT_DIR"

QUANTS=("q4km" "q5km" "q8_0")
CONTEXTS=(512 768 1024)
GPU_LAYERS=(0 32 64)

# Simple timestamped log
log() { echo "[$(date +'%H:%M:%S')] $*"; }

for quant in "${QUANTS[@]}"; do
  for ctx in "${CONTEXTS[@]}"; do
    for gl in "${GPU_LAYERS[@]}"; do
      model_path="${MODEL_ROOT}/tinyllama-${quant}.gguf"
      out_csv="${OUT_DIR}/tinyllama_${quant}_c${ctx}_gl${gl}.csv"
      server_log="${OUT_DIR}/server_${quant}_c${ctx}_gl${gl}.log"

      log "=== Running ${quant} | ctx=${ctx} | gpu-layers=${gl} ==="
      log "Model path: ${model_path}"
      log "Output CSV: ${out_csv}"

      # Kill any old server
      pkill -f llama-server || true
      sleep 2

      if [[ ! -f "$model_path" ]]; then
        log "⚠️ Model file not found: $model_path"
        continue
      fi

      # Start llama.cpp server
      log "Starting server..."
      cd /media/ubuntu/ssd_drive/llama.cpp
      ./build/bin/llama-server \
        -m "$model_path" \
        -c "$ctx" \
        --gpu-layers "$gl" \
        --host 0.0.0.0 \
        --port "$PORT" \
        >"$server_log" 2>&1 &

      server_pid=$!
      sleep 6  # give it time to load

      # Check server health
      if ! curl -s --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null; then
        log "❌ Server not responding on port ${PORT}. Check ${server_log}"
        pkill -f llama-server || true
        continue
      fi

      log "✅ Server responding. Starting benchmark..."

      # Run benchmark
      cd /media/ubuntu/ssd_drive/projects/synaptica
      if ! python bench/bench_llamacpp.py \
        --prompts "$PROMPTS" \
        --out "$out_csv" \
        --n_predict "$N_PREDICT"; then
        log "❌ Benchmark failed for ${quant}_c${ctx}_gl${gl}"
      else
        log "✅ Benchmark completed for ${quant}_c${ctx}_gl${gl}"
      fi

      log "Stopping server..."
      pkill -f llama-server || true
      sleep 2
      log "---------------------------------------------"
    done
  done
done

log "All benchmarks complete."
log "Results: $OUT_DIR"
