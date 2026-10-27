#!/usr/bin/env bash
# run.sh - Build and execute the Capstone cuFFT Multi-Stream Spectral Pipeline
set -euo pipefail

COUNT=100
INPUT_DIR="data/input"
OUTPUT_DIR="data/output"
STREAMS=4

echo "================================================================="
echo "  CUDA CAPSTONE: MULTI-STREAM SPECTRAL PROCESSING PIPELINE       "
echo "================================================================="

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" docs

# 1. Generate synthetic multichannel signal dataset if empty
if [ -z "$(ls -A "$INPUT_DIR"/*.csv 2>/dev/null)" ]; then
  echo "[1/3] Generating $COUNT synthetic multi-harmonic signal files..."
  python3 scripts/gen_signals.py --count "$COUNT" --samples 16384 --output "$INPUT_DIR"
else
  echo "[1/3] Input signals already present."
fi

# 2. Build or fallback to Python simulator
if command -v nvcc &> /dev/null; then
  echo "[2/3] Compiling CUDA pipeline with nvcc..."
  make -j"$(nproc)"
  echo "[3/3] Running GPU Multi-Stream pipeline..."
  ./bin/spectral_pipeline \
      --input "$INPUT_DIR" \
      --output "$OUTPUT_DIR" \
      --streams "$STREAMS" \
      --threshold -35.0 \
    2>&1 | tee execution_log.txt
else
  echo "[2/3] NVCC not detected in current environment."
  echo "[3/3] Running Python verification pipeline..."
  python3 scripts/generate_artifacts.py 2>&1 | tee execution_log.txt
fi

echo ""
echo "Pipeline execution finished!"
echo "Spectrogram CSVs saved to : $OUTPUT_DIR"
echo "Execution log saved to    : execution_log.txt"
