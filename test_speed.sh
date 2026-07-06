#!/bin/bash

# ==============================================================================
# Script to benchmark the GGUF model speed using CPU and RAM only
# ==============================================================================

MODEL_PATH="./Gemma-4-E4B-it-PARL-Q4_K_M.gguf"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file $MODEL_PATH not found."
    echo "Please make sure the conversion finished and the GGUF file exists in this directory."
    exit 1
fi

# Detect CPU cores to optimize thread usage
CORES=$(nproc 2>/dev/null || sysctl -n hw.physicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
# Optimal thread count is usually the number of physical cores
THREADS=$((CORES > 8 ? 8 : CORES))

echo "=== System Info ==="
echo "Detected CPU Cores: $CORES"
echo "Recommended threads for inference: $THREADS"
echo "Using model: $MODEL_PATH"
echo "--------------------------------------------------------"

# Find llama-cli and llama-bench
CLI_BIN=$(find llama.cpp/build -name "llama-cli" -type f -print -quit 2>/dev/null || echo "")
BENCH_BIN=$(find llama.cpp/build -name "llama-bench" -type f -print -quit 2>/dev/null || echo "")

if [ -z "$CLI_BIN" ]; then
    CLI_BIN="./llama.cpp/build/bin/llama-cli"
fi

if [ -z "$BENCH_BIN" ]; then
    BENCH_BIN="./llama.cpp/build/bin/llama-bench"
fi

echo "=== TEST 1: Running Speed Benchmark (llama-bench) ==="
if [ -f "$BENCH_BIN" ]; then
    # -ngl 0: offloads 0 layers to GPU (runs 100% on CPU / RAM)
    # -t: threads
    # -m: model path
    $BENCH_BIN -m "$MODEL_PATH" -ngl 0 -t $THREADS -p 512 -n 128
else
    echo "llama-bench binary not found. Skipping Test 1."
fi

echo ""
echo "=== TEST 2: Running Prompt Generation Test (llama-cli) ==="
if [ -f "$CLI_BIN" ]; then
    # Run a real prompt to see text output and speed stats
    # -n 128: Limit response length to 128 tokens
    # --predict: number of tokens to predict
    $CLI_BIN -m "$MODEL_PATH" \
        -p "สวัสดีครับ ช่วยบอกความสามารถเด่น 3 ข้อของคุณในฐานะโมเดลภาษาภาษาไทย" \
        -ngl 0 \
        -t $THREADS \
        -n 128
else
    echo "llama-cli binary not found. Skipping Test 2."
fi
