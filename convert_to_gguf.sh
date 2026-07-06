#!/bin/bash

# ==============================================================================
# Script for converting Hugging Face model to GGUF and quantizing to 4-bit (Q4_K_M)
# Target Model: Phonsiri/Gemma-4-E4B-it-PARL
# Designed to run on Cloud environments (Google Colab, RunPod, Vast.ai, etc.)
# ==============================================================================

# Hugging Face Configuration (Fill these in to automatically upload the converted model)
HF_USERNAME="Phonsiri"                 # Your Hugging Face username
HF_TOKEN=""                            # Your Hugging Face write token (starts with hf_...)
HF_REPO_NAME="Gemma-4-E4B-it-PARL-GGUF" # Name of the target Hugging Face repository to push to

# Exit immediately if a command exits with a non-zero status
set -e

MODEL_ID="Phonsiri/Gemma-4-E4B-it-PARL"
MODEL_NAME="Gemma-4-E4B-it-PARL"
QUANT_TYPE="Q4_K_M" # Recommended 4-bit quantization method

echo "=== 1. Installing System Dependencies and Python Packages ==="
# Update and install system dependencies if needed (for Debian/Ubuntu based cloud instances)
if [ -x "$(command -v apt-get)" ]; then
    sudo apt-get update && sudo apt-get install -y git build-essential python3-pip python3-venv cmake
fi

# Create a virtual environment to keep things clean
python3 -m venv venv
source venv/bin/activate

# Install required python packages including cmake
pip install --upgrade pip
pip install huggingface_hub torch transformers accelerator sentencepiece cmake

echo "=== 2. Cloning llama.cpp and Compiling via CMake ==="
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
fi
cd llama.cpp

# Install llama.cpp conversion dependencies
pip install -r requirements.txt

# Compile llama.cpp using CMake (Makefile has been deprecated in llama.cpp)
cmake -B build
cmake --build build --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)

cd ..

echo "=== 3. Downloading Model from Hugging Face ==="
# Download the model weights and config files from HF Hub
python3 -c "
from huggingface_hub import snapshot_download
print('Downloading model $MODEL_ID...')
snapshot_download(
    repo_id='$MODEL_ID',
    local_dir='./$MODEL_NAME',
    local_dir_use_symlinks=False,
    ignore_patterns=['*.gguf', '*.bin'] # Skip any existing GGUF or large custom bin formats to save bandwidth
)
print('Download complete!')
"

echo "=== 4. Converting Model to f16 GGUF Format ==="
# Run llama.cpp conversion script
python3 llama.cpp/convert_hf_to_gguf.py ./$MODEL_NAME \
    --outfile ./${MODEL_NAME}-f16.gguf \
    --outtype f16

echo "=== 5. Quantizing model to 4-bit ($QUANT_TYPE) ==="
# Dynamically locate the llama-quantize binary built by CMake
QUANTIZE_BIN=$(find llama.cpp/build -name "llama-quantize" -type f -print -quit 2>/dev/null || echo "")
if [ -z "$QUANTIZE_BIN" ]; then
    QUANTIZE_BIN="./llama.cpp/build/bin/llama-quantize"
fi

echo "Using llama-quantize binary at: $QUANTIZE_BIN"
# Run llama-quantize
$QUANTIZE_BIN ./${MODEL_NAME}-f16.gguf ./${MODEL_NAME}-${QUANT_TYPE}.gguf $QUANT_TYPE

echo "=== 6. Conversion Complete! ==="
echo "Converted model saved to: ./${MODEL_NAME}-${QUANT_TYPE}.gguf"
ls -lh ./${MODEL_NAME}-${QUANT_TYPE}.gguf

echo "=== 7. Uploading Model to Hugging Face ==="
if [ -n "$HF_TOKEN" ] && [ -n "$HF_USERNAME" ] && [ -n "$HF_REPO_NAME" ]; then
    python3 -c "
from huggingface_hub import HfApi
api = HfApi()
repo_id = '$HF_USERNAME/$HF_REPO_NAME'
print(f'Creating repo {repo_id} if it does not exist...')
api.create_repo(repo_id=repo_id, repo_type='model', exist_ok=True, token='$HF_TOKEN')
print(f'Uploading ${MODEL_NAME}-${QUANT_TYPE}.gguf to HF Hub Repository {repo_id}...')
api.upload_file(
    path_or_fileobj='./${MODEL_NAME}-${QUANT_TYPE}.gguf',
    path_in_repo='${MODEL_NAME}-${QUANT_TYPE}.gguf',
    repo_id=repo_id,
    repo_type='model',
    token='$HF_TOKEN'
)
print('Upload complete!')
"
else
    echo "Hugging Face credentials (HF_USERNAME, HF_TOKEN, or HF_REPO_NAME) are empty."
    echo "Skipping automatic upload. You can manually upload the file ./${MODEL_NAME}-${QUANT_TYPE}.gguf"
fi
