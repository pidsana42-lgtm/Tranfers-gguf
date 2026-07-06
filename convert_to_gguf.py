#!/usr/bin/env python3
"""
Python script to convert Phonsiri/Gemma-4-E4B-it-PARL to GGUF format 
and perform 4-bit quantization (Q4_K_M) on a cloud server.
Now includes automatic vision projector (mmproj) extraction and upload.
"""

import os
import subprocess
import sys
import json

# Hugging Face Configuration (Fill these in to automatically upload the converted model)
HF_USERNAME = "Phonsiri"                      # Your Hugging Face username
HF_TOKEN = ""                                 # Your Hugging Face write token (starts with hf_...)
HF_REPO_NAME = "Gemma-4-E4B-it-PARL-GGUF"     # Name of the target Hugging Face repository to push to

MODEL_ID = "Phonsiri/Gemma-4-E4B-it-PARL"
MODEL_NAME = "Gemma-4-E4B-it-PARL"
QUANT_TYPE = "Q4_K_M"

def run_command(command, cwd=None):
    print(f"\nExecuting: {command}")
    process = subprocess.Popen(command, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"Error: Command failed with exit code {process.returncode}")
        sys.exit(1)

def patch_tokenizer_config():
    tokenizer_config_path = os.path.join(MODEL_NAME, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        print(f"\n=== Patching {tokenizer_config_path} for compatibility ===")
        try:
            with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            
            patched = False
            # Fix 'extra_special_tokens' if it's a list instead of a dict
            if "extra_special_tokens" in config_data:
                if isinstance(config_data["extra_special_tokens"], list):
                    print("Found 'extra_special_tokens' as a list. Converting to an empty dict to prevent AttributeError...")
                    config_data["extra_special_tokens"] = {}
                    patched = True
                    
            if patched:
                with open(tokenizer_config_path, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                print("Successfully patched tokenizer_config.json!")
            else:
                print("No patching required.")
        except Exception as e:
            print(f"Warning: Failed to patch tokenizer_config.json: {e}")

def main():
    print("=== Step 1: Installing dependencies ===")
    run_command("pip install --upgrade pip")
    run_command("pip install huggingface_hub torch transformers accelerator sentencepiece cmake")

    print("\n=== Step 2: Cloning and compiling llama.cpp via CMake ===")
    if not os.path.exists("llama.cpp"):
        run_command("git clone https://github.com/ggerganov/llama.cpp.git")
    
    # Install conversion requirements
    run_command("pip install -r requirements.txt", cwd="llama.cpp")
    
    # Compile llama.cpp using CMake
    run_command("cmake -B build", cwd="llama.cpp")
    import multiprocessing
    cores = multiprocessing.cpu_count()
    run_command(f"cmake --build build --config Release -j {cores}", cwd="llama.cpp")

    print("\n=== Step 3: Downloading model from Hugging Face ===")
    from huggingface_hub import snapshot_download
    print(f"Downloading {MODEL_ID} to ./{MODEL_NAME}...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=f"./{MODEL_NAME}",
        local_dir_use_symlinks=False,
        ignore_patterns=["*.gguf", "*.bin"]
    )
    print("Download finished.")

    # Patch tokenizer before converting to avoid AttributeError in transformers
    patch_tokenizer_config()

    print("\n=== Step 4: Converting HF model to GGUF f16 ===")
    convert_script = os.path.join("llama.cpp", "convert_hf_to_gguf.py")
    output_f16 = f"./{MODEL_NAME}-f16.gguf"
    run_command(f"python3 {convert_script} ./{MODEL_NAME} --outfile {output_f16} --outtype f16")

    print("\n=== Step 4b: Extracting and Converting Multimodal Vision Projector (mmproj) ===")
    output_mmproj = f"./{MODEL_NAME}-mmproj.gguf"
    run_command(f"python3 {convert_script} ./{MODEL_NAME} --mmproj --outfile {output_mmproj}")

    print(f"\n=== Step 5: Quantizing to 4-bit ({QUANT_TYPE}) ===")
    # Dynamically find the llama-quantize binary in the build directory
    quantize_bin = None
    build_dir = os.path.join("llama.cpp", "build")
    for root, dirs, files in os.walk(build_dir):
        for file in files:
            if file == "llama-quantize" or file == "llama-quantize.exe":
                quantize_bin = os.path.join(root, file)
                break
        if quantize_bin:
            break
            
    if not quantize_bin:
        # Fallback to default CMake output path
        quantize_bin = os.path.join("llama.cpp", "build", "bin", "llama-quantize")
        
    print(f"Found llama-quantize at: {quantize_bin}")
    output_quant = f"./{MODEL_NAME}-{QUANT_TYPE}.gguf"
    run_command(f"{quantize_bin} {output_f16} {output_quant} {QUANT_TYPE}")

    print("\n=== Conversion Completed Successfully! ===")
    print(f"Final Quantized GGUF model: {output_quant}")
    if os.path.exists(output_quant):
        size_gb = os.path.getsize(output_quant) / (1024 * 1024 * 1024)
        print(f"File size: {size_gb:.2f} GB")
    if os.path.exists(output_mmproj):
        size_mb = os.path.getsize(output_mmproj) / (1024 * 1024)
        print(f"Vision Projector size: {size_mb:.2f} MB")

    print("\n=== Step 6: Uploading to Hugging Face ===")
    if HF_USERNAME and HF_TOKEN and HF_REPO_NAME:
        from huggingface_hub import HfApi
        api = HfApi()
        repo_id = f"{HF_USERNAME}/{HF_REPO_NAME}"
        print(f"Creating repo {repo_id} if it does not exist...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=HF_TOKEN)
        
        # 1. Upload main language model GGUF (4-bit)
        print(f"Uploading {output_quant} to Hugging Face Repository {repo_id}...")
        api.upload_file(
            path_or_fileobj=output_quant,
            path_in_repo=f"{MODEL_NAME}-{QUANT_TYPE}.gguf",
            repo_id=repo_id,
            repo_type="model",
            token=HF_TOKEN
        )
        
        # 2. Upload vision projector GGUF (mmproj)
        if os.path.exists(output_mmproj):
            print(f"Uploading vision projector {output_mmproj} to Hugging Face Repository {repo_id}...")
            api.upload_file(
                path_or_fileobj=output_mmproj,
                path_in_repo=f"{MODEL_NAME}-mmproj.gguf",
                repo_id=repo_id,
                repo_type="model",
                token=HF_TOKEN
            )
            
        print("Upload completed successfully!")
    else:
        print("Hugging Face credentials (HF_USERNAME, HF_TOKEN, or HF_REPO_NAME) are empty.")
        print("Skipping automatic upload.")

if __name__ == "__main__":
    main()
