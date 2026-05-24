import os
import sys
import argparse
from huggingface_hub import hf_hub_download, list_repo_files

# Enable faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Default models configuration
# Format: "GenericName": ("RepoID", "FilenamePattern")
RECOMMENDED_MODELS = {
    "qwen-14b": ("Qwen/Qwen2.5-14B-Instruct-GGUF", "qwen2.5-14b-instruct-q4_k_m.gguf"),
    "qwen-7b": ("Qwen/Qwen2.5-7B-Instruct-GGUF", "qwen2.5-7b-instruct-q4_k_m.gguf"),
    "deepseek-r1": ("ishikli/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    "llama3-8b": ("MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF", "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"),
    "phi3-mini": ("microsoft/Phi-3-mini-4k-instruct-gguf", "Phi-3-mini-4k-instruct-q4.gguf"),
}

def download_model(repo_id, filename, output_dir):
    print(f"⬇️  Downloading {filename} from {repo_id}...")
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir
        )
        print(f"✅ Successfully downloaded to: {file_path}")
        return file_path
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return None

def list_models():
    print("📋 Recommended Models for TARS:")
    for name, (repo, file) in RECOMMENDED_MODELS.items():
        print(f"  - {name:<15} : {repo} ({file})")

def main():
    parser = argparse.ArgumentParser(description="TARS Model Manager - Download GGUF models for local inference.")
    
    parser.add_argument("--list", action="store_true", help="List recommended models")
    parser.add_argument("--download", type=str, help="Name of the recommended model to download (e.g., qwen-14b)")
    parser.add_argument("--repo", type=str, help="Custom HuggingFace Repo ID")
    parser.add_argument("--file", type=str, help="Custom GGUF filename")
    parser.add_argument("--out", type=str, default="models", help="Output directory (default: models/)")
    
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    if args.download:
        if args.download in RECOMMENDED_MODELS:
            repo, filename = RECOMMENDED_MODELS[args.download]
            download_model(repo, filename, args.out)
        else:
            print(f"❌ Unknown model alias: {args.download}")
            print("Use --list to see available aliases.")
    
    elif args.repo and args.file:
        download_model(args.repo, args.file, args.out)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
