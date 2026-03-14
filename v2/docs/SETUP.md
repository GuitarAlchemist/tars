# TARS v2 - Setup Guide

This guide covers setting up TARS from scratch on a fresh Windows machine.

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Windows | 10/11 | 64-bit required |
| RAM | 16GB+ | 32GB recommended for larger models |
| GPU | NVIDIA RTX 3060+ | For fast inference (optional but recommended) |
| Disk | 50GB+ free | For models and dependencies |

## Quick Start (Automated)

Run this single command in **PowerShell as Administrator**:

```powershell
irm https://raw.githubusercontent.com/GuitarAlchemist/tars/main/scripts/setup-tars.ps1 | iex
```

Or manually download and run:

```powershell
.\scripts\setup-tars.ps1
```

---

## Manual Setup

### Step 1: Install Package Manager

**Option A: Winget (Recommended - Built into Windows 11)**
```powershell
# Winget is pre-installed on Windows 11
# For Windows 10, install from Microsoft Store: "App Installer"
winget --version
```

**Option B: Chocolatey**
```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

### Step 2: Install .NET 10 SDK

```powershell
# Using Winget
winget install Microsoft.DotNet.SDK.10

# Or using Chocolatey
choco install dotnet-sdk -y

# Verify
dotnet --version
```

### Step 3: Install Git

```powershell
# Using Winget
winget install Git.Git

# Or using Chocolatey
choco install git -y

# Verify
git --version
```

### Step 4: Install NVIDIA CUDA (For GPU Acceleration)

```powershell
# Using Winget
winget install Nvidia.CUDA

# Or download from: https://developer.nvidia.com/cuda-downloads
# Verify
nvidia-smi
```

### Step 5: Install Ollama (Recommended Model Runner)

```powershell
# Using Winget
winget install Ollama.Ollama

# Or download from: https://ollama.com/download
# Verify
ollama --version
```

### Step 6: Install llama.cpp (Optional - Faster Inference)

```powershell
# Create tools directory
mkdir C:\tools\llama-cpp -Force

# Download latest release (CUDA version)
$release = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
$asset = $release.assets | Where-Object { $_.name -like "*win-cuda-12.4*" -and $_.name -notlike "*cudart*" } | Select-Object -First 1
Invoke-WebRequest -Uri $asset.browser_download_url -OutFile "$env:TEMP\llama.zip"
Expand-Archive "$env:TEMP\llama.zip" -DestinationPath "C:\tools\llama-cpp" -Force

# Download CUDA runtime
$cudart = $release.assets | Where-Object { $_.name -like "*cudart*cuda-12.4*" } | Select-Object -First 1
Invoke-WebRequest -Uri $cudart.browser_download_url -OutFile "$env:TEMP\cudart.zip"
Expand-Archive "$env:TEMP\cudart.zip" -DestinationPath "C:\tools\llama-cpp" -Force

# Add to PATH
$env:Path += ";C:\tools\llama-cpp"
[Environment]::SetEnvironmentVariable("Path", $env:Path, "User")
```

### Step 7: Install Docker Desktop (For Sandbox)

```powershell
# Using Winget
winget install Docker.DockerDesktop

# After installation, restart and enable WSL2 if prompted
```

### Step 8: Clone TARS Repository

```powershell
git clone https://github.com/GuitarAlchemist/tars.git
cd tars/v2
```

### Step 9: Build TARS

```powershell
dotnet restore
dotnet build
```

### Step 10: Download Models

**For Ollama (Recommended):**
```powershell
# Best thinking model for reasoning
ollama pull qwen3:14b

# Best for math/logic
ollama pull deepseek-r1:14b

# Fast inference
ollama pull magistral

# Embedding model
ollama pull nomic-embed-text
```

**For llama.cpp (Fastest):**
```powershell
mkdir C:\models -Force
# Download Qwen3-8B (5GB, fast)
curl -L -o "C:\models\Qwen3-8B-Q4_K_M.gguf" "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
```

### Step 11: Start Services

**Terminal 1 - llama.cpp Server (Optional):**
```powershell
C:\tools\llama-cpp\llama-server.exe -m "C:\models\Qwen3-8B-Q4_K_M.gguf" --host 127.0.0.1 --port 8080 -c 4096 -ngl 99
```

**Terminal 2 - TARS UI:**
```powershell
cd tars/v2
dotnet run --project src/Tars.Interface.Ui/Tars.Interface.Ui.fsproj
```

**Access TARS:** http://localhost:5000

---

## Docker Setup (Alternative)

For a fully containerized setup:

```powershell
# Build the TARS container
docker build -t tars:latest .

# Run with GPU support
docker run --gpus all -p 5000:5000 -p 11434:11434 tars:latest
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | - |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude | - |
| `GOOGLE_API_KEY` | Google API key for Gemini | - |
| `TARS_LLM_BACKEND` | Default backend: `ollama`, `llamacpp`, `openai` | `llamacpp` |

### Model Configuration

Edit `src/Tars.Interface.Ui/Program.fs` to change default models:

```fsharp
DefaultOllamaModel = "qwen3:14b"       // For Ollama
DefaultLlamaCppModel = Some "Qwen3-8B-Q4_K_M.gguf"  // For llama.cpp
```

---

## Troubleshooting

### "CUDA not found"
- Install NVIDIA CUDA Toolkit 12.4+
- Restart your terminal after installation

### "Model not found"
- Ensure Ollama is running: `ollama serve`
- Pull the model: `ollama pull qwen3:14b`

### "Port 5000 in use"
- Stop other applications using port 5000
- Or change the port in `launchSettings.json`

### "llama-server crashes"
- Use CPU version instead of CUDA
- Reduce context size: `-c 2048` instead of `-c 4096`

---

## Performance Tips

1. **Use llama.cpp for fastest inference** (~1.8x faster than Ollama)
2. **GPU offload all layers**: `-ngl 99`
3. **Use quantized models**: Q4_K_M offers best speed/quality balance
4. **Flash Attention**: Enabled automatically on compatible GPUs

---

## Next Steps

1. Open http://localhost:5000
2. Go to **Chat** and test the LLM
3. Go to **Tools** to explore available capabilities
4. Check **Agents** for registered system agents
5. Scan your codebase in **Knowledge**

For more information, see the [TARS Documentation](docs/README.md).
