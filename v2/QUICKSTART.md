# TARS v2 - Quick Reference

## 🚀 One-Line Setup (Windows)

```powershell
# Run as Administrator
irm https://raw.githubusercontent.com/GuitarAlchemist/tars/main/v2/scripts/setup-tars.ps1 | iex
```

## 📋 Prerequisites Checklist

| Component | Command to Install | Verify |
|-----------|-------------------|--------|
| .NET 10 SDK | `winget install Microsoft.DotNet.SDK.10` | `dotnet --version` |
| Git | `winget install Git.Git` | `git --version` |
| Ollama | `winget install Ollama.Ollama` | `ollama --version` |
| CUDA (GPU) | `winget install Nvidia.CUDA` | `nvidia-smi` |

## 🎯 Quick Start

```powershell
# Clone and build
git clone https://github.com/GuitarAlchemist/tars.git
cd tars/v2
dotnet build

# Option 1: Start everything
.\start-all.bat

# Option 2: Manual start
# Terminal 1: .\start-llama.bat
# Terminal 2: .\start-tars.bat

# Open browser
start http://localhost:5000
```

## 📦 Model Downloads

```powershell
# Recommended models via Ollama
ollama pull qwen3:14b          # Best thinking model
ollama pull deepseek-r1:14b    # Best for math/reasoning
ollama pull magistral          # Fastest inference
ollama pull nomic-embed-text   # Embeddings

# Direct GGUF download for llama.cpp
curl -L -o C:\models\Qwen3-8B-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf
```

## 🐳 Docker Setup

```bash
# Build and run
docker-compose up -d

# With GPU support
docker-compose --profile gpu up -d
```

## ⚡ Performance Tiers

| Backend | Speed | Setup Complexity |
|---------|-------|------------------|
| llama.cpp | ~100 TPS | Medium |
| Ollama | ~60 TPS | Easy |
| Cloud APIs | Variable | API Key needed |

## 🔧 Environment Variables

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:ANTHROPIC_API_KEY = "..."
$env:GOOGLE_API_KEY = "..."
```

## 🌐 Default Ports

| Service | Port | URL |
|---------|------|-----|
| TARS UI | 5000 | http://localhost:5000 |
| llama.cpp | 8080 | http://localhost:8080 |
| Ollama | 11434 | http://localhost:11434 |

## 🆘 Troubleshooting

```powershell
# Check Ollama status
ollama list
ollama serve  # Start if not running

# Check llama.cpp
curl http://localhost:8080/health

# Check TARS
curl http://localhost:5000

# View logs
dotnet run --project src/Tars.Interface.Ui/Tars.Interface.Ui.fsproj --verbosity detailed
```

## 📁 Key Paths

```
C:\tools\llama-cpp\     # llama.cpp binaries
C:\models\              # GGUF model files
~\.ollama\              # Ollama models & config
~\.tars\skills\         # Agent Skills
```

## 🔗 Useful Links

- [TARS Docs](docs/README.md)
- [Setup Guide](docs/SETUP.md)
- [Ollama Models](https://ollama.com/library)
- [HuggingFace GGUF](https://huggingface.co/models?sort=trending&search=gguf)
