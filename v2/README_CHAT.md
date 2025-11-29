# TARS v2 Chat - Quick Start Guide

## ⚠️ Important: LLM Service Required

TARS chat requires a running LLM service (Ollama by default).

## How to Start TARS Chat

### Option 1: Using the Startup Script (Recommended)

```powershell
# From c:\Users\spare\source\repos\tars\v2
.\start-chat.ps1
```

The script will:

- Check if Ollama is running
- Offer to start it automatically if not
- Launch TARS chat when ready

### Option 2: Manual Setup

#### Step 1: Start Ollama

Open a **separate terminal** and run:

```powershell
ollama serve
```

Leave this terminal open!

#### Step 2: Pull a Model (First Time Only)

In another terminal:

```powershell
ollama pull qwen2.5-coder:latest
```

#### Step 3: Start TARS Chat

```powershell
cd c:\Users\spare\source\repos\tars\v2
dotnet run --project src/Tars.Interface.Cli -- chat
```

## 🐛 Troubleshooting

### Error: "Cannot connect to LLM service"

**Cause**: Ollama is not running

**Solution**:

1. Open a new terminal
2. Run: `ollama serve`
3. Wait a few seconds
4. Try starting TARS chat again

### Error: "Model not found"

**Cause**: The model hasn't been downloaded

**Solution**:

```powershell
ollama pull qwen2.5-coder:latest
```

### Check Ollama Status

```powershell
# List downloaded models
ollama list

# Test Ollama is responding
curl http://localhost:11434/api/tags
```

## 📝 Other TARS Commands

```powershell
# Demo ping (no LLM required)
dotnet run --project src/Tars.Interface.Cli -- demo-ping

# Run a metascript
dotnet run --project src/Tars.Interface.Cli -- run sample.tars

# Evolution loop
dotnet run --project src/Tars.Interface.Cli -- evolve --max-iterations 5
```

## ⚙️ Configuration

TARS is configured to use:

- **Ollama** on `http://localhost:11434/`
- **Default Model**: `qwen2.5-coder:latest`

To change this, edit: `src/Tars.Interface.Cli/Commands/Chat.fs` (lines 27-38)

## 🎯 What We Built Today

- ✅ SafetyGate module with Docker sandbox execution  
- ✅ Constraint enforcement in EventBus
- ✅ Universal versioning for all entities
- ✅ Comprehensive architecture documentation
- ✅ 39/39 tests passing

See `demos/architecture_showcase.ps1` for a visual summary!
