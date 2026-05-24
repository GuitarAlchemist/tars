# RTX 5080 Optimization Report for TARS v2

**Date:** December 25, 2024
**Hardware:** NVIDIA GeForce RTX 5080 (16GB VRAM, Blackwell Architecture)
**Objective:** Optimize LLM backend performance for >75 tok/s.

## 🏆 Final Result
**Achieved Speed:** **151.0 tok/s** 🚀
**Configuration:**
- **Provider:** Ollama
- **Model:** `mistral:7b`
- **Context Window:** 4096 (Strictly enforced)
- **Status:** ✅ Passing "River Crossing" Logic Puzzle (Score: 100%)

---

## 🔍 Findings & Diagnosis

### 1. Blackwell Architecture Compatibility
The RTX 5080 (Compute Capability 12.0) is not yet fully supported by standard pre-built `llama.cpp` binaries compiled with CUDA 11/12. Usage resulted in either crashes or CPU fallback (18 tok/s).
**Solution:** `Ollama` provided a more compatible runtime that successfully utilizes the GPU.

### 2. VRAM Saturation & CPU Thrashing
Initial tests showed speeds of **12.3 tok/s** despite high GPU utilization.
- **Root Cause:** Ollama defaults to the context window defined in the model file (typically 32k). 
- **Impact:** `Mistral 7B` (4.5GB) + 32k Context KV Cache (>12GB) exceeded the 16GB VRAM limit, causing massive swapping to system RAM.
- **Diagnosis:** `ollama ps` showed `67%/33% CPU/GPU` split.

### 3. Routing Configuration Bug
TARS was not correctly propagating the configured `ContextWindow` from `appsettings.json` to the internal `RoutingConfig`.
- **Bug:** `DefaultContextWindow` was defaulting to `None`.
- **Effect:** Ollama received no `num_ctx` parameter and used the default 32k, causing the VRAM spill.

---

## 🛠️ Implementation Details

### Optimized Configuration (`appsettings.json`)
```json
"Llm": {
    "Provider": "Ollama",
    "Model": "mistral:7b",
    "ContextWindow": 4096,  // CRITICAL: Fits model + KV cache in 16GB
    "Temperature": 0.2
}
```

### Code Fixes
Updated `RoutingConfig` initialization in:
1. `src/Tars.Interface.Cli/Commands/PuzzleDemo.fs`
2. `src/Tars.Interface.Cli/Commands/Evolve.fs`
3. `src/Tars.Interface.Cli/Commands/NeuroSymbolicBenchmark.fs`
To explicitly pass:
```fsharp
DefaultContextWindow = if config.Llm.ContextWindow > 0 then Some config.Llm.ContextWindow else None
```

### UX Improvements
- **Warmup Phase:** Added to `PuzzleDemo` to ensure model is loaded before measuring speed.
- **Reasoning Display:** Unlocked full reasoning output in `PuzzleDemo` for better transparency.

---

## 📊 Performance Benchmarks

| Configuration | Speed | Result | Notes |
| :--- | :--- | :--- | :--- |
| **Mistral 7B (Optimized)** | **151.0 tok/s** | ✅ PASS | **Recommended Daily Driver** |
| Llama 3.2 3B | 274.0 tok/s | ❌ FAIL | Too small for reasoning tasks |
| Mistral 7B (Unoptimized) | 12.3 tok/s | ✅ PASS | CPU/RAM Bottleneck (32k context) |
| Llama Server (Ext) | 18.3 tok/s | ✅ PASS | CUDA 12.4 Binary Compat Mode |

## 📝 Recommendations
For current RTX 5080 hardware:
1. Always use **Ollama** backend.
2. Keep **Context Window <= 4096** for 7B models.
3. Use **Mistral 7B** for best balance of speed and intelligence.
