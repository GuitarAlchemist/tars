@echo off
echo.
echo ╔════════════════════════════════════════╗
echo ║     Starting llama.cpp server          ║
echo ╚════════════════════════════════════════╝
echo.
echo Model: Qwen3-8B-Q4_K_M.gguf
echo Port:  8080
echo GPU:   All layers offloaded
echo.

C:\tools\llama-cpp\llama-server.exe -m "C:\models\Qwen3-8B-Q4_K_M.gguf" --host 127.0.0.1 --port 8080 -c 4096 -ngl 99

if errorlevel 1 (
    echo.
    echo ❌ llama.cpp failed to start. Trying CPU mode...
    C:\tools\llama-cpp\llama-server.exe -m "C:\models\Qwen3-8B-Q4_K_M.gguf" --host 127.0.0.1 --port 8080 -c 4096
)

pause
