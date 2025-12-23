#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    TARS v2 Container                        ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Start Ollama in background
echo "Starting Ollama..."
ollama serve &
sleep 3

# Pull default model if not present
if ! ollama list | grep -q "qwen3"; then
    echo "Pulling default model (qwen3:8b)..."
    ollama pull qwen3:8b
fi

# Pull embedding model
if ! ollama list | grep -q "nomic-embed-text"; then
    echo "Pulling embedding model..."
    ollama pull nomic-embed-text
fi

echo "Starting TARS UI on port 5000..."
exec dotnet Tars.Interface.Ui.dll
