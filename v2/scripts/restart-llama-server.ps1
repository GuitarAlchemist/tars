# Restart llama-cpp server with MAXIMUM context size for TARS evolution

Write-Host "🔄 Restarting llama-server with MAXIMUM context..." -ForegroundColor Cyan

# Stop existing llama-server
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

# Maximum context size - Qwen3-8B supports up to 32K
Write-Host "Starting llama-server with 32768 context size (32K tokens)..." -ForegroundColor Green

Start-Process -FilePath "C:\tools\llama-cpp\llama-server.exe" `
    -ArgumentList "-m `"C:\models\Qwen3-8B-Q4_K_M.gguf`" --host localhost --port 11434 --ctx-size 32768 --batch-size 2048 --threads 12 --n-gpu-layers 35 --cont-batching" `
    -WindowStyle Hidden

Start-Sleep -Seconds 8
Write-Host "✅ Server restarted with 32K token context window" -ForegroundColor Green
Write-Host "   (16x larger than default - supports ~24,000 words)" -ForegroundColor Cyan
Write-Host "   This should handle any tool output size!" -ForegroundColor Yellow
