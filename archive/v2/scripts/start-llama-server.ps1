# Start llama-server with Magistral model for maximum GPU performance
# Run this after the model download completes

$llamaServerPath = "C:\Users\spare\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
$modelPath = "C:\Users\spare\source\repos\tars\v2\models\magistral\Magistral-Small-2507-Q4_K_M.gguf"

Write-Host "Starting llama-server with maximum GPU performance..." -ForegroundColor Green
Write-Host "  Model: Magistral-Small Q4_K_M (14GB)" -ForegroundColor Cyan
Write-Host "  GPU Layers: ALL (-1 = offload everything to GPU)" -ForegroundColor Cyan
Write-Host "  Parallel Slots: 1" -ForegroundColor Cyan
Write-Host "  Context Size: 8192 tokens" -ForegroundColor Cyan
Write-Host "  Port: 8080" -ForegroundColor Cyan
Write-Host ""

& $llamaServerPath `
    -m $modelPath `
    --port 8080 `
    -ngl -1 `
    -np 1 `
    -c 8192 `
    --host 0.0.0.0 `
    --n-predict -1 `
    --verbose
