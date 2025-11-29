#!/usr/bin/env pwsh
# TARS v2 - Quick Start Script with LLM Check

Write-Host ""
Write-Host "╭────────────────────────────────────────────╮" -ForegroundColor Cyan
Write-Host "│  TARS v2 Chat - Startup Assistant         │" -ForegroundColor Cyan
Write-Host "╰────────────────────────────────────────────╯" -ForegroundColor Cyan
Write-Host ""

Write-Host "🔍 Checking LLM service..." -ForegroundColor Yellow
Write-Host ""

# Check if Ollama is running
$ollamaRunning = $false
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        $ollamaRunning = $true
        Write-Host "✅ Ollama is running on port 11434" -ForegroundColor Green
        
        # List available models
        $models = ($response.Content | ConvertFrom-Json).models
        if ($models) {
            Write-Host "📦 Available models:" -ForegroundColor Cyan
            foreach ($model in $models) {
                Write-Host "   - $($model.name)" -ForegroundColor Gray
            }
        }
    }
}
catch {
    Write-Host "❌ Ollama is NOT running" -ForegroundColor Red
}

Write-Host ""

if (-not $ollamaRunning) {
    Write-Host "⚠️  TARS requires Ollama to be running!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To start Ollama:" -ForegroundColor Cyan
    Write-Host "   1. Open a new terminal" -ForegroundColor Gray
    Write-Host "   2. Run: ollama serve" -ForegroundColor Gray
    Write-Host "   3. In another terminal, run: ollama pull qwen2.5-coder:latest" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Then run this script again!" -ForegroundColor Cyan
    Write-Host ""
    
    $choice = Read-Host "Try to start Ollama automatically? (Y/n)"
    if ($choice -ne "n" -and $choice -ne "N") {
        Write-Host ""
        Write-Host "🚀 Starting Ollama..." -ForegroundColor Yellow
        Start-Process -FilePath "ollama" -ArgumentList "serve" -NoNewWindow
        Write-Host "Waiting for Ollama to start..." -ForegroundColor Gray
        Start-Sleep -Seconds 5
        
        # Check again
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 5 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Host "✅ Ollama started successfully!" -ForegroundColor Green
                $ollamaRunning = $true
            }
        }
        catch {
            Write-Host "❌ Failed to start Ollama automatically" -ForegroundColor Red
            Write-Host "Please start it manually: ollama serve" -ForegroundColor Yellow
            exit 1
        }
    }
    else {
        exit 1
    }
}

Write-Host ""
Write-Host "🚀 Starting TARS Chat..." -ForegroundColor Green
Write-Host ""

# Start TARS
dotnet run --project src/Tars.Interface.Cli -- chat
