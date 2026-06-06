#!/usr/bin/env pwsh

# Test script for enhanced TARS CLI chatbot
Write-Host "Testing Enhanced TARS CLI Chatbot..." -ForegroundColor Green

# Create test commands file
$testCommands = @"
help
version
status
moe status
llm models
agent status
flux status
diagnostics
exit
"@

# Save test commands to file
$testCommands | Out-File -FilePath "chatbot_test_commands.txt" -Encoding UTF8

Write-Host "`n=== Testing Enhanced CLI Chatbot ===" -ForegroundColor Yellow
Write-Host "Commands to test:" -ForegroundColor Cyan
Write-Host $testCommands -ForegroundColor White

Write-Host "`nStarting chatbot test..." -ForegroundColor Green
Write-Host "Note: The chatbot will run interactively. You can test the commands manually." -ForegroundColor Yellow

# Start the chatbot
dotnet run --project TarsEngine.FSharp.Cli -- chat
