#!/usr/bin/env pwsh

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   TARS Interactive Chatbot Demo" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This demonstrates something " -NoNewline
Write-Host "EXTRAORDINARY" -ForegroundColor Yellow -NoNewline
Write-Host ":"
Write-Host "A real AI chatbot using MoE system for intelligent task execution!" -ForegroundColor Green
Write-Host ""
Write-Host "Features:" -ForegroundColor Yellow
Write-Host "  ✅ Interactive conversation with TARS AI" -ForegroundColor Green
Write-Host "  ✅ Intelligent task routing to expert models" -ForegroundColor Green
Write-Host "  ✅ Real-time MoE system integration" -ForegroundColor Green
Write-Host "  ✅ Command execution and system analysis" -ForegroundColor Green
Write-Host "  ✅ Natural language processing" -ForegroundColor Green
Write-Host ""
Write-Host "Commands you can try:" -ForegroundColor Yellow
Write-Host "  • " -NoNewline -ForegroundColor Cyan
Write-Host '"moe status"' -NoNewline -ForegroundColor White
Write-Host " - Check expert system status" -ForegroundColor Gray
Write-Host "  • " -NoNewline -ForegroundColor Cyan
Write-Host '"list agents"' -NoNewline -ForegroundColor White
Write-Host " - Show available AI agents" -ForegroundColor Gray
Write-Host "  • " -NoNewline -ForegroundColor Cyan
Write-Host '"analyze datastore"' -NoNewline -ForegroundColor White
Write-Host " - Analyze in-memory data" -ForegroundColor Gray
Write-Host "  • " -NoNewline -ForegroundColor Cyan
Write-Host '"run demo transformer"' -NoNewline -ForegroundColor White
Write-Host " - Execute demos" -ForegroundColor Gray
Write-Host "  • " -NoNewline -ForegroundColor Cyan
Write-Host '"help me solve a complex reasoning problem"' -NoNewline -ForegroundColor White
Write-Host " - MoE routing" -ForegroundColor Gray
Write-Host "  • " -NoNewline -ForegroundColor Cyan
Write-Host '"translate this to Chinese"' -NoNewline -ForegroundColor White
Write-Host " - Multilingual expert" -ForegroundColor Gray
Write-Host "  • " -NoNewline -ForegroundColor Cyan
Write-Host '"create an autonomous agent"' -NoNewline -ForegroundColor White
Write-Host " - Agentic expert" -ForegroundColor Gray
Write-Host "  • " -NoNewline -ForegroundColor Cyan
Write-Host '"help"' -NoNewline -ForegroundColor White
Write-Host " - Show all commands" -ForegroundColor Gray
Write-Host "  • " -NoNewline -ForegroundColor Cyan
Write-Host '"exit"' -NoNewline -ForegroundColor White
Write-Host " - Exit chatbot" -ForegroundColor Gray
Write-Host ""
Write-Host "Starting TARS Interactive Chatbot..." -ForegroundColor Yellow
Write-Host ""

# Start the chatbot in a new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "dotnet run chat"

Write-Host ""
Write-Host "Chatbot launched in new window!" -ForegroundColor Green
Write-Host ""
Write-Host "Try these example conversations:" -ForegroundColor Yellow
Write-Host "  1. Ask for MoE status" -ForegroundColor Cyan
Write-Host "  2. Request system analysis" -ForegroundColor Cyan
Write-Host "  3. Ask for reasoning help" -ForegroundColor Cyan
Write-Host "  4. Request multilingual translation" -ForegroundColor Cyan
Write-Host "  5. Ask to create agents" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
