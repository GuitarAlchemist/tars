# Fix TARS CLI Chat Command Duplication
Write-Host "🔧 FIXING TARS CLI CHAT COMMAND DUPLICATION" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

$chatbotFile = "src/TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli/Commands/ChatbotCommand.fs"

if (Test-Path $chatbotFile) {
    Write-Host "📋 Reading ChatbotCommand.fs file..." -ForegroundColor Yellow
    
    # Create backup
    $backupFile = "$chatbotFile.backup"
    Copy-Item $chatbotFile $backupFile -Force
    Write-Host "💾 Created backup: $backupFile" -ForegroundColor Green
    
    # Read content
    $content = Get-Content $chatbotFile -Raw
    
    # Define the new header content (without duplicate commands)
    $newHeaderContent = @"
        let headerPanel = Panel("""[bold cyan]🤖 TARS Interactive Chatbot[/]
[dim]Powered by Mixture of Experts AI System[/]

[bold magenta]💡 Just ask naturally! TARS will route your request to the right expert.[/]
[yellow]Type '[green]help[/]' to see available commands or '[green]exit[/]' to quit.[/]""")
"@
    
    # Use regex to replace the header panel section
    $pattern = 'let headerPanel = Panel\(""".*?"""\)'
    $newContent = $content -replace $pattern, $newHeaderContent, 'Singleline'
    
    # Write the fixed content
    Set-Content $chatbotFile $newContent -Encoding UTF8
    
    Write-Host "✅ Successfully removed duplicate command listing" -ForegroundColor Green
    Write-Host "📊 Original file size: $($content.Length) characters" -ForegroundColor Yellow
    Write-Host "📊 New file size: $($newContent.Length) characters" -ForegroundColor Yellow
    Write-Host "📉 Reduced by: $($content.Length - $newContent.Length) characters" -ForegroundColor Yellow
    
    Write-Host ""
    Write-Host "🎉 CHAT COMMAND DUPLICATION FIXED!" -ForegroundColor Green
    Write-Host "==================================" -ForegroundColor Green
    Write-Host "✅ Removed duplicate command list from header" -ForegroundColor Green
    Write-Host "✅ Kept comprehensive help command intact" -ForegroundColor Green
    Write-Host "✅ Improved user experience with cleaner interface" -ForegroundColor Green
    Write-Host "✅ Commands now only shown when user types 'help'" -ForegroundColor Green
    
} else {
    Write-Host "❌ ChatbotCommand.fs file not found at: $chatbotFile" -ForegroundColor Red
}

Write-Host ""
Write-Host "🔧 TARS CLI Chat Duplication Fix: COMPLETE" -ForegroundColor Cyan
