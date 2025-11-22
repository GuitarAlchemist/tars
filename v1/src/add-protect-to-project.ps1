#!/usr/bin/env pwsh

# Add ProtectCommand.fs to TARS CLI Project File

Write-Host "🔧 Adding ProtectCommand.fs to Project File" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

$projectPath = "TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj"

if (Test-Path $projectPath) {
    Write-Host "📝 Reading project file..." -ForegroundColor Yellow
    
    $content = Get-Content $projectPath -Raw
    $updated = $false
    
    Write-Host "🔧 Adding ProtectCommand.fs..." -ForegroundColor Yellow
    
    # Add ProtectCommand.fs after QACommand.fs
    if ($content -match '<Compile Include="Commands\\QACommand\.fs" />') {
        $content = $content -replace '<Compile Include="Commands\\QACommand\.fs" />', '<Compile Include="Commands\QACommand.fs" />' + [Environment]::NewLine + '    <Compile Include="Commands\ProtectCommand.fs" />'
        $updated = $true
        Write-Host "  ✅ Added ProtectCommand.fs to project file" -ForegroundColor Green
    }
    
    if ($updated) {
        Set-Content -Path $projectPath -Value $content -Encoding UTF8
        Write-Host "  📦 Project file updated successfully" -ForegroundColor Green
    } else {
        Write-Host "  ℹ️  No updates needed or QACommand.fs not found" -ForegroundColor Blue
    }
    
    Write-Host ""
} else {
    Write-Host "❌ Project file not found: $projectPath" -ForegroundColor Red
    exit 1
}

Write-Host "🎯 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Run 'dotnet build TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj' to test compilation" -ForegroundColor White
Write-Host "2. Test with 'dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj protect help'" -ForegroundColor White
Write-Host ""

Write-Host "✅ ProtectCommand.fs added to project!" -ForegroundColor Green
