#!/usr/bin/env pwsh

# Integrate TARS Protect Command into CLI Application
# Adds the ProtectCommand to the dependency injection and command registry

Write-Host "🔧 Integrating TARS Protect Command" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

$cliAppPath = "TarsEngine.FSharp.Cli/Core/CliApplication.fs"

if (Test-Path $cliAppPath) {
    Write-Host "📝 Reading CLI application file..." -ForegroundColor Yellow
    
    $content = Get-Content $cliAppPath -Raw
    $updated = $false
    
    Write-Host "🔧 Adding ProtectCommand to dependency injection..." -ForegroundColor Yellow
    
    # Add ProtectCommand to service registration
    if ($content -match 'services\.AddTransient<QACommand>\(\)') {
        $content = $content -replace 'services\.AddTransient<QACommand>\(\) \|> ignore', 'services.AddTransient<QACommand>() |> ignore' + [Environment]::NewLine + '        services.AddTransient<ProtectCommand>() |> ignore'
        $updated = $true
        Write-Host "  ✅ Added ProtectCommand to service registration" -ForegroundColor Green
    }
    
    Write-Host "🔧 Adding protect command to command registry..." -ForegroundColor Yellow
    
    # Add protect command to the command registry
    if ($content -match '\| "qa" ->') {
        $protectCommandBlock = @'
        | "protect" ->
            let cmd = serviceProvider.GetRequiredService<ProtectCommand>()
            Some (box cmd |> unbox<ICommand>)
'@
        $content = $content -replace '(\| "qa" ->[^|]*Some \(box cmd \|> unbox<ICommand>\))', '$1' + [Environment]::NewLine + $protectCommandBlock
        $updated = $true
        Write-Host "  ✅ Added protect command to command registry" -ForegroundColor Green
    }
    
    Write-Host "🔧 Adding protect to command list..." -ForegroundColor Yellow
    
    # Add protect to the command names list
    if ($content -match 'let commandNames = \[') {
        $content = $content -replace '(\["[^"]*"[^]]*)"qa"([^]]*\])', '$1"qa"; "protect"$2'
        $updated = $true
        Write-Host "  ✅ Added protect to command names list" -ForegroundColor Green
    }
    
    if ($updated) {
        Set-Content -Path $cliAppPath -Value $content -Encoding UTF8
        Write-Host "  📦 CLI application updated successfully" -ForegroundColor Green
    } else {
        Write-Host "  ℹ️  No updates needed or patterns not found" -ForegroundColor Blue
    }
    
    Write-Host ""
} else {
    Write-Host "❌ CLI application file not found: $cliAppPath" -ForegroundColor Red
    exit 1
}

Write-Host "🎯 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Add 'open TarsEngine.FSharp.Cli.Commands' to imports if needed" -ForegroundColor White
Write-Host "2. Run 'dotnet build TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj' to test integration" -ForegroundColor White
Write-Host "3. Test with 'dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj protect help'" -ForegroundColor White
Write-Host ""

Write-Host "✅ TARS Protect Command integration completed!" -ForegroundColor Green
