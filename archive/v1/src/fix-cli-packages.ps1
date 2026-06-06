#!/usr/bin/env pwsh

# Fix TARS CLI Package References
# Fixes escaped package names and version compatibility issues

Write-Host "🔧 Fixing TARS CLI Package References" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

$projectPath = "TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj"

if (Test-Path $projectPath) {
    Write-Host "📝 Fixing packages in $projectPath..." -ForegroundColor Yellow
    
    $content = Get-Content $projectPath -Raw
    $updated = $false
    
    # Fix escaped package names and versions
    $packageFixes = @{
        'Microsoft\.Extensions\.DependencyInjection" Version="10\.0\.0"' = 'Microsoft.Extensions.DependencyInjection" Version="9.0.0"'
        'Microsoft\.Extensions\.Http" Version="10\.0\.0"' = 'Microsoft.Extensions.Http" Version="9.0.0"'
        'Microsoft\.Extensions\.Logging" Version="10\.0\.0"' = 'Microsoft.Extensions.Logging" Version="9.0.0"'
        'System\.Threading\.Channels" Version="10\.0\.0"' = 'System.Threading.Channels" Version="9.0.0"'
        'System\.Text\.Json" Version="10\.0\.0"' = 'System.Text.Json" Version="9.0.0"'
        'System\.ServiceProcess\.ServiceController" Version="10\.0\.0"' = 'System.ServiceProcess.ServiceController" Version="9.0.0"'
        'System\.Management" Version="10\.0\.0"' = 'System.Management" Version="9.0.0"'
    }
    
    foreach ($pattern in $packageFixes.Keys) {
        $replacement = $packageFixes[$pattern]
        if ($content -match $pattern) {
            $content = $content -replace $pattern, $replacement
            $updated = $true
            Write-Host "  ✅ Fixed: $pattern -> $replacement" -ForegroundColor Green
        }
    }
    
    if ($updated) {
        Set-Content -Path $projectPath -Value $content
        Write-Host "  📦 Package references fixed successfully" -ForegroundColor Green
    } else {
        Write-Host "  ℹ️  No package fixes needed" -ForegroundColor Blue
    }
} else {
    Write-Host "❌ Project file not found: $projectPath" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🎯 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Run 'dotnet restore TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj'" -ForegroundColor White
Write-Host "2. Run 'dotnet build TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj'" -ForegroundColor White
Write-Host ""

Write-Host "✅ CLI package fixes completed!" -ForegroundColor Green
