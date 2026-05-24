# Phase 6.2 Integration Verification Script
# Tests all integration points for Budget Governance + Episode Ingestion

$ErrorActionPreference = "Stop"
$baseDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location $baseDir

Write-Host "`n=== Phase 6.2 Integration Verification ===" -ForegroundColor Cyan
Write-Host "Testing Budget Governance + Episode Ingestion`n" -ForegroundColor Cyan

$testsPassed = 0
$testsFailed = 0

function Test-Step {
    param($Name, $ScriptBlock)
    Write-Host "Testing: $Name..." -ForegroundColor Yellow
    try {
        & $ScriptBlock
        Write-Host "  ✅ PASSED" -ForegroundColor Green
        $script:testsPassed++
    } catch {
        Write-Host "  ❌ FAILED: $_" -ForegroundColor Red
        $script:testsFailed++
    }
}

# Test 1: Build succeeds
Test-Step "Build succeeds" {
    dotnet build src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj --nologo -v quiet
    if ($LASTEXITCODE -ne 0) { throw "Build failed" }
}

# Test 2: Evolution tests pass
Test-Step "Evolution tests pass" {
    $output = dotnet test --filter "Evolution" --no-build --nologo -v quiet 2>&1 | Out-String
    if ($output -notmatch "succeeded: 4") { throw "Tests did not pass" }
}

# Test 3: Help text includes budget
Test-Step "Help text includes budget parameter" {
    $help = dotnet run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj --no-build -- --help 2>&1 | Out-String
    if ($help -notmatch "--budget") { throw "Budget parameter not in help" }
}

# Test 4: Ollama is available
Test-Step "Ollama service is accessible" {
    $models = ollama list 2>&1 | Out-String
    if ($models -notmatch "qwen2.5-coder") { throw "Required model not available" }
}

# Test 5: MetascriptContext fields updated
Test-Step "All MetascriptContext sites have EpisodeService field" {
    $files = @(
        "src/Tars.Interface.Cli/Commands/MacroDemo.fs",
        "src/Tars.Interface.Cli/Commands/Run.fs",
        "src/Tars.Interface.Cli/Commands/RunCommand.fs",
        "src/Tars.Interface.Cli/Commands/RagDemo.fs"
    )
    
    foreach ($file in $files) {
        $content = Get-Content $file -Raw
        if ($content -notmatch "EpisodeService\s*=") {
            throw "$file missing EpisodeService field"
        }
    }
}

# Test 6: Budget field in EvolveOptions
Test-Step "EvolveOptions has Budget field" {
    $content = Get-Content "src/Tars.Interface.Cli/Commands/Evolve.fs" -Raw
    if ($content -notmatch "Budget:\s*decimal\s*option") {
        throw "Budget field not found in EvolveOptions"
    }
}

# Test 7: Episode service integration
Test-Step "Episode service integration present" {
    $content = Get-Content "src/Tars.Interface.Cli/Commands/Evolve.fs" -Raw
    if ($content -notmatch "GRAPHITI_URL") {
        throw "Graphiti URL detection not found"
    }
}

# Test 8: Quick evolution run (if Ollama running)
Test-Step "Quick evolution execution (30s timeout)" {
    $job = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        dotnet run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj --no-build -- evolve --max-iterations 1 --budget 0.5 --quiet 2>&1
    }
    
    Wait-Job $job -Timeout 30 | Out-Null
    $output = Receive-Job $job
    Stop-Job $job -ErrorAction SilentlyContinue
    Remove-Job $job -ErrorAction SilentlyContinue
    
    # Check if it at least started (don't require completion)
    $combined = $output -join "`n"
    if ($combined -notmatch "Starting TARS") {
        throw "Evolution did not start: $combined"
    }
}

# Summary
Write-Host "`n=== Test Summary ===" -ForegroundColor Cyan
Write-Host "Passed: $testsPassed" -ForegroundColor Green
Write-Host "Failed: $testsFailed" -ForegroundColor $(if($testsFailed -eq 0){"Green"}else{"Red"})

if ($testsFailed -eq 0) {
    Write-Host "`n✨ All Phase 6.2 integration tests PASSED!" -ForegroundColor Green
    Write-Host "Budget Governance + Episode Ingestion: READY ✅`n" -ForegroundColor Green
    Pop-Location
    exit 0
} else {
    Write-Host "`n⚠️ Some tests failed. Review errors above.`n" -ForegroundColor Red
    Pop-Location
    exit 1
}
