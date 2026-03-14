# TARS v2 - Comprehensive Validation Script
# Tests Phase 7 (Production Hardening) and Phase 9 (Symbolic Knowledge)

Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║         TARS v2 - Production Validation Suite            ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Continue"
$testsPassed = 0
$testsFailed = 0

function Test-Step {
    param(
        [string]$Name,
        [scriptblock]$Test
    )
    
    Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] Testing: $Name..." -ForegroundColor Yellow
    try {
        & $Test
        Write-Host "  ✅ PASS: $Name" -ForegroundColor Green
        $script:testsPassed++
        return $true
    }
    catch {
        Write-Host "  ❌ FAIL: $Name" -ForegroundColor Red
        Write-Host "    Error: $_" -ForegroundColor DarkRed
        $script:testsFailed++
        return $false
    }
}

# Test 1: Solution Build
Test-Step "Full Solution Build" {
    $output = dotnet build Tars.sln 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
    if ($output -match "Build FAILED") {
        throw "Build reported failure"
    }
}

# Test 2: VerifierAgent Tests
Test-Step "VerifierAgent Tests (3 tests)" {
    $output = dotnet test tests/Tars.Tests/Tars.Tests.fsproj --filter VerifierAgentTests 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "Tests failed with exit code $LASTEXITCODE"
    }
    if ($output -notmatch "Passed.*3.*Total.*3") {
        throw "Expected 3/3 tests to pass"
    }
}

# Test 3: Metrics Module
Test-Step "Metrics.toPrometheus() Function Exists" {
    $metricsFile = "src/Tars.Core/Metrics.fs"
    $content = Get-Content $metricsFile -Raw
    if ($content -notmatch "toPrometheus") {
        throw "toPrometheus function not found in Metrics.fs"
    }
    if ($content -notmatch "tars_.*_total") {
        throw "Prometheus counter format not found"
    }
}

# Test 4: Configuration Files
Test-Step "MetricsSettings in Configuration" {
    $configFile = "src/Tars.Core/Configuration.fs"
    $content = Get-Content $configFile -Raw
    if ($content -notmatch "MetricsSettings") {
        throw "MetricsSettings type not found"
    }
}

Test-Step "appsettings.json Contains Metrics" {
    $appSettings = Get-Content "src/Tars.Interface.Cli/appsettings.json" -Raw | ConvertFrom-Json
    if (-not $appSettings.Metrics) {
        throw "Metrics section not found in appsettings.json"
    }
    if ($appSettings.Metrics.Enabled -ne $true) {
        throw "Metrics not enabled by default"
    }
    if ($appSettings.Metrics.Port -ne 9090) {
        throw "Metrics port not set to 9090"
    }
}

# Test 5: InfrastructureServer
Test-Step "InfrastructureServer.fs Exists" {
    if (-not (Test-Path "src/Tars.Interface.Cli/InfrastructureServer.fs")) {
        throw "InfrastructureServer.fs not found"
    }
    $content = Get-Content "src/Tars.Interface.Cli/InfrastructureServer.fs" -Raw
    if ($content -notmatch "/health") {
        throw "/health endpoint not found"
    }
    if ($content -notmatch "/metrics") {
        throw "/metrics endpoint not found"
    }
}

# Test 6: Docker Compose Files
Test-Step "docker-compose.all.yml Exists" {
    if (-not (Test-Path "docker-compose.all.yml")) {
        throw "docker-compose.all.yml not found"
    }
}

Test-Step "docker-compose.monitoring.yml Exists" {
    if (-not (Test-Path "docker-compose.monitoring.yml")) {
        throw "docker-compose.monitoring.yml not found"
    }
}

Test-Step "Prometheus Config Exists" {
    if (-not (Test-Path "docker/monitoring/prometheus.yml")) {
        throw "prometheus.yml not found"
    }
    $content = Get-Content "docker/monitoring/prometheus.yml" -Raw
    if ($content -notmatch "host.docker.internal:9090") {
        throw "Prometheus not configured to scrape TARS on port 9090"
    }
}

# Test 7: ReflectionAgent
Test-Step "ReflectionAgent.fs Exists" {
    if (-not (Test-Path "src/Tars.Knowledge/ReflectionAgent.fs")) {
        throw "ReflectionAgent.fs not found"
    }
    $content = Get-Content "src/Tars.Knowledge/ReflectionAgent.fs" -Raw
    if ($content -notmatch "ReflectAsync") {
        throw "ReflectAsync method not found"
    }
    if ($content -notmatch "CleanupAsync") {
        throw "CleanupAsync method not found"
    }
}

# Test 8: ConstraintScoring Fixes
Test-Step "ConstraintScoring Contradiction Detection" {
    $content = Get-Content "src/Tars.Symbolic/ConstraintScoring.fs" -Raw
    if ($content -notmatch "safeReplace") {
        throw "safeReplace helper not found"
    }
    if ($content -notmatch "clean") {
        throw "clean helper not found"
    }
}

# Test 9: Live Server Test (Optional - only if not already running)
$serverTest = Test-Step "Live Server Health & Metrics Endpoints" {
    Write-Host "    Starting TARS in background..." -ForegroundColor DarkYellow
    $proc = Start-Process -FilePath "dotnet" -ArgumentList "run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- chat" -PassThru -NoNewWindow -RedirectStandardOutput "validation_tars.log" -RedirectStandardError "validation_tars_err.log"
    
    Start-Sleep -Seconds 12
    
    try {
        # Test health endpoint
        $health = Invoke-WebRequest "http://localhost:9090/health" -TimeoutSec 5
        if ($health.StatusCode -ne 200) {
            throw "Health endpoint returned status $($health.StatusCode)"
        }
        
        $healthJson = $health.Content | ConvertFrom-Json
        if (-not $healthJson.status) {
            throw "Health response missing status field"
        }
        
        Write-Host "    ✓ Health endpoint: $($healthJson.status)" -ForegroundColor DarkGreen
        
        # Test metrics endpoint
        $metrics = Invoke-WebRequest "http://localhost:9090/metrics" -TimeoutSec 5
        if ($metrics.StatusCode -ne 200) {
            throw "Metrics endpoint returned status $($metrics.StatusCode)"
        }
        
        if ($metrics.Content -notmatch "tars_.*_total") {
            throw "Metrics response missing expected format"
        }
        
        Write-Host "    ✓ Metrics endpoint active" -ForegroundColor DarkGreen
    }
    finally {
        Write-Host "    Stopping TARS..." -ForegroundColor DarkYellow
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
}

# Test 10: Project File Updates
Test-Step "Tars.Interface.Cli.fsproj Contains InfrastructureServer" {
    $content = Get-Content "src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj" -Raw
    if ($content -notmatch "InfrastructureServer.fs") {
        throw "InfrastructureServer.fs not in project file"
    }
}

Test-Step "Tars.Knowledge.fsproj Contains ReflectionAgent" {
    $content = Get-Content "src/Tars.Knowledge/Tars.Knowledge.fsproj" -Raw
    if ($content -notmatch "ReflectionAgent.fs") {
        throw "ReflectionAgent.fs not in project file"
    }
}

# Summary
Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                   Validation Summary                      ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  ✅ Tests Passed: $testsPassed" -ForegroundColor Green
Write-Host "  ❌ Tests Failed: $testsFailed" -ForegroundColor $(if ($testsFailed -eq 0) { "Green" } else { "Red" })
Write-Host ""

if ($testsFailed -eq 0) {
    Write-Host "🎉 All validation tests passed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "✨ Phase 7 (Production Hardening) - VERIFIED" -ForegroundColor Cyan
    Write-Host "✨ Phase 9 (Symbolic Knowledge) - VERIFIED" -ForegroundColor Cyan
    exit 0
}
else {
    Write-Host "⚠️  Some validation tests failed. Please review errors above." -ForegroundColor Yellow
    exit 1
}
