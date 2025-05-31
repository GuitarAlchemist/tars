# TARS Comprehensive Test Runner
# Runs all TARS tests and validates functionality

param(
    [switch]$Verbose = $false,
    [switch]$SkipBuild = $false,
    [string]$TestCategory = "all"
)

Write-Host "🧪 TARS Comprehensive Test Runner" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Set verbose preference
if ($Verbose) {
    $VerbosePreference = "Continue"
}

$TestResults = @{
    Passed = 0
    Failed = 0
    Skipped = 0
    Details = @()
}

function Add-TestResult {
    param(
        [string]$TestName,
        [string]$Status,
        [string]$Message = ""
    )
    
    $TestResults.Details += @{
        Name = $TestName
        Status = $Status
        Message = $Message
        Timestamp = Get-Date
    }
    
    switch ($Status) {
        "PASSED" { $TestResults.Passed++ }
        "FAILED" { $TestResults.Failed++ }
        "SKIPPED" { $TestResults.Skipped++ }
    }
}

function Test-Build {
    Write-Host "🔨 Testing Build Process..." -ForegroundColor Yellow
    
    try {
        $buildOutput = dotnet build TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ Build successful" -ForegroundColor Green
            Add-TestResult -TestName "Build Process" -Status "PASSED" -Message "Clean build with no errors"
            return $true
        } else {
            Write-Host "   ❌ Build failed" -ForegroundColor Red
            Add-TestResult -TestName "Build Process" -Status "FAILED" -Message "Build errors detected"
            return $false
        }
    } catch {
        Write-Host "   ❌ Build exception: $($_.Exception.Message)" -ForegroundColor Red
        Add-TestResult -TestName "Build Process" -Status "FAILED" -Message $_.Exception.Message
        return $false
    }
}

function Test-CliCommands {
    Write-Host "🎮 Testing CLI Commands..." -ForegroundColor Yellow
    
    $commands = @(
        @{ Name = "version"; Args = "version"; Expected = "TARS" }
        @{ Name = "help"; Args = "help"; Expected = "Available commands" }
        @{ Name = "intelligence"; Args = "intelligence measure"; Expected = "Intelligence measurement" }
        @{ Name = "ml"; Args = "ml train"; Expected = "ML model training" }
        @{ Name = "metascript-list"; Args = "metascript-list --discover"; Expected = "metascripts" }
    )
    
    foreach ($cmd in $commands) {
        try {
            Write-Host "   Testing: $($cmd.Name)" -ForegroundColor Gray
            
            $output = dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- $($cmd.Args) 2>&1
            
            if ($output -match $cmd.Expected) {
                Write-Host "   ✅ $($cmd.Name) command working" -ForegroundColor Green
                Add-TestResult -TestName "CLI Command: $($cmd.Name)" -Status "PASSED"
            } else {
                Write-Host "   ❌ $($cmd.Name) command failed" -ForegroundColor Red
                Add-TestResult -TestName "CLI Command: $($cmd.Name)" -Status "FAILED" -Message "Expected output not found"
            }
        } catch {
            Write-Host "   ❌ $($cmd.Name) command exception" -ForegroundColor Red
            Add-TestResult -TestName "CLI Command: $($cmd.Name)" -Status "FAILED" -Message $_.Exception.Message
        }
    }
}

function Test-MetascriptDiscovery {
    Write-Host "📜 Testing Metascript Discovery..." -ForegroundColor Yellow
    
    try {
        $output = dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- metascript-list --discover 2>&1
        
        if ($output -match "metascripts" -and $output -match "Loaded from") {
            $metascriptCount = ($output | Select-String "Loaded from").Count
            Write-Host "   ✅ Metascript discovery working ($metascriptCount scripts found)" -ForegroundColor Green
            Add-TestResult -TestName "Metascript Discovery" -Status "PASSED" -Message "$metascriptCount metascripts discovered"
        } else {
            Write-Host "   ❌ Metascript discovery failed" -ForegroundColor Red
            Add-TestResult -TestName "Metascript Discovery" -Status "FAILED" -Message "No metascripts discovered"
        }
    } catch {
        Write-Host "   ❌ Metascript discovery exception" -ForegroundColor Red
        Add-TestResult -TestName "Metascript Discovery" -Status "FAILED" -Message $_.Exception.Message
    }
}

function Test-IntelligenceServices {
    Write-Host "🧠 Testing Intelligence Services..." -ForegroundColor Yellow
    
    try {
        $output = dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- intelligence measure 2>&1
        
        if ($output -match "Intelligence measurement" -and $output -match "Learning Rate") {
            Write-Host "   ✅ Intelligence services working" -ForegroundColor Green
            Add-TestResult -TestName "Intelligence Services" -Status "PASSED"
        } else {
            Write-Host "   ❌ Intelligence services failed" -ForegroundColor Red
            Add-TestResult -TestName "Intelligence Services" -Status "FAILED" -Message "Intelligence measurement not working"
        }
    } catch {
        Write-Host "   ❌ Intelligence services exception" -ForegroundColor Red
        Add-TestResult -TestName "Intelligence Services" -Status "FAILED" -Message $_.Exception.Message
    }
}

function Test-MLServices {
    Write-Host "🤖 Testing ML Services..." -ForegroundColor Yellow
    
    try {
        $output = dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- ml train 2>&1
        
        if ($output -match "ML model training" -and $output -match "Training completed") {
            Write-Host "   ✅ ML services working" -ForegroundColor Green
            Add-TestResult -TestName "ML Services" -Status "PASSED"
        } else {
            Write-Host "   ❌ ML services failed" -ForegroundColor Red
            Add-TestResult -TestName "ML Services" -Status "FAILED" -Message "ML training not working"
        }
    } catch {
        Write-Host "   ❌ ML services exception" -ForegroundColor Red
        Add-TestResult -TestName "ML Services" -Status "FAILED" -Message $_.Exception.Message
    }
}

function Test-ProjectStructure {
    Write-Host "🗂️ Testing Project Structure..." -ForegroundColor Yellow
    
    $requiredFiles = @(
        "TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj",
        "TarsEngine.FSharp.Metascripts/TarsEngine.FSharp.Metascripts.fsproj",
        ".tars/tars.yaml",
        ".tars/plans/README.md",
        ".tars/metascripts",
        ".tars/docs"
    )
    
    $missingFiles = @()
    
    foreach ($file in $requiredFiles) {
        if (Test-Path $file) {
            Write-Host "   ✅ $file exists" -ForegroundColor Green
        } else {
            Write-Host "   ❌ $file missing" -ForegroundColor Red
            $missingFiles += $file
        }
    }
    
    if ($missingFiles.Count -eq 0) {
        Add-TestResult -TestName "Project Structure" -Status "PASSED"
    } else {
        Add-TestResult -TestName "Project Structure" -Status "FAILED" -Message "Missing files: $($missingFiles -join ', ')"
    }
}

function Show-TestSummary {
    Write-Host ""
    Write-Host "📊 Test Summary" -ForegroundColor Cyan
    Write-Host "===============" -ForegroundColor Cyan
    Write-Host ""
    
    $total = $TestResults.Passed + $TestResults.Failed + $TestResults.Skipped
    
    Write-Host "Total Tests: $total" -ForegroundColor White
    Write-Host "Passed: $($TestResults.Passed)" -ForegroundColor Green
    Write-Host "Failed: $($TestResults.Failed)" -ForegroundColor Red
    Write-Host "Skipped: $($TestResults.Skipped)" -ForegroundColor Yellow
    Write-Host ""
    
    if ($TestResults.Failed -gt 0) {
        Write-Host "❌ Failed Tests:" -ForegroundColor Red
        foreach ($test in $TestResults.Details | Where-Object { $_.Status -eq "FAILED" }) {
            Write-Host "   • $($test.Name): $($test.Message)" -ForegroundColor Red
        }
        Write-Host ""
    }
    
    $successRate = if ($total -gt 0) { [math]::Round(($TestResults.Passed / $total) * 100, 1) } else { 0 }
    Write-Host "Success Rate: $successRate%" -ForegroundColor $(if ($successRate -ge 90) { "Green" } elseif ($successRate -ge 70) { "Yellow" } else { "Red" })
    
    if ($successRate -eq 100) {
        Write-Host ""
        Write-Host "🎉 All tests passed! TARS is working perfectly!" -ForegroundColor Green
    } elseif ($successRate -ge 90) {
        Write-Host ""
        Write-Host "✅ Most tests passed! TARS is working well with minor issues." -ForegroundColor Yellow
    } else {
        Write-Host ""
        Write-Host "⚠️ Some tests failed. Please review and fix issues." -ForegroundColor Red
    }
}

# Main test execution
Write-Host "🎯 Starting TARS Test Suite" -ForegroundColor Green
Write-Host ""

# Build test (unless skipped)
if (-not $SkipBuild) {
    if (-not (Test-Build)) {
        Write-Host "❌ Build failed. Stopping tests." -ForegroundColor Red
        Show-TestSummary
        exit 1
    }
} else {
    Write-Host "⏭️ Skipping build test" -ForegroundColor Yellow
    Add-TestResult -TestName "Build Process" -Status "SKIPPED" -Message "Skipped by user request"
}

# Run tests based on category
switch ($TestCategory.ToLower()) {
    "all" {
        Test-ProjectStructure
        Test-CliCommands
        Test-MetascriptDiscovery
        Test-IntelligenceServices
        Test-MLServices
    }
    "cli" {
        Test-CliCommands
    }
    "metascripts" {
        Test-MetascriptDiscovery
    }
    "services" {
        Test-IntelligenceServices
        Test-MLServices
    }
    "structure" {
        Test-ProjectStructure
    }
    default {
        Write-Host "❌ Unknown test category: $TestCategory" -ForegroundColor Red
        Write-Host "Available categories: all, cli, metascripts, services, structure" -ForegroundColor Yellow
        exit 1
    }
}

Show-TestSummary

# Exit with appropriate code
exit $(if ($TestResults.Failed -eq 0) { 0 } else { 1 })
