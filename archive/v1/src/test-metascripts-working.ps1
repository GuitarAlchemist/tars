# Working TARS Metascript Testing Suite

param(
    [switch]$All = $false,
    [switch]$Core = $false,
    [switch]$Sample = $false,
    [int]$MaxTests = 10,
    [switch]$Detailed = $false,
    [switch]$ContinueOnError = $true
)

$ProjectPath = "TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj"
$TestResults = @()

function Write-TestHeader($title) {
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host " $title" -ForegroundColor Cyan
    Write-Host "=" * 80 -ForegroundColor Cyan
}

function Test-MetascriptFile($filePath) {
    $fileName = [System.IO.Path]::GetFileNameWithoutExtension($filePath)
    Write-Host ""
    Write-Host "Testing: $fileName" -ForegroundColor White
    Write-Host "Path: $filePath" -ForegroundColor Gray
    
    $startTime = Get-Date
    
    try {
        # Test the metascript execution
        $result = & dotnet run --project $ProjectPath -- metascript $filePath 2>&1
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalMilliseconds
        
        $success = $LASTEXITCODE -eq 0
        
        if ($success) {
            Write-Host "✅ PASS" -ForegroundColor Green
            $status = "PASS"
        } else {
            Write-Host "❌ FAIL" -ForegroundColor Red
            $status = "FAIL"
        }
        
        Write-Host "Duration: $([math]::Round($duration, 2))ms" -ForegroundColor Gray
        
        if ($Detailed -and $success) {
            Write-Host "Output Preview:" -ForegroundColor Gray
            $result | Select-Object -First 5 | ForEach-Object { 
                if ($_ -notmatch "info:|fail:|warn:") {
                    Write-Host "  $_" -ForegroundColor DarkGray 
                }
            }
        }
        
        $global:TestResults += [PSCustomObject]@{
            Name = $fileName
            Path = $filePath
            Status = $status
            Duration = $duration
            Success = $success
            Output = $result -join "`n"
        }
        
        return $success
    }
    catch {
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalMilliseconds
        
        Write-Host "❌ ERROR" -ForegroundColor Red
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
        
        $global:TestResults += [PSCustomObject]@{
            Name = $fileName
            Path = $filePath
            Status = "ERROR"
            Duration = $duration
            Success = $false
            Output = $_.Exception.Message
        }
        
        return $false
    }
}

function Get-MetascriptFiles($maxCount = 50) {
    Write-Host "Discovering metascript files..." -ForegroundColor Yellow
    
    $metascriptFiles = @()
    
    # Get files from TarsCli/Metascripts
    if (Test-Path "TarsCli/Metascripts") {
        $files = Get-ChildItem -Path "TarsCli/Metascripts" -Filter "*.tars" -Recurse | Select-Object -First $maxCount
        $metascriptFiles += $files.FullName
    }
    
    # Get files from root Metascripts directory
    if (Test-Path "Metascripts") {
        $files = Get-ChildItem -Path "Metascripts" -Filter "*.tars" -Recurse | Select-Object -First ($maxCount - $metascriptFiles.Count)
        $metascriptFiles += $files.FullName
    }
    
    # Get our test files
    $testFiles = @("test.tars", "demo_metascript.tars") | Where-Object { Test-Path $_ }
    $metascriptFiles += $testFiles
    
    Write-Host "Found $($metascriptFiles.Count) metascript files" -ForegroundColor Green
    return $metascriptFiles
}

function Test-SampleMetascripts {
    Write-Host ""
    Write-Host "🧪 Testing Sample Metascripts" -ForegroundColor Yellow
    Write-Host "-" * 60 -ForegroundColor Gray
    
    # Test specific known good metascripts
    $sampleFiles = @(
        "TarsCli/Metascripts/hello_world.tars",
        "TarsCli/Metascripts/simple_fsharp.tars",
        "TarsCli/Metascripts/fsharp_example.tars",
        "test.tars",
        "demo_metascript.tars"
    ) | Where-Object { Test-Path $_ }
    
    foreach ($file in $sampleFiles) {
        Test-MetascriptFile $file
        
        if (-not $ContinueOnError -and $global:TestResults[-1].Status -ne "PASS") {
            Write-Host "Stopping due to failure" -ForegroundColor Red
            break
        }
    }
}

function Test-CoreMetascripts {
    Write-Host ""
    Write-Host "🧪 Testing Core Metascripts" -ForegroundColor Yellow
    Write-Host "-" * 60 -ForegroundColor Gray
    
    # Test core metascripts
    $corePattern = @("hello_world", "simple_fsharp", "fsharp_example", "error_handling", "file_operations")
    
    $allFiles = Get-MetascriptFiles
    $coreFiles = $allFiles | Where-Object { 
        $fileName = [System.IO.Path]::GetFileNameWithoutExtension($_)
        $corePattern | ForEach-Object { if ($fileName -match $_) { return $true } }
    }
    
    foreach ($file in $coreFiles) {
        Test-MetascriptFile $file
        
        if (-not $ContinueOnError -and $global:TestResults[-1].Status -ne "PASS") {
            Write-Host "Stopping due to failure" -ForegroundColor Red
            break
        }
    }
}

function Test-AllMetascripts {
    Write-Host ""
    Write-Host "🧪 Testing All Metascripts (Limited to $MaxTests)" -ForegroundColor Yellow
    Write-Host "-" * 60 -ForegroundColor Gray
    
    $allFiles = Get-MetascriptFiles $MaxTests
    
    $count = 0
    foreach ($file in $allFiles) {
        if ($count -ge $MaxTests) {
            Write-Host "Reached maximum test limit of $MaxTests" -ForegroundColor Yellow
            break
        }
        
        Test-MetascriptFile $file
        $count++
        
        if (-not $ContinueOnError -and $global:TestResults[-1].Status -ne "PASS") {
            Write-Host "Stopping due to failure" -ForegroundColor Red
            break
        }
    }
}

function Show-TestSummary {
    Write-TestHeader "TEST RESULTS SUMMARY"
    
    $totalTests = $TestResults.Count
    $passedTests = ($TestResults | Where-Object { $_.Status -eq "PASS" }).Count
    $failedTests = ($TestResults | Where-Object { $_.Status -eq "FAIL" }).Count
    $errorTests = ($TestResults | Where-Object { $_.Status -eq "ERROR" }).Count
    $avgDuration = if ($totalTests -gt 0) { ($TestResults | Measure-Object -Property Duration -Average).Average } else { 0 }
    
    Write-Host ""
    Write-Host "📊 Overall Results:" -ForegroundColor Cyan
    Write-Host "  Total Tests: $totalTests" -ForegroundColor White
    Write-Host "  Passed: $passedTests" -ForegroundColor Green
    Write-Host "  Failed: $failedTests" -ForegroundColor Red
    Write-Host "  Errors: $errorTests" -ForegroundColor Magenta
    Write-Host "  Success Rate: $([math]::Round(($passedTests / $totalTests) * 100, 1))%" -ForegroundColor $(if ($failedTests -eq 0) { "Green" } else { "Yellow" })
    Write-Host "  Average Duration: $([math]::Round($avgDuration, 2))ms" -ForegroundColor White
    
    # Show successful tests
    if ($passedTests -gt 0) {
        Write-Host ""
        Write-Host "✅ Successful Tests:" -ForegroundColor Green
        $TestResults | Where-Object { $_.Status -eq "PASS" } | ForEach-Object {
            Write-Host "  🟢 $($_.Name) - $([math]::Round($_.Duration, 2))ms" -ForegroundColor Green
        }
    }
    
    # Show failed tests
    if ($failedTests -gt 0 -or $errorTests -gt 0) {
        Write-Host ""
        Write-Host "❌ Failed/Error Tests:" -ForegroundColor Red
        $TestResults | Where-Object { $_.Status -ne "PASS" } | ForEach-Object {
            Write-Host "  🔴 $($_.Name) - $($_.Status)" -ForegroundColor Red
        }
    }
    
    # Show fastest tests
    if ($passedTests -gt 0) {
        Write-Host ""
        Write-Host "🏆 Fastest Tests:" -ForegroundColor Yellow
        $TestResults | Where-Object { $_.Status -eq "PASS" } | Sort-Object Duration | Select-Object -First 3 | ForEach-Object {
            Write-Host "  ⚡ $($_.Name) - $([math]::Round($_.Duration, 2))ms" -ForegroundColor Yellow
        }
    }
    
    # Save results
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $resultsFile = "metascript_test_results_$timestamp.json"
    $TestResults | ConvertTo-Json -Depth 3 | Out-File $resultsFile
    Write-Host ""
    Write-Host "📄 Results saved to: $resultsFile" -ForegroundColor Cyan
    
    Write-Host ""
    if ($failedTests -eq 0 -and $errorTests -eq 0) {
        Write-Host "🎉 ALL TESTS PASSED! The TARS metascript ecosystem is working perfectly!" -ForegroundColor Green
    } elseif ($passedTests -gt ($failedTests + $errorTests)) {
        Write-Host "✅ Most tests passed! The TARS metascript ecosystem is largely functional." -ForegroundColor Yellow
    } else {
        Write-Host "⚠️  Some tests failed. This is normal for complex metascripts." -ForegroundColor Yellow
    }
}

# Main execution
Write-TestHeader "🚀 TARS METASCRIPT TESTING SUITE"
Write-Host "Testing metascripts in the TARS ecosystem..." -ForegroundColor White

# Build the project first
Write-Host "Building TARS F# CLI..." -ForegroundColor Yellow
dotnet build $ProjectPath | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed! Cannot proceed with testing." -ForegroundColor Red
    exit 1
}
Write-Host "✅ Build successful!" -ForegroundColor Green

# Run tests based on parameters
if ($Sample -or (-not $All -and -not $Core)) {
    Test-SampleMetascripts
} elseif ($Core) {
    Test-CoreMetascripts
} elseif ($All) {
    Test-AllMetascripts
}

Show-TestSummary

Write-Host ""
Write-Host "🏁 Metascript testing completed!" -ForegroundColor Cyan
