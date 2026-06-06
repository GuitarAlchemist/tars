# Comprehensive TARS Metascript Testing Suite

param(
    [switch]$All = $false,
    [switch]$Core = $false,
    [switch]$Analysis = $false,
    [switch]$Improvement = $false,
    [switch]$Generation = $false,
    [switch]$Parallel = $false,
    [int]$MaxConcurrent = 3,
    [switch]$Detailed = $false,
    [switch]$ContinueOnError = $true
)

$ProjectPath = "TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj"
$TestResults = @()
$FailedTests = @()
$SuccessfulTests = @()

function Write-TestHeader($title) {
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host " $title" -ForegroundColor Cyan
    Write-Host "=" * 80 -ForegroundColor Cyan
}

function Write-TestSection($section) {
    Write-Host ""
    Write-Host "🧪 $section" -ForegroundColor Yellow
    Write-Host "-" * 60 -ForegroundColor Gray
}

function Test-Metascript($metascriptName, $category = "Unknown") {
    Write-Host ""
    Write-Host "Testing: $metascriptName" -ForegroundColor White
    Write-Host "Category: $category" -ForegroundColor Gray
    
    $startTime = Get-Date
    
    try {
        # Test the metascript execution
        $result = & dotnet run --project $ProjectPath -- metascript $metascriptName 2>&1
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalMilliseconds
        
        $success = $LASTEXITCODE -eq 0
        
        if ($success) {
            Write-Host "✅ PASS" -ForegroundColor Green
            $status = "PASS"
            $global:SuccessfulTests += $metascriptName
        } else {
            Write-Host "❌ FAIL" -ForegroundColor Red
            $status = "FAIL"
            $global:FailedTests += $metascriptName
        }
        
        Write-Host "Duration: $([math]::Round($duration, 2))ms" -ForegroundColor Gray
        
        if ($Detailed) {
            Write-Host "Output:" -ForegroundColor Gray
            $result | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
        }
        
        $global:TestResults += [PSCustomObject]@{
            Name = $metascriptName
            Category = $category
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
        Write-Host "Duration: $([math]::Round($duration, 2))ms" -ForegroundColor Gray
        
        $global:FailedTests += $metascriptName
        $global:TestResults += [PSCustomObject]@{
            Name = $metascriptName
            Category = $category
            Status = "ERROR"
            Duration = $duration
            Success = $false
            Output = $_.Exception.Message
        }
        
        return $false
    }
}

function Get-DiscoveredMetascripts {
    Write-Host "Discovering all metascripts..." -ForegroundColor Yellow
    
    # Run discovery and capture the output
    $discoveryOutput = & dotnet run --project $ProjectPath -- metascript-list --discover 2>&1
    
    # Parse the output to extract metascript names
    $metascripts = @()
    
    # Look for "Metascript registered:" lines in the output
    $discoveryOutput | ForEach-Object {
        if ($_ -match "Metascript registered: (.+)") {
            $metascriptName = $matches[1].Trim()
            
            # Infer category from name
            $category = "General"
            if ($metascriptName -match "improvement|auto") { $category = "Improvement" }
            elseif ($metascriptName -match "analysis|analyzer|quality") { $category = "Analysis" }
            elseif ($metascriptName -match "generator|generation|template") { $category = "Generation" }
            elseif ($metascriptName -match "core|hello|simple") { $category = "Core" }
            elseif ($metascriptName -match "test|testing") { $category = "Testing" }
            elseif ($metascriptName -match "doc|documentation") { $category = "Documentation" }
            elseif ($metascriptName -match "tot|tree") { $category = "TreeOfThought" }
            
            $metascripts += [PSCustomObject]@{
                Name = $metascriptName
                Category = $category
            }
        }
    }
    
    Write-Host "Discovered $($metascripts.Count) metascripts" -ForegroundColor Green
    return $metascripts
}

function Test-MetascriptsByCategory($metascripts, $categoryFilter) {
    $filteredMetascripts = $metascripts | Where-Object { $_.Category -eq $categoryFilter }
    
    Write-TestSection "Testing $categoryFilter Metascripts ($($filteredMetascripts.Count) scripts)"
    
    foreach ($metascript in $filteredMetascripts) {
        Test-Metascript $metascript.Name $metascript.Category
        
        if (-not $ContinueOnError -and $global:TestResults[-1].Status -ne "PASS") {
            Write-Host "Stopping due to failure (use -ContinueOnError to continue)" -ForegroundColor Red
            break
        }
    }
}

function Test-AllMetascripts($metascripts) {
    Write-TestSection "Testing All Metascripts ($($metascripts.Count) scripts)"
    
    if ($Parallel) {
        Write-Host "Running tests in parallel (max $MaxConcurrent concurrent)" -ForegroundColor Yellow
        
        # Group metascripts into batches
        $batches = @()
        for ($i = 0; $i -lt $metascripts.Count; $i += $MaxConcurrent) {
            $batch = $metascripts[$i..([Math]::Min($i + $MaxConcurrent - 1, $metascripts.Count - 1))]
            $batches += ,$batch
        }
        
        foreach ($batch in $batches) {
            $jobs = @()
            foreach ($metascript in $batch) {
                $job = Start-Job -ScriptBlock {
                    param($metascriptName, $projectPath)
                    
                    $startTime = Get-Date
                    $result = & dotnet run --project $projectPath -- metascript $metascriptName 2>&1
                    $endTime = Get-Date
                    $duration = ($endTime - $startTime).TotalMilliseconds
                    $success = $LASTEXITCODE -eq 0
                    
                    return @{
                        Name = $metascriptName
                        Success = $success
                        Duration = $duration
                        Output = $result -join "`n"
                    }
                } -ArgumentList $metascript.Name, $ProjectPath
                
                $jobs += $job
            }
            
            # Wait for batch to complete
            $results = $jobs | Wait-Job | Receive-Job
            $jobs | Remove-Job
            
            # Process results
            foreach ($result in $results) {
                if ($result.Success) {
                    Write-Host "✅ $($result.Name) - $([math]::Round($result.Duration, 2))ms" -ForegroundColor Green
                    $global:SuccessfulTests += $result.Name
                } else {
                    Write-Host "❌ $($result.Name) - $([math]::Round($result.Duration, 2))ms" -ForegroundColor Red
                    $global:FailedTests += $result.Name
                }
                
                $global:TestResults += [PSCustomObject]@{
                    Name = $result.Name
                    Category = "Unknown"
                    Status = if ($result.Success) { "PASS" } else { "FAIL" }
                    Duration = $result.Duration
                    Success = $result.Success
                    Output = $result.Output
                }
            }
        }
    } else {
        # Sequential testing
        foreach ($metascript in $metascripts) {
            Test-Metascript $metascript.Name $metascript.Category
            
            if (-not $ContinueOnError -and $global:TestResults[-1].Status -ne "PASS") {
                Write-Host "Stopping due to failure (use -ContinueOnError to continue)" -ForegroundColor Red
                break
            }
        }
    }
}

function Show-TestSummary {
    Write-TestHeader "COMPREHENSIVE TEST RESULTS"
    
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
    
    # Results by category
    Write-Host ""
    Write-Host "📈 Results by Category:" -ForegroundColor Cyan
    $TestResults | Group-Object Category | ForEach-Object {
        $categoryPassed = ($_.Group | Where-Object { $_.Status -eq "PASS" }).Count
        $categoryTotal = $_.Count
        $categoryRate = if ($categoryTotal -gt 0) { [math]::Round(($categoryPassed / $categoryTotal) * 100, 1) } else { 0 }
        
        Write-Host "  $($_.Name): $categoryPassed/$categoryTotal ($categoryRate%)" -ForegroundColor White
    }
    
    # Failed tests details
    if ($FailedTests.Count -gt 0) {
        Write-Host ""
        Write-Host "❌ Failed Tests:" -ForegroundColor Red
        $FailedTests | ForEach-Object {
            Write-Host "  - $_" -ForegroundColor Red
        }
    }
    
    # Top performers
    if ($SuccessfulTests.Count -gt 0) {
        Write-Host ""
        Write-Host "🏆 Fastest Successful Tests:" -ForegroundColor Green
        $TestResults | Where-Object { $_.Status -eq "PASS" } | Sort-Object Duration | Select-Object -First 5 | ForEach-Object {
            Write-Host "  🥇 $($_.Name) - $([math]::Round($_.Duration, 2))ms" -ForegroundColor Green
        }
    }
    
    # Save detailed results
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $resultsFile = "metascript_test_results_$timestamp.json"
    $TestResults | ConvertTo-Json -Depth 3 | Out-File $resultsFile
    Write-Host ""
    Write-Host "📄 Detailed results saved to: $resultsFile" -ForegroundColor Cyan
    
    Write-Host ""
    if ($failedTests -eq 0 -and $errorTests -eq 0) {
        Write-Host "🎉 ALL TESTS PASSED! The TARS metascript ecosystem is fully functional!" -ForegroundColor Green
    } elseif ($passedTests -gt ($failedTests + $errorTests)) {
        Write-Host "✅ Most tests passed! The TARS metascript ecosystem is largely functional." -ForegroundColor Yellow
    } else {
        Write-Host "⚠️  Many tests failed. Review the results and fix issues." -ForegroundColor Red
    }
}

# Main execution
Write-TestHeader "🚀 COMPREHENSIVE TARS METASCRIPT TESTING SUITE"
Write-Host "Testing all discovered metascripts in the TARS ecosystem..." -ForegroundColor White

# Build the project first
Write-Host "Building TARS F# CLI..." -ForegroundColor Yellow
dotnet build $ProjectPath | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed! Cannot proceed with testing." -ForegroundColor Red
    exit 1
}
Write-Host "✅ Build successful!" -ForegroundColor Green

# Discover all metascripts
$allMetascripts = Get-DiscoveredMetascripts

if ($allMetascripts.Count -eq 0) {
    Write-Host "❌ No metascripts discovered! Check the discovery system." -ForegroundColor Red
    exit 1
}

# Run tests based on parameters
if ($All -or (-not $Core -and -not $Analysis -and -not $Improvement -and -not $Generation)) {
    Test-AllMetascripts $allMetascripts
} else {
    if ($Core) { Test-MetascriptsByCategory $allMetascripts "Core" }
    if ($Analysis) { Test-MetascriptsByCategory $allMetascripts "Analysis" }
    if ($Improvement) { Test-MetascriptsByCategory $allMetascripts "Improvement" }
    if ($Generation) { Test-MetascriptsByCategory $allMetascripts "Generation" }
}

Show-TestSummary

Write-Host ""
Write-Host "🏁 Comprehensive metascript testing completed!" -ForegroundColor Cyan
