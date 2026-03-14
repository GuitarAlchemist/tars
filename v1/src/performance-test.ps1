# TARS F# Performance Testing Script

param(
    [int]$Iterations = 10,
    [switch]$Detailed = $false
)

$ProjectPath = "TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj"

Write-Host "🚀 TARS F# Performance Testing" -ForegroundColor Cyan
Write-Host "Testing with $Iterations iterations" -ForegroundColor White
Write-Host "=" * 50 -ForegroundColor Gray

# Build first
Write-Host "Building project..." -ForegroundColor Yellow
dotnet build $ProjectPath | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Build successful!" -ForegroundColor Green

$commands = @(
    "version",
    "help",
    "analyze .",
    "compile test.tars",
    "test",
    "metascript test.tars"
)

$results = @{}

foreach ($command in $commands) {
    Write-Host ""
    Write-Host "Testing: tars $command" -ForegroundColor Cyan
    
    $times = @()
    $successCount = 0
    
    for ($i = 1; $i -le $Iterations; $i++) {
        if ($Detailed) {
            Write-Host "  Run $i..." -ForegroundColor Gray
        }
        
        $startTime = Get-Date
        $result = & dotnet run --project $ProjectPath -- $command.Split(' ') 2>&1
        $endTime = Get-Date
        
        $duration = ($endTime - $startTime).TotalMilliseconds
        $times += $duration
        
        if ($LASTEXITCODE -eq 0) {
            $successCount++
        }
        
        if ($Detailed) {
            $status = if ($LASTEXITCODE -eq 0) { "✅" } else { "❌" }
            Write-Host "    $status $([math]::Round($duration, 2))ms" -ForegroundColor Gray
        }
    }
    
    $avgTime = ($times | Measure-Object -Average).Average
    $minTime = ($times | Measure-Object -Minimum).Minimum
    $maxTime = ($times | Measure-Object -Maximum).Maximum
    $successRate = ($successCount / $Iterations) * 100
    
    $results[$command] = @{
        Average = $avgTime
        Min = $minTime
        Max = $maxTime
        SuccessRate = $successRate
        Times = $times
    }
    
    Write-Host "  Average: $([math]::Round($avgTime, 2))ms" -ForegroundColor Green
    Write-Host "  Range: $([math]::Round($minTime, 2))ms - $([math]::Round($maxTime, 2))ms" -ForegroundColor White
    Write-Host "  Success Rate: $([math]::Round($successRate, 1))%" -ForegroundColor $(if ($successRate -eq 100) { "Green" } else { "Yellow" })
}

Write-Host ""
Write-Host "=" * 50 -ForegroundColor Gray
Write-Host "📊 Performance Summary" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Gray

$overallAvg = ($results.Values | ForEach-Object { $_.Average } | Measure-Object -Average).Average
$overallSuccessRate = ($results.Values | ForEach-Object { $_.SuccessRate } | Measure-Object -Average).Average

Write-Host ""
Write-Host "Overall Performance:" -ForegroundColor Yellow
Write-Host "  Average Response Time: $([math]::Round($overallAvg, 2))ms" -ForegroundColor White
Write-Host "  Overall Success Rate: $([math]::Round($overallSuccessRate, 1))%" -ForegroundColor $(if ($overallSuccessRate -eq 100) { "Green" } else { "Yellow" })

Write-Host ""
Write-Host "Command Performance Ranking:" -ForegroundColor Yellow
$sortedResults = $results.GetEnumerator() | Sort-Object { $_.Value.Average }

$rank = 1
foreach ($result in $sortedResults) {
    $command = $result.Key
    $avg = $result.Value.Average
    $successRate = $result.Value.SuccessRate
    
    $medal = switch ($rank) {
        1 { "🥇" }
        2 { "🥈" }
        3 { "🥉" }
        default { "  " }
    }
    
    Write-Host "  $medal $rank. $command - $([math]::Round($avg, 2))ms ($([math]::Round($successRate, 1))%)" -ForegroundColor White
    $rank++
}

Write-Host ""
if ($overallSuccessRate -eq 100 -and $overallAvg -lt 2000) {
    Write-Host "🎉 Excellent Performance! All tests passed with good response times." -ForegroundColor Green
} elseif ($overallSuccessRate -ge 90) {
    Write-Host "✅ Good Performance! Most tests passed with acceptable response times." -ForegroundColor Yellow
} else {
    Write-Host "⚠️  Performance Issues Detected. Review the results above." -ForegroundColor Red
}

Write-Host ""
Write-Host "Performance testing completed!" -ForegroundColor Cyan
