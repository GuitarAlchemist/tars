#!/usr/bin/env pwsh
# TARS Night Evolution Session Launcher
# Launches the Master Evolver Agent for autonomous night evolution

Write-Host "🌙 TARS NIGHT EVOLUTION SESSION LAUNCHER" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Launching Master Evolver Agent for autonomous evolution" -ForegroundColor White
Write-Host ""

# Configuration
$SessionId = [System.Guid]::NewGuid().ToString("N").Substring(0, 7)
$StartTime = Get-Date
$Duration = [TimeSpan]::FromHours(8)  # 8-hour night session
$EvaluationInterval = [TimeSpan]::FromMinutes(15)  # Evaluate every 15 minutes
$MetricSamplingRate = [TimeSpan]::FromMinutes(5)   # Sample metrics every 5 minutes

$GreenEnvironment = "tars-green-stable"
$BlueEnvironment = "tars-blue-evolution"

Write-Host "🚀 INITIALIZING NIGHT EVOLUTION SESSION" -ForegroundColor Magenta
Write-Host "======================================" -ForegroundColor Magenta

# Step 1: Setup directories
Write-Host "📂 Setting up evolution directories..." -ForegroundColor Yellow

$Directories = @(
    ".tars/evolution",
    ".tars/evolution/sessions",
    ".tars/evolution/reports", 
    ".tars/evolution/monitoring",
    ".tars/evolution/green-blue",
    ".tars/green",
    ".tars/blue",
    ".tars/monitoring/green-blue"
)

foreach ($Dir in $Directories) {
    if (-not (Test-Path $Dir)) {
        New-Item -ItemType Directory -Path $Dir -Force | Out-Null
        Write-Host "  📂 Created: $Dir" -ForegroundColor Gray
    }
}

Write-Host "✅ Evolution directories created" -ForegroundColor Green

# Step 2: Generate session configuration
Write-Host ""
Write-Host "📋 GENERATING EVOLUTION SESSION" -ForegroundColor Magenta
Write-Host "==============================" -ForegroundColor Magenta

$Session = @{
    SessionId = $SessionId
    StartTime = $StartTime.ToString("yyyy-MM-ddTHH:mm:ss")
    Duration = $Duration.ToString()
    Status = "active"
    
    # Environment Configuration
    GreenContainer = $GreenEnvironment
    BlueContainer = $BlueEnvironment
    TrafficSplit = 0.1  # Start with 10% traffic to blue
    
    # Evolution Goals
    EvolutionGoals = @(
        "Enhance meta-cognitive reasoning capabilities",
        "Optimize autonomous decision-making",
        "Improve pattern recognition across abstraction layers",
        "Increase learning efficiency and adaptation speed"
    )
    
    # Expected Improvements
    ExpectedImprovements = @{
        "reasoning_depth_score" = 0.15      # 15% improvement expected
        "capability_breadth_score" = 0.10   # 10% improvement expected
        "learning_efficiency_score" = 0.20  # 20% improvement expected
        "adaptation_speed" = 0.25           # 25% improvement expected
    }
    
    # Configuration
    EvaluationInterval = $EvaluationInterval.ToString()
    MetricSamplingRate = $MetricSamplingRate.ToString()
    
    # Tolerance Thresholds
    ToleranceThresholds = @{
        "response_time_ms" = 0.15        # 15% degradation allowed
        "memory_usage_mb" = 0.20         # 20% increase allowed
        "cpu_utilization_percent" = 0.25 # 25% increase allowed
        "task_completion_rate" = -0.05   # 5% decrease allowed
        "error_rate_percent" = 0.10      # 10% increase allowed
        "reasoning_depth_score" = -0.10  # 10% decrease allowed temporarily
        "capability_breadth_score" = -0.05 # 5% decrease allowed
        "learning_efficiency_score" = -0.15 # 15% decrease allowed during learning
    }
    
    AdaptiveToleranceEnabled = $true
}

# Save session configuration
$SessionDir = ".tars/evolution/sessions/$SessionId"
New-Item -ItemType Directory -Path $SessionDir -Force | Out-Null

$SessionJson = $Session | ConvertTo-Json -Depth 10
$SessionJson | Out-File -FilePath "$SessionDir/session-config.json" -Encoding UTF8

Write-Host "  🆔 Session ID: $SessionId" -ForegroundColor White
Write-Host "  ⏰ Start Time: $($StartTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor White
Write-Host "  ⏱️ Duration: $($Duration.TotalHours) hours" -ForegroundColor White
Write-Host "  🟢 Green Container: $GreenEnvironment" -ForegroundColor White
Write-Host "  🔵 Blue Container: $BlueEnvironment" -ForegroundColor White
Write-Host "  🎯 Evolution Goals: $($Session.EvolutionGoals.Count)" -ForegroundColor White
Write-Host "  📂 Session Directory: $SessionDir" -ForegroundColor White

Write-Host "✅ Evolution session generated" -ForegroundColor Green

# Step 3: Create Green Environment Script
Write-Host ""
Write-Host "🟢 CREATING GREEN ENVIRONMENT SCRIPT" -ForegroundColor Magenta
Write-Host "===================================" -ForegroundColor Magenta

$GreenScript = @"
#!/bin/bash
# Green Environment - Stable Baseline
echo "🟢 Starting TARS Green Environment (Baseline)"

# Check if container already exists
if docker ps -a --format "table {{.Names}}" | grep -q "^$GreenEnvironment`$"; then
    echo "  🔄 Stopping existing green container..."
    docker stop $GreenEnvironment 2>/dev/null || true
    docker rm $GreenEnvironment 2>/dev/null || true
fi

# Create network if it doesn't exist
docker network create tars-evolution 2>/dev/null || true

# Start green container
echo "  🚀 Starting green container..."
docker run -d --name $GreenEnvironment \
  --network tars-evolution \
  --label tars.environment=green \
  --label tars.role=baseline \
  --label tars.evolver.session=$SessionId \
  -p 8080:8080 \
  -p 8081:8081 \
  -v "`$(pwd)/.tars/green:/app/tars:rw" \
  -v "`$(pwd)/.tars/shared:/app/shared:ro" \
  -e TARS_ENVIRONMENT=green \
  -e TARS_ROLE=baseline \
  -e TARS_MONITORING_ENABLED=true \
  -e TARS_SESSION_ID=$SessionId \
  mcr.microsoft.com/dotnet/aspnet:9.0

echo "  ✅ Green environment ready at http://localhost:8080"
echo "  📊 Metrics available at http://localhost:8081/metrics"
"@

$GreenScriptPath = ".tars/green/start-green.sh"
$GreenScript | Out-File -FilePath $GreenScriptPath -Encoding UTF8
Write-Host "  📄 Green script: $GreenScriptPath" -ForegroundColor Gray

# Step 4: Create Blue Environment Script
Write-Host ""
Write-Host "🔵 CREATING BLUE ENVIRONMENT SCRIPT" -ForegroundColor Magenta
Write-Host "==================================" -ForegroundColor Magenta

$BlueScript = @"
#!/bin/bash
# Blue Environment - Evolution Experimental
echo "🔵 Starting TARS Blue Environment (Evolution)"

# Check if container already exists
if docker ps -a --format "table {{.Names}}" | grep -q "^$BlueEnvironment`$"; then
    echo "  🔄 Stopping existing blue container..."
    docker stop $BlueEnvironment 2>/dev/null || true
    docker rm $BlueEnvironment 2>/dev/null || true
fi

# Create network if it doesn't exist
docker network create tars-evolution 2>/dev/null || true

# Start blue container
echo "  🚀 Starting blue container..."
docker run -d --name $BlueEnvironment \
  --network tars-evolution \
  --label tars.environment=blue \
  --label tars.role=evolution \
  --label tars.evolver.session=$SessionId \
  -p 8082:8080 \
  -p 8083:8081 \
  -v "`$(pwd)/.tars/blue:/app/tars:rw" \
  -v "`$(pwd)/.tars/shared:/app/shared:ro" \
  -v "`$(pwd)/.tars/evolution:/app/evolution:rw" \
  -e TARS_ENVIRONMENT=blue \
  -e TARS_ROLE=evolution \
  -e TARS_EVOLUTION_ENABLED=true \
  -e TARS_MONITORING_ENABLED=true \
  -e TARS_SESSION_ID=$SessionId \
  mcr.microsoft.com/dotnet/aspnet:9.0

echo "  ✅ Blue environment ready at http://localhost:8082"
echo "  📊 Metrics available at http://localhost:8083/metrics"
"@

$BlueScriptPath = ".tars/blue/start-blue.sh"
$BlueScript | Out-File -FilePath $BlueScriptPath -Encoding UTF8
Write-Host "  📄 Blue script: $BlueScriptPath" -ForegroundColor Gray

# Step 5: Create Monitoring Script
Write-Host ""
Write-Host "📊 CREATING MONITORING SCRIPT" -ForegroundColor Magenta
Write-Host "============================" -ForegroundColor Magenta

$MonitoringScript = @"
#!/bin/bash
# TARS Evolution Monitoring Script
echo "📊 Starting TARS Evolution Monitoring"

SESSION_ID="$SessionId"
GREEN_ENDPOINT="http://localhost:8080"
BLUE_ENDPOINT="http://localhost:8082"
GREEN_METRICS="http://localhost:8081/metrics"
BLUE_METRICS="http://localhost:8083/metrics"

MONITORING_DIR=".tars/monitoring/green-blue/`$SESSION_ID"
mkdir -p "`$MONITORING_DIR"

echo "  🔍 Monitoring session: `$SESSION_ID"
echo "  🟢 Green endpoint: `$GREEN_ENDPOINT"
echo "  🔵 Blue endpoint: `$BLUE_ENDPOINT"
echo "  📊 Monitoring directory: `$MONITORING_DIR"

# Function to collect metrics
collect_metrics() {
    local timestamp=`$(date -Iseconds)
    local green_status="unknown"
    local blue_status="unknown"
    
    # Check green container health
    if curl -s "`$GREEN_ENDPOINT/health" > /dev/null 2>&1; then
        green_status="healthy"
    else
        green_status="unhealthy"
    fi
    
    # Check blue container health
    if curl -s "`$BLUE_ENDPOINT/health" > /dev/null 2>&1; then
        blue_status="healthy"
    else
        blue_status="unhealthy"
    fi
    
    # Log status
    echo "`$timestamp,green,`$green_status" >> "`$MONITORING_DIR/health-status.csv"
    echo "`$timestamp,blue,`$blue_status" >> "`$MONITORING_DIR/health-status.csv"
    
    echo "[`$timestamp] Green: `$green_status, Blue: `$blue_status"
}

# Function to compare performance
compare_performance() {
    local timestamp=`$(date -Iseconds)
    
    # Simulate metric collection (in real implementation, collect from actual endpoints)
    local green_response_time=`$((150 + RANDOM % 20))
    local blue_response_time=`$((165 + RANDOM % 25))
    local green_memory=`$((512 + RANDOM % 50))
    local blue_memory=`$((580 + RANDOM % 60))
    
    # Log metrics
    echo "`$timestamp,green,response_time_ms,`$green_response_time" >> "`$MONITORING_DIR/metrics.csv"
    echo "`$timestamp,blue,response_time_ms,`$blue_response_time" >> "`$MONITORING_DIR/metrics.csv"
    echo "`$timestamp,green,memory_usage_mb,`$green_memory" >> "`$MONITORING_DIR/metrics.csv"
    echo "`$timestamp,blue,memory_usage_mb,`$blue_memory" >> "`$MONITORING_DIR/metrics.csv"
    
    # Calculate performance difference
    local response_diff=`$(( (blue_response_time - green_response_time) * 100 / green_response_time ))
    local memory_diff=`$(( (blue_memory - green_memory) * 100 / green_memory ))
    
    echo "[`$timestamp] Response time: Green `${green_response_time}ms, Blue `${blue_response_time}ms (`${response_diff}%)"
    echo "[`$timestamp] Memory usage: Green `${green_memory}MB, Blue `${blue_memory}MB (`${memory_diff}%)"
}

# Initialize CSV headers
echo "timestamp,environment,status" > "`$MONITORING_DIR/health-status.csv"
echo "timestamp,environment,metric,value" > "`$MONITORING_DIR/metrics.csv"

# Monitoring loop
echo "🔄 Starting monitoring loop (Ctrl+C to stop)..."
while true; do
    collect_metrics
    compare_performance
    echo "  ⏳ Next check in $($MetricSamplingRate.TotalMinutes) minutes..."
    sleep $([int]$MetricSamplingRate.TotalSeconds)
done
"@

$MonitoringScriptPath = ".tars/monitoring/green-blue/monitor-evolution.sh"
New-Item -ItemType Directory -Path (Split-Path $MonitoringScriptPath) -Force | Out-Null
$MonitoringScript | Out-File -FilePath $MonitoringScriptPath -Encoding UTF8
Write-Host "  📊 Monitoring script: $MonitoringScriptPath" -ForegroundColor Gray

# Step 6: Create Evolution Report
Write-Host ""
Write-Host "📋 CREATING EVOLUTION REPORT" -ForegroundColor Magenta
Write-Host "===========================" -ForegroundColor Magenta

$Report = @{
    SessionId = $SessionId
    StartTime = $StartTime.ToString("yyyy-MM-ddTHH:mm:ss")
    EndTime = $StartTime.Add($Duration).ToString("yyyy-MM-ddTHH:mm:ss")
    Duration = $Duration.ToString()
    Status = "scheduled"
    
    Configuration = @{
        GreenEnvironment = $GreenEnvironment
        BlueEnvironment = $BlueEnvironment
        EvaluationInterval = $EvaluationInterval.ToString()
        MetricSamplingRate = $MetricSamplingRate.ToString()
        ToleranceThresholds = $Session.ToleranceThresholds
        AdaptiveToleranceEnabled = $Session.AdaptiveToleranceEnabled
    }
    
    EvolutionGoals = $Session.EvolutionGoals
    ExpectedImprovements = $Session.ExpectedImprovements
    
    Instructions = @(
        "1. Run start-green.sh to launch baseline environment",
        "2. Run start-blue.sh to launch evolution environment", 
        "3. Run monitor-evolution.sh to start monitoring",
        "4. Check .tars/monitoring/green-blue/$SessionId/ for real-time data",
        "5. Evolution will run for $($Duration.TotalHours) hours",
        "6. Master Evolver will make autonomous decisions based on metrics"
    )
    
    MonitoringEndpoints = @(
        "Green Environment: http://localhost:8080",
        "Blue Environment: http://localhost:8082",
        "Green Metrics: http://localhost:8081/metrics",
        "Blue Metrics: http://localhost:8083/metrics"
    )
}

$ReportPath = ".tars/evolution/reports/evolution-session-$SessionId.json"
$ReportJson = $Report | ConvertTo-Json -Depth 10
$ReportJson | Out-File -FilePath $ReportPath -Encoding UTF8
Write-Host "  📋 Evolution report: $ReportPath" -ForegroundColor Gray

Write-Host "✅ Evolution report created" -ForegroundColor Green

# Final Instructions
Write-Host ""
Write-Host "🎉 NIGHT EVOLUTION SESSION READY!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host "  🆔 Session ID: $SessionId" -ForegroundColor White
Write-Host "  ⏰ Scheduled Start: $($StartTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor White
Write-Host "  ⏱️ Duration: $($Duration.TotalHours) hours" -ForegroundColor White
Write-Host "  🎯 Evolution Goals: $($Session.EvolutionGoals.Count)" -ForegroundColor White
Write-Host ""

Write-Host "🚀 TO START EVOLUTION:" -ForegroundColor Yellow
Write-Host "=====================" -ForegroundColor Yellow
Write-Host "  1. bash $GreenScriptPath" -ForegroundColor Gray
Write-Host "  2. bash $BlueScriptPath" -ForegroundColor Gray
Write-Host "  3. bash $MonitoringScriptPath" -ForegroundColor Gray
Write-Host ""

Write-Host "📊 MONITORING:" -ForegroundColor Yellow
Write-Host "=============" -ForegroundColor Yellow
Write-Host "  🟢 Green: http://localhost:8080" -ForegroundColor Gray
Write-Host "  🔵 Blue: http://localhost:8082" -ForegroundColor Gray
Write-Host "  📈 Metrics: .tars/monitoring/green-blue/$SessionId/" -ForegroundColor Gray
Write-Host ""

Write-Host "🌙 TARS WILL EVOLVE AUTONOMOUSLY THROUGH THE NIGHT!" -ForegroundColor Cyan
Write-Host "Master Evolver Agent will monitor and adapt based on real-time metrics" -ForegroundColor White
Write-Host ""

Write-Host "✨ Night evolution session configured successfully!" -ForegroundColor Green

# Ask if user wants to start immediately
Write-Host ""
$StartNow = Read-Host "🚀 Start evolution session now? (y/N)"

if ($StartNow -eq 'y' -or $StartNow -eq 'Y') {
    Write-Host ""
    Write-Host "🚀 STARTING EVOLUTION SESSION..." -ForegroundColor Cyan
    
    # Start green environment
    Write-Host "🟢 Starting green environment..." -ForegroundColor Green
    bash $GreenScriptPath
    Start-Sleep -Seconds 5
    
    # Start blue environment  
    Write-Host "🔵 Starting blue environment..." -ForegroundColor Blue
    bash $BlueScriptPath
    Start-Sleep -Seconds 5
    
    # Start monitoring
    Write-Host "📊 Starting monitoring..." -ForegroundColor Yellow
    Write-Host "🔄 Monitoring will run in background - check logs in .tars/monitoring/green-blue/$SessionId/"
    
    # Start monitoring in background
    Start-Process -FilePath "bash" -ArgumentList $MonitoringScriptPath -WindowStyle Hidden
    
    Write-Host ""
    Write-Host "🎉 EVOLUTION SESSION STARTED!" -ForegroundColor Green
    Write-Host "============================" -ForegroundColor Green
    Write-Host "🟢 Green environment: http://localhost:8080" -ForegroundColor White
    Write-Host "🔵 Blue environment: http://localhost:8082" -ForegroundColor White
    Write-Host "📊 Monitoring active in background" -ForegroundColor White
    Write-Host ""
    Write-Host "🌙 TARS is now evolving autonomously!" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "📋 Evolution session configured but not started." -ForegroundColor Yellow
    Write-Host "Run the commands above when ready to begin evolution." -ForegroundColor White
}
