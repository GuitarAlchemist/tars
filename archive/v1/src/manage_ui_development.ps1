# TARS UI Development Manager
# Manages parallel Green (stable) and Blue (experimental) UI development tracks

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("status", "start", "pause", "resume", "green-start", "green-pause", "green-resume", "blue-start", "blue-pause", "blue-resume", "comparison", "monitor")]
    [string]$Action = "status",
    
    [Parameter(Mandatory=$false)]
    [string]$ServiceUrl = "http://localhost:5000",
    
    [Parameter(Mandatory=$false)]
    [int]$MonitorInterval = 5
)

Write-Host "TARS UI DEVELOPMENT MANAGER" -ForegroundColor Cyan
Write-Host "===========================" -ForegroundColor Cyan
Write-Host ""

# Check if TARS service is running
Write-Host "Checking TARS Windows Service..." -ForegroundColor Yellow
$tarsService = Get-Service -Name "TarsService" -ErrorAction SilentlyContinue

if ($tarsService -and $tarsService.Status -eq "Running") {
    Write-Host "  TARS Windows Service is running" -ForegroundColor Green
} else {
    Write-Host "  TARS Windows Service is not running" -ForegroundColor Red
    Write-Host "  Please start the service first:" -ForegroundColor Yellow
    Write-Host "    dotnet run --project TarsServiceManager -- service start" -ForegroundColor Gray
    exit 1
}

Write-Host ""

# Function to make API calls
function Invoke-UIAPI {
    param(
        [string]$Endpoint,
        [string]$Method = "GET"
    )
    
    try {
        $url = "$ServiceUrl/api/ui/$Endpoint"
        Write-Host "  Calling: $Method $url" -ForegroundColor Gray
        
        if ($Method -eq "GET") {
            $response = Invoke-RestMethod -Uri $url -Method $Method -ContentType "application/json"
        } else {
            $response = Invoke-RestMethod -Uri $url -Method $Method -ContentType "application/json" -Body "{}"
        }
        
        return $response
    } catch {
        Write-Host "  API call failed: $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

# Function to display UI status
function Show-UIStatus {
    param($StatusData)
    
    if ($StatusData -and $StatusData.success) {
        Write-Host "UI DEVELOPMENT TRACKS STATUS" -ForegroundColor Cyan
        Write-Host "============================" -ForegroundColor Cyan
        Write-Host ""
        
        if ($StatusData.status.GreenTrack) {
            # Overall status
            $greenTrack = $StatusData.status.GreenTrack
            $blueTrack = $StatusData.status.BlueTrack
            
            Write-Host "GREEN UI TRACK (STABLE PRODUCTION)" -ForegroundColor Green
            Write-Host "==================================" -ForegroundColor Green
            Write-Host "  Status: $($greenTrack.State)" -ForegroundColor $(if ($greenTrack.State -eq "Running") { "Green" } elseif ($greenTrack.State -eq "Paused") { "Yellow" } else { "White" })
            Write-Host "  Progress: $($greenTrack.Progress.CompletedTasks)/$($greenTrack.Progress.TotalTasks) tasks" -ForegroundColor White
            Write-Host "  Percentage: $([math]::Round(($greenTrack.Progress.CompletedTasks / $greenTrack.Progress.TotalTasks) * 100, 1))%" -ForegroundColor White
            Write-Host "  Current: $($greenTrack.Progress.CurrentTask)" -ForegroundColor White
            Write-Host "  Purpose: Production maintenance and stability" -ForegroundColor Gray
            Write-Host ""
            
            Write-Host "BLUE UI TRACK (EXPERIMENTAL)" -ForegroundColor Blue
            Write-Host "============================" -ForegroundColor Blue
            Write-Host "  Status: $($blueTrack.State)" -ForegroundColor $(if ($blueTrack.State -eq "Running") { "Green" } elseif ($blueTrack.State -eq "Paused") { "Yellow" } else { "White" })
            Write-Host "  Progress: $($blueTrack.Progress.CompletedTasks)/$($blueTrack.Progress.TotalTasks) tasks" -ForegroundColor White
            Write-Host "  Percentage: $([math]::Round(($blueTrack.Progress.CompletedTasks / $blueTrack.Progress.TotalTasks) * 100, 1))%" -ForegroundColor White
            Write-Host "  Current: $($blueTrack.Progress.CurrentTask)" -ForegroundColor White
            Write-Host "  Purpose: Next-generation UI development" -ForegroundColor Gray
            Write-Host ""
            
        } else {
            # Single track status
            $track = $StatusData.status
            $trackColor = if ($StatusData.track -eq "Green (Stable)") { "Green" } else { "Blue" }
            
            Write-Host "$($StatusData.track.ToUpper()) TRACK" -ForegroundColor $trackColor
            Write-Host "=" * ($StatusData.track.Length + 6) -ForegroundColor $trackColor
            Write-Host "  Status: $($track.State)" -ForegroundColor $(if ($track.State -eq "Running") { "Green" } elseif ($track.State -eq "Paused") { "Yellow" } else { "White" })
            Write-Host "  Progress: $($track.Progress.CompletedTasks)/$($track.Progress.TotalTasks) tasks" -ForegroundColor White
            Write-Host "  Percentage: $([math]::Round(($track.Progress.CompletedTasks / $track.Progress.TotalTasks) * 100, 1))%" -ForegroundColor White
            Write-Host "  Current: $($track.Progress.CurrentTask)" -ForegroundColor White
            Write-Host ""
        }
        
        Write-Host "AVAILABLE ACTIONS:" -ForegroundColor Yellow
        Write-Host "  start        - Start both UI tracks" -ForegroundColor Green
        Write-Host "  green-start  - Start Green UI maintenance" -ForegroundColor Green
        Write-Host "  blue-start   - Start Blue UI development" -ForegroundColor Green
        Write-Host "  pause        - Pause both UI tracks" -ForegroundColor Yellow
        Write-Host "  green-pause  - Pause Green UI maintenance" -ForegroundColor Yellow
        Write-Host "  blue-pause   - Pause Blue UI development" -ForegroundColor Yellow
        Write-Host "  resume       - Resume both UI tracks" -ForegroundColor Green
        Write-Host "  green-resume - Resume Green UI maintenance" -ForegroundColor Green
        Write-Host "  blue-resume  - Resume Blue UI development" -ForegroundColor Green
        Write-Host "  comparison   - Compare both tracks" -ForegroundColor Blue
        Write-Host "  monitor      - Live monitoring" -ForegroundColor Blue
        
    } else {
        Write-Host "Failed to get UI development status" -ForegroundColor Red
    }
}

# Function to display comparison
function Show-UIComparison {
    param($ComparisonData)
    
    if ($ComparisonData -and $ComparisonData.success) {
        $green = $ComparisonData.comparison.green
        $blue = $ComparisonData.comparison.blue
        
        Write-Host "UI DEVELOPMENT TRACKS COMPARISON" -ForegroundColor Cyan
        Write-Host "================================" -ForegroundColor Cyan
        Write-Host ""
        
        Write-Host "TRACK COMPARISON TABLE" -ForegroundColor Yellow
        Write-Host "=====================" -ForegroundColor Yellow
        Write-Host ""
        
        $format = "{0,-20} | {1,-25} | {2,-25}"
        Write-Host ($format -f "Aspect", "Green (Stable)", "Blue (Experimental)") -ForegroundColor White
        Write-Host ("-" * 75) -ForegroundColor Gray
        Write-Host ($format -f "Purpose", $green.purpose, $blue.purpose) -ForegroundColor White
        Write-Host ($format -f "Status", $green.status, $blue.status) -ForegroundColor White
        Write-Host ($format -f "Progress", $green.progress, $blue.progress) -ForegroundColor White
        Write-Host ($format -f "Percentage", "$([math]::Round($green.percentage, 1))%", "$([math]::Round($blue.percentage, 1))%") -ForegroundColor White
        Write-Host ($format -f "Focus", $green.focus, $blue.focus) -ForegroundColor White
        Write-Host ""
        
        Write-Host "CURRENT ACTIVITIES:" -ForegroundColor Yellow
        Write-Host "  Green Track: $($green.currentTask)" -ForegroundColor Green
        Write-Host "  Blue Track:  $($blue.currentTask)" -ForegroundColor Blue
        Write-Host ""
        
        Write-Host "STRATEGY: $($ComparisonData.comparison.strategy)" -ForegroundColor Cyan
        
    } else {
        Write-Host "Failed to get UI comparison data" -ForegroundColor Red
    }
}

# Execute the requested action
switch ($Action.ToLower()) {
    "status" {
        Write-Host "Getting UI development status..." -ForegroundColor Yellow
        $statusData = Invoke-UIAPI -Endpoint "status"
        Show-UIStatus -StatusData $statusData
    }
    
    "start" {
        Write-Host "Starting both UI development tracks..." -ForegroundColor Green
        $result = Invoke-UIAPI -Endpoint "start" -Method "POST"
        
        if ($result -and $result.success) {
            Write-Host "  Both UI tracks started successfully!" -ForegroundColor Green
            Write-Host "  Green Track: $($result.tracks.green)" -ForegroundColor Green
            Write-Host "  Blue Track: $($result.tracks.blue)" -ForegroundColor Blue
        } else {
            Write-Host "  Failed to start UI tracks" -ForegroundColor Red
        }
    }
    
    "green-start" {
        Write-Host "Starting Green UI maintenance..." -ForegroundColor Green
        $result = Invoke-UIAPI -Endpoint "green/start" -Method "POST"
        
        if ($result -and $result.success) {
            Write-Host "  Green UI maintenance started successfully!" -ForegroundColor Green
            Write-Host "  Status: $($result.status)" -ForegroundColor White
        } else {
            Write-Host "  Failed to start Green UI maintenance" -ForegroundColor Red
        }
    }
    
    "blue-start" {
        Write-Host "Starting Blue UI development..." -ForegroundColor Blue
        $result = Invoke-UIAPI -Endpoint "blue/start" -Method "POST"
        
        if ($result -and $result.success) {
            Write-Host "  Blue UI development started successfully!" -ForegroundColor Blue
            Write-Host "  Status: $($result.status)" -ForegroundColor White
        } else {
            Write-Host "  Failed to start Blue UI development" -ForegroundColor Red
        }
    }
    
    "green-pause" {
        Write-Host "Pausing Green UI maintenance..." -ForegroundColor Yellow
        $result = Invoke-UIAPI -Endpoint "green/pause" -Method "POST"
        
        if ($result -and $result.success) {
            Write-Host "  Green UI maintenance paused successfully!" -ForegroundColor Yellow
            Write-Host "  Status: $($result.status)" -ForegroundColor White
        } else {
            Write-Host "  Failed to pause Green UI maintenance" -ForegroundColor Red
        }
    }
    
    "blue-pause" {
        Write-Host "Pausing Blue UI development..." -ForegroundColor Yellow
        $result = Invoke-UIAPI -Endpoint "blue/pause" -Method "POST"
        
        if ($result -and $result.success) {
            Write-Host "  Blue UI development paused successfully!" -ForegroundColor Yellow
            Write-Host "  Status: $($result.status)" -ForegroundColor White
        } else {
            Write-Host "  Failed to pause Blue UI development" -ForegroundColor Red
        }
    }
    
    "green-resume" {
        Write-Host "Resuming Green UI maintenance..." -ForegroundColor Green
        $result = Invoke-UIAPI -Endpoint "green/resume" -Method "POST"
        
        if ($result -and $result.success) {
            Write-Host "  Green UI maintenance resumed successfully!" -ForegroundColor Green
            Write-Host "  Status: $($result.status)" -ForegroundColor White
        } else {
            Write-Host "  Failed to resume Green UI maintenance" -ForegroundColor Red
        }
    }
    
    "blue-resume" {
        Write-Host "Resuming Blue UI development..." -ForegroundColor Green
        $result = Invoke-UIAPI -Endpoint "blue/resume" -Method "POST"
        
        if ($result -and $result.success) {
            Write-Host "  Blue UI development resumed successfully!" -ForegroundColor Green
            Write-Host "  Status: $($result.status)" -ForegroundColor White
        } else {
            Write-Host "  Failed to resume Blue UI development" -ForegroundColor Red
        }
    }
    
    "comparison" {
        Write-Host "Getting UI tracks comparison..." -ForegroundColor Blue
        $comparisonData = Invoke-UIAPI -Endpoint "comparison"
        Show-UIComparison -ComparisonData $comparisonData
    }
    
    "monitor" {
        Write-Host "Starting UI development monitoring (Press Ctrl+C to stop)..." -ForegroundColor Blue
        Write-Host "Refresh interval: $MonitorInterval seconds" -ForegroundColor Gray
        Write-Host ""
        
        try {
            while ($true) {
                Clear-Host
                Write-Host "TARS UI DEVELOPMENT MONITOR" -ForegroundColor Cyan
                Write-Host "===========================" -ForegroundColor Cyan
                Write-Host "Last updated: $(Get-Date)" -ForegroundColor Gray
                Write-Host ""
                
                $statusData = Invoke-UIAPI -Endpoint "status"
                Show-UIStatus -StatusData $statusData
                
                Write-Host ""
                Write-Host "Press Ctrl+C to stop monitoring..." -ForegroundColor Gray
                
                Start-Sleep -Seconds $MonitorInterval
            }
        } catch {
            Write-Host ""
            Write-Host "Monitoring stopped." -ForegroundColor Yellow
        }
    }
    
    default {
        Write-Host "Invalid action: $Action" -ForegroundColor Red
        Write-Host ""
        Write-Host "Available actions:" -ForegroundColor Yellow
        Write-Host "  status       - Show UI development status" -ForegroundColor White
        Write-Host "  start        - Start both UI tracks" -ForegroundColor White
        Write-Host "  green-start  - Start Green UI maintenance" -ForegroundColor White
        Write-Host "  blue-start   - Start Blue UI development" -ForegroundColor White
        Write-Host "  green-pause  - Pause Green UI maintenance" -ForegroundColor White
        Write-Host "  blue-pause   - Pause Blue UI development" -ForegroundColor White
        Write-Host "  green-resume - Resume Green UI maintenance" -ForegroundColor White
        Write-Host "  blue-resume  - Resume Blue UI development" -ForegroundColor White
        Write-Host "  comparison   - Compare both UI tracks" -ForegroundColor White
        Write-Host "  monitor      - Live monitoring" -ForegroundColor White
        Write-Host ""
        Write-Host "Usage examples:" -ForegroundColor Yellow
        Write-Host "  .\manage_ui_development.ps1 -Action status" -ForegroundColor Gray
        Write-Host "  .\manage_ui_development.ps1 -Action start" -ForegroundColor Gray
        Write-Host "  .\manage_ui_development.ps1 -Action comparison" -ForegroundColor Gray
        Write-Host "  .\manage_ui_development.ps1 -Action monitor -MonitorInterval 10" -ForegroundColor Gray
    }
}

Write-Host ""
