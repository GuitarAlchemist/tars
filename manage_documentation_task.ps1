# TARS Documentation Task Manager
# Manages pausable and resumable documentation generation in Windows service

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("status", "start", "pause", "resume", "stop", "progress", "departments", "monitor")]
    [string]$Action = "status",
    
    [Parameter(Mandatory=$false)]
    [string]$ServiceUrl = "http://localhost:5000",
    
    [Parameter(Mandatory=$false)]
    [int]$MonitorInterval = 5
)

Write-Host "TARS DOCUMENTATION TASK MANAGER" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan
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
function Invoke-TarsAPI {
    param(
        [string]$Endpoint,
        [string]$Method = "GET"
    )
    
    try {
        $url = "$ServiceUrl/api/documentation/$Endpoint"
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

# Function to display status
function Show-Status {
    param($StatusData)
    
    if ($StatusData -and $StatusData.success) {
        $status = $StatusData.status
        $progress = $status.Progress
        
        Write-Host "DOCUMENTATION TASK STATUS" -ForegroundColor Cyan
        Write-Host "=========================" -ForegroundColor Cyan
        Write-Host "  State: $($status.State)" -ForegroundColor $(if ($status.State -eq "Running") { "Green" } elseif ($status.State -eq "Paused") { "Yellow" } else { "White" })
        Write-Host "  Progress: $($progress.CompletedTasks)/$($progress.TotalTasks) tasks" -ForegroundColor White
        Write-Host "  Percentage: $([math]::Round(($progress.CompletedTasks / $progress.TotalTasks) * 100, 1))%" -ForegroundColor White
        Write-Host "  Current Task: $($progress.CurrentTask)" -ForegroundColor White
        Write-Host "  Start Time: $($progress.StartTime)" -ForegroundColor Gray
        Write-Host "  Last Update: $($progress.LastUpdateTime)" -ForegroundColor Gray
        
        if ($progress.EstimatedCompletion) {
            Write-Host "  Estimated Completion: $($progress.EstimatedCompletion)" -ForegroundColor Gray
        }
        
        Write-Host ""
        Write-Host "AVAILABLE ACTIONS:" -ForegroundColor Yellow
        if ($status.CanStart) { Write-Host "  start   - Start documentation generation" -ForegroundColor Green }
        if ($status.CanPause) { Write-Host "  pause   - Pause documentation generation" -ForegroundColor Yellow }
        if ($status.CanResume) { Write-Host "  resume  - Resume documentation generation" -ForegroundColor Green }
        Write-Host "  stop    - Stop documentation generation" -ForegroundColor Red
        Write-Host "  progress - Show detailed progress" -ForegroundColor Blue
        Write-Host "  departments - Show department progress" -ForegroundColor Blue
        Write-Host "  monitor - Continuous monitoring" -ForegroundColor Blue
    } else {
        Write-Host "Failed to get status information" -ForegroundColor Red
    }
}

# Function to display progress
function Show-Progress {
    param($ProgressData)
    
    if ($ProgressData -and $ProgressData.success) {
        $progress = $ProgressData.progress
        
        Write-Host "DETAILED PROGRESS INFORMATION" -ForegroundColor Cyan
        Write-Host "=============================" -ForegroundColor Cyan
        Write-Host "  Total Tasks: $($progress.totalTasks)" -ForegroundColor White
        Write-Host "  Completed: $($progress.completedTasks)" -ForegroundColor Green
        Write-Host "  Remaining: $($progress.totalTasks - $progress.completedTasks)" -ForegroundColor Yellow
        Write-Host "  Percentage: $([math]::Round($progress.percentage, 1))%" -ForegroundColor White
        Write-Host "  Current Task: $($progress.currentTask)" -ForegroundColor White
        Write-Host "  Elapsed Time: $($progress.elapsedTime)" -ForegroundColor Gray
        
        if ($progress.estimatedCompletion) {
            Write-Host "  Estimated Completion: $($progress.estimatedCompletion)" -ForegroundColor Gray
        }
        
        Write-Host ""
        Write-Host "DEPARTMENT PROGRESS:" -ForegroundColor Yellow
        foreach ($dept in $progress.departments) {
            $deptName = $dept[0]
            $deptProgress = $dept[1]
            $color = if ($deptProgress -eq 100) { "Green" } elseif ($deptProgress -gt 0) { "Yellow" } else { "Gray" }
            Write-Host "  $deptName`: $deptProgress%" -ForegroundColor $color
        }
    } else {
        Write-Host "Failed to get progress information" -ForegroundColor Red
    }
}

# Function to display department details
function Show-Departments {
    param($DepartmentData)
    
    if ($DepartmentData -and $DepartmentData.success) {
        Write-Host "UNIVERSITY DEPARTMENT STATUS" -ForegroundColor Cyan
        Write-Host "============================" -ForegroundColor Cyan
        Write-Host ""
        
        foreach ($dept in $DepartmentData.departments) {
            $color = switch ($dept.status) {
                "Completed" { "Green" }
                "In Progress" { "Yellow" }
                default { "Gray" }
            }
            
            Write-Host "$($dept.name) Department" -ForegroundColor $color
            Write-Host "  Status: $($dept.status)" -ForegroundColor $color
            Write-Host "  Progress: $($dept.progress)%" -ForegroundColor $color
            Write-Host "  Role: $($dept.description)" -ForegroundColor Gray
            Write-Host ""
        }
        
        Write-Host "Overall State: $($DepartmentData.overallState)" -ForegroundColor White
    } else {
        Write-Host "Failed to get department information" -ForegroundColor Red
    }
}

# Execute the requested action
switch ($Action.ToLower()) {
    "status" {
        Write-Host "Getting documentation task status..." -ForegroundColor Yellow
        $statusData = Invoke-TarsAPI -Endpoint "status"
        Show-Status -StatusData $statusData
    }
    
    "start" {
        Write-Host "Starting documentation generation..." -ForegroundColor Green
        $result = Invoke-TarsAPI -Endpoint "start" -Method "POST"
        
        if ($result -and $result.success) {
            Write-Host "  Documentation generation started successfully!" -ForegroundColor Green
            Write-Host "  Status: $($result.status)" -ForegroundColor White
            Write-Host "  Message: $($result.message)" -ForegroundColor Gray
        } else {
            Write-Host "  Failed to start documentation generation" -ForegroundColor Red
            if ($result.error) {
                Write-Host "  Error: $($result.error)" -ForegroundColor Red
            }
        }
    }
    
    "pause" {
        Write-Host "Pausing documentation generation..." -ForegroundColor Yellow
        $result = Invoke-TarsAPI -Endpoint "pause" -Method "POST"
        
        if ($result -and $result.success) {
            Write-Host "  Documentation generation paused successfully!" -ForegroundColor Yellow
            Write-Host "  Status: $($result.status)" -ForegroundColor White
            Write-Host "  Message: $($result.message)" -ForegroundColor Gray
        } else {
            Write-Host "  Failed to pause documentation generation" -ForegroundColor Red
            if ($result.error) {
                Write-Host "  Error: $($result.error)" -ForegroundColor Red
            }
        }
    }
    
    "resume" {
        Write-Host "Resuming documentation generation..." -ForegroundColor Green
        $result = Invoke-TarsAPI -Endpoint "resume" -Method "POST"
        
        if ($result -and $result.success) {
            Write-Host "  Documentation generation resumed successfully!" -ForegroundColor Green
            Write-Host "  Status: $($result.status)" -ForegroundColor White
            Write-Host "  Message: $($result.message)" -ForegroundColor Gray
        } else {
            Write-Host "  Failed to resume documentation generation" -ForegroundColor Red
            if ($result.error) {
                Write-Host "  Error: $($result.error)" -ForegroundColor Red
            }
        }
    }
    
    "stop" {
        Write-Host "Stopping documentation generation..." -ForegroundColor Red
        $result = Invoke-TarsAPI -Endpoint "stop" -Method "POST"
        
        if ($result -and $result.success) {
            Write-Host "  Documentation generation stopped successfully!" -ForegroundColor Red
            Write-Host "  Status: $($result.status)" -ForegroundColor White
            Write-Host "  Message: $($result.message)" -ForegroundColor Gray
        } else {
            Write-Host "  Failed to stop documentation generation" -ForegroundColor Red
            if ($result.error) {
                Write-Host "  Error: $($result.error)" -ForegroundColor Red
            }
        }
    }
    
    "progress" {
        Write-Host "Getting detailed progress information..." -ForegroundColor Blue
        $progressData = Invoke-TarsAPI -Endpoint "progress"
        Show-Progress -ProgressData $progressData
    }
    
    "departments" {
        Write-Host "Getting department progress..." -ForegroundColor Blue
        $departmentData = Invoke-TarsAPI -Endpoint "departments"
        Show-Departments -DepartmentData $departmentData
    }
    
    "monitor" {
        Write-Host "Starting continuous monitoring (Press Ctrl+C to stop)..." -ForegroundColor Blue
        Write-Host "Refresh interval: $MonitorInterval seconds" -ForegroundColor Gray
        Write-Host ""
        
        try {
            while ($true) {
                Clear-Host
                Write-Host "TARS DOCUMENTATION TASK MONITOR" -ForegroundColor Cyan
                Write-Host "===============================" -ForegroundColor Cyan
                Write-Host "Last updated: $(Get-Date)" -ForegroundColor Gray
                Write-Host ""
                
                $statusData = Invoke-TarsAPI -Endpoint "status"
                Show-Status -StatusData $statusData
                
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
        Write-Host "  status      - Show current task status" -ForegroundColor White
        Write-Host "  start       - Start documentation generation" -ForegroundColor White
        Write-Host "  pause       - Pause documentation generation" -ForegroundColor White
        Write-Host "  resume      - Resume documentation generation" -ForegroundColor White
        Write-Host "  stop        - Stop documentation generation" -ForegroundColor White
        Write-Host "  progress    - Show detailed progress" -ForegroundColor White
        Write-Host "  departments - Show department progress" -ForegroundColor White
        Write-Host "  monitor     - Continuous monitoring" -ForegroundColor White
        Write-Host ""
        Write-Host "Usage examples:" -ForegroundColor Yellow
        Write-Host "  .\manage_documentation_task.ps1 -Action status" -ForegroundColor Gray
        Write-Host "  .\manage_documentation_task.ps1 -Action start" -ForegroundColor Gray
        Write-Host "  .\manage_documentation_task.ps1 -Action monitor -MonitorInterval 10" -ForegroundColor Gray
    }
}

Write-Host ""
