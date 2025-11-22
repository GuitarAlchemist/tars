# TARS Experimental Tracks Manager
# Comprehensive management of parallel experimental tracks across all domains

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("overview", "all", "green", "blue", "ui", "backend", "ai", "infrastructure", "security", "data", "devops", "research", "start-all", "start-green", "start-blue", "comparison", "monitor", "help")]
    [string]$Action = "overview",
    
    [Parameter(Mandatory=$false)]
    [string]$TrackId = "",
    
    [Parameter(Mandatory=$false)]
    [string]$ServiceUrl = "http://localhost:5000",
    
    [Parameter(Mandatory=$false)]
    [int]$MonitorInterval = 5
)

Write-Host "🔬 TARS EXPERIMENTAL TRACKS MANAGER" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
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
function Invoke-TracksAPI {
    param(
        [string]$Endpoint,
        [string]$Method = "GET"
    )
    
    try {
        $url = "$ServiceUrl/api/experimentaltracks/$Endpoint"
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

# Function to display system overview
function Show-SystemOverview {
    param($OverviewData)
    
    if ($OverviewData -and $OverviewData.success) {
        $overview = $OverviewData.overview
        
        Write-Host "EXPERIMENTAL TRACKS SYSTEM OVERVIEW" -ForegroundColor Cyan
        Write-Host "====================================" -ForegroundColor Cyan
        Write-Host ""
        
        Write-Host "📊 SYSTEM STATISTICS:" -ForegroundColor Yellow
        Write-Host "  Total Tracks: $($overview.TotalTracks)" -ForegroundColor White
        Write-Host "  Running: $($overview.RunningTracks)" -ForegroundColor Green
        Write-Host "  Paused: $($overview.PausedTracks)" -ForegroundColor Yellow
        Write-Host "  Completed: $($overview.CompletedTracks)" -ForegroundColor Blue
        Write-Host ""
        
        Write-Host "🎯 DOMAIN BREAKDOWN:" -ForegroundColor Yellow
        foreach ($domain in $overview.DomainBreakdown.PSObject.Properties) {
            Write-Host "  $($domain.Name): $($domain.Value) tracks" -ForegroundColor White
        }
        Write-Host ""
        
        Write-Host "🔄 TRACK TYPE BREAKDOWN:" -ForegroundColor Yellow
        foreach ($type in $overview.TypeBreakdown.PSObject.Properties) {
            $color = if ($type.Name -eq "Green") { "Green" } else { "Blue" }
            Write-Host "  $($type.Name): $($type.Value) tracks" -ForegroundColor $color
        }
        Write-Host ""
        
        Write-Host "📈 RESOURCE ALLOCATION:" -ForegroundColor Yellow
        foreach ($allocation in $overview.ResourceAllocation.PSObject.Properties) {
            $color = if ($allocation.Name -eq "Green") { "Green" } else { "Blue" }
            Write-Host "  $($allocation.Name): $([math]::Round($allocation.Value, 1))%" -ForegroundColor $color
        }
        
    } else {
        Write-Host "Failed to get system overview" -ForegroundColor Red
    }
}

# Function to display tracks
function Show-Tracks {
    param($TracksData, $Title = "TRACKS")
    
    if ($TracksData -and $TracksData.success) {
        Write-Host "$Title" -ForegroundColor Cyan
        Write-Host ("=" * $Title.Length) -ForegroundColor Cyan
        Write-Host ""
        
        foreach ($track in $TracksData.tracks) {
            $statusColor = switch ($track.Status) {
                "Running" { "Green" }
                "Paused" { "Yellow" }
                "Completed" { "Blue" }
                "NotStarted" { "Gray" }
                default { "Red" }
            }
            
            $typeColor = if ($track.Type -eq "Green") { "Green" } else { "Blue" }
            $percentage = if ($track.TotalTasks -gt 0) { [math]::Round(($track.Progress / $track.TotalTasks) * 100, 1) } else { 0 }
            
            Write-Host "🔬 $($track.Name)" -ForegroundColor $typeColor
            Write-Host "   ID: $($track.Id)" -ForegroundColor Gray
            Write-Host "   Domain: $($track.Domain) | Type: $($track.Type)" -ForegroundColor White
            Write-Host "   Status: $($track.Status)" -ForegroundColor $statusColor
            Write-Host "   Progress: $($track.Progress)/$($track.TotalTasks) ($percentage%)" -ForegroundColor White
            Write-Host "   Current: $($track.CurrentTask)" -ForegroundColor Gray
            Write-Host "   Technologies: $($track.Technologies -join ', ')" -ForegroundColor Gray
            Write-Host "   Resource Allocation: $($track.ResourceAllocation)%" -ForegroundColor Gray
            Write-Host ""
        }
        
        Write-Host "Total: $($TracksData.count) tracks" -ForegroundColor White
        
    } else {
        Write-Host "Failed to get tracks data" -ForegroundColor Red
    }
}

# Function to display comparison
function Show-Comparison {
    param($ComparisonData)
    
    if ($ComparisonData -and $ComparisonData.success) {
        $green = $ComparisonData.comparison.green
        $blue = $ComparisonData.comparison.blue
        
        Write-Host "EXPERIMENTAL TRACKS COMPARISON" -ForegroundColor Cyan
        Write-Host "==============================" -ForegroundColor Cyan
        Write-Host ""
        
        Write-Host "GREEN TRACKS (STABLE)" -ForegroundColor Green
        Write-Host "====================" -ForegroundColor Green
        Write-Host "  Purpose: $($green.purpose)" -ForegroundColor White
        Write-Host "  Track Count: $($green.trackCount)" -ForegroundColor White
        Write-Host "  Progress: $($green.totalProgress)/$($green.totalTasks) ($([math]::Round($green.percentage, 1))%)" -ForegroundColor White
        Write-Host "  Running Tracks: $($green.runningTracks)" -ForegroundColor White
        Write-Host "  Resource Allocation: $([math]::Round($green.resourceAllocation, 1))%" -ForegroundColor White
        Write-Host ""
        
        Write-Host "BLUE TRACKS (EXPERIMENTAL)" -ForegroundColor Blue
        Write-Host "=========================" -ForegroundColor Blue
        Write-Host "  Purpose: $($blue.purpose)" -ForegroundColor White
        Write-Host "  Track Count: $($blue.trackCount)" -ForegroundColor White
        Write-Host "  Progress: $($blue.totalProgress)/$($blue.totalTasks) ($([math]::Round($blue.percentage, 1))%)" -ForegroundColor White
        Write-Host "  Running Tracks: $($blue.runningTracks)" -ForegroundColor White
        Write-Host "  Resource Allocation: $([math]::Round($blue.resourceAllocation, 1))%" -ForegroundColor White
        Write-Host ""
        
        Write-Host "STRATEGY: $($ComparisonData.comparison.strategy)" -ForegroundColor Cyan
        
    } else {
        Write-Host "Failed to get comparison data" -ForegroundColor Red
    }
}

# Execute the requested action
switch ($Action.ToLower()) {
    "overview" {
        Write-Host "Getting experimental tracks system overview..." -ForegroundColor Yellow
        $overviewData = Invoke-TracksAPI -Endpoint "overview"
        Show-SystemOverview -OverviewData $overviewData
    }
    
    "all" {
        Write-Host "Getting all experimental tracks..." -ForegroundColor Yellow
        $tracksData = Invoke-TracksAPI -Endpoint "all"
        Show-Tracks -TracksData $tracksData -Title "ALL EXPERIMENTAL TRACKS"
    }
    
    "green" {
        Write-Host "Getting green (stable) tracks..." -ForegroundColor Green
        $tracksData = Invoke-TracksAPI -Endpoint "type/green"
        Show-Tracks -TracksData $tracksData -Title "GREEN TRACKS (STABLE)"
    }
    
    "blue" {
        Write-Host "Getting blue (experimental) tracks..." -ForegroundColor Blue
        $tracksData = Invoke-TracksAPI -Endpoint "type/blue"
        Show-Tracks -TracksData $tracksData -Title "BLUE TRACKS (EXPERIMENTAL)"
    }
    
    "ui" {
        Write-Host "Getting UI domain tracks..." -ForegroundColor Magenta
        $tracksData = Invoke-TracksAPI -Endpoint "domain/ui"
        Show-Tracks -TracksData $tracksData -Title "UI DOMAIN TRACKS"
    }
    
    "backend" {
        Write-Host "Getting Backend domain tracks..." -ForegroundColor DarkGreen
        $tracksData = Invoke-TracksAPI -Endpoint "domain/backend"
        Show-Tracks -TracksData $tracksData -Title "BACKEND DOMAIN TRACKS"
    }
    
    "ai" {
        Write-Host "Getting AI/ML domain tracks..." -ForegroundColor DarkMagenta
        $tracksData = Invoke-TracksAPI -Endpoint "domain/ai"
        Show-Tracks -TracksData $tracksData -Title "AI/ML DOMAIN TRACKS"
    }
    
    "infrastructure" {
        Write-Host "Getting Infrastructure domain tracks..." -ForegroundColor DarkCyan
        $tracksData = Invoke-TracksAPI -Endpoint "domain/infrastructure"
        Show-Tracks -TracksData $tracksData -Title "INFRASTRUCTURE DOMAIN TRACKS"
    }
    
    "security" {
        Write-Host "Getting Security domain tracks..." -ForegroundColor Red
        $tracksData = Invoke-TracksAPI -Endpoint "domain/security"
        Show-Tracks -TracksData $tracksData -Title "SECURITY DOMAIN TRACKS"
    }
    
    "data" {
        Write-Host "Getting Data domain tracks..." -ForegroundColor DarkBlue
        $tracksData = Invoke-TracksAPI -Endpoint "domain/data"
        Show-Tracks -TracksData $tracksData -Title "DATA DOMAIN TRACKS"
    }
    
    "devops" {
        Write-Host "Getting DevOps domain tracks..." -ForegroundColor DarkYellow
        $tracksData = Invoke-TracksAPI -Endpoint "domain/devops"
        Show-Tracks -TracksData $tracksData -Title "DEVOPS DOMAIN TRACKS"
    }
    
    "research" {
        Write-Host "Getting Research domain tracks..." -ForegroundColor Cyan
        $tracksData = Invoke-TracksAPI -Endpoint "domain/research"
        Show-Tracks -TracksData $tracksData -Title "RESEARCH DOMAIN TRACKS"
    }
    
    "start-all" {
        Write-Host "Starting all experimental tracks..." -ForegroundColor Green
        
        # Start green tracks
        $greenResult = Invoke-TracksAPI -Endpoint "type/green/start" -Method "POST"
        if ($greenResult -and $greenResult.success) {
            Write-Host "  Green tracks: $($greenResult.startedTracks)/$($greenResult.totalTracks) started" -ForegroundColor Green
        }
        
        # Start blue tracks
        $blueResult = Invoke-TracksAPI -Endpoint "type/blue/start" -Method "POST"
        if ($blueResult -and $blueResult.success) {
            Write-Host "  Blue tracks: $($blueResult.startedTracks)/$($blueResult.totalTracks) started" -ForegroundColor Blue
        }
    }
    
    "start-green" {
        Write-Host "Starting all green (stable) tracks..." -ForegroundColor Green
        $result = Invoke-TracksAPI -Endpoint "type/green/start" -Method "POST"
        
        if ($result -and $result.success) {
            Write-Host "  Green tracks started: $($result.startedTracks)/$($result.totalTracks)" -ForegroundColor Green
        } else {
            Write-Host "  Failed to start green tracks" -ForegroundColor Red
        }
    }
    
    "start-blue" {
        Write-Host "Starting all blue (experimental) tracks..." -ForegroundColor Blue
        $result = Invoke-TracksAPI -Endpoint "type/blue/start" -Method "POST"
        
        if ($result -and $result.success) {
            Write-Host "  Blue tracks started: $($result.startedTracks)/$($result.totalTracks)" -ForegroundColor Blue
        } else {
            Write-Host "  Failed to start blue tracks" -ForegroundColor Red
        }
    }
    
    "comparison" {
        Write-Host "Getting experimental tracks comparison..." -ForegroundColor Cyan
        $comparisonData = Invoke-TracksAPI -Endpoint "comparison"
        Show-Comparison -ComparisonData $comparisonData
    }
    
    "monitor" {
        Write-Host "Starting experimental tracks monitoring (Press Ctrl+C to stop)..." -ForegroundColor Blue
        Write-Host "Refresh interval: $MonitorInterval seconds" -ForegroundColor Gray
        Write-Host ""
        
        try {
            while ($true) {
                Clear-Host
                Write-Host "🔬 TARS EXPERIMENTAL TRACKS MONITOR" -ForegroundColor Cyan
                Write-Host "===================================" -ForegroundColor Cyan
                Write-Host "Last updated: $(Get-Date)" -ForegroundColor Gray
                Write-Host ""
                
                $overviewData = Invoke-TracksAPI -Endpoint "overview"
                Show-SystemOverview -OverviewData $overviewData
                
                Write-Host ""
                Write-Host "Press Ctrl+C to stop monitoring..." -ForegroundColor Gray
                
                Start-Sleep -Seconds $MonitorInterval
            }
        } catch {
            Write-Host ""
            Write-Host "Monitoring stopped." -ForegroundColor Yellow
        }
    }
    
    "help" {
        Write-Host "TARS EXPERIMENTAL TRACKS MANAGER - HELP" -ForegroundColor Cyan
        Write-Host "=======================================" -ForegroundColor Cyan
        Write-Host ""
        
        Write-Host "AVAILABLE ACTIONS:" -ForegroundColor Yellow
        Write-Host "  overview      - Show system overview and statistics" -ForegroundColor White
        Write-Host "  all           - Show all experimental tracks" -ForegroundColor White
        Write-Host "  green         - Show green (stable) tracks only" -ForegroundColor White
        Write-Host "  blue          - Show blue (experimental) tracks only" -ForegroundColor White
        Write-Host ""
        
        Write-Host "DOMAIN-SPECIFIC ACTIONS:" -ForegroundColor Yellow
        Write-Host "  ui            - Show UI domain tracks" -ForegroundColor White
        Write-Host "  backend       - Show Backend domain tracks" -ForegroundColor White
        Write-Host "  ai            - Show AI/ML domain tracks" -ForegroundColor White
        Write-Host "  infrastructure- Show Infrastructure domain tracks" -ForegroundColor White
        Write-Host "  security      - Show Security domain tracks" -ForegroundColor White
        Write-Host "  data          - Show Data domain tracks" -ForegroundColor White
        Write-Host "  devops        - Show DevOps domain tracks" -ForegroundColor White
        Write-Host "  research      - Show Research domain tracks" -ForegroundColor White
        Write-Host ""
        
        Write-Host "CONTROL ACTIONS:" -ForegroundColor Yellow
        Write-Host "  start-all     - Start all experimental tracks" -ForegroundColor White
        Write-Host "  start-green   - Start all green (stable) tracks" -ForegroundColor White
        Write-Host "  start-blue    - Start all blue (experimental) tracks" -ForegroundColor White
        Write-Host ""
        
        Write-Host "ANALYSIS ACTIONS:" -ForegroundColor Yellow
        Write-Host "  comparison    - Compare green vs blue tracks" -ForegroundColor White
        Write-Host "  monitor       - Live monitoring dashboard" -ForegroundColor White
        Write-Host "  help          - Show this help message" -ForegroundColor White
        Write-Host ""
        
        Write-Host "USAGE EXAMPLES:" -ForegroundColor Yellow
        Write-Host "  .\manage_experimental_tracks.ps1 -Action overview" -ForegroundColor Gray
        Write-Host "  .\manage_experimental_tracks.ps1 -Action start-all" -ForegroundColor Gray
        Write-Host "  .\manage_experimental_tracks.ps1 -Action ui" -ForegroundColor Gray
        Write-Host "  .\manage_experimental_tracks.ps1 -Action monitor -MonitorInterval 10" -ForegroundColor Gray
    }
    
    default {
        Write-Host "Invalid action: $Action" -ForegroundColor Red
        Write-Host "Use -Action help to see available actions" -ForegroundColor Yellow
    }
}

Write-Host ""
