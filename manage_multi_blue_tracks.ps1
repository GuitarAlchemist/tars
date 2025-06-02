# TARS Multi-Blue Tracks Manager
# Advanced management of multiple experimental paths per domain

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("overview", "all", "ui", "backend", "ai", "infrastructure", "security", "data", "devops", "research", "competition", "strategy", "start-blue", "alpha", "beta", "gamma", "delta", "monitor", "help")]
    [string]$Action = "overview",
    
    [Parameter(Mandatory=$false)]
    [string]$Domain = "",
    
    [Parameter(Mandatory=$false)]
    [string]$Strategy = "",
    
    [Parameter(Mandatory=$false)]
    [string]$ServiceUrl = "http://localhost:5000",
    
    [Parameter(Mandatory=$false)]
    [int]$MonitorInterval = 5
)

Write-Host "🔬 TARS MULTI-BLUE TRACKS MANAGER" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
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
function Invoke-MultiBluAPI {
    param(
        [string]$Endpoint,
        [string]$Method = "GET"
    )
    
    try {
        $url = "$ServiceUrl/api/multiblutracks/$Endpoint"
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

# Function to display multi-blue overview
function Show-MultiBluOverview {
    param($OverviewData)
    
    if ($OverviewData -and $OverviewData.success) {
        $overview = $OverviewData.overview
        
        Write-Host "MULTI-BLUE EXPERIMENTAL TRACKS OVERVIEW" -ForegroundColor Cyan
        Write-Host "=======================================" -ForegroundColor Cyan
        Write-Host ""
        
        Write-Host "📊 SYSTEM STATISTICS:" -ForegroundColor Yellow
        Write-Host "  Total Tracks: $($overview.TotalTracks)" -ForegroundColor White
        Write-Host "  Green Tracks: $($overview.GreenTracks)" -ForegroundColor Green
        Write-Host "  Blue-Alpha Tracks: $($overview.BlueAlphaTracks)" -ForegroundColor Blue
        Write-Host "  Blue-Beta Tracks: $($overview.BlueBetaTracks)" -ForegroundColor Blue
        Write-Host "  Blue-Gamma Tracks: $($overview.BlueGammaTracks)" -ForegroundColor Blue
        Write-Host "  Blue-Delta Tracks: $($overview.BlueDeltaTracks)" -ForegroundColor Blue
        Write-Host "  Running Tracks: $($overview.RunningTracks)" -ForegroundColor Green
        Write-Host ""
        
        Write-Host "🎯 DOMAIN BREAKDOWN:" -ForegroundColor Yellow
        foreach ($domain in $overview.DomainBreakdown.PSObject.Properties) {
            $domainData = $domain.Value
            Write-Host "  $($domain.Name):" -ForegroundColor White
            Write-Host "    Total: $($domainData.Total) | Green: $($domainData.Green) | Blue: $($domainData.BlueCount)" -ForegroundColor Gray
            if ($domainData.BlueTypes.Count -gt 0) {
                Write-Host "    Blue Types: $($domainData.BlueTypes -join ', ')" -ForegroundColor Blue
            }
        }
        Write-Host ""
        
        Write-Host "🔄 STRATEGY BREAKDOWN:" -ForegroundColor Yellow
        foreach ($strategy in $overview.StrategyBreakdown.PSObject.Properties) {
            Write-Host "  $($strategy.Name): $($strategy.Value) tracks" -ForegroundColor White
        }
        Write-Host ""
        
        Write-Host "⚠️ RISK BREAKDOWN:" -ForegroundColor Yellow
        foreach ($risk in $overview.RiskBreakdown.PSObject.Properties) {
            $color = switch ($risk.Name) {
                "Low" { "Green" }
                "Medium" { "Yellow" }
                "High" { "Red" }
                "VeryHigh" { "Magenta" }
                default { "White" }
            }
            Write-Host "  $($risk.Name): $($risk.Value) tracks" -ForegroundColor $color
        }
        
        Write-Host ""
        Write-Host "🌟 CAPABILITIES:" -ForegroundColor Cyan
        $capabilities = $OverviewData.capabilities
        Write-Host "  Multiple Blue Paths: $($capabilities.multipleBluePathsPerDomain)" -ForegroundColor Green
        Write-Host "  Max Blue Tracks/Domain: $($capabilities.maxBlueTracksPerDomain)" -ForegroundColor White
        Write-Host "  Supported Strategies: $($capabilities.supportedStrategies.Count)" -ForegroundColor White
        Write-Host "  Risk Levels: $($capabilities.supportedRiskLevels.Count)" -ForegroundColor White
        
    } else {
        Write-Host "Failed to get multi-blue overview" -ForegroundColor Red
    }
}

# Function to display domain analysis
function Show-DomainAnalysis {
    param($AnalysisData, $DomainName)
    
    if ($AnalysisData -and $AnalysisData.success) {
        $analysis = $AnalysisData.analysis
        
        Write-Host "$($DomainName.ToUpper()) DOMAIN MULTI-BLUE ANALYSIS" -ForegroundColor Cyan
        Write-Host ("=" * ($DomainName.Length + 30)) -ForegroundColor Cyan
        Write-Host ""
        
        # Green Track
        if ($analysis.GreenTrack) {
            $green = $analysis.GreenTrack
            Write-Host "🟢 GREEN TRACK (STABLE)" -ForegroundColor Green
            Write-Host "  Name: $($green.Name)" -ForegroundColor White
            Write-Host "  Progress: $($green.Progress)/$($green.TotalTasks)" -ForegroundColor White
            Write-Host "  Resource Allocation: $($green.ResourceAllocation)%" -ForegroundColor White
            Write-Host ""
        }
        
        # Blue Tracks Competition
        Write-Host "🔵 BLUE TRACKS COMPETITION" -ForegroundColor Blue
        Write-Host "=========================" -ForegroundColor Blue
        
        foreach ($track in $analysis.CompetitionAnalysis) {
            $statusColor = switch ($track.RiskLevel) {
                "Low" { "Green" }
                "Medium" { "Yellow" }
                "High" { "Red" }
                "VeryHigh" { "Magenta" }
                default { "White" }
            }
            
            Write-Host ""
            Write-Host "  🔬 $($track.Name)" -ForegroundColor Blue
            Write-Host "     ID: $($track.TrackId)" -ForegroundColor Gray
            Write-Host "     Progress: $($track.Progress)/$($track.TotalTasks) ($([math]::Round($track.Percentage, 1))%)" -ForegroundColor White
            Write-Host "     Risk Level: $($track.RiskLevel)" -ForegroundColor $statusColor
            Write-Host "     Resource Allocation: $($track.ResourceAllocation)%" -ForegroundColor White
            Write-Host "     Timeline: $($track.Timeline)" -ForegroundColor Gray
            Write-Host "     Strategy: $($track.Strategy)" -ForegroundColor Gray
            if ($track.SuccessMetrics.Count -gt 0) {
                Write-Host "     Success Metrics: $($track.SuccessMetrics -join ', ')" -ForegroundColor Gray
            }
            if ($track.CompetingWith.Count -gt 0) {
                Write-Host "     Competing With: $($track.CompetingWith -join ', ')" -ForegroundColor Yellow
            }
        }
        
        Write-Host ""
        Write-Host "📊 COMPETITION SUMMARY:" -ForegroundColor Yellow
        Write-Host "  Total Blue Resources: $([math]::Round($analysis.TotalBlueResources, 1))%" -ForegroundColor White
        if ($analysis.LeadingTrack) {
            Write-Host "  Current Leader: $($analysis.LeadingTrack)" -ForegroundColor Green
        }
        
    } else {
        Write-Host "Failed to get domain analysis" -ForegroundColor Red
    }
}

# Function to display competitive analysis
function Show-CompetitiveAnalysis {
    param($CompetitionData, $DomainName)
    
    if ($CompetitionData -and $CompetitionData.success) {
        $analysis = $CompetitionData.competitiveAnalysis
        
        Write-Host "$($DomainName.ToUpper()) COMPETITIVE ANALYSIS" -ForegroundColor Cyan
        Write-Host ("=" * ($DomainName.Length + 22)) -ForegroundColor Cyan
        Write-Host ""
        
        Write-Host "🏆 LEADERBOARD:" -ForegroundColor Yellow
        for ($i = 0; $i -lt $analysis.leaderboard.Count; $i++) {
            $track = $analysis.leaderboard[$i]
            $position = $i + 1
            $medal = switch ($position) {
                1 { "🥇" }
                2 { "🥈" }
                3 { "🥉" }
                default { "  " }
            }
            
            Write-Host "  $medal $position. $($track.name)" -ForegroundColor White
            Write-Host "      Progress: $([math]::Round($track.progressPercentage, 1))%" -ForegroundColor Gray
            Write-Host "      Competitive Score: $([math]::Round($track.competitiveScore, 1))" -ForegroundColor Gray
            Write-Host "      Risk Level: $($track.riskLevel)" -ForegroundColor Gray
            Write-Host ""
        }
        
        Write-Host "📈 RESOURCE DISTRIBUTION:" -ForegroundColor Yellow
        foreach ($resource in $analysis.resourceDistribution) {
            $barLength = [math]::Round($resource.allocation / 5)
            $bar = "█" * $barLength
            Write-Host "  $($resource.name): $bar $($resource.allocation)%" -ForegroundColor Blue
        }
        
        Write-Host ""
        Write-Host "🎯 STRATEGY BREAKDOWN:" -ForegroundColor Yellow
        foreach ($strategy in $analysis.strategyBreakdown) {
            Write-Host "  $($strategy.strategy): $($strategy.count) tracks" -ForegroundColor White
        }
        
        Write-Host ""
        Write-Host "⚠️ RISK PROFILE:" -ForegroundColor Yellow
        foreach ($risk in $analysis.riskProfile) {
            $color = switch ($risk.riskLevel) {
                "Low" { "Green" }
                "Medium" { "Yellow" }
                "High" { "Red" }
                "VeryHigh" { "Magenta" }
                default { "White" }
            }
            Write-Host "  $($risk.riskLevel): $($risk.count) tracks" -ForegroundColor $color
        }
        
        if ($analysis.currentLeader) {
            Write-Host ""
            Write-Host "👑 CURRENT LEADER:" -ForegroundColor Green
            $leader = $analysis.currentLeader
            Write-Host "  $($leader.name)" -ForegroundColor Green
            Write-Host "  Score: $([math]::Round($leader.competitiveScore, 1))" -ForegroundColor White
            Write-Host "  Progress: $([math]::Round($leader.progressPercentage, 1))%" -ForegroundColor White
        }
        
    } else {
        Write-Host "Failed to get competitive analysis" -ForegroundColor Red
    }
}

# Execute the requested action
switch ($Action.ToLower()) {
    "overview" {
        Write-Host "Getting multi-blue tracks system overview..." -ForegroundColor Yellow
        $overviewData = Invoke-MultiBluAPI -Endpoint "overview"
        Show-MultiBluOverview -OverviewData $overviewData
    }
    
    "all" {
        Write-Host "Getting all multi-blue tracks..." -ForegroundColor Yellow
        $tracksData = Invoke-MultiBluAPI -Endpoint "all"
        
        if ($tracksData -and $tracksData.success) {
            Write-Host "ALL MULTI-BLUE TRACKS" -ForegroundColor Cyan
            Write-Host "=====================" -ForegroundColor Cyan
            Write-Host ""
            
            Write-Host "📊 TRACK TYPE SUMMARY:" -ForegroundColor Yellow
            $types = $tracksData.trackTypes
            Write-Host "  Green: $($types.green)" -ForegroundColor Green
            Write-Host "  Blue-Alpha: $($types.blueAlpha)" -ForegroundColor Blue
            Write-Host "  Blue-Beta: $($types.blueBeta)" -ForegroundColor Blue
            Write-Host "  Blue-Gamma: $($types.blueGamma)" -ForegroundColor Blue
            Write-Host "  Blue-Delta: $($types.blueDelta)" -ForegroundColor Blue
            Write-Host ""
            
            Write-Host "Total Tracks: $($tracksData.count)" -ForegroundColor White
        } else {
            Write-Host "Failed to get all tracks" -ForegroundColor Red
        }
    }
    
    { $_ -in @("ui", "backend", "ai", "infrastructure", "security", "data", "devops", "research") } {
        Write-Host "Getting $Action domain multi-blue analysis..." -ForegroundColor Yellow
        $analysisData = Invoke-MultiBluAPI -Endpoint "domain/$Action"
        Show-DomainAnalysis -AnalysisData $analysisData -DomainName $Action
    }
    
    "competition" {
        if ($Domain) {
            Write-Host "Getting competitive analysis for $Domain domain..." -ForegroundColor Yellow
            $competitionData = Invoke-MultiBluAPI -Endpoint "competition/$Domain"
            Show-CompetitiveAnalysis -CompetitionData $competitionData -DomainName $Domain
        } else {
            Write-Host "Please specify a domain with -Domain parameter" -ForegroundColor Red
            Write-Host "Example: .\manage_multi_blue_tracks.ps1 -Action competition -Domain ui" -ForegroundColor Yellow
        }
    }
    
    "strategy" {
        if ($Strategy) {
            Write-Host "Getting tracks for $Strategy strategy..." -ForegroundColor Yellow
            $strategyData = Invoke-MultiBluAPI -Endpoint "strategy/$Strategy"
            
            if ($strategyData -and $strategyData.success) {
                Write-Host "$($Strategy.ToUpper()) STRATEGY TRACKS" -ForegroundColor Cyan
                Write-Host ("=" * ($Strategy.Length + 16)) -ForegroundColor Cyan
                Write-Host ""
                
                Write-Host "📊 STRATEGY SUMMARY:" -ForegroundColor Yellow
                Write-Host "  Total Tracks: $($strategyData.count)" -ForegroundColor White
                Write-Host "  Average Resource Allocation: $([math]::Round($strategyData.averageResourceAllocation, 1))%" -ForegroundColor White
                Write-Host ""
                
                Write-Host "🎯 DOMAIN DISTRIBUTION:" -ForegroundColor Yellow
                foreach ($domain in $strategyData.domainDistribution) {
                    Write-Host "  $($domain.domain): $($domain.count) tracks" -ForegroundColor White
                }
            } else {
                Write-Host "Failed to get strategy tracks" -ForegroundColor Red
            }
        } else {
            Write-Host "Please specify a strategy with -Strategy parameter" -ForegroundColor Red
            Write-Host "Available strategies: technology, risk, timeline, feature, competitive" -ForegroundColor Yellow
        }
    }
    
    "start-blue" {
        if ($Domain) {
            Write-Host "Starting all blue tracks for $Domain domain..." -ForegroundColor Blue
            $result = Invoke-MultiBluAPI -Endpoint "domain/$Domain/start-blue" -Method "POST"
            
            if ($result -and $result.success) {
                Write-Host "  Blue tracks started: $($result.startedTracks)/$($result.totalBluTracks)" -ForegroundColor Blue
                Write-Host ""
                Write-Host "  Track Details:" -ForegroundColor Yellow
                foreach ($track in $result.trackDetails) {
                    Write-Host "    $($track.name) ($($track.type))" -ForegroundColor Blue
                    Write-Host "      Strategy: $($track.strategy)" -ForegroundColor Gray
                    Write-Host "      Risk: $($track.riskLevel)" -ForegroundColor Gray
                }
            } else {
                Write-Host "  Failed to start blue tracks" -ForegroundColor Red
            }
        } else {
            Write-Host "Please specify a domain with -Domain parameter" -ForegroundColor Red
            Write-Host "Example: .\manage_multi_blue_tracks.ps1 -Action start-blue -Domain ui" -ForegroundColor Yellow
        }
    }
    
    { $_ -in @("alpha", "beta", "gamma", "delta") } {
        $trackType = "blue-$Action"
        Write-Host "Getting $trackType tracks..." -ForegroundColor Blue
        $tracksData = Invoke-MultiBluAPI -Endpoint "type/$trackType"
        
        if ($tracksData -and $tracksData.success) {
            Write-Host "$($trackType.ToUpper()) TRACKS" -ForegroundColor Blue
            Write-Host ("=" * ($trackType.Length + 7)) -ForegroundColor Blue
            Write-Host ""
            
            Write-Host "📊 SUMMARY:" -ForegroundColor Yellow
            Write-Host "  Total Tracks: $($tracksData.count)" -ForegroundColor White
            Write-Host "  Total Resource Allocation: $([math]::Round($tracksData.totalResourceAllocation, 1))%" -ForegroundColor White
            Write-Host "  Average Risk Level: $($tracksData.averageRiskLevel)" -ForegroundColor White
        } else {
            Write-Host "Failed to get $trackType tracks" -ForegroundColor Red
        }
    }
    
    "monitor" {
        Write-Host "Starting multi-blue tracks monitoring (Press Ctrl+C to stop)..." -ForegroundColor Blue
        Write-Host "Refresh interval: $MonitorInterval seconds" -ForegroundColor Gray
        Write-Host ""
        
        try {
            while ($true) {
                Clear-Host
                Write-Host "🔬 TARS MULTI-BLUE TRACKS MONITOR" -ForegroundColor Cyan
                Write-Host "=================================" -ForegroundColor Cyan
                Write-Host "Last updated: $(Get-Date)" -ForegroundColor Gray
                Write-Host ""
                
                $overviewData = Invoke-MultiBluAPI -Endpoint "overview"
                Show-MultiBluOverview -OverviewData $overviewData
                
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
        Write-Host "TARS MULTI-BLUE TRACKS MANAGER - HELP" -ForegroundColor Cyan
        Write-Host "=====================================" -ForegroundColor Cyan
        Write-Host ""
        
        Write-Host "OVERVIEW ACTIONS:" -ForegroundColor Yellow
        Write-Host "  overview      - Show multi-blue system overview" -ForegroundColor White
        Write-Host "  all           - Show all multi-blue tracks summary" -ForegroundColor White
        Write-Host "  monitor       - Live monitoring dashboard" -ForegroundColor White
        Write-Host ""
        
        Write-Host "DOMAIN ANALYSIS:" -ForegroundColor Yellow
        Write-Host "  ui            - UI domain multi-blue analysis" -ForegroundColor White
        Write-Host "  backend       - Backend domain analysis" -ForegroundColor White
        Write-Host "  ai            - AI/ML domain analysis" -ForegroundColor White
        Write-Host "  infrastructure- Infrastructure domain analysis" -ForegroundColor White
        Write-Host "  security      - Security domain analysis" -ForegroundColor White
        Write-Host "  data          - Data domain analysis" -ForegroundColor White
        Write-Host "  devops        - DevOps domain analysis" -ForegroundColor White
        Write-Host "  research      - Research domain analysis" -ForegroundColor White
        Write-Host ""
        
        Write-Host "TRACK TYPE ANALYSIS:" -ForegroundColor Yellow
        Write-Host "  alpha         - Blue-Alpha tracks analysis" -ForegroundColor White
        Write-Host "  beta          - Blue-Beta tracks analysis" -ForegroundColor White
        Write-Host "  gamma         - Blue-Gamma tracks analysis" -ForegroundColor White
        Write-Host "  delta         - Blue-Delta tracks analysis" -ForegroundColor White
        Write-Host ""
        
        Write-Host "COMPETITIVE ANALYSIS:" -ForegroundColor Yellow
        Write-Host "  competition   - Competitive analysis (requires -Domain)" -ForegroundColor White
        Write-Host "  strategy      - Strategy-based grouping (requires -Strategy)" -ForegroundColor White
        Write-Host ""
        
        Write-Host "CONTROL ACTIONS:" -ForegroundColor Yellow
        Write-Host "  start-blue    - Start all blue tracks for domain (requires -Domain)" -ForegroundColor White
        Write-Host ""
        
        Write-Host "PARAMETERS:" -ForegroundColor Yellow
        Write-Host "  -Domain       - Specify domain for domain-specific actions" -ForegroundColor White
        Write-Host "  -Strategy     - Specify strategy (technology, risk, timeline, feature, competitive)" -ForegroundColor White
        Write-Host "  -MonitorInterval - Monitoring refresh interval in seconds" -ForegroundColor White
        Write-Host ""
        
        Write-Host "USAGE EXAMPLES:" -ForegroundColor Yellow
        Write-Host "  .\manage_multi_blue_tracks.ps1 -Action overview" -ForegroundColor Gray
        Write-Host "  .\manage_multi_blue_tracks.ps1 -Action ui" -ForegroundColor Gray
        Write-Host "  .\manage_multi_blue_tracks.ps1 -Action competition -Domain ai" -ForegroundColor Gray
        Write-Host "  .\manage_multi_blue_tracks.ps1 -Action strategy -Strategy technology" -ForegroundColor Gray
        Write-Host "  .\manage_multi_blue_tracks.ps1 -Action start-blue -Domain ui" -ForegroundColor Gray
    }
    
    default {
        Write-Host "Invalid action: $Action" -ForegroundColor Red
        Write-Host "Use -Action help to see available actions" -ForegroundColor Yellow
    }
}

Write-Host ""
