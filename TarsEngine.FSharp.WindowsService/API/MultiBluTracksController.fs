namespace TarsEngine.FSharp.WindowsService.API

open System
open Microsoft.AspNetCore.Mvc
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Tasks

/// <summary>
/// REST API Controller for managing TARS multi-blue experimental tracks
/// Supports multiple experimental paths per domain for advanced innovation strategies
/// </summary>
[<ApiController>]
[<Route("api/[controller]")>]
type MultiBluTracksController(logger: ILogger<MultiBluTracksController>, tracksManager: MultiBluTracksManager) =
    inherit ControllerBase()
    
    /// Get multi-blue system overview
    [<HttpGet("overview")>]
    member this.GetMultiBluSystemOverview() =
        try
            let overview = tracksManager.GetMultiBluSystemOverview()
            logger.LogInformation("📊 Multi-blue tracks system overview requested")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                overview = overview
                message = "Multi-blue tracks system overview retrieved"
                capabilities = {|
                    multipleBluePathsPerDomain = true
                    supportedStrategies = ["TechnologyComparison"; "RiskDiversification"; "TimelineVariation"; "FeatureExploration"; "CompetitiveEvaluation"]
                    supportedRiskLevels = ["Low"; "Medium"; "High"; "VeryHigh"]
                    maxBlueTracksPerDomain = 4
                |}
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error getting multi-blue tracks overview")
            this.StatusCode(500, {|
                success = false
                error = "Failed to get multi-blue tracks overview"
                details = ex.Message
            |})
    
    /// Get all multi-blue tracks
    [<HttpGet("all")>]
    member this.GetAllMultiBluTracks() =
        try
            let tracks = tracksManager.GetAllTracks()
            logger.LogInformation($"📋 All multi-blue tracks requested ({tracks.Length} tracks)")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                tracks = tracks
                count = tracks.Length
                trackTypes = {|
                    green = tracks |> Array.filter (fun t -> t.Type = Green) |> Array.length
                    blueAlpha = tracks |> Array.filter (fun t -> t.Type = BlueAlpha) |> Array.length
                    blueBeta = tracks |> Array.filter (fun t -> t.Type = BlueBeta) |> Array.length
                    blueGamma = tracks |> Array.filter (fun t -> t.Type = BlueGamma) |> Array.length
                    blueDelta = tracks |> Array.filter (fun t -> t.Type = BlueDelta) |> Array.length
                |}
                message = "All multi-blue tracks retrieved"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error getting all multi-blue tracks")
            this.StatusCode(500, {|
                success = false
                error = "Failed to get all multi-blue tracks"
                details = ex.Message
            |})
    
    /// Get tracks by domain with multi-blue analysis
    [<HttpGet("domain/{domain}")>]
    member this.GetDomainMultiBluTracks(domain: string) =
        try
            let parsedDomain = 
                match domain.ToLower() with
                | "ui" -> Some TrackDomain.UI
                | "backend" -> Some TrackDomain.Backend
                | "ai" | "ai_ml" -> Some TrackDomain.AI_ML
                | "infrastructure" | "infra" -> Some TrackDomain.Infrastructure
                | "security" -> Some TrackDomain.Security
                | "data" -> Some TrackDomain.Data
                | "devops" -> Some TrackDomain.DevOps
                | "research" -> Some TrackDomain.Research
                | _ -> None
            
            match parsedDomain with
            | Some d ->
                let analysis = tracksManager.GetCompetingTracksAnalysis(d)
                logger.LogInformation($"🎯 {domain} domain multi-blue analysis requested")
                
                this.Ok({|
                    success = true
                    timestamp = DateTime.UtcNow
                    domain = domain
                    analysis = analysis
                    message = $"{domain} domain multi-blue tracks analysis retrieved"
                |})
            | None ->
                this.BadRequest({|
                    success = false
                    error = $"Invalid domain: {domain}"
                    validDomains = ["ui"; "backend"; "ai"; "infrastructure"; "security"; "data"; "devops"; "research"]
                |})
        with
        | ex ->
            logger.LogError(ex, $"❌ Error getting multi-blue analysis for domain {domain}")
            this.StatusCode(500, {|
                success = false
                error = $"Failed to get multi-blue analysis for domain {domain}"
                details = ex.Message
            |})
    
    /// Get tracks by extended type (green, blue-alpha, blue-beta, etc.)
    [<HttpGet("type/{trackType}")>]
    member this.GetTracksByExtendedType(trackType: string) =
        try
            let parsedType = 
                match trackType.ToLower() with
                | "green" | "stable" -> Some Green
                | "blue-alpha" | "bluealpha" | "alpha" -> Some BlueAlpha
                | "blue-beta" | "bluebeta" | "beta" -> Some BlueBeta
                | "blue-gamma" | "bluegamma" | "gamma" -> Some BlueGamma
                | "blue-delta" | "bluedelta" | "delta" -> Some BlueDelta
                | _ -> None
            
            match parsedType with
            | Some t ->
                let tracks = tracksManager.GetAllTracks() |> Array.filter (fun track -> track.Type = t)
                logger.LogInformation($"🔄 {trackType} type tracks requested ({tracks.Length} tracks)")
                
                this.Ok({|
                    success = true
                    timestamp = DateTime.UtcNow
                    trackType = trackType
                    tracks = tracks
                    count = tracks.Length
                    totalResourceAllocation = tracks |> Array.sumBy (fun t -> t.ResourceAllocation)
                    averageRiskLevel = 
                        if tracks.Length > 0 then
                            tracks |> Array.groupBy (fun t -> t.RiskLevel) |> Array.maxBy (fun (_, group) -> group.Length) |> fst |> string
                        else "N/A"
                    message = $"{trackType} type tracks retrieved"
                |})
            | None ->
                this.BadRequest({|
                    success = false
                    error = $"Invalid track type: {trackType}"
                    validTypes = ["green"; "blue-alpha"; "blue-beta"; "blue-gamma"; "blue-delta"]
                |})
        with
        | ex ->
            logger.LogError(ex, $"❌ Error getting tracks for type {trackType}")
            this.StatusCode(500, {|
                success = false
                error = $"Failed to get tracks for type {trackType}"
                details = ex.Message
            |})
    
    /// Start all blue tracks for a domain
    [<HttpPost("domain/{domain}/start-blue")>]
    member this.StartDomainBluTracks(domain: string) =
        try
            let parsedDomain = 
                match domain.ToLower() with
                | "ui" -> Some TrackDomain.UI
                | "backend" -> Some TrackDomain.Backend
                | "ai" | "ai_ml" -> Some TrackDomain.AI_ML
                | "infrastructure" | "infra" -> Some TrackDomain.Infrastructure
                | "security" -> Some TrackDomain.Security
                | "data" -> Some TrackDomain.Data
                | "devops" -> Some TrackDomain.DevOps
                | "research" -> Some TrackDomain.Research
                | _ -> None
            
            match parsedDomain with
            | Some d ->
                let allTracks = tracksManager.GetAllTracks()
                let domainBluTracks = allTracks |> Array.filter (fun t -> t.Domain = d && t.Type <> Green)
                let mutable successCount = 0
                
                for track in domainBluTracks do
                    if tracksManager.StartTrack(track.Id) then
                        successCount <- successCount + 1
                
                logger.LogInformation($"🚀 Started {successCount}/{domainBluTracks.Length} blue tracks for {domain} domain")
                
                this.Ok({|
                    success = true
                    timestamp = DateTime.UtcNow
                    domain = domain
                    totalBluTracks = domainBluTracks.Length
                    startedTracks = successCount
                    trackDetails = domainBluTracks |> Array.map (fun t -> {|
                        id = t.Id
                        name = t.Name
                        type = t.Type.ToString()
                        strategy = t.Strategy.ToString()
                        riskLevel = t.RiskLevel.ToString()
                    |})
                    message = $"Started {successCount}/{domainBluTracks.Length} blue tracks for {domain} domain"
                |})
            | None ->
                this.BadRequest({|
                    success = false
                    error = $"Invalid domain: {domain}"
                    validDomains = ["ui"; "backend"; "ai"; "infrastructure"; "security"; "data"; "devops"; "research"]
                |})
        with
        | ex ->
            logger.LogError(ex, $"❌ Error starting blue tracks for domain {domain}")
            this.StatusCode(500, {|
                success = false
                error = $"Failed to start blue tracks for domain {domain}"
                details = ex.Message
            |})
    
    /// Get competitive analysis between blue tracks
    [<HttpGet("competition/{domain}")>]
    member this.GetCompetitiveAnalysis(domain: string) =
        try
            let parsedDomain = 
                match domain.ToLower() with
                | "ui" -> Some TrackDomain.UI
                | "backend" -> Some TrackDomain.Backend
                | "ai" | "ai_ml" -> Some TrackDomain.AI_ML
                | "infrastructure" | "infra" -> Some TrackDomain.Infrastructure
                | "security" -> Some TrackDomain.Security
                | "data" -> Some TrackDomain.Data
                | "devops" -> Some TrackDomain.DevOps
                | "research" -> Some TrackDomain.Research
                | _ -> None
            
            match parsedDomain with
            | Some d ->
                let analysis = tracksManager.GetCompetingTracksAnalysis(d)
                logger.LogInformation($"🏆 Competitive analysis requested for {domain} domain")
                
                // Calculate competitive metrics
                let bluTracks = analysis.BlueTracks
                let leaderboard = 
                    bluTracks 
                    |> Array.map (fun track -> {|
                        trackId = track.Id
                        name = track.Name
                        progressPercentage = (float track.Progress / float track.TotalTasks) * 100.0
                        resourceAllocation = track.ResourceAllocation
                        riskLevel = track.RiskLevel.ToString()
                        timeline = track.Timeline
                        strategy = track.Strategy.ToString()
                        competitiveScore = 
                            let progressScore = (float track.Progress / float track.TotalTasks) * 40.0
                            let resourceScore = track.ResourceAllocation * 0.3
                            let riskScore = match track.RiskLevel with
                                           | Low -> 30.0
                                           | Medium -> 20.0
                                           | High -> 10.0
                                           | VeryHigh -> 5.0
                            progressScore + resourceScore + riskScore
                    |})
                    |> Array.sortByDescending (fun t -> t.competitiveScore)
                
                this.Ok({|
                    success = true
                    timestamp = DateTime.UtcNow
                    domain = domain
                    competitiveAnalysis = {|
                        totalCompetitors = bluTracks.Length
                        leaderboard = leaderboard
                        resourceDistribution = bluTracks |> Array.map (fun t -> {| 
                            name = t.Name
                            allocation = t.ResourceAllocation 
                        |})
                        strategyBreakdown = 
                            bluTracks 
                            |> Array.groupBy (fun t -> t.Strategy.ToString())
                            |> Array.map (fun (strategy, tracks) -> {| 
                                strategy = strategy
                                count = tracks.Length 
                            |})
                        riskProfile = 
                            bluTracks 
                            |> Array.groupBy (fun t -> t.RiskLevel.ToString())
                            |> Array.map (fun (risk, tracks) -> {| 
                                riskLevel = risk
                                count = tracks.Length 
                            |})
                        currentLeader = leaderboard |> Array.tryHead
                    |}
                    message = $"Competitive analysis for {domain} domain retrieved"
                |})
            | None ->
                this.BadRequest({|
                    success = false
                    error = $"Invalid domain: {domain}"
                    validDomains = ["ui"; "backend"; "ai"; "infrastructure"; "security"; "data"; "devops"; "research"]
                |})
        with
        | ex ->
            logger.LogError(ex, $"❌ Error getting competitive analysis for domain {domain}")
            this.StatusCode(500, {|
                success = false
                error = $"Failed to get competitive analysis for domain {domain}"
                details = ex.Message
            |})
    
    /// Get strategy-based track grouping
    [<HttpGet("strategy/{strategy}")>]
    member this.GetTracksByStrategy(strategy: string) =
        try
            let parsedStrategy = 
                match strategy.ToLower() with
                | "technology" | "technologycomparison" -> Some TechnologyComparison
                | "risk" | "riskdiversification" -> Some RiskDiversification
                | "timeline" | "timelinevariation" -> Some TimelineVariation
                | "feature" | "featureexploration" -> Some FeatureExploration
                | "competitive" | "competitiveevaluation" -> Some CompetitiveEvaluation
                | _ -> None
            
            match parsedStrategy with
            | Some s ->
                let tracks = tracksManager.GetAllTracks() |> Array.filter (fun t -> t.Strategy = s)
                logger.LogInformation($"📊 {strategy} strategy tracks requested ({tracks.Length} tracks)")
                
                this.Ok({|
                    success = true
                    timestamp = DateTime.UtcNow
                    strategy = strategy
                    tracks = tracks
                    count = tracks.Length
                    domainDistribution = 
                        tracks 
                        |> Array.groupBy (fun t -> t.Domain.ToString())
                        |> Array.map (fun (domain, domainTracks) -> {| 
                            domain = domain
                            count = domainTracks.Length 
                        |})
                    averageResourceAllocation = 
                        if tracks.Length > 0 then tracks |> Array.averageBy (fun t -> t.ResourceAllocation) else 0.0
                    message = $"{strategy} strategy tracks retrieved"
                |})
            | None ->
                this.BadRequest({|
                    success = false
                    error = $"Invalid strategy: {strategy}"
                    validStrategies = ["technology"; "risk"; "timeline"; "feature"; "competitive"]
                |})
        with
        | ex ->
            logger.LogError(ex, $"❌ Error getting tracks for strategy {strategy}")
            this.StatusCode(500, {|
                success = false
                error = $"Failed to get tracks for strategy {strategy}"
                details = ex.Message
            |})
    
    /// Health check for multi-blue tracks system
    [<HttpGet("health")>]
    member this.MultiBluHealthCheck() =
        try
            let overview = tracksManager.GetMultiBluSystemOverview()
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                service = "Multi-Blue Tracks Manager"
                status = "Healthy"
                overview = overview
                capabilities = {|
                    multiBluSupport = true
                    maxBlueTracksPerDomain = 4
                    supportedStrategies = 5
                    supportedRiskLevels = 4
                |}
                message = "Multi-blue tracks system is operational"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Multi-blue tracks health check failed")
            this.StatusCode(500, {|
                success = false
                service = "Multi-Blue Tracks Manager"
                status = "Unhealthy"
                error = ex.Message
            |})
