namespace TarsEngine.FSharp.WindowsService.API

open System
open Microsoft.AspNetCore.Mvc
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Tasks

/// <summary>
/// REST API Controller for managing TARS experimental tracks across all domains
/// Provides comprehensive control over parallel development tracks
/// </summary>
[<ApiController>]
[<Route("api/[controller]")>]
type ExperimentalTracksController(logger: ILogger<ExperimentalTracksController>, tracksManager: ExperimentalTracksManager) =
    inherit ControllerBase()
    
    /// Get system overview of all experimental tracks
    [<HttpGet("overview")>]
    member this.GetSystemOverview() =
        try
            let overview = tracksManager.GetSystemOverview()
            logger.LogInformation("📊 Experimental tracks system overview requested")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                overview = overview
                message = "Experimental tracks system overview retrieved"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error getting experimental tracks overview")
            this.StatusCode(500, {|
                success = false
                error = "Failed to get experimental tracks overview"
                details = ex.Message
            |})
    
    /// Get all tracks
    [<HttpGet("all")>]
    member this.GetAllTracks() =
        try
            let tracks = tracksManager.GetAllTracks()
            logger.LogInformation($"📋 All experimental tracks requested ({tracks.Length} tracks)")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                tracks = tracks
                count = tracks.Length
                message = "All experimental tracks retrieved"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error getting all experimental tracks")
            this.StatusCode(500, {|
                success = false
                error = "Failed to get all experimental tracks"
                details = ex.Message
            |})
    
    /// Get tracks by domain
    [<HttpGet("domain/{domain}")>]
    member this.GetTracksByDomain(domain: string) =
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
                let tracks = tracksManager.GetTracksByDomain(d)
                logger.LogInformation($"🎯 {domain} domain tracks requested ({tracks.Length} tracks)")
                
                this.Ok({|
                    success = true
                    timestamp = DateTime.UtcNow
                    domain = domain
                    tracks = tracks
                    count = tracks.Length
                    message = $"{domain} domain tracks retrieved"
                |})
            | None ->
                this.BadRequest({|
                    success = false
                    error = $"Invalid domain: {domain}"
                    validDomains = ["ui"; "backend"; "ai"; "infrastructure"; "security"; "data"; "devops"; "research"]
                |})
        with
        | ex ->
            logger.LogError(ex, $"❌ Error getting tracks for domain {domain}")
            this.StatusCode(500, {|
                success = false
                error = $"Failed to get tracks for domain {domain}"
                details = ex.Message
            |})
    
    /// Get tracks by type (green/blue)
    [<HttpGet("type/{trackType}")>]
    member this.GetTracksByType(trackType: string) =
        try
            let parsedType = 
                match trackType.ToLower() with
                | "green" | "stable" -> Some TrackType.Green
                | "blue" | "experimental" -> Some TrackType.Blue
                | _ -> None
            
            match parsedType with
            | Some t ->
                let tracks = tracksManager.GetTracksByType(t)
                logger.LogInformation($"🔄 {trackType} type tracks requested ({tracks.Length} tracks)")
                
                this.Ok({|
                    success = true
                    timestamp = DateTime.UtcNow
                    trackType = trackType
                    tracks = tracks
                    count = tracks.Length
                    message = $"{trackType} type tracks retrieved"
                |})
            | None ->
                this.BadRequest({|
                    success = false
                    error = $"Invalid track type: {trackType}"
                    validTypes = ["green"; "blue"; "stable"; "experimental"]
                |})
        with
        | ex ->
            logger.LogError(ex, $"❌ Error getting tracks for type {trackType}")
            this.StatusCode(500, {|
                success = false
                error = $"Failed to get tracks for type {trackType}"
                details = ex.Message
            |})
    
    /// Get specific track status
    [<HttpGet("{trackId}")>]
    member this.GetTrackStatus(trackId: string) =
        try
            match tracksManager.GetTrackStatus(trackId) with
            | Some track ->
                logger.LogInformation($"📊 Track status requested: {trackId}")
                
                this.Ok({|
                    success = true
                    timestamp = DateTime.UtcNow
                    track = track
                    message = $"Track {trackId} status retrieved"
                |})
            | None ->
                this.NotFound({|
                    success = false
                    error = $"Track not found: {trackId}"
                    message = "The specified track ID does not exist"
                |})
        with
        | ex ->
            logger.LogError(ex, $"❌ Error getting track status for {trackId}")
            this.StatusCode(500, {|
                success = false
                error = $"Failed to get track status for {trackId}"
                details = ex.Message
            |})
    
    /// Start specific track
    [<HttpPost("{trackId}/start")>]
    member this.StartTrack(trackId: string) =
        try
            let success = tracksManager.StartTrack(trackId)
            
            if success then
                logger.LogInformation($"🚀 Track started: {trackId}")
                
                this.Ok({|
                    success = true
                    timestamp = DateTime.UtcNow
                    trackId = trackId
                    action = "start"
                    message = $"Track {trackId} started successfully"
                |})
            else
                this.BadRequest({|
                    success = false
                    error = $"Failed to start track {trackId}"
                    message = "Track may not exist or may not be in a startable state"
                |})
        with
        | ex ->
            logger.LogError(ex, $"❌ Error starting track {trackId}")
            this.StatusCode(500, {|
                success = false
                error = $"Failed to start track {trackId}"
                details = ex.Message
            |})
    
    /// Pause specific track
    [<HttpPost("{trackId}/pause")>]
    member this.PauseTrack(trackId: string) =
        try
            let success = tracksManager.PauseTrack(trackId)
            
            if success then
                logger.LogInformation($"⏸️ Track paused: {trackId}")
                
                this.Ok({|
                    success = true
                    timestamp = DateTime.UtcNow
                    trackId = trackId
                    action = "pause"
                    message = $"Track {trackId} paused successfully"
                |})
            else
                this.BadRequest({|
                    success = false
                    error = $"Failed to pause track {trackId}"
                    message = "Track may not exist or may not be in a pausable state"
                |})
        with
        | ex ->
            logger.LogError(ex, $"❌ Error pausing track {trackId}")
            this.StatusCode(500, {|
                success = false
                error = $"Failed to pause track {trackId}"
                details = ex.Message
            |})
    
    /// Resume specific track
    [<HttpPost("{trackId}/resume")>]
    member this.ResumeTrack(trackId: string) =
        try
            let success = tracksManager.ResumeTrack(trackId)
            
            if success then
                logger.LogInformation($"▶️ Track resumed: {trackId}")
                
                this.Ok({|
                    success = true
                    timestamp = DateTime.UtcNow
                    trackId = trackId
                    action = "resume"
                    message = $"Track {trackId} resumed successfully"
                |})
            else
                this.BadRequest({|
                    success = false
                    error = $"Failed to resume track {trackId}"
                    message = "Track may not exist or may not be in a resumable state"
                |})
        with
        | ex ->
            logger.LogError(ex, $"❌ Error resuming track {trackId}")
            this.StatusCode(500, {|
                success = false
                error = $"Failed to resume track {trackId}"
                details = ex.Message
            |})
    
    /// Start all tracks of a specific type
    [<HttpPost("type/{trackType}/start")>]
    member this.StartTracksByType(trackType: string) =
        try
            let parsedType = 
                match trackType.ToLower() with
                | "green" | "stable" -> Some TrackType.Green
                | "blue" | "experimental" -> Some TrackType.Blue
                | _ -> None
            
            match parsedType with
            | Some t ->
                let tracks = tracksManager.GetTracksByType(t)
                let mutable successCount = 0
                
                for track in tracks do
                    if tracksManager.StartTrack(track.Id) then
                        successCount <- successCount + 1
                
                logger.LogInformation($"🚀 Started {successCount}/{tracks.Length} {trackType} tracks")
                
                this.Ok({|
                    success = true
                    timestamp = DateTime.UtcNow
                    trackType = trackType
                    totalTracks = tracks.Length
                    startedTracks = successCount
                    message = $"Started {successCount}/{tracks.Length} {trackType} tracks"
                |})
            | None ->
                this.BadRequest({|
                    success = false
                    error = $"Invalid track type: {trackType}"
                    validTypes = ["green"; "blue"; "stable"; "experimental"]
                |})
        with
        | ex ->
            logger.LogError(ex, $"❌ Error starting {trackType} tracks")
            this.StatusCode(500, {|
                success = false
                error = $"Failed to start {trackType} tracks"
                details = ex.Message
            |})
    
    /// Start all tracks in a domain
    [<HttpPost("domain/{domain}/start")>]
    member this.StartTracksByDomain(domain: string) =
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
                let tracks = tracksManager.GetTracksByDomain(d)
                let mutable successCount = 0
                
                for track in tracks do
                    if tracksManager.StartTrack(track.Id) then
                        successCount <- successCount + 1
                
                logger.LogInformation($"🚀 Started {successCount}/{tracks.Length} {domain} domain tracks")
                
                this.Ok({|
                    success = true
                    timestamp = DateTime.UtcNow
                    domain = domain
                    totalTracks = tracks.Length
                    startedTracks = successCount
                    message = $"Started {successCount}/{tracks.Length} {domain} domain tracks"
                |})
            | None ->
                this.BadRequest({|
                    success = false
                    error = $"Invalid domain: {domain}"
                    validDomains = ["ui"; "backend"; "ai"; "infrastructure"; "security"; "data"; "devops"; "research"]
                |})
        with
        | ex ->
            logger.LogError(ex, $"❌ Error starting {domain} domain tracks")
            this.StatusCode(500, {|
                success = false
                error = $"Failed to start {domain} domain tracks"
                details = ex.Message
            |})
    
    /// Get comprehensive comparison of green vs blue tracks
    [<HttpGet("comparison")>]
    member this.GetTracksComparison() =
        try
            let greenTracks = tracksManager.GetTracksByType(TrackType.Green)
            let blueTracks = tracksManager.GetTracksByType(TrackType.Blue)
            
            let greenProgress = greenTracks |> Array.sumBy (fun t -> t.Progress)
            let greenTotal = greenTracks |> Array.sumBy (fun t -> t.TotalTasks)
            let blueProgress = blueTracks |> Array.sumBy (fun t -> t.Progress)
            let blueTotal = blueTracks |> Array.sumBy (fun t -> t.TotalTasks)
            
            logger.LogInformation("📊 Tracks comparison requested")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                comparison = {|
                    green = {|
                        trackType = "Green (Stable)"
                        purpose = "Production stability and reliability"
                        trackCount = greenTracks.Length
                        totalProgress = greenProgress
                        totalTasks = greenTotal
                        percentage = if greenTotal > 0 then (float greenProgress / float greenTotal) * 100.0 else 0.0
                        runningTracks = greenTracks |> Array.filter (fun t -> t.Status = Running) |> Array.length
                        resourceAllocation = greenTracks |> Array.sumBy (fun t -> t.ResourceAllocation)
                    |}
                    blue = {|
                        trackType = "Blue (Experimental)"
                        purpose = "Innovation and future development"
                        trackCount = blueTracks.Length
                        totalProgress = blueProgress
                        totalTasks = blueTotal
                        percentage = if blueTotal > 0 then (float blueProgress / float blueTotal) * 100.0 else 0.0
                        runningTracks = blueTracks |> Array.filter (fun t -> t.Status = Running) |> Array.length
                        resourceAllocation = blueTracks |> Array.sumBy (fun t -> t.ResourceAllocation)
                    |}
                    strategy = "Parallel experimental tracks for risk-free innovation"
                |}
                message = "Tracks comparison retrieved"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error getting tracks comparison")
            this.StatusCode(500, {|
                success = false
                error = "Failed to get tracks comparison"
                details = ex.Message
            |})
    
    /// Health check for experimental tracks system
    [<HttpGet("health")>]
    member this.HealthCheck() =
        try
            let overview = tracksManager.GetSystemOverview()
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                service = "Experimental Tracks Manager"
                status = "Healthy"
                overview = overview
                message = "Experimental tracks system is operational"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Experimental tracks health check failed")
            this.StatusCode(500, {|
                success = false
                service = "Experimental Tracks Manager"
                status = "Unhealthy"
                error = ex.Message
            |})
