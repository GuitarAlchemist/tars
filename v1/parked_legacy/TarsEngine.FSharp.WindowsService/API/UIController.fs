namespace TarsEngine.FSharp.WindowsService.API

open System
open Microsoft.AspNetCore.Mvc
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Tasks

/// <summary>
/// REST API Controller for managing TARS UI development tracks
/// Provides endpoints to control Green (stable) and Blue (experimental) UI development
/// </summary>
[<ApiController>]
[<Route("api/[controller]")>]
type UIController(logger: ILogger<UIController>, uiTaskManager: UITaskManager) =
    inherit ControllerBase()
    
    /// Get overall UI development status
    [<HttpGet("status")>]
    member this.GetOverallStatus() =
        try
            let status = uiTaskManager.GetUIStatus(None)
            logger.LogInformation("📊 UI development status requested")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                status = status
                message = "UI development tracks status retrieved"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error getting UI development status")
            this.StatusCode(500, {|
                success = false
                error = "Failed to get UI development status"
                details = ex.Message
            |})
    
    /// Get Green UI (stable) status
    [<HttpGet("green/status")>]
    member this.GetGreenStatus() =
        try
            let status = uiTaskManager.GetUIStatus(Some UITrack.GreenStable)
            logger.LogInformation("🟢 Green UI status requested")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                track = "Green (Stable)"
                status = status
                message = "Green UI development status retrieved"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error getting Green UI status")
            this.StatusCode(500, {|
                success = false
                error = "Failed to get Green UI status"
                details = ex.Message
            |})
    
    /// Get Blue UI (experimental) status
    [<HttpGet("blue/status")>]
    member this.GetBlueStatus() =
        try
            let status = uiTaskManager.GetUIStatus(Some UITrack.BlueExperimental)
            logger.LogInformation("🔵 Blue UI status requested")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                track = "Blue (Experimental)"
                status = status
                message = "Blue UI development status retrieved"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error getting Blue UI status")
            this.StatusCode(500, {|
                success = false
                error = "Failed to get Blue UI status"
                details = ex.Message
            |})
    
    /// Start Green UI maintenance tasks
    [<HttpPost("green/start")>]
    member this.StartGreenUI() =
        try
            uiTaskManager.StartUITasks(Some UITrack.GreenStable)
            logger.LogInformation("🟢 Green UI maintenance started via API")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                track = "Green (Stable)"
                message = "Green UI maintenance tasks started successfully"
                status = "Running"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error starting Green UI tasks")
            this.StatusCode(500, {|
                success = false
                error = "Failed to start Green UI tasks"
                details = ex.Message
            |})
    
    /// Start Blue UI development tasks
    [<HttpPost("blue/start")>]
    member this.StartBlueUI() =
        try
            uiTaskManager.StartUITasks(Some UITrack.BlueExperimental)
            logger.LogInformation("🔵 Blue UI development started via API")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                track = "Blue (Experimental)"
                message = "Blue UI development tasks started successfully"
                status = "Running"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error starting Blue UI tasks")
            this.StatusCode(500, {|
                success = false
                error = "Failed to start Blue UI tasks"
                details = ex.Message
            |})
    
    /// Start both UI tracks
    [<HttpPost("start")>]
    member this.StartBothTracks() =
        try
            uiTaskManager.StartUITasks(None)
            logger.LogInformation("🎨 Both UI tracks started via API")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                message = "Both UI development tracks started successfully"
                tracks = {|
                    green = "Green (Stable) maintenance started"
                    blue = "Blue (Experimental) development started"
                |}
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error starting UI tracks")
            this.StatusCode(500, {|
                success = false
                error = "Failed to start UI tracks"
                details = ex.Message
            |})
    
    /// Pause Green UI tasks
    [<HttpPost("green/pause")>]
    member this.PauseGreenUI() =
        try
            uiTaskManager.PauseUITasks(Some UITrack.GreenStable)
            logger.LogInformation("⏸️ Green UI maintenance paused via API")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                track = "Green (Stable)"
                message = "Green UI maintenance tasks paused successfully"
                status = "Paused"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error pausing Green UI tasks")
            this.StatusCode(500, {|
                success = false
                error = "Failed to pause Green UI tasks"
                details = ex.Message
            |})
    
    /// Pause Blue UI tasks
    [<HttpPost("blue/pause")>]
    member this.PauseBlueUI() =
        try
            uiTaskManager.PauseUITasks(Some UITrack.BlueExperimental)
            logger.LogInformation("⏸️ Blue UI development paused via API")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                track = "Blue (Experimental)"
                message = "Blue UI development tasks paused successfully"
                status = "Paused"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error pausing Blue UI tasks")
            this.StatusCode(500, {|
                success = false
                error = "Failed to pause Blue UI tasks"
                details = ex.Message
            |})
    
    /// Resume Green UI tasks
    [<HttpPost("green/resume")>]
    member this.ResumeGreenUI() =
        try
            uiTaskManager.ResumeUITasks(Some UITrack.GreenStable)
            logger.LogInformation("▶️ Green UI maintenance resumed via API")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                track = "Green (Stable)"
                message = "Green UI maintenance tasks resumed successfully"
                status = "Running"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error resuming Green UI tasks")
            this.StatusCode(500, {|
                success = false
                error = "Failed to resume Green UI tasks"
                details = ex.Message
            |})
    
    /// Resume Blue UI tasks
    [<HttpPost("blue/resume")>]
    member this.ResumeBlueUI() =
        try
            uiTaskManager.ResumeUITasks(Some UITrack.BlueExperimental)
            logger.LogInformation("▶️ Blue UI development resumed via API")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                track = "Blue (Experimental)"
                message = "Blue UI development tasks resumed successfully"
                status = "Running"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error resuming Blue UI tasks")
            this.StatusCode(500, {|
                success = false
                error = "Failed to resume Blue UI tasks"
                details = ex.Message
            |})
    
    /// Get UI development comparison
    [<HttpGet("comparison")>]
    member this.GetUIComparison() =
        try
            let greenStatus = uiTaskManager.GetUIStatus(Some UITrack.GreenStable)
            let blueStatus = uiTaskManager.GetUIStatus(Some UITrack.BlueExperimental)
            
            logger.LogInformation("📊 UI development comparison requested")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                comparison = {|
                    green = {|
                        track = "Green (Stable)"
                        purpose = "Production maintenance and stability"
                        status = greenStatus.State
                        progress = $"{greenStatus.Progress.CompletedTasks}/{greenStatus.Progress.TotalTasks}"
                        percentage = (float greenStatus.Progress.CompletedTasks / float greenStatus.Progress.TotalTasks) * 100.0
                        currentTask = greenStatus.Progress.CurrentTask
                        focus = "Stability, security, performance"
                    |}
                    blue = {|
                        track = "Blue (Experimental)"
                        purpose = "Next-generation UI development"
                        status = blueStatus.State
                        progress = $"{blueStatus.Progress.CompletedTasks}/{blueStatus.Progress.TotalTasks}"
                        percentage = (float blueStatus.Progress.CompletedTasks / float blueStatus.Progress.TotalTasks) * 100.0
                        currentTask = blueStatus.Progress.CurrentTask
                        focus = "Innovation, advanced features, experimentation"
                    |}
                    strategy = "Parallel development tracks for risk-free innovation"
                |}
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error getting UI comparison")
            this.StatusCode(500, {|
                success = false
                error = "Failed to get UI comparison"
                details = ex.Message
            |})
    
    /// Health check for UI development
    [<HttpGet("health")>]
    member this.UIHealthCheck() =
        try
            let greenStatus = uiTaskManager.GetUIStatus(Some UITrack.GreenStable)
            let blueStatus = uiTaskManager.GetUIStatus(Some UITrack.BlueExperimental)
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                service = "UI Development Manager"
                status = "Healthy"
                tracks = {|
                    green = {|
                        status = greenStatus.State
                        healthy = greenStatus.State <> "Error"
                        lastUpdate = greenStatus.Progress.LastUpdateTime
                    |}
                    blue = {|
                        status = blueStatus.State
                        healthy = blueStatus.State <> "Error"
                        lastUpdate = blueStatus.Progress.LastUpdateTime
                    |}
                |}
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ UI health check failed")
            this.StatusCode(500, {|
                success = false
                service = "UI Development Manager"
                status = "Unhealthy"
                error = ex.Message
            |})
