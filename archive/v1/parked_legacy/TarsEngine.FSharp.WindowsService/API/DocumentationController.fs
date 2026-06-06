namespace TarsEngine.FSharp.WindowsService.API

open System
open Microsoft.AspNetCore.Mvc
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Tasks

/// <summary>
/// REST API Controller for managing TARS documentation generation tasks
/// Provides endpoints to start, pause, resume, and monitor documentation generation
/// </summary>
[<ApiController>]
[<Route("api/[controller]")>]
type DocumentationController(logger: ILogger<DocumentationController>, taskManager: DocumentationTaskManager) =
    inherit ControllerBase()
    
    /// Get current documentation task status
    [<HttpGet("status")>]
    member this.GetStatus() =
        try
            let status = taskManager.GetStatus()
            logger.LogInformation($"📊 Documentation status requested: {status.State}")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                status = status
                message = $"Documentation task is {status.State.ToLower()}"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error getting documentation status")
            this.StatusCode(500, {|
                success = false
                error = "Failed to get documentation status"
                details = ex.Message
            |})
    
    /// Start documentation generation task
    [<HttpPost("start")>]
    member this.StartTask() =
        try
            let status = taskManager.GetStatus()
            
            if status.CanStart then
                taskManager.StartTask()
                logger.LogInformation("🚀 Documentation task started via API")
                
                this.Ok({|
                    success = true
                    timestamp = DateTime.UtcNow
                    message = "Documentation generation started successfully"
                    status = "Running"
                |})
            else
                logger.LogWarning($"⚠️ Cannot start documentation task in current state: {status.State}")
                
                this.BadRequest({|
                    success = false
                    error = $"Cannot start task in current state: {status.State}"
                    currentState = status.State
                    canStart = status.CanStart
                |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error starting documentation task")
            this.StatusCode(500, {|
                success = false
                error = "Failed to start documentation task"
                details = ex.Message
            |})
    
    /// Pause documentation generation task
    [<HttpPost("pause")>]
    member this.PauseTask() =
        try
            let status = taskManager.GetStatus()
            
            if status.CanPause then
                taskManager.PauseTask()
                logger.LogInformation("⏸️ Documentation task paused via API")
                
                this.Ok({|
                    success = true
                    timestamp = DateTime.UtcNow
                    message = "Documentation generation paused successfully"
                    status = "Paused"
                |})
            else
                logger.LogWarning($"⚠️ Cannot pause documentation task in current state: {status.State}")
                
                this.BadRequest({|
                    success = false
                    error = $"Cannot pause task in current state: {status.State}"
                    currentState = status.State
                    canPause = status.CanPause
                |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error pausing documentation task")
            this.StatusCode(500, {|
                success = false
                error = "Failed to pause documentation task"
                details = ex.Message
            |})
    
    /// Resume documentation generation task
    [<HttpPost("resume")>]
    member this.ResumeTask() =
        try
            let status = taskManager.GetStatus()
            
            if status.CanResume then
                taskManager.ResumeTask()
                logger.LogInformation("▶️ Documentation task resumed via API")
                
                this.Ok({|
                    success = true
                    timestamp = DateTime.UtcNow
                    message = "Documentation generation resumed successfully"
                    status = "Running"
                |})
            else
                logger.LogWarning($"⚠️ Cannot resume documentation task in current state: {status.State}")
                
                this.BadRequest({|
                    success = false
                    error = $"Cannot resume task in current state: {status.State}"
                    currentState = status.State
                    canResume = status.CanResume
                |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error resuming documentation task")
            this.StatusCode(500, {|
                success = false
                error = "Failed to resume documentation task"
                details = ex.Message
            |})
    
    /// Stop documentation generation task
    [<HttpPost("stop")>]
    member this.StopTask() =
        try
            taskManager.StopTask()
            logger.LogInformation("⏹️ Documentation task stopped via API")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                message = "Documentation generation stopped successfully"
                status = "Paused"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error stopping documentation task")
            this.StatusCode(500, {|
                success = false
                error = "Failed to stop documentation task"
                details = ex.Message
            |})
    
    /// Get detailed progress information
    [<HttpGet("progress")>]
    member this.GetProgress() =
        try
            let status = taskManager.GetStatus()
            let progress = status.Progress
            
            logger.LogDebug($"📈 Documentation progress requested: {progress.CompletedTasks}/{progress.TotalTasks}")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                progress = {|
                    totalTasks = progress.TotalTasks
                    completedTasks = progress.CompletedTasks
                    percentage = (float progress.CompletedTasks / float progress.TotalTasks) * 100.0
                    currentTask = progress.CurrentTask
                    startTime = progress.StartTime
                    lastUpdateTime = progress.LastUpdateTime
                    estimatedCompletion = progress.EstimatedCompletion
                    departments = progress.Departments |> Map.toList
                    elapsedTime = DateTime.UtcNow - progress.StartTime
                |}
                state = status.State
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error getting documentation progress")
            this.StatusCode(500, {|
                success = false
                error = "Failed to get documentation progress"
                details = ex.Message
            |})
    
    /// Get department-specific progress
    [<HttpGet("departments")>]
    member this.GetDepartmentProgress() =
        try
            let status = taskManager.GetStatus()
            let departments = status.Progress.Departments
            
            logger.LogDebug("🏛️ Department progress requested")
            
            let departmentDetails = [
                ("Technical Writing", "User manuals, guides, and documentation coordination")
                ("Development", "API documentation, code examples, and technical guides")
                ("AI Research", "Jupyter notebooks, AI tutorials, and research documentation")
                ("Quality Assurance", "Testing guides, validation procedures, and quality metrics")
                ("DevOps", "Deployment guides, infrastructure docs, and service management")
            ]
            
            let departmentStatus = 
                departmentDetails
                |> List.map (fun (name, description) ->
                    let progress = departments |> Map.tryFind name |> Option.defaultValue 0
                    {|
                        name = name
                        description = description
                        progress = progress
                        status = if progress = 100 then "Completed" elif progress > 0 then "In Progress" else "Not Started"
                    |})
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                departments = departmentStatus
                overallState = status.State
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error getting department progress")
            this.StatusCode(500, {|
                success = false
                error = "Failed to get department progress"
                details = ex.Message
            |})
    
    /// Health check endpoint
    [<HttpGet("health")>]
    member this.HealthCheck() =
        try
            let status = taskManager.GetStatus()
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                service = "Documentation Task Manager"
                status = "Healthy"
                taskState = status.State
                lastUpdate = status.Progress.LastUpdateTime
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Health check failed")
            this.StatusCode(500, {|
                success = false
                service = "Documentation Task Manager"
                status = "Unhealthy"
                error = ex.Message
            |})
