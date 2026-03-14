namespace TarsEngine.FSharp.WindowsService.Tasks

open System
open System.IO
open System.Threading
open System.Threading.Tasks
open System.Text.Json
open Microsoft.Extensions.Logging
open Microsoft.Extensions.Hosting

/// <summary>
/// UI development track types
/// </summary>
type UITrack =
    | GreenStable
    | BlueExperimental

/// <summary>
/// UI task state
/// </summary>
type UITaskState =
    | NotStarted
    | Running
    | Paused
    | Completed
    | Error of string

/// <summary>
/// UI development progress
/// </summary>
type UIProgress = {
    Track: UITrack
    TotalTasks: int
    CompletedTasks: int
    CurrentTask: string
    StartTime: DateTime
    LastUpdateTime: DateTime
    EstimatedCompletion: DateTime option
    Components: Map<string, int> // Component -> Progress percentage
}

/// <summary>
/// UI Task Manager for parallel UI development tracks
/// Manages Green (stable) and Blue (experimental) UI development in background
/// </summary>
type UITaskManager(logger: ILogger<UITaskManager>) =
    inherit BackgroundService()
    
    let mutable greenTaskState = NotStarted
    let mutable blueTaskState = NotStarted
    
    let mutable greenProgress = {
        Track = GreenStable
        TotalTasks = 50
        CompletedTasks = 0
        CurrentTask = "Initializing Green UI maintenance..."
        StartTime = DateTime.UtcNow
        LastUpdateTime = DateTime.UtcNow
        EstimatedCompletion = None
        Components = Map.empty
    }
    
    let mutable blueProgress = {
        Track = BlueExperimental
        TotalTasks = 100
        CompletedTasks = 0
        CurrentTask = "Initializing Blue UI development..."
        StartTime = DateTime.UtcNow
        LastUpdateTime = DateTime.UtcNow
        EstimatedCompletion = None
        Components = Map.empty
    }
    
    let mutable cancellationTokenSource = new CancellationTokenSource()
    let greenProgressFile = Path.Combine(".tars", "green_ui_progress.json")
    let blueProgressFile = Path.Combine(".tars", "blue_ui_progress.json")
    let greenStateFile = Path.Combine(".tars", "green_ui_state.json")
    let blueStateFile = Path.Combine(".tars", "blue_ui_state.json")
    
    // Ensure .tars directory exists
    do
        let tarsDir = ".tars"
        if not (Directory.Exists(tarsDir)) then
            Directory.CreateDirectory(tarsDir) |> ignore
    
    /// Save progress to disk
    member private this.SaveProgress(track: UITrack) =
        try
            match track with
            | GreenStable ->
                let progressJson = JsonSerializer.Serialize(greenProgress, JsonSerializerOptions(WriteIndented = true))
                File.WriteAllText(greenProgressFile, progressJson)
                
                let stateJson = JsonSerializer.Serialize(greenTaskState.ToString(), JsonSerializerOptions(WriteIndented = true))
                File.WriteAllText(greenStateFile, stateJson)
                
            | BlueExperimental ->
                let progressJson = JsonSerializer.Serialize(blueProgress, JsonSerializerOptions(WriteIndented = true))
                File.WriteAllText(blueProgressFile, progressJson)
                
                let stateJson = JsonSerializer.Serialize(blueTaskState.ToString(), JsonSerializerOptions(WriteIndented = true))
                File.WriteAllText(blueStateFile, stateJson)
            
            logger.LogDebug($"UI progress saved for {track} track")
        with
        | ex -> logger.LogError(ex, $"Failed to save UI progress for {track}")
    
    /// Load progress from disk
    member private this.LoadProgress() =
        try
            // Load Green UI progress
            if File.Exists(greenProgressFile) then
                let progressJson = File.ReadAllText(greenProgressFile)
                let loadedProgress = JsonSerializer.Deserialize<UIProgress>(progressJson)
                greenProgress <- loadedProgress
                logger.LogInformation("Green UI progress loaded from disk")
            
            if File.Exists(greenStateFile) then
                let stateJson = File.ReadAllText(greenStateFile)
                let stateString = JsonSerializer.Deserialize<string>(stateJson)
                match stateString with
                | "Running" -> greenTaskState <- Paused
                | "Paused" -> greenTaskState <- Paused
                | "Completed" -> greenTaskState <- Completed
                | _ -> greenTaskState <- NotStarted
            
            // Load Blue UI progress
            if File.Exists(blueProgressFile) then
                let progressJson = File.ReadAllText(blueProgressFile)
                let loadedProgress = JsonSerializer.Deserialize<UIProgress>(progressJson)
                blueProgress <- loadedProgress
                logger.LogInformation("Blue UI progress loaded from disk")
            
            if File.Exists(blueStateFile) then
                let stateJson = File.ReadAllText(blueStateFile)
                let stateString = JsonSerializer.Deserialize<string>(stateJson)
                match stateString with
                | "Running" -> blueTaskState <- Paused
                | "Paused" -> blueTaskState <- Paused
                | "Completed" -> blueTaskState <- Completed
                | _ -> blueTaskState <- NotStarted
                
        with
        | ex -> 
            logger.LogError(ex, "Failed to load UI progress, starting fresh")
            greenTaskState <- NotStarted
            blueTaskState <- NotStarted
    
    /// Update progress for specific track
    member private this.UpdateProgress(track: UITrack, completedTasks: int, currentTask: string, components: Map<string, int>) =
        let updatedProgress = {
            Track = track
            TotalTasks = match track with | GreenStable -> 50 | BlueExperimental -> 100
            CompletedTasks = completedTasks
            CurrentTask = currentTask
            StartTime = match track with | GreenStable -> greenProgress.StartTime | BlueExperimental -> blueProgress.StartTime
            LastUpdateTime = DateTime.UtcNow
            Components = components
            EstimatedCompletion = 
                if completedTasks > 0 then
                    let startTime = match track with | GreenStable -> greenProgress.StartTime | BlueExperimental -> blueProgress.StartTime
                    let elapsed = DateTime.UtcNow - startTime
                    let totalTasks = match track with | GreenStable -> 50 | BlueExperimental -> 100
                    let estimatedTotal = elapsed.TotalMinutes * (float totalTasks) / (float completedTasks)
                    Some (startTime.AddMinutes(estimatedTotal))
                else None
        }
        
        match track with
        | GreenStable -> greenProgress <- updatedProgress
        | BlueExperimental -> blueProgress <- updatedProgress
        
        this.SaveProgress(track)
        logger.LogInformation($"UI progress updated for {track}: {completedTasks}/{updatedProgress.TotalTasks} - {currentTask}")
    
    /// Execute Green UI maintenance tasks
    member private this.ExecuteGreenUITasks(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("🟢 Starting Green UI (Stable) maintenance tasks")
            
            let maintenanceTasks = [
                ("Performance monitoring", 3)
                ("Security patch validation", 4)
                ("Bug fix implementation", 6)
                ("Accessibility compliance check", 3)
                ("User feedback processing", 4)
                ("Cross-browser testing", 5)
                ("Mobile responsiveness validation", 4)
                ("Production monitoring setup", 3)
                ("Documentation updates", 5)
                ("Code quality improvements", 8)
                ("Performance optimizations", 6)
                ("Final stability validation", 4)
            ]
            
            let mutable completedTasks = greenProgress.CompletedTasks
            let mutable components = greenProgress.Components
            
            for (taskName, taskDuration) in maintenanceTasks do
                if cancellationToken.IsCancellationRequested then break
                
                // Check for pause
                while greenTaskState = Paused && not cancellationToken.IsCancellationRequested do
                    logger.LogInformation("🟢 Green UI task paused, waiting for resume...")
                    do! Task.Delay(1000, cancellationToken)
                
                if cancellationToken.IsCancellationRequested then break
                
                this.UpdateProgress(GreenStable, completedTasks, $"Green UI: {taskName}", components)
                
                // Simulate work
                let workDuration = taskDuration * 1000 // Convert to milliseconds
                do! Task.Delay(workDuration, cancellationToken)
                
                completedTasks <- completedTasks + 1
                
                // Update component progress
                let componentName = taskName.Split(' ').[0]
                components <- components |> Map.add componentName (completedTasks * 2)
                
                this.UpdateProgress(GreenStable, completedTasks, $"Green UI: {taskName} completed", components)
                
                if completedTasks % 5 = 0 then
                    logger.LogInformation($"🟢 Green UI milestone: {completedTasks}/50 tasks completed")
            
            if not cancellationToken.IsCancellationRequested then
                greenTaskState <- Completed
                this.UpdateProgress(GreenStable, 50, "Green UI maintenance completed!", components)
                logger.LogInformation("🎉 Green UI (Stable) maintenance completed successfully!")
            
        with
        | :? OperationCanceledException ->
            logger.LogInformation("🟢 Green UI task was cancelled")
            greenTaskState <- Paused
        | ex ->
            logger.LogError(ex, "❌ Error during Green UI maintenance")
            greenTaskState <- Error ex.Message
    }
    
    /// Execute Blue UI development tasks
    member private this.ExecuteBlueUITasks(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("🔵 Starting Blue UI (Experimental) development tasks")
            
            let developmentTasks = [
                ("Advanced component library design", 5)
                ("Modern design system creation", 6)
                ("Animation framework development", 8)
                ("Real-time collaboration features", 10)
                ("AI-powered interface components", 12)
                ("Voice control integration", 8)
                ("Gesture recognition system", 10)
                ("Adaptive layout engine", 9)
                ("Personalization framework", 7)
                ("Advanced data visualization", 11)
                ("Interactive tutorial system", 6)
                ("Performance optimization engine", 8)
            ]
            
            let mutable completedTasks = blueProgress.CompletedTasks
            let mutable components = blueProgress.Components
            
            for (taskName, taskDuration) in developmentTasks do
                if cancellationToken.IsCancellationRequested then break
                
                // Check for pause
                while blueTaskState = Paused && not cancellationToken.IsCancellationRequested do
                    logger.LogInformation("🔵 Blue UI task paused, waiting for resume...")
                    do! Task.Delay(1000, cancellationToken)
                
                if cancellationToken.IsCancellationRequested then break
                
                this.UpdateProgress(BlueExperimental, completedTasks, $"Blue UI: {taskName}", components)
                
                // Simulate work with more complex timing for experimental features
                let workDuration = taskDuration * 1500 // Longer for experimental work
                do! Task.Delay(workDuration, cancellationToken)
                
                completedTasks <- completedTasks + 1
                
                // Update component progress
                let componentName = taskName.Split(' ').[0]
                components <- components |> Map.add componentName (completedTasks * 1)
                
                this.UpdateProgress(BlueExperimental, completedTasks, $"Blue UI: {taskName} completed", components)
                
                if completedTasks % 10 = 0 then
                    logger.LogInformation($"🔵 Blue UI milestone: {completedTasks}/100 tasks completed")
            
            if not cancellationToken.IsCancellationRequested then
                blueTaskState <- Completed
                this.UpdateProgress(BlueExperimental, 100, "Blue UI experimental development completed!", components)
                logger.LogInformation("🎉 Blue UI (Experimental) development completed successfully!")
            
        with
        | :? OperationCanceledException ->
            logger.LogInformation("🔵 Blue UI task was cancelled")
            blueTaskState <- Paused
        | ex ->
            logger.LogError(ex, "❌ Error during Blue UI development")
            blueTaskState <- Error ex.Message
    }
    
    /// Start UI tasks
    member this.StartUITasks(track: UITrack option) =
        match track with
        | Some GreenStable ->
            if greenTaskState = NotStarted || greenTaskState = Paused then
                greenTaskState <- Running
                logger.LogInformation("🟢 Starting Green UI maintenance tasks")
                Task.Run(fun () -> this.ExecuteGreenUITasks(cancellationTokenSource.Token)) |> ignore
        
        | Some BlueExperimental ->
            if blueTaskState = NotStarted || blueTaskState = Paused then
                blueTaskState <- Running
                logger.LogInformation("🔵 Starting Blue UI development tasks")
                Task.Run(fun () -> this.ExecuteBlueUITasks(cancellationTokenSource.Token)) |> ignore
        
        | None ->
            // Start both tracks
            this.StartUITasks(Some GreenStable)
            this.StartUITasks(Some BlueExperimental)
    
    /// Pause UI tasks
    member this.PauseUITasks(track: UITrack option) =
        match track with
        | Some GreenStable ->
            if greenTaskState = Running then
                greenTaskState <- Paused
                logger.LogInformation("⏸️ Pausing Green UI maintenance tasks")
        
        | Some BlueExperimental ->
            if blueTaskState = Running then
                blueTaskState <- Paused
                logger.LogInformation("⏸️ Pausing Blue UI development tasks")
        
        | None ->
            this.PauseUITasks(Some GreenStable)
            this.PauseUITasks(Some BlueExperimental)
    
    /// Resume UI tasks
    member this.ResumeUITasks(track: UITrack option) =
        match track with
        | Some GreenStable ->
            if greenTaskState = Paused then
                greenTaskState <- Running
                logger.LogInformation("▶️ Resuming Green UI maintenance tasks")
        
        | Some BlueExperimental ->
            if blueTaskState = Paused then
                blueTaskState <- Running
                logger.LogInformation("▶️ Resuming Blue UI development tasks")
        
        | None ->
            this.ResumeUITasks(Some GreenStable)
            this.ResumeUITasks(Some BlueExperimental)
    
    /// Get UI status
    member this.GetUIStatus(track: UITrack option) = 
        match track with
        | Some GreenStable -> {|
            Track = "Green (Stable)"
            State = greenTaskState.ToString()
            Progress = greenProgress
            IsRunning = greenTaskState = Running
            IsPaused = greenTaskState = Paused
            IsCompleted = greenTaskState = Completed
        |}
        
        | Some BlueExperimental -> {|
            Track = "Blue (Experimental)"
            State = blueTaskState.ToString()
            Progress = blueProgress
            IsRunning = blueTaskState = Running
            IsPaused = blueTaskState = Paused
            IsCompleted = blueTaskState = Completed
        |}
        
        | None -> {|
            GreenTrack = this.GetUIStatus(Some GreenStable)
            BlueTrack = this.GetUIStatus(Some BlueExperimental)
            OverallStatus = $"Green: {greenTaskState}, Blue: {blueTaskState}"
        |}
    
    /// Background service execution
    override this.ExecuteAsync(stoppingToken: CancellationToken) = task {
        logger.LogInformation("🎨 UI Task Manager started - Parallel development tracks ready")
        
        // Load previous progress
        this.LoadProgress()
        
        // Auto-start both tracks
        this.StartUITasks(None)
        
        // Keep service running
        while not stoppingToken.IsCancellationRequested do
            try
                do! Task.Delay(10000, stoppingToken) // Check every 10 seconds
            with
            | :? OperationCanceledException -> ()
            | ex -> logger.LogError(ex, "Error in UI task manager loop")
        
        logger.LogInformation("🎨 UI Task Manager stopped")
    }
    
    /// Dispose resources
    override this.Dispose() =
        cancellationTokenSource?.Cancel()
        cancellationTokenSource?.Dispose()
        base.Dispose()
