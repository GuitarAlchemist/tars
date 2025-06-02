namespace TarsEngine.FSharp.WindowsService.Tasks

open System
open System.IO
open System.Threading
open System.Threading.Tasks
open System.Text.Json
open System.Collections.Concurrent
open Microsoft.Extensions.Logging
open Microsoft.Extensions.Hosting

/// <summary>
/// Extended track types for multi-blue experimental paths
/// </summary>
type ExtendedTrackType =
    | Green                 // Stable production
    | BlueAlpha            // Primary experimental path
    | BlueBeta             // Secondary experimental path  
    | BlueGamma            // Tertiary experimental path
    | BlueDelta            // Quaternary experimental path

/// <summary>
/// Risk levels for experimental tracks
/// </summary>
type RiskLevel =
    | Low
    | Medium
    | High
    | VeryHigh

/// <summary>
/// Experimental strategy types
/// </summary>
type ExperimentalStrategy =
    | TechnologyComparison
    | RiskDiversification
    | TimelineVariation
    | FeatureExploration
    | CompetitiveEvaluation

/// <summary>
/// Multi-blue experimental track
/// </summary>
type MultiBluTrack = {
    Id: string
    Domain: TrackDomain
    Type: ExtendedTrackType
    Name: string
    Description: string
    Technologies: string list
    Priorities: string list
    Status: TrackStatus
    Progress: int
    TotalTasks: int
    CurrentTask: string
    StartTime: DateTime
    LastUpdateTime: DateTime
    ResourceAllocation: float
    RiskLevel: RiskLevel
    Timeline: string
    Strategy: ExperimentalStrategy
    EstimatedCompletion: DateTime option
    SuccessMetrics: string list
    CompetingTracks: string list // IDs of tracks this competes with
}

/// <summary>
/// Multi-Blue Tracks Manager for advanced experimental development
/// Supports multiple blue experimental paths per domain
/// </summary>
type MultiBluTracksManager(logger: ILogger<MultiBluTracksManager>) =
    inherit BackgroundService()
    
    let tracks = ConcurrentDictionary<string, MultiBluTrack>()
    let mutable cancellationTokenSource = new CancellationTokenSource()
    let tracksFile = Path.Combine(".tars", "multi_blue_tracks.json")
    
    // Ensure .tars directory exists
    do
        let tarsDir = ".tars"
        if not (Directory.Exists(tarsDir)) then
            Directory.CreateDirectory(tarsDir) |> ignore
    
    /// Initialize multi-blue tracks
    member private this.InitializeMultiBluTracks() =
        let multiBluTracks = [
            // UI Domain Multi-Blue Tracks
            {
                Id = "ui-green"
                Domain = UI
                Type = Green
                Name = "UI Stable Production"
                Description = "Production interface stability and maintenance"
                Technologies = ["React 18"; "CSS Modules"; "Redux Toolkit"]
                Priorities = ["Accessibility"; "Performance"; "Security"]
                Status = NotStarted
                Progress = 0
                TotalTasks = 50
                CurrentTask = "Initializing UI stable track..."
                StartTime = DateTime.UtcNow
                LastUpdateTime = DateTime.UtcNow
                ResourceAllocation = 25.0
                RiskLevel = Low
                Timeline = "Continuous"
                Strategy = TechnologyComparison
                EstimatedCompletion = None
                SuccessMetrics = ["99.9% Uptime"; "Lighthouse 95+"; "Zero Critical Bugs"]
                CompetingTracks = []
            }
            {
                Id = "ui-blue-alpha"
                Domain = UI
                Type = BlueAlpha
                Name = "React 19 + AI Integration"
                Description = "Next-generation React with AI-powered components"
                Technologies = ["React 19"; "AI Components"; "Voice Control"; "Gesture Recognition"]
                Priorities = ["AI Integration"; "Voice Control"; "Advanced Interactions"]
                Status = NotStarted
                Progress = 0
                TotalTasks = 80
                CurrentTask = "Initializing React 19 + AI track..."
                StartTime = DateTime.UtcNow
                LastUpdateTime = DateTime.UtcNow
                ResourceAllocation = 30.0
                RiskLevel = Medium
                Timeline = "6 months"
                Strategy = TechnologyComparison
                EstimatedCompletion = None
                SuccessMetrics = ["AI Response <100ms"; "Voice Accuracy 95%"; "User Engagement +50%"]
                CompetingTracks = ["ui-blue-beta"; "ui-blue-gamma"]
            }
            {
                Id = "ui-blue-beta"
                Domain = UI
                Type = BlueBeta
                Name = "Vue 3 + WebAssembly Performance"
                Description = "High-performance UI with Vue 3 and WebAssembly components"
                Technologies = ["Vue 3"; "WebAssembly"; "Rust Components"; "WASM Modules"]
                Priorities = ["Performance"; "Memory Efficiency"; "Native Speed"]
                Status = NotStarted
                Progress = 0
                TotalTasks = 90
                CurrentTask = "Initializing Vue 3 + WASM track..."
                StartTime = DateTime.UtcNow
                LastUpdateTime = DateTime.UtcNow
                ResourceAllocation = 25.0
                RiskLevel = High
                Timeline = "9 months"
                Strategy = TechnologyComparison
                EstimatedCompletion = None
                SuccessMetrics = ["50% Faster Rendering"; "30% Less Memory"; "Native Performance"]
                CompetingTracks = ["ui-blue-alpha"; "ui-blue-gamma"]
            }
            {
                Id = "ui-blue-gamma"
                Domain = UI
                Type = BlueGamma
                Name = "Svelte + Edge Computing"
                Description = "Lightweight UI with edge computing capabilities"
                Technologies = ["Svelte"; "Edge Workers"; "Real-time Sync"; "CDN Integration"]
                Priorities = ["Bundle Size"; "Edge Performance"; "Real-time Updates"]
                Status = NotStarted
                Progress = 0
                TotalTasks = 60
                CurrentTask = "Initializing Svelte + Edge track..."
                StartTime = DateTime.UtcNow
                LastUpdateTime = DateTime.UtcNow
                ResourceAllocation = 20.0
                RiskLevel = Low
                Timeline = "3 months"
                Strategy = TechnologyComparison
                EstimatedCompletion = None
                SuccessMetrics = ["<50KB Bundle"; "Edge Latency <10ms"; "Real-time Sync 99%"]
                CompetingTracks = ["ui-blue-alpha"; "ui-blue-beta"]
            }
            
            // AI Domain Multi-Blue Tracks
            {
                Id = "ai-green"
                Domain = AI_ML
                Type = Green
                Name = "Production AI Systems"
                Description = "Stable AI/ML production inference and processing"
                Technologies = ["OpenAI API"; "Stable Models"; "Production Inference"]
                Priorities = ["Reliability"; "Cost Efficiency"; "Response Time"]
                Status = NotStarted
                Progress = 0
                TotalTasks = 40
                CurrentTask = "Initializing AI production track..."
                StartTime = DateTime.UtcNow
                LastUpdateTime = DateTime.UtcNow
                ResourceAllocation = 20.0
                RiskLevel = Low
                Timeline = "Continuous"
                Strategy = RiskDiversification
                EstimatedCompletion = None
                SuccessMetrics = ["99.5% Availability"; "Response <2s"; "Cost <$0.01/request"]
                CompetingTracks = []
            }
            {
                Id = "ai-blue-alpha"
                Domain = AI_ML
                Type = BlueAlpha
                Name = "Custom Transformer Models"
                Description = "Custom-trained transformer models for TARS-specific tasks"
                Technologies = ["PyTorch"; "Custom Training"; "Fine-tuning"; "Model Optimization"]
                Priorities = ["Task Specialization"; "Performance"; "Cost Reduction"]
                Status = NotStarted
                Progress = 0
                TotalTasks = 120
                CurrentTask = "Initializing custom transformer track..."
                StartTime = DateTime.UtcNow
                LastUpdateTime = DateTime.UtcNow
                ResourceAllocation = 30.0
                RiskLevel = High
                Timeline = "12 months"
                Strategy = RiskDiversification
                EstimatedCompletion = None
                SuccessMetrics = ["50% Better Accuracy"; "70% Cost Reduction"; "Custom Task Performance"]
                CompetingTracks = ["ai-blue-beta"; "ai-blue-gamma"]
            }
            {
                Id = "ai-blue-beta"
                Domain = AI_ML
                Type = BlueBeta
                Name = "Multimodal AI Integration"
                Description = "Integrated text, vision, and audio AI capabilities"
                Technologies = ["Vision Models"; "Audio Processing"; "Text-to-Speech"; "Multimodal Fusion"]
                Priorities = ["Multimodal Understanding"; "Context Awareness"; "Rich Interactions"]
                Status = NotStarted
                Progress = 0
                TotalTasks = 100
                CurrentTask = "Initializing multimodal AI track..."
                StartTime = DateTime.UtcNow
                LastUpdateTime = DateTime.UtcNow
                ResourceAllocation = 25.0
                RiskLevel = Medium
                Timeline = "8 months"
                Strategy = FeatureExploration
                EstimatedCompletion = None
                SuccessMetrics = ["Multimodal Accuracy 90%"; "Context Understanding"; "Rich UX"]
                CompetingTracks = ["ai-blue-alpha"; "ai-blue-gamma"]
            }
            {
                Id = "ai-blue-gamma"
                Domain = AI_ML
                Type = BlueGamma
                Name = "Edge AI Deployment"
                Description = "AI inference at the edge for low-latency processing"
                Technologies = ["ONNX"; "TensorFlow Lite"; "Edge Inference"; "Model Quantization"]
                Priorities = ["Low Latency"; "Privacy"; "Offline Capability"]
                Status = NotStarted
                Progress = 0
                TotalTasks = 70
                CurrentTask = "Initializing edge AI track..."
                StartTime = DateTime.UtcNow
                LastUpdateTime = DateTime.UtcNow
                ResourceAllocation = 15.0
                RiskLevel = Low
                Timeline = "6 months"
                Strategy = RiskDiversification
                EstimatedCompletion = None
                SuccessMetrics = ["Latency <50ms"; "Offline Capability"; "Privacy Compliance"]
                CompetingTracks = ["ai-blue-alpha"; "ai-blue-beta"]
            }
            {
                Id = "ai-blue-delta"
                Domain = AI_ML
                Type = BlueDelta
                Name = "Quantum ML Algorithms"
                Description = "Quantum machine learning for optimization problems"
                Technologies = ["Qiskit"; "Quantum Circuits"; "Hybrid Algorithms"; "Quantum Advantage"]
                Priorities = ["Quantum Advantage"; "Optimization"; "Research"]
                Status = NotStarted
                Progress = 0
                TotalTasks = 150
                CurrentTask = "Initializing quantum ML track..."
                StartTime = DateTime.UtcNow
                LastUpdateTime = DateTime.UtcNow
                ResourceAllocation = 10.0
                RiskLevel = VeryHigh
                Timeline = "18 months"
                Strategy = CompetitiveEvaluation
                EstimatedCompletion = None
                SuccessMetrics = ["Quantum Speedup"; "Optimization Improvement"; "Research Publications"]
                CompetingTracks = []
            }
        ]
        
        for track in multiBluTracks do
            tracks.TryAdd(track.Id, track) |> ignore
        
        logger.LogInformation($"🔬 Initialized {multiBluTracks.Length} multi-blue tracks with {multiBluTracks |> List.filter (fun t -> t.Type <> Green) |> List.length} experimental paths")
    
    /// Save tracks to disk
    member private this.SaveTracks() =
        try
            let tracksData = tracks.Values |> Seq.toArray
            let json = JsonSerializer.Serialize(tracksData, JsonSerializerOptions(WriteIndented = true))
            File.WriteAllText(tracksFile, json)
            logger.LogDebug("💾 Multi-blue tracks saved to disk")
        with
        | ex -> logger.LogError(ex, "❌ Failed to save multi-blue tracks")
    
    /// Load tracks from disk
    member private this.LoadTracks() =
        try
            if File.Exists(tracksFile) then
                let json = File.ReadAllText(tracksFile)
                let tracksData = JsonSerializer.Deserialize<MultiBluTrack[]>(json)
                
                tracks.Clear()
                for track in tracksData do
                    tracks.TryAdd(track.Id, track) |> ignore
                
                logger.LogInformation($"📂 Loaded {tracksData.Length} multi-blue tracks from disk")
            else
                this.InitializeMultiBluTracks()
        with
        | ex -> 
            logger.LogError(ex, "❌ Failed to load multi-blue tracks, initializing defaults")
            this.InitializeMultiBluTracks()
    
    /// Update track progress
    member private this.UpdateTrackProgress(trackId: string, progress: int, currentTask: string) =
        match tracks.TryGetValue(trackId) with
        | true, track ->
            let updatedTrack = {
                track with
                    Progress = progress
                    CurrentTask = currentTask
                    LastUpdateTime = DateTime.UtcNow
                    EstimatedCompletion = 
                        if progress > 0 then
                            let elapsed = DateTime.UtcNow - track.StartTime
                            let estimatedTotal = elapsed.TotalMinutes * (float track.TotalTasks) / (float progress)
                            Some (track.StartTime.AddMinutes(estimatedTotal))
                        else None
            }
            tracks.TryUpdate(trackId, updatedTrack, track) |> ignore
            this.SaveTracks()
            logger.LogDebug($"📈 Updated {trackId}: {progress}/{track.TotalTasks} - {currentTask}")
        | false, _ ->
            logger.LogWarning($"⚠️ Track not found: {trackId}")
    
    /// Execute track tasks with strategy-specific behavior
    member private this.ExecuteTrackTasks(track: MultiBluTrack, cancellationToken: CancellationToken) = task {
        try
            let trackTypeStr = match track.Type with
                              | Green -> "Green"
                              | BlueAlpha -> "Blue-Alpha"
                              | BlueBeta -> "Blue-Beta"
                              | BlueGamma -> "Blue-Gamma"
                              | BlueDelta -> "Blue-Delta"
            
            logger.LogInformation($"🚀 Starting {track.Domain} {trackTypeStr} track: {track.Name}")
            
            let taskDuration = match track.Type, track.RiskLevel with
                              | Green, _ -> 2000
                              | BlueAlpha, Low -> 2500
                              | BlueAlpha, Medium -> 3000
                              | BlueAlpha, High -> 4000
                              | BlueBeta, _ -> 3500
                              | BlueGamma, _ -> 3000
                              | BlueDelta, VeryHigh -> 5000
                              | _, _ -> 3000
            
            for i in 1 to track.TotalTasks do
                if cancellationToken.IsCancellationRequested then break
                
                // Check if track is paused
                let currentTrack = tracks.[track.Id]
                while currentTrack.Status = Paused && not cancellationToken.IsCancellationRequested do
                    do! Task.Delay(1000, cancellationToken)
                
                if cancellationToken.IsCancellationRequested then break
                
                // Simulate work with risk-adjusted duration
                let adjustedDuration = match track.RiskLevel with
                                      | Low -> taskDuration
                                      | Medium -> int (float taskDuration * 1.2)
                                      | High -> int (float taskDuration * 1.5)
                                      | VeryHigh -> int (float taskDuration * 2.0)
                
                do! Task.Delay(adjustedDuration, cancellationToken)
                
                let taskName = $"{trackTypeStr} {track.Domain} task {i}: {track.Strategy}"
                this.UpdateTrackProgress(track.Id, i, taskName)
                
                if i % 10 = 0 then
                    logger.LogInformation($"📊 {track.Name} milestone: {i}/{track.TotalTasks} completed")
            
            if not cancellationToken.IsCancellationRequested then
                let completedTrack = { tracks.[track.Id] with Status = Completed }
                tracks.TryUpdate(track.Id, completedTrack, tracks.[track.Id]) |> ignore
                this.SaveTracks()
                logger.LogInformation($"🎉 {track.Name} completed successfully!")
            
        with
        | :? OperationCanceledException ->
            logger.LogInformation($"⏸️ {track.Name} was cancelled")
            let pausedTrack = { tracks.[track.Id] with Status = Paused }
            tracks.TryUpdate(track.Id, pausedTrack, tracks.[track.Id]) |> ignore
        | ex ->
            logger.LogError(ex, $"❌ Error in {track.Name}")
            let errorTrack = { tracks.[track.Id] with Status = Error ex.Message }
            tracks.TryUpdate(track.Id, errorTrack, tracks.[track.Id]) |> ignore
    }
    
    /// Start track
    member this.StartTrack(trackId: string) =
        match tracks.TryGetValue(trackId) with
        | true, track when track.Status = NotStarted || track.Status = Paused ->
            let runningTrack = { track with Status = Running; StartTime = DateTime.UtcNow }
            tracks.TryUpdate(trackId, runningTrack, track) |> ignore
            this.SaveTracks()
            
            Task.Run(fun () -> this.ExecuteTrackTasks(runningTrack, cancellationTokenSource.Token)) |> ignore
            logger.LogInformation($"▶️ Started track: {track.Name}")
            true
        | true, track ->
            logger.LogWarning($"⚠️ Cannot start track {trackId} in state {track.Status}")
            false
        | false, _ ->
            logger.LogError($"❌ Track not found: {trackId}")
            false
    
    /// Get tracks by domain and type
    member this.GetTracksByDomainAndType(domain: TrackDomain, trackType: ExtendedTrackType) =
        tracks.Values 
        |> Seq.filter (fun t -> t.Domain = domain && t.Type = trackType) 
        |> Seq.toArray
    
    /// Get competing tracks analysis
    member this.GetCompetingTracksAnalysis(domain: TrackDomain) =
        let domainTracks = tracks.Values |> Seq.filter (fun t -> t.Domain = domain) |> Seq.toArray
        let blueTracks = domainTracks |> Array.filter (fun t -> t.Type <> Green)
        
        {|
            Domain = domain.ToString()
            GreenTrack = domainTracks |> Array.tryFind (fun t -> t.Type = Green)
            BlueTracks = blueTracks
            CompetitionAnalysis = 
                blueTracks 
                |> Array.map (fun track -> {|
                    TrackId = track.Id
                    Name = track.Name
                    Progress = track.Progress
                    TotalTasks = track.TotalTasks
                    Percentage = (float track.Progress / float track.TotalTasks) * 100.0
                    RiskLevel = track.RiskLevel.ToString()
                    ResourceAllocation = track.ResourceAllocation
                    Timeline = track.Timeline
                    Strategy = track.Strategy.ToString()
                    SuccessMetrics = track.SuccessMetrics
                    CompetingWith = track.CompetingTracks
                |})
            TotalBlueResources = blueTracks |> Array.sumBy (fun t -> t.ResourceAllocation)
            LeadingTrack = 
                blueTracks 
                |> Array.sortByDescending (fun t -> float t.Progress / float t.TotalTasks)
                |> Array.tryHead
                |> Option.map (fun t -> t.Id)
        |}
    
    /// Get all tracks
    member this.GetAllTracks() = tracks.Values |> Seq.toArray
    
    /// Get system overview with multi-blue insights
    member this.GetMultiBluSystemOverview() = {|
        TotalTracks = tracks.Count
        GreenTracks = tracks.Values |> Seq.filter (fun t -> t.Type = Green) |> Seq.length
        BlueAlphaTracks = tracks.Values |> Seq.filter (fun t -> t.Type = BlueAlpha) |> Seq.length
        BlueBetaTracks = tracks.Values |> Seq.filter (fun t -> t.Type = BlueBeta) |> Seq.length
        BlueGammaTracks = tracks.Values |> Seq.filter (fun t -> t.Type = BlueGamma) |> Seq.length
        BlueDeltaTracks = tracks.Values |> Seq.filter (fun t -> t.Type = BlueDelta) |> Seq.length
        RunningTracks = tracks.Values |> Seq.filter (fun t -> t.Status = Running) |> Seq.length
        DomainBreakdown = 
            tracks.Values 
            |> Seq.groupBy (fun t -> t.Domain)
            |> Seq.map (fun (domain, tracks) -> 
                let trackList = tracks |> Seq.toArray
                domain.ToString(), {|
                    Total = trackList.Length
                    Green = trackList |> Array.filter (fun t -> t.Type = Green) |> Array.length
                    BlueCount = trackList |> Array.filter (fun t -> t.Type <> Green) |> Array.length
                    BlueTypes = trackList |> Array.filter (fun t -> t.Type <> Green) |> Array.map (fun t -> t.Type.ToString())
                |})
            |> Map.ofSeq
        StrategyBreakdown = 
            tracks.Values 
            |> Seq.groupBy (fun t -> t.Strategy)
            |> Seq.map (fun (strategy, tracks) -> strategy.ToString(), Seq.length tracks)
            |> Map.ofSeq
        RiskBreakdown = 
            tracks.Values 
            |> Seq.groupBy (fun t -> t.RiskLevel)
            |> Seq.map (fun (risk, tracks) -> risk.ToString(), Seq.length tracks)
            |> Map.ofSeq
    |}
    
    /// Background service execution
    override this.ExecuteAsync(stoppingToken: CancellationToken) = task {
        logger.LogInformation("🔬 Multi-Blue Tracks Manager started")
        
        // Load existing tracks
        this.LoadTracks()
        
        // Keep service running
        while not stoppingToken.IsCancellationRequested do
            try
                do! Task.Delay(10000, stoppingToken)
            with
            | :? OperationCanceledException -> ()
            | ex -> logger.LogError(ex, "Error in multi-blue tracks manager loop")
        
        logger.LogInformation("🔬 Multi-Blue Tracks Manager stopped")
    }
    
    /// Dispose resources
    override this.Dispose() =
        cancellationTokenSource?.Cancel()
        cancellationTokenSource?.Dispose()
        base.Dispose()
