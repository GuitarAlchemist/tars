namespace TarsEngine.FSharp.Cli.Core

open System
open System.IO
open System.Text.Json
open System.Threading
open System.Collections.Concurrent
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedTypes

/// TARS Unified State Manager - Thread-safe state management with persistence and versioning
module UnifiedStateManager =
    
    /// State change event
    type StateChangeEvent = {
        ChangeId: string
        ComponentName: string
        ChangeType: string
        OldValue: obj option
        NewValue: obj
        Timestamp: DateTime
        CorrelationId: string
        UserId: string option
    }

    /// State snapshot for versioning
    type StateSnapshot = {
        SnapshotId: string
        Timestamp: DateTime
        State: TarsUnifiedState
        Version: int64
        Description: string
        CreatedBy: string
    }

    /// State persistence options
    type PersistenceOptions = {
        EnableAutoPersistence: bool
        PersistenceIntervalMs: int
        MaxSnapshots: int
        CompressionEnabled: bool
        EncryptionEnabled: bool
        BackupEnabled: bool
    }

    /// Thread-safe state manager
    type TarsStateManager(initialState: TarsUnifiedState, persistenceOptions: PersistenceOptions, logger: ITarsLogger option) =
        let mutable currentState = initialState
        let stateHistory = ConcurrentQueue<StateChangeEvent>()
        let snapshots = ConcurrentQueue<StateSnapshot>()
        let mutable version = 0L
        let stateLock = new ReaderWriterLockSlim()
        let persistenceTimer = new Timer((fun _ -> ()), null, Timeout.Infinite, Timeout.Infinite)
        
        /// Generate unique change ID
        let generateChangeId() =
            let timestamp = DateTime.Now.ToString("yyyyMMddHHmmss")
            let guid = Guid.NewGuid().ToString("N").Substring(0, 8)
            $"change-{timestamp}-{guid}"

        /// Generate unique snapshot ID
        let generateSnapshotId() =
            let timestamp = DateTime.Now.ToString("yyyyMMddHHmmss")
            let guid = Guid.NewGuid().ToString("N").Substring(0, 8)
            $"snapshot-{timestamp}-{guid}"

        /// Get current state (read-only)
        member this.GetCurrentState() : TarsUnifiedState =
            stateLock.EnterReadLock()
            try
                currentState
            finally
                stateLock.ExitReadLock()

        /// Get state version
        member this.GetVersion() : int64 =
            Interlocked.Read(&version)

        /// Update state with change tracking
        member this.UpdateState(componentName: string, changeType: string, updater: TarsUnifiedState -> TarsUnifiedState, correlationId: string, userId: string option) : TarsResult<unit> =
            stateLock.EnterWriteLock()
            try
                let oldState = currentState
                let newState = updater oldState
                
                // Validate state change
                match this.ValidateStateChange(oldState, newState) with
                | Success _ ->
                    currentState <- { newState with LastStateUpdate = DateTime.Now }
                    Interlocked.Increment(&version) |> ignore
                    
                    // Record change event
                    let changeEvent = {
                        ChangeId = generateChangeId()
                        ComponentName = componentName
                        ChangeType = changeType
                        OldValue = Some (box oldState)
                        NewValue = box newState
                        Timestamp = DateTime.Now
                        CorrelationId = correlationId
                        UserId = userId
                    }
                    
                    stateHistory.Enqueue(changeEvent)
                    
                    // Trigger persistence if enabled
                    if persistenceOptions.EnableAutoPersistence then
                        this.TriggerPersistence() |> ignore
                    
                    logger |> Option.iter (fun l -> l.LogDebug(correlationId, $"State updated by {componentName}: {changeType}"))
                    Success ((), Map [("version", version); ("component", componentName)])
                
                | Failure (error, corrId) ->
                    logger |> Option.iter (fun l -> l.LogError(corrId, error))
                    Failure (error, corrId)
            finally
                stateLock.ExitWriteLock()

        /// Validate state change
        member private this.ValidateStateChange(oldState: TarsUnifiedState, newState: TarsUnifiedState) : TarsResult<unit> =
            try
                // Basic validation rules
                if newState.Configuration.MaxConcurrentOperations <= 0 then
                    Failure (ValidationError ("MaxConcurrentOperations must be positive", Map.empty), generateCorrelationId())
                elif newState.Configuration.OperationTimeoutMs <= 0 then
                    Failure (ValidationError ("OperationTimeoutMs must be positive", Map.empty), generateCorrelationId())
                elif String.IsNullOrWhiteSpace(newState.Configuration.DataDirectory) then
                    Failure (ValidationError ("DataDirectory cannot be empty", Map.empty), generateCorrelationId())
                else
                    Success ((), Map.empty)
            with
            | ex -> Failure (ValidationError ("State validation failed", Map [("error", ex.Message)]), generateCorrelationId())

        /// Get specific component state
        member this.GetComponentState<'T>(componentName: string, key: string) : TarsResult<'T option> =
            stateLock.EnterReadLock()
            try
                let state = currentState
                match componentName.ToLower() with
                | "flux" -> 
                    match state.FluxVariables.TryGetValue(key) with
                    | true, value -> 
                        match TypeConversion.tryConvert<'T> value with
                        | Some converted -> Success (Some converted, Map [("component", componentName); ("key", key)])
                        | None -> Success (None, Map [("component", componentName); ("key", key); ("reason", "type_mismatch")])
                    | false, _ -> Success (None, Map [("component", componentName); ("key", key); ("reason", "not_found")])
                
                | "agent" ->
                    match state.AgentStates.TryGetValue(key) with
                    | true, value ->
                        match TypeConversion.tryConvert<'T> value with
                        | Some converted -> Success (Some converted, Map [("component", componentName); ("key", key)])
                        | None -> Success (None, Map [("component", componentName); ("key", key); ("reason", "type_mismatch")])
                    | false, _ -> Success (None, Map [("component", componentName); ("key", key); ("reason", "not_found")])
                
                | "cache" ->
                    match state.CacheEntries.TryGetValue(key) with
                    | true, (value, _) ->
                        match TypeConversion.tryConvert<'T> value with
                        | Some converted -> Success (Some converted, Map [("component", componentName); ("key", key)])
                        | None -> Success (None, Map [("component", componentName); ("key", key); ("reason", "type_mismatch")])
                    | false, _ -> Success (None, Map [("component", componentName); ("key", key); ("reason", "not_found")])
                
                | _ -> Failure (ValidationError ($"Unknown component: {componentName}", Map [("component", componentName)]), generateCorrelationId())
            finally
                stateLock.ExitReadLock()

        /// Set component state
        member this.SetComponentState<'T>(componentName: string, key: string, value: 'T, correlationId: string, userId: string option) : TarsResult<unit> =
            this.UpdateState(componentName, "set_state", (fun state ->
                match componentName.ToLower() with
                | "flux" -> 
                    state.FluxVariables.[key] <- box value
                    state
                | "agent" ->
                    state.AgentStates.[key] <- box value
                    state
                | "cache" ->
                    state.CacheEntries.[key] <- (box value, DateTime.Now)
                    state
                | _ -> state
            ), correlationId, userId)

        /// Remove component state
        member this.RemoveComponentState(componentName: string, key: string, correlationId: string, userId: string option) : TarsResult<bool> =
            stateLock.EnterWriteLock()
            try
                let removed = 
                    match componentName.ToLower() with
                    | "flux" -> currentState.FluxVariables.TryRemove(key) |> fst
                    | "agent" -> currentState.AgentStates.TryRemove(key) |> fst
                    | "cache" -> currentState.CacheEntries.TryRemove(key) |> fst
                    | _ -> false
                
                if removed then
                    Interlocked.Increment(&version) |> ignore
                    currentState <- { currentState with LastStateUpdate = DateTime.Now }
                    
                    let changeEvent = {
                        ChangeId = generateChangeId()
                        ComponentName = componentName
                        ChangeType = "remove_state"
                        OldValue = None
                        NewValue = box key
                        Timestamp = DateTime.Now
                        CorrelationId = correlationId
                        UserId = userId
                    }
                    
                    stateHistory.Enqueue(changeEvent)
                    logger |> Option.iter (fun l -> l.LogDebug(correlationId, $"State removed from {componentName}: {key}"))
                
                Success (removed, Map [("component", componentName); ("key", key); ("removed", removed)])
            finally
                stateLock.ExitWriteLock()

        /// Create state snapshot
        member this.CreateSnapshot(description: string, createdBy: string) : TarsResult<string> =
            stateLock.EnterReadLock()
            try
                let snapshot = {
                    SnapshotId = generateSnapshotId()
                    Timestamp = DateTime.Now
                    State = currentState
                    Version = version
                    Description = description
                    CreatedBy = createdBy
                }
                
                snapshots.Enqueue(snapshot)
                
                // Maintain max snapshots limit
                while snapshots.Count > persistenceOptions.MaxSnapshots do
                    snapshots.TryDequeue() |> ignore
                
                logger |> Option.iter (fun l -> l.LogInformation(generateCorrelationId(), $"State snapshot created: {snapshot.SnapshotId}"))
                Success (snapshot.SnapshotId, Map [("version", version); ("description", description)])
            finally
                stateLock.ExitReadLock()

        /// Restore from snapshot
        member this.RestoreFromSnapshot(snapshotId: string, correlationId: string, userId: string option) : TarsResult<unit> =
            let snapshot = 
                snapshots
                |> Seq.tryFind (fun s -> s.SnapshotId = snapshotId)
            
            match snapshot with
            | Some snap ->
                stateLock.EnterWriteLock()
                try
                    currentState <- snap.State
                    version <- snap.Version
                    
                    let changeEvent = {
                        ChangeId = generateChangeId()
                        ComponentName = "StateManager"
                        ChangeType = "restore_snapshot"
                        OldValue = None
                        NewValue = box snapshotId
                        Timestamp = DateTime.Now
                        CorrelationId = correlationId
                        UserId = userId
                    }
                    
                    stateHistory.Enqueue(changeEvent)
                    logger |> Option.iter (fun l -> l.LogInformation(correlationId, $"State restored from snapshot: {snapshotId}"))
                    Success ((), Map [("snapshotId", snapshotId); ("version", version)])
                finally
                    stateLock.ExitWriteLock()
            | None ->
                Failure (ValidationError ($"Snapshot not found: {snapshotId}", Map [("snapshotId", snapshotId)]), correlationId)

        /// Get state change history
        member this.GetChangeHistory(limit: int option) : StateChangeEvent list =
            let events = stateHistory |> Seq.toList |> List.rev
            match limit with
            | Some l -> events |> List.take (Math.Min(l, events.Length))
            | None -> events

        /// Get available snapshots
        member this.GetSnapshots() : StateSnapshot list =
            snapshots |> Seq.toList |> List.rev

        /// Persist state to disk
        member this.PersistState() : TarsResult<string> =
            try
                let state = this.GetCurrentState()
                let dataDir = state.Configuration.DataDirectory
                
                if not (Directory.Exists(dataDir)) then
                    Directory.CreateDirectory(dataDir) |> ignore
                
                let timestamp = DateTime.Now.ToString("yyyyMMdd-HHmmss")
                let stateFile = Path.Combine(dataDir, $"tars-state-{timestamp}.json")
                let json = JsonSerializer.Serialize(state, JsonSerializerOptions(WriteIndented = true))
                
                File.WriteAllText(stateFile, json)
                
                logger |> Option.iter (fun l -> l.LogInformation(generateCorrelationId(), $"State persisted to: {stateFile}"))
                Success (stateFile, Map [("file", stateFile); ("size", File.ReadAllBytes(stateFile).Length)])
            with
            | ex ->
                let correlationId = generateCorrelationId()
                logger |> Option.iter (fun l -> l.LogError(correlationId, FileSystemError ("State persistence failed", ex.Message)))
                Failure (FileSystemError ("State persistence failed", ex.Message), correlationId)

        /// Load state from disk
        member this.LoadState(stateFile: string) : TarsResult<unit> =
            try
                if not (File.Exists(stateFile)) then
                    Failure (FileSystemError ("State file not found", stateFile), generateCorrelationId())
                else
                    let json = File.ReadAllText(stateFile)
                    let loadedState = JsonSerializer.Deserialize<TarsUnifiedState>(json)
                    
                    stateLock.EnterWriteLock()
                    try
                        currentState <- loadedState
                        Interlocked.Exchange(&version, loadedState.LastStateUpdate.Ticks) |> ignore
                        
                        logger |> Option.iter (fun l -> l.LogInformation(generateCorrelationId(), $"State loaded from: {stateFile}"))
                        Success ((), Map [("file", stateFile)])
                    finally
                        stateLock.ExitWriteLock()
            with
            | ex ->
                let correlationId = generateCorrelationId()
                logger |> Option.iter (fun l -> l.LogError(correlationId, FileSystemError ("State loading failed", ex.Message)))
                Failure (FileSystemError ("State loading failed", ex.Message), correlationId)

        /// Trigger persistence (internal)
        member private this.TriggerPersistence() : TarsResult<unit> =
            try
                persistenceTimer.Change(persistenceOptions.PersistenceIntervalMs, Timeout.Infinite) |> ignore
                Success ((), Map.empty)
            with
            | ex -> Failure (ExecutionError ("Failed to trigger persistence", Some ex), generateCorrelationId())

        /// Get state statistics
        member this.GetStatistics() : Map<string, obj> =
            let state = this.GetCurrentState()
            Map [
                ("version", version)
                ("lastUpdate", state.LastStateUpdate)
                ("activeOperations", state.ActiveOperations.Count)
                ("fluxVariables", state.FluxVariables.Count)
                ("agentStates", state.AgentStates.Count)
                ("cacheEntries", state.CacheEntries.Count)
                ("changeHistory", stateHistory.Count)
                ("snapshots", snapshots.Count)
                ("uptime", box (DateTime.Now - state.StartTime))
            ]

        interface IDisposable with
            member this.Dispose() =
                persistenceTimer.Dispose()
                stateLock.Dispose()

    /// Default persistence options
    let defaultPersistenceOptions = {
        EnableAutoPersistence = true
        PersistenceIntervalMs = 300000 // 5 minutes
        MaxSnapshots = 10
        CompressionEnabled = true
        EncryptionEnabled = false
        BackupEnabled = true
    }

    /// Create state manager with default options
    let createStateManager (initialState: TarsUnifiedState) (logger: ITarsLogger option) =
        new TarsStateManager(initialState, defaultPersistenceOptions, logger)

    /// Create state manager with custom options
    let createStateManagerWithOptions (initialState: TarsUnifiedState) (options: PersistenceOptions) (logger: ITarsLogger option) =
        new TarsStateManager(initialState, options, logger)
