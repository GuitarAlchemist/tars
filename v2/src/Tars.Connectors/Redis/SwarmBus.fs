namespace Tars.Connectors.Redis

open System
open System.Text.Json
open System.Text.Json.Serialization
open StackExchange.Redis

// =========================================================================
// Swarm Message Types
// =========================================================================

/// A job posted to the work queue for a worker to pick up.
type SwarmJob =
    { JobId: string
      Goal: string
      PatternHint: string option
      MaxSteps: int
      Priority: int
      PostedBy: string
      PostedAt: DateTime }

/// Result submitted by a worker after completing a job.
type SwarmResult =
    { JobId: string
      WorkerId: string
      Success: bool
      Output: string
      PatternUsed: string
      DurationMs: int64
      StepCount: int
      CompletedAt: DateTime }

/// Heartbeat from a worker instance.
type SwarmHeartbeat =
    { WorkerId: string
      Status: string // "idle" | "busy" | "stopping"
      CurrentJobId: string option
      UptimeMs: int64
      CompletedJobs: int
      Timestamp: DateTime }

/// A belief to sync across the swarm.
type SwarmBelief =
    { Subject: string
      Predicate: string
      Object: string
      Confidence: float
      SourceWorker: string }

// =========================================================================
// Channel/Key constants
// =========================================================================

module SwarmChannels =
    [<Literal>]
    let WorkQueue = "tars:work"

    [<Literal>]
    let Results = "tars:results"

    [<Literal>]
    let Heartbeat = "tars:heartbeat"

    [<Literal>]
    let Knowledge = "tars:knowledge"

    [<Literal>]
    let Control = "tars:control"

    let workerKey (workerId: string) = $"tars:worker:{workerId}"
    let jobKey (jobId: string) = $"tars:job:{jobId}"
    let resultKey (jobId: string) = $"tars:result:{jobId}"

// =========================================================================
// SwarmBus - thin wrapper over StackExchange.Redis
// =========================================================================

module private SwarmSerialization =
    let jsonOptions =
        let opts = JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase)
        opts.Converters.Add(JsonFSharpConverter())
        opts

    let serialize value = JsonSerializer.Serialize(value, jsonOptions)
    let inline deserialize<'T> (s: string) : 'T = JsonSerializer.Deserialize<'T>(s, jsonOptions)

/// Connection manager and message bus for TARS swarm communication.
type SwarmBus(connectionString: string) =

    let mutable connection: IConnectionMultiplexer option = None
    let mutable subscriber: ISubscriber option = None

    let ensureConnected () =
        match connection with
        | Some c when c.IsConnected -> c
        | _ ->
            let c = ConnectionMultiplexer.Connect(connectionString)
            connection <- Some c
            subscriber <- Some(c.GetSubscriber())
            c

    let db () = (ensureConnected()).GetDatabase()
    let sub () =
        ensureConnected () |> ignore
        subscriber.Value

    // =====================================================================
    // Connection
    // =====================================================================

    /// Connect to Redis/Sider. Returns true if successful.
    member _.Connect() =
        try
            ensureConnected () |> ignore
            true
        with _ -> false

    /// Check if connected.
    member _.IsConnected =
        match connection with
        | Some c -> c.IsConnected
        | None -> false

    /// Disconnect and clean up.
    member _.Disconnect() =
        match connection with
        | Some c ->
            c.Dispose()
            connection <- None
            subscriber <- None
        | None -> ()

    // =====================================================================
    // Work Queue (List-based FIFO)
    // =====================================================================

    /// Post a job to the work queue.
    member _.PostJob(job: SwarmJob) =
        let json = SwarmSerialization.serialize job
        db().ListRightPush(RedisKey SwarmChannels.WorkQueue, RedisValue json) |> ignore
        // Also store job metadata for lookup
        db().StringSet(RedisKey(SwarmChannels.jobKey job.JobId), RedisValue json) |> ignore

    /// Take the next job from the queue (blocking pop with timeout).
    member _.TakeJob(timeoutSeconds: int) : SwarmJob option =
        let result = db().ListLeftPop(RedisKey SwarmChannels.WorkQueue)
        if result.IsNull then None
        else Some(SwarmSerialization.deserialize<SwarmJob>(string result))

    /// Peek at the queue length.
    member _.QueueLength() =
        db().ListLength(RedisKey SwarmChannels.WorkQueue) |> int

    // =====================================================================
    // Results
    // =====================================================================

    /// Submit a job result.
    member _.SubmitResult(result: SwarmResult) =
        let json = SwarmSerialization.serialize result
        // Store result
        db().StringSet(RedisKey(SwarmChannels.resultKey result.JobId), RedisValue json) |> ignore
        // Publish notification
        sub().Publish(RedisChannel(SwarmChannels.Results, RedisChannel.PatternMode.Literal), RedisValue json) |> ignore

    /// Get result for a job (if available).
    member _.GetResult(jobId: string) : SwarmResult option =
        let value = db().StringGet(RedisKey(SwarmChannels.resultKey jobId))
        if value.IsNull then None
        else Some(SwarmSerialization.deserialize<SwarmResult>(string value))

    /// Subscribe to results as they arrive.
    member _.OnResult(handler: SwarmResult -> unit) =
        sub().Subscribe(
            RedisChannel(SwarmChannels.Results, RedisChannel.PatternMode.Literal),
            fun _channel message ->
                try
                    let result = SwarmSerialization.deserialize<SwarmResult>(string message)
                    handler result
                with _ -> ())

    // =====================================================================
    // Heartbeat
    // =====================================================================

    /// Send a heartbeat.
    member _.SendHeartbeat(heartbeat: SwarmHeartbeat) =
        let json = SwarmSerialization.serialize heartbeat
        // Store with TTL (expire after 30s if no update)
        db().StringSet(
            RedisKey(SwarmChannels.workerKey heartbeat.WorkerId),
            RedisValue json,
            expiry = Expiration.op_Implicit(TimeSpan.FromSeconds(30.0))) |> ignore
        // Publish for live monitoring
        sub().Publish(
            RedisChannel(SwarmChannels.Heartbeat, RedisChannel.PatternMode.Literal),
            RedisValue json) |> ignore

    /// Get all known workers (with active heartbeats).
    member _.GetWorkers() : SwarmHeartbeat list =
        let server = (ensureConnected()).GetServers() |> Seq.head
        let keys =
            server.Keys(pattern = RedisValue "tars:worker:*")
            |> Seq.toList
        keys
        |> List.choose (fun key ->
            let value = db().StringGet(key)
            if value.IsNull then None
            else
                try Some(SwarmSerialization.deserialize<SwarmHeartbeat>(string value))
                with _ -> None)

    /// Subscribe to heartbeats.
    member _.OnHeartbeat(handler: SwarmHeartbeat -> unit) =
        sub().Subscribe(
            RedisChannel(SwarmChannels.Heartbeat, RedisChannel.PatternMode.Literal),
            fun _channel message ->
                try
                    let hb = SwarmSerialization.deserialize<SwarmHeartbeat>(string message)
                    handler hb
                with _ -> ())

    // =====================================================================
    // Knowledge Sync (Pub/Sub)
    // =====================================================================

    /// Broadcast a belief to all instances.
    member _.BroadcastBelief(belief: SwarmBelief) =
        let json = SwarmSerialization.serialize belief
        sub().Publish(
            RedisChannel(SwarmChannels.Knowledge, RedisChannel.PatternMode.Literal),
            RedisValue json) |> ignore

    /// Subscribe to belief broadcasts.
    member _.OnBelief(handler: SwarmBelief -> unit) =
        sub().Subscribe(
            RedisChannel(SwarmChannels.Knowledge, RedisChannel.PatternMode.Literal),
            fun _channel message ->
                try
                    let belief = SwarmSerialization.deserialize<SwarmBelief>(string message)
                    handler belief
                with _ -> ())

    // =====================================================================
    // Control Channel
    // =====================================================================

    /// Send a control command ("shutdown", "pause", "resume").
    member _.SendControl(command: string) =
        sub().Publish(
            RedisChannel(SwarmChannels.Control, RedisChannel.PatternMode.Literal),
            RedisValue command) |> ignore

    /// Subscribe to control commands.
    member _.OnControl(handler: string -> unit) =
        sub().Subscribe(
            RedisChannel(SwarmChannels.Control, RedisChannel.PatternMode.Literal),
            fun _channel message -> handler (string message))

    // =====================================================================
    // Utility
    // =====================================================================

    /// Flush all TARS swarm keys (for testing/reset).
    member _.FlushSwarm() =
        let server = (ensureConnected()).GetServers() |> Seq.head
        let keys = server.Keys(pattern = RedisValue "tars:*") |> Seq.toArray
        if keys.Length > 0 then
            db().KeyDelete(keys) |> ignore

    /// Get swarm stats.
    member this.Stats() =
        {| QueueLength = this.QueueLength()
           ActiveWorkers = this.GetWorkers().Length
           IsConnected = this.IsConnected |}

    interface IDisposable with
        member this.Dispose() = this.Disconnect()
