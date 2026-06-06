namespace TarsEngine.FSharp.Cli.BeliefPropagation

open System
open System.Collections.Concurrent
open System.Threading.Channels
open System.Threading.Tasks
open System.Collections.Generic

// ============================================================================
// TARS REAL-TIME BELIEF PROPAGATION SYSTEM
// ============================================================================

type BeliefType = 
    | Performance
    | Confidence  
    | State
    | Insight
    | Prediction
    | Alert

type BeliefStrength =
    | Weak = 1
    | Moderate = 2
    | Strong = 3
    | Critical = 4

type SubsystemId =
    | CognitivePsychology
    | CudaAcceleration
    | VectorStores
    | AiEngine
    | FluxSystem
    | AgentTeams
    | TarsNodes
    | ApiSandbox
    | SelfEvolution

type Belief = {
    Id: string
    Source: SubsystemId
    BeliefType: BeliefType
    Strength: BeliefStrength
    Confidence: float
    Message: string
    Data: Map<string, obj>
    Timestamp: DateTime
    ExpiresAt: DateTime option
    TargetSubsystems: SubsystemId list option // None = broadcast to all
}

type BeliefResponse = {
    OriginalBeliefId: string
    Responder: SubsystemId
    ResponseType: string // "validate", "challenge", "enhance", "act"
    Message: string
    Data: Map<string, obj>
    Timestamp: DateTime
}

// ============================================================================
// BELIEF BUS - CENTRAL COMMUNICATION HUB
// ============================================================================

type TarsBeliefBus() =
    let beliefChannel = Channel.CreateUnbounded<Belief>()
    let responseChannel = Channel.CreateUnbounded<BeliefResponse>()
    let subscribers = ConcurrentDictionary<SubsystemId, Channel<Belief>>()
    let responseSubscribers = ConcurrentDictionary<SubsystemId, Channel<BeliefResponse>>()
    let activeBeliefsCache = ConcurrentDictionary<string, Belief>()
    let beliefHistory = ConcurrentQueue<Belief>()
    let mutable isRunning = false

    // Belief strength thresholds for propagation
    let propagationThresholds = Map [
        (Performance, BeliefStrength.Moderate)
        (Confidence, BeliefStrength.Weak)
        (State, BeliefStrength.Strong)
        (Insight, BeliefStrength.Moderate)
        (Prediction, BeliefStrength.Strong)
        (Alert, BeliefStrength.Weak)
    ]

    member this.Subscribe(subsystemId: SubsystemId) =
        let channel = Channel.CreateUnbounded<Belief>()
        subscribers.TryAdd(subsystemId, channel) |> ignore
        channel.Reader

    member this.SubscribeToResponses(subsystemId: SubsystemId) =
        let channel = Channel.CreateUnbounded<BeliefResponse>()
        responseSubscribers.TryAdd(subsystemId, channel) |> ignore
        channel.Reader

    member this.PublishBelief(belief: Belief) =
        task {
            // Add to cache and history
            activeBeliefsCache.TryAdd(belief.Id, belief) |> ignore
            beliefHistory.Enqueue(belief)
            
            // Keep history manageable
            if beliefHistory.Count > 1000 then
                beliefHistory.TryDequeue() |> ignore

            // Check if belief meets propagation threshold
            let threshold = propagationThresholds.TryFind(belief.BeliefType) |> Option.defaultValue BeliefStrength.Moderate
            if belief.Strength >= threshold then
                do! beliefChannel.Writer.WriteAsync(belief)
        }

    member this.PublishResponse(response: BeliefResponse) =
        task {
            do! responseChannel.Writer.WriteAsync(response)
        }

    member this.StartPropagation() =
        if not isRunning then
            isRunning <- true
            
            // Start belief propagation task
            Task.Run(System.Func<Task>(fun () ->
                task {
                    while isRunning do
                        try
                            let! belief = beliefChannel.Reader.ReadAsync()
                            
                            // Determine target subsystems
                            let targets = 
                                match belief.TargetSubsystems with
                                | Some targets -> targets
                                | None -> subscribers.Keys |> List.ofSeq
                            
                            // Propagate to target subsystems (except source)
                            for target in targets do
                                if target <> belief.Source then
                                    match subscribers.TryGetValue(target) with
                                    | true, channel -> 
                                        try
                                            do! channel.Writer.WriteAsync(belief)
                                        with
                                        | _ -> () // Ignore if channel is closed
                                    | false, _ -> ()
                        with
                        | _ -> () // Continue on errors
                }
            )) |> ignore

            // Start response propagation task
            Task.Run(System.Func<Task>(fun () ->
                task {
                    while isRunning do
                        try
                            let! response = responseChannel.Reader.ReadAsync()
                            
                            // Propagate responses to all subscribers
                            for kvp in responseSubscribers do
                                if kvp.Key <> response.Responder then
                                    try
                                        do! kvp.Value.Writer.WriteAsync(response)
                                    with
                                    | _ -> () // Ignore if channel is closed
                        with
                        | _ -> () // Continue on errors
                }
            )) |> ignore

    member this.Stop() =
        isRunning <- false
        beliefChannel.Writer.Complete()
        responseChannel.Writer.Complete()

    member this.GetActiveBeliefsCount() = activeBeliefsCache.Count
    member this.GetBeliefHistory() = beliefHistory |> List.ofSeq
    member this.GetActiveBeliefs() = activeBeliefsCache.Values |> List.ofSeq

// ============================================================================
// BELIEF FACTORY - HELPER FOR CREATING BELIEFS
// ============================================================================

type BeliefFactory() =
    static member CreatePerformanceBelief(source: SubsystemId, metric: string, value: float, message: string) =
        {
            Id = Guid.NewGuid().ToString()
            Source = source
            BeliefType = Performance
            Strength = if value > 80.0 then BeliefStrength.Strong elif value > 60.0 then BeliefStrength.Moderate else BeliefStrength.Weak
            Confidence = value / 100.0
            Message = message
            Data = Map [("metric", metric :> obj); ("value", value :> obj)]
            Timestamp = DateTime.UtcNow
            ExpiresAt = Some(DateTime.UtcNow.AddMinutes(5.0))
            TargetSubsystems = None
        }

    static member CreateConfidenceBelief(source: SubsystemId, confidence: float, context: string, message: string) =
        {
            Id = Guid.NewGuid().ToString()
            Source = source
            BeliefType = Confidence
            Strength = if confidence > 0.8 then BeliefStrength.Strong elif confidence > 0.5 then BeliefStrength.Moderate else BeliefStrength.Weak
            Confidence = confidence
            Message = message
            Data = Map [("context", context :> obj); ("confidence", confidence :> obj)]
            Timestamp = DateTime.UtcNow
            ExpiresAt = Some(DateTime.UtcNow.AddMinutes(2.0))
            TargetSubsystems = None
        }

    static member CreateInsightBelief(source: SubsystemId, insight: string, evidence: string list, confidence: float) =
        {
            Id = Guid.NewGuid().ToString()
            Source = source
            BeliefType = Insight
            Strength = BeliefStrength.Strong
            Confidence = confidence
            Message = insight
            Data = Map [("evidence", evidence :> obj); ("insight_type", "pattern_discovery" :> obj)]
            Timestamp = DateTime.UtcNow
            ExpiresAt = Some(DateTime.UtcNow.AddMinutes(10.0))
            TargetSubsystems = None
        }

    static member CreateAlertBelief(source: SubsystemId, alertType: string, severity: string, message: string) =
        let strength = 
            match severity.ToLower() with
            | "critical" -> BeliefStrength.Critical
            | "high" -> BeliefStrength.Strong
            | "medium" -> BeliefStrength.Moderate
            | _ -> BeliefStrength.Weak

        {
            Id = Guid.NewGuid().ToString()
            Source = source
            BeliefType = Alert
            Strength = strength
            Confidence = 0.9
            Message = message
            Data = Map [("alert_type", alertType :> obj); ("severity", severity :> obj)]
            Timestamp = DateTime.UtcNow
            ExpiresAt = Some(DateTime.UtcNow.AddMinutes(1.0))
            TargetSubsystems = None
        }

// ============================================================================
// SUBSYSTEM BELIEF INTERFACE
// ============================================================================

type IBeliefAwareSubsystem =
    abstract member ProcessBelief: Belief -> Task<BeliefResponse option>
    abstract member GetSubsystemId: unit -> SubsystemId
    abstract member GetCurrentBeliefs: unit -> Belief list
