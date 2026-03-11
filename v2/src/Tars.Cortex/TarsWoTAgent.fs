namespace Tars.Cortex

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.Runtime.CompilerServices
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading
open System.Threading.Tasks
open Microsoft.Agents.AI
open Microsoft.Extensions.AI
open Tars.Core
open Tars.Cortex.WoTTypes

// ─────────────────────────────────────────────────────────────────────
// Session with persistent conversation history and execution traces
// ─────────────────────────────────────────────────────────────────────

/// Serializable conversation turn for session persistence.
[<CLIMutable>]
type SessionTurn = {
    [<JsonPropertyName("role")>] Role: string
    [<JsonPropertyName("content")>] Content: string
    [<JsonPropertyName("timestamp")>] Timestamp: string
}

/// Serializable execution trace summary for session persistence.
[<CLIMutable>]
type SessionTraceSummary = {
    [<JsonPropertyName("run_id")>] RunId: string
    [<JsonPropertyName("pattern")>] Pattern: string
    [<JsonPropertyName("steps")>] Steps: int
    [<JsonPropertyName("success")>] Success: bool
    [<JsonPropertyName("tokens_used")>] TokensUsed: int
    [<JsonPropertyName("duration_ms")>] DurationMs: int64
    [<JsonPropertyName("timestamp")>] Timestamp: string
}

/// Serializable session state.
[<CLIMutable>]
type SessionState = {
    [<JsonPropertyName("conversation")>] Conversation: SessionTurn list
    [<JsonPropertyName("traces")>] Traces: SessionTraceSummary list
    [<JsonPropertyName("created_at")>] CreatedAt: string
    [<JsonPropertyName("last_active")>] LastActive: string
}

/// <summary>
/// Session for the TarsWoTAgent that persists conversation history
/// and execution trace summaries across serialization boundaries.
/// </summary>
type TarsWoTAgentSession() =
    inherit AgentSession()

    let mutable conversation: SessionTurn list = []
    let mutable traces: SessionTraceSummary list = []
    let createdAt = DateTime.UtcNow

    /// Add a user message to the conversation history.
    member _.AddUserMessage(content: string) =
        conversation <- conversation @ [
            { Role = "user"; Content = content; Timestamp = DateTime.UtcNow.ToString("o") }
        ]

    /// Add an assistant response to the conversation history.
    member _.AddAssistantMessage(content: string) =
        conversation <- conversation @ [
            { Role = "assistant"; Content = content; Timestamp = DateTime.UtcNow.ToString("o") }
        ]

    /// Record a WoT execution trace summary.
    member _.AddTrace(result: WoTResult) =
        let summary: SessionTraceSummary = {
            RunId = result.Trace.RunId.ToString("N").[..7]
            Pattern = result.Trace.Plan.Metadata.Kind.ToString()
            Steps = result.Metrics.TotalSteps
            Success = result.Success
            TokensUsed = result.Metrics.TotalTokens
            DurationMs = result.Metrics.TotalDurationMs
            Timestamp = DateTime.UtcNow.ToString("o")
        }
        traces <- traces @ [ summary ]

    /// Get conversation history as ChatMessages for context.
    member _.GetConversationMessages() : ChatMessage list =
        conversation
        |> List.map (fun turn ->
            let role = if turn.Role = "assistant" then ChatRole.Assistant else ChatRole.User
            ChatMessage(role, turn.Content))

    /// Get the full serializable state.
    member _.GetState() : SessionState =
        { Conversation = conversation
          Traces = traces
          CreatedAt = createdAt.ToString("o")
          LastActive = DateTime.UtcNow.ToString("o") }

    /// Restore state from a deserialized SessionState.
    member _.RestoreState(state: SessionState) =
        conversation <- state.Conversation
        traces <- state.Traces

/// <summary>
/// Wraps the TARS Workflow-of-Thought executor as a Microsoft Agent Framework AIAgent.
/// This adapter allows the WoT engine to participate in MAF orchestrations.
/// </summary>
type TarsWoTAgent
    (
        executor: IWoTExecutor,
        compiler: IPatternCompiler,
        selector: IPatternSelector,
        agentContext: AgentContext,
        ?name: string,
        ?description: string
    ) =
    inherit AIAgent()

    let mutable agentName = defaultArg name "TarsWoT"
    let mutable agentDescription = defaultArg description "TARS Workflow-of-Thought reasoning agent"

    /// Gets or sets the display name of the agent.
    override _.Name = agentName
    /// Gets a description of the agent's purpose.
    override _.Description = agentDescription

    /// <summary>
    /// Extract the user's goal from the incoming chat messages.
    /// Takes the text content of the last user message.
    /// </summary>
    static member private ExtractGoal(messages: IEnumerable<ChatMessage>) : string =
        let userMessages =
            messages
            |> Seq.filter (fun m -> m.Role = ChatRole.User)
            |> Seq.toList

        match userMessages with
        | [] -> "No goal specified"
        | msgs ->
            let lastMsg = msgs |> List.last
            lastMsg.Text |> Option.ofObj |> Option.defaultValue "No goal specified"

    /// <summary>
    /// Run the WoT pipeline: select pattern -> compile -> execute.
    /// </summary>
    member private this.ExecuteWoT(goal: string, ct: CancellationToken) : Async<WoTResult> =
        async {
            // 1. Build a default cognitive state for pattern selection
            let cogState: WoTCognitiveState =
                { Mode = Exploratory
                  Eigenvalue = 1.0
                  Entropy = 0.5
                  BranchingFactor = 1.0
                  ActivePattern = None
                  WoTRunId = None
                  StepCount = 0
                  TokenBudget = None
                  LastTransition = DateTime.UtcNow
                  ConstraintScore = None
                  SuccessRate = 1.0 }

            // 2. Select the best pattern for this goal
            let patternKind = selector.Recommend(goal, cogState)

            // 3. Compile the pattern into a WoTPlan
            let plan =
                match patternKind with
                | ChainOfThought -> compiler.CompileChainOfThought(5, goal)
                | ReAct -> compiler.CompileReAct([ "search"; "calculate"; "read" ], 10, goal)
                | GraphOfThoughts -> compiler.CompileGraphOfThoughts(3, 3, goal)
                | TreeOfThoughts -> compiler.CompileTreeOfThoughts(3, 2, goal)
                | PlanAndExecute -> compiler.CompileChainOfThought(3, goal)
                | WorkflowOfThought -> compiler.CompileChainOfThought(5, goal)
                | Custom _ -> compiler.CompileChainOfThought(3, goal)

            // 4. Execute the plan through the WoT engine
            let ctx =
                { agentContext with
                    CancellationToken = ct }

            return! executor.Execute(plan, ctx)
        }

    /// <summary>
    /// Run the WoT pipeline with per-step progress reporting.
    /// Steps are collected into the provided ConcurrentQueue as they complete.
    /// </summary>
    member private this.ExecuteWoTWithProgress
        (
            goal: string,
            ct: CancellationToken,
            progressQueue: ConcurrentQueue<WoTTraceStep>
        ) : Async<WoTResult> =
        async {
            let cogState: WoTCognitiveState =
                { Mode = Exploratory
                  Eigenvalue = 1.0
                  Entropy = 0.5
                  BranchingFactor = 1.0
                  ActivePattern = None
                  WoTRunId = None
                  StepCount = 0
                  TokenBudget = None
                  LastTransition = DateTime.UtcNow
                  ConstraintScore = None
                  SuccessRate = 1.0 }

            let patternKind = selector.Recommend(goal, cogState)

            let plan =
                match patternKind with
                | ChainOfThought -> compiler.CompileChainOfThought(5, goal)
                | ReAct -> compiler.CompileReAct([ "search"; "calculate"; "read" ], 10, goal)
                | GraphOfThoughts -> compiler.CompileGraphOfThoughts(3, 3, goal)
                | TreeOfThoughts -> compiler.CompileTreeOfThoughts(3, 2, goal)
                | PlanAndExecute -> compiler.CompileChainOfThought(3, goal)
                | WorkflowOfThought -> compiler.CompileChainOfThought(5, goal)
                | Custom _ -> compiler.CompileChainOfThought(3, goal)

            let ctx =
                { agentContext with
                    CancellationToken = ct }

            let onProgress (step: WoTTraceStep) =
                progressQueue.Enqueue(step)

            return! executor.ExecuteWithProgress(plan, ctx, onProgress)
        }

    // =========================================================================
    // MAF AIAgent abstract member overrides
    // =========================================================================

    /// <summary>
    /// Creates a new in-memory session for this agent.
    /// </summary>
    override _.CreateSessionCoreAsync(_cancellationToken: CancellationToken) : ValueTask<AgentSession> =
        ValueTask<AgentSession>(TarsWoTAgentSession() :> AgentSession)

    /// <summary>
    /// Serializes a session to JSON, persisting conversation history and trace summaries.
    /// </summary>
    override _.SerializeSessionCoreAsync
        (
            session: AgentSession,
            jsonSerializerOptions: JsonSerializerOptions,
            _cancellationToken: CancellationToken
        ) : ValueTask<JsonElement> =
        let state =
            match session with
            | :? TarsWoTAgentSession as s -> s.GetState()
            | _ -> { Conversation = []; Traces = []; CreatedAt = DateTime.UtcNow.ToString("o"); LastActive = DateTime.UtcNow.ToString("o") }
        let json = JsonSerializer.Serialize(state, jsonSerializerOptions)
        let doc = JsonDocument.Parse(json)
        ValueTask<JsonElement>(doc.RootElement.Clone())

    /// <summary>
    /// Deserializes a session from JSON, restoring conversation history and trace summaries.
    /// </summary>
    override _.DeserializeSessionCoreAsync
        (
            serializedState: JsonElement,
            jsonSerializerOptions: JsonSerializerOptions,
            _cancellationToken: CancellationToken
        ) : ValueTask<AgentSession> =
        let session = TarsWoTAgentSession()
        try
            let json = serializedState.GetRawText()
            if json <> "{}" && json <> "null" then
                let state = JsonSerializer.Deserialize<SessionState>(json, jsonSerializerOptions)
                session.RestoreState(state)
        with _ -> () // Fresh session on deserialization failure
        ValueTask<AgentSession>(session :> AgentSession)

    /// <summary>
    /// Core implementation: runs the WoT executor to completion and returns
    /// the result as an AgentResponse containing a single assistant ChatMessage.
    /// </summary>
    override this.RunCoreAsync
        (
            messages: IEnumerable<ChatMessage>,
            session: AgentSession,
            _options: AgentRunOptions,
            cancellationToken: CancellationToken
        ) : Task<AgentResponse> =
        task {
            let goal = TarsWoTAgent.ExtractGoal(messages)

            // Track the user message in the session (safe cast)
            let wotSession =
                match session with
                | :? TarsWoTAgentSession as s -> Some s
                | _ -> None
            wotSession |> Option.iter (fun s -> s.AddUserMessage(goal))

            let! result =
                this.ExecuteWoT(goal, cancellationToken)
                |> Async.StartAsTask

            // Track the result in the session
            wotSession |> Option.iter (fun s ->
                s.AddAssistantMessage(result.Output)
                s.AddTrace(result))

            // Build the response message
            let responseMsg = ChatMessage(ChatRole.Assistant, result.Output)
            responseMsg.AuthorName <- agentName

            let response = AgentResponse(responseMsg)
            response.AgentId <- this.Id
            response.ResponseId <- Guid.NewGuid().ToString()
            return response
        }

    /// <summary>
    /// Streaming implementation: runs the WoT execution in the background and yields
    /// per-step AgentResponseUpdate items as each step completes, followed by a final
    /// update with the complete result.
    /// </summary>
    override this.RunCoreStreamingAsync
        (
            messages: IEnumerable<ChatMessage>,
            _session: AgentSession,
            _options: AgentRunOptions,
            [<EnumeratorCancellation>] cancellationToken: CancellationToken
        ) : IAsyncEnumerable<AgentResponseUpdate> =
        { new IAsyncEnumerable<AgentResponseUpdate> with
            member _.GetAsyncEnumerator(ct) =
                let effectiveCt =
                    if ct <> CancellationToken.None then ct
                    else cancellationToken

                let progressQueue = ConcurrentQueue<WoTTraceStep>()
                let completionSource = TaskCompletionSource<WoTResult>()
                let mutable stepIndex = 0
                let mutable currentUpdate: AgentResponseUpdate = Unchecked.defaultof<AgentResponseUpdate>
                let mutable executionStarted = false
                let mutable finalYielded = false
                let responseId = Guid.NewGuid().ToString()

                { new IAsyncEnumerator<AgentResponseUpdate> with
                    member _.Current = currentUpdate

                    member _.MoveNextAsync() =
                        let t = task {
                            // Start the background execution on first call
                            if not executionStarted then
                                executionStarted <- true
                                let goal = TarsWoTAgent.ExtractGoal(messages)
                                Task.Run((fun () ->
                                    task {
                                        try
                                            let! result =
                                                this.ExecuteWoTWithProgress(goal, effectiveCt, progressQueue)
                                                |> Async.StartAsTask
                                            completionSource.SetResult(result)
                                        with ex ->
                                            completionSource.SetException(ex)
                                    } :> Task), effectiveCt) |> ignore

                            // If we already yielded the final result, we are done
                            if finalYielded then
                                return false
                            else
                                // Poll for progress steps or completion
                                let mutable found = false
                                while not found do
                                    effectiveCt.ThrowIfCancellationRequested()

                                    // Try to dequeue a progress step
                                    let mutable step = Unchecked.defaultof<WoTTraceStep>
                                    if progressQueue.TryDequeue(&step) then
                                        stepIndex <- stepIndex + 1
                                        let outputText = step.Output |> Option.defaultValue ""
                                        let preview =
                                            if String.IsNullOrWhiteSpace(outputText) then "(no output)"
                                            elif outputText.Length > 120 then outputText.[..119] + "..."
                                            else outputText
                                        let text = sprintf "[Step %d] %s: %s" stepIndex step.NodeType preview
                                        let update = AgentResponseUpdate(Nullable ChatRole.Assistant, text)
                                        update.AgentId <- this.Id
                                        update.AuthorName <- agentName
                                        update.ResponseId <- responseId
                                        update.MessageId <- Guid.NewGuid().ToString()
                                        currentUpdate <- update
                                        found <- true
                                    elif completionSource.Task.IsCompleted then
                                        // Execution finished — drain any remaining steps first
                                        if progressQueue.TryDequeue(&step) then
                                            stepIndex <- stepIndex + 1
                                            let outputText = step.Output |> Option.defaultValue ""
                                            let preview =
                                                if String.IsNullOrWhiteSpace(outputText) then "(no output)"
                                                elif outputText.Length > 120 then outputText.[..119] + "..."
                                                else outputText
                                            let text = sprintf "[Step %d] %s: %s" stepIndex step.NodeType preview
                                            let update = AgentResponseUpdate(Nullable ChatRole.Assistant, text)
                                            update.AgentId <- this.Id
                                            update.AuthorName <- agentName
                                            update.ResponseId <- responseId
                                            update.MessageId <- Guid.NewGuid().ToString()
                                            currentUpdate <- update
                                            found <- true
                                        else
                                            // No more steps — yield the final result
                                            let result = completionSource.Task.Result
                                            let update = AgentResponseUpdate(Nullable ChatRole.Assistant, result.Output)
                                            update.AgentId <- this.Id
                                            update.AuthorName <- agentName
                                            update.ResponseId <- responseId
                                            update.MessageId <- Guid.NewGuid().ToString()
                                            currentUpdate <- update
                                            finalYielded <- true
                                            found <- true
                                    else
                                        // Nothing yet — brief yield before re-polling
                                        do! Task.Delay(25, effectiveCt)

                                return true
                        }
                        ValueTask<bool>(t)

                    member _.DisposeAsync() = ValueTask()
                }
        }
