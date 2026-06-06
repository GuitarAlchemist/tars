namespace Tars.Tests

open System
open System.Collections.Generic
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open Xunit
open Microsoft.Agents.AI
open Microsoft.Extensions.AI
open Tars.Cortex

/// Stub session for testing.
type StubSession() =
    inherit AgentSession()

/// Minimal stub AIAgent for testing orchestrator routing and pipeline logic.
/// RunCoreAsync returns a canned response containing the agent name and input.
type StubAgent(agentName: string, agentId: string) =
    inherit AIAgent()

    override _.Name = agentName
    override _.Description = "stub"

    override _.CreateSessionCoreAsync(_ct: CancellationToken) : ValueTask<AgentSession> =
        ValueTask<AgentSession>(StubSession() :> AgentSession)

    override _.SerializeSessionCoreAsync
        (_session: AgentSession, _opts: JsonSerializerOptions, _ct: CancellationToken)
        : ValueTask<JsonElement> =
        let doc = JsonDocument.Parse("{}")
        ValueTask<JsonElement>(doc.RootElement.Clone())

    override _.DeserializeSessionCoreAsync
        (_data: JsonElement, _opts: JsonSerializerOptions, _ct: CancellationToken)
        : ValueTask<AgentSession> =
        ValueTask<AgentSession>(StubSession() :> AgentSession)

    override _.RunCoreAsync
        (messages: IEnumerable<ChatMessage>, _session: AgentSession,
         _options: AgentRunOptions, _ct: CancellationToken)
        : Task<AgentResponse> =
        task {
            let inputText =
                messages
                |> Seq.tryLast
                |> Option.bind (fun (m: ChatMessage) -> m.Text |> Option.ofObj)
                |> Option.defaultValue ""
            let responseMsg = ChatMessage(ChatRole.Assistant, $"[{agentName}] processed: {inputText}")
            let response = AgentResponse(responseMsg)
            response.AgentId <- agentId
            return response
        }

    override _.RunCoreStreamingAsync
        (_messages: IEnumerable<ChatMessage>, _session: AgentSession,
         _options: AgentRunOptions, _ct: CancellationToken)
        : IAsyncEnumerable<AgentResponseUpdate> =
        { new IAsyncEnumerable<AgentResponseUpdate> with
            member _.GetAsyncEnumerator(_ct) =
                { new IAsyncEnumerator<AgentResponseUpdate> with
                    member _.Current = Unchecked.defaultof<AgentResponseUpdate>
                    member _.MoveNextAsync() = ValueTask<bool>(false)
                    member _.DisposeAsync() = ValueTask() } }


type OrchestratorRoutingTests() =

    let mkOrchestrator () =
        let orch = AgentOrchestrator()
        let agentA = StubAgent("analyzer", "agent-1")
        let agentB = StubAgent("coder", "agent-2")
        let agentC = StubAgent("reviewer", "agent-3")
        orch.Register(agentA, [ "analyze"; "complexity"; "metrics" ], priority = 1)
        orch.Register(agentB, [ "code"; "implement"; "refactor" ], priority = 2)
        orch.Register(agentC, [ "review"; "check"; "verify" ], priority = 0)
        orch

    [<Fact>]
    member _.``Route matches agent by capability keyword``() =
        let orch = mkOrchestrator ()
        let result = orch.Route("analyze the code complexity")
        Assert.True(result.IsSome, "Should find a matching agent")
        Assert.Equal("analyzer", result.Value.Agent.Name)

    [<Fact>]
    member _.``Route matches coder for implement goal``() =
        let orch = mkOrchestrator ()
        let result = orch.Route("implement a new feature in code")
        Assert.True(result.IsSome)
        Assert.Equal("coder", result.Value.Agent.Name)

    [<Fact>]
    member _.``Route matches reviewer for verify goal``() =
        let orch = mkOrchestrator ()
        let result = orch.Route("verify the output is correct")
        Assert.True(result.IsSome)
        Assert.Equal("reviewer", result.Value.Agent.Name)

    [<Fact>]
    member _.``Route falls back to highest priority when no keyword matches``() =
        let orch = mkOrchestrator ()
        let result = orch.Route("do something completely unrelated")
        Assert.True(result.IsSome, "Should fall back to highest priority agent")
        // Coder has priority 2, which is highest
        Assert.Equal("coder", result.Value.Agent.Name)

    [<Fact>]
    member _.``Route returns None when no agents registered``() =
        let orch = AgentOrchestrator()
        let result = orch.Route("anything")
        Assert.True(result.IsNone, "Should return None when no agents registered")

    [<Fact>]
    member _.``GetRegisteredAgents returns all registered names``() =
        let orch = mkOrchestrator ()
        let names = orch.GetRegisteredAgents()
        Assert.Contains("analyzer", names)
        Assert.Contains("coder", names)
        Assert.Contains("reviewer", names)
        Assert.Equal(3, names.Length)

    // ─────────────────────────────────────────────────────────────────────
    // Pipeline tests
    // ─────────────────────────────────────────────────────────────────────

    [<Fact>]
    member _.``Pipeline executes agents in sequence``() =
        let orch = mkOrchestrator ()
        let results =
            orch.Pipeline([ "analyzer"; "coder" ], "initial goal")
            |> Async.AwaitTask
            |> Async.RunSynchronously
        Assert.Equal(2, results.Length)
        // First agent should process the initial goal
        Assert.True(results.[0].Success)
        Assert.Equal("analyzer", results.[0].AgentName)
        Assert.Contains("initial goal", results.[0].Response)
        // Second agent should process the first agent's output (chaining)
        Assert.True(results.[1].Success)
        Assert.Equal("coder", results.[1].AgentName)
        Assert.Contains("[analyzer]", results.[1].Response)

    [<Fact>]
    member _.``Pipeline handles missing agent gracefully``() =
        let orch = mkOrchestrator ()
        let results =
            orch.Pipeline([ "analyzer"; "nonexistent"; "coder" ], "test")
            |> Async.AwaitTask
            |> Async.RunSynchronously
        Assert.Equal(3, results.Length)
        Assert.True(results.[0].Success)
        Assert.False(results.[1].Success)
        Assert.Contains("not found", results.[1].Response)
        // Coder should still execute (pipeline continues)
        Assert.True(results.[2].Success)

    // ─────────────────────────────────────────────────────────────────────
    // FanOut tests
    // ─────────────────────────────────────────────────────────────────────

    [<Fact>]
    member _.``FanOut executes agents in parallel on same goal``() =
        let orch = mkOrchestrator ()
        let results =
            orch.FanOut([ "analyzer"; "coder"; "reviewer" ], "shared goal")
            |> Async.AwaitTask
            |> Async.RunSynchronously
        Assert.Equal(3, results.Length)
        // All should have processed the same goal
        for r in results do
            Assert.True(r.Success)
            Assert.Contains("shared goal", r.Response)

    [<Fact>]
    member _.``FanOut skips missing agents without failing``() =
        let orch = mkOrchestrator ()
        let results =
            orch.FanOut([ "analyzer"; "ghost_agent"; "coder" ], "test")
            |> Async.AwaitTask
            |> Async.RunSynchronously
        // Only 2 results since "ghost_agent" is not registered (skipped via List.choose)
        Assert.Equal(2, results.Length)

    [<Fact>]
    member _.``Execute routes to best agent and runs``() =
        let orch = mkOrchestrator ()
        let result =
            orch.Execute("analyze something")
            |> Async.AwaitTask
            |> Async.RunSynchronously
        Assert.True(result.Success)
        Assert.Equal("analyzer", result.AgentName)

    [<Fact>]
    member _.``Execute returns failure when no agents registered``() =
        let orch = AgentOrchestrator()
        let result =
            orch.Execute("anything")
            |> Async.AwaitTask
            |> Async.RunSynchronously
        Assert.False(result.Success)
        Assert.Contains("No agent registered", result.Response)
