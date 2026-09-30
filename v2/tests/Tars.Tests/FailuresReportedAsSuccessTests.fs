namespace Tars.Tests

open System
open Xunit
open Tars.Core
open Tars.Cortex
open Tars.Cortex.WoTTypes
open Tars.Llm.LlmService
open Tars.Llm

// Three places where something that had failed came back looking like a success.
// Each test is written from the failure it would have hidden.

/// Answers instantly, reporting the token usage it was given.
type CountingLlm(tokensPerCall: int) =
    interface ILlmService with
        member _.CompleteAsync(_req) =
            task {
                return
                    { Text = "VERIFIED"
                      FinishReason = Some "stop"
                      Usage =
                        Some
                            { PromptTokens = tokensPerCall
                              CompletionTokens = 0
                              TotalTokens = tokensPerCall }
                      Raw = None }
            }

        member this.CompleteStreamAsync(req, _onToken) =
            (this :> ILlmService).CompleteAsync(req)

        member _.EmbedAsync(_text) = task { return [| 0.1f |] }

        member _.RouteAsync(_) =
            task {
                return
                    { Backend = Ollama "mock"
                      Endpoint = Uri "http://localhost:11434"
                      ApiKey = None }
            }

type FailuresReportedAsSuccessTests() =

    static let noTools _name _args : Async<Result<string, string>> =
        async { return Result.Error "no tools here" }

    // -------------------------------------------------- a schema nobody could parse

    [<Fact>]
    member _.``A schema that is not valid JSON verifies nothing``() =
        // One unquoted key — a typo, or a schema a model wrote — and this used to
        // report every payload verified, including ones the schema would plainly
        // have rejected.
        let broken = "{ type: object, required: [answer] }"

        match
            Verification.verify """{"anything":1}""" (Schema broken) noTools
            |> Async.RunSynchronously
        with
        | Result.Ok verdict -> failwith $"expected an error, got {verdict}"
        | Result.Error message -> Assert.Contains("not valid JSON", message)

    [<Fact>]
    member _.``A schema that parses is still applied``() =
        let schema = """{"type":"object","required":["answer"]}"""

        let verdict (content: string) =
            match Verification.verify content (Schema schema) noTools |> Async.RunSynchronously with
            | Result.Ok value -> value
            | Result.Error message -> failwith message

        Assert.True(verdict """{"answer":"yes"}""")
        Assert.False(verdict """{"other":"no"}""")

    // ------------------------------------------------------------ a check nobody knows

    [<Fact>]
    member _.``A custom check nobody knows verifies nothing``() =
        // `Ok true` before #284: `CustomOp "valid_jsn"` passed every payload.
        match Verification.verify "anything" (CustomOp "valid_jsn") noTools |> Async.RunSynchronously with
        | Result.Ok verdict -> failwith $"expected an error, got {verdict}"
        | Result.Error message -> Assert.Contains("valid_jsn", message)

    [<Fact>]
    member _.``The MCP validator never counts an unchecked invariant as a pass``() =
        let invariant name op : WoTInvariant = { Name = name; Op = op; Weight = 1.0 }

        // Each of these answered `true // placeholder` in `ClaudeCodeBridge.validateStep`,
        // whatever the content: an MCP caller was told its payload satisfied a schema
        // nothing had read, and a tool check nothing had run.
        let results =
            ClaudeCodeBridge.checkInvariants
                """{"other":"no"}"""
                [ invariant "has answer" (Schema """{"type":"object","required":["answer"]}""")
                  invariant "made up" (CustomOp "valid_jsn")
                  invariant "file exists" (ToolCheck("file_exists", Map.empty))
                  invariant "non empty" (CustomOp "non_empty") ]
            |> Map.ofList

        // Checked, and the content is wrong.
        Assert.Equal(Result.Ok false, results["has answer"])

        // Not checked, and it says why.
        match results["made up"], results["file exists"] with
        | Result.Error unknown, Result.Error tool ->
            Assert.Contains("valid_jsn", unknown)
            Assert.Contains("was not run", tool)
        | other -> failwith $"expected both to be reported unchecked, got %A{other}"

        // And a check that holds still holds.
        Assert.Equal(Result.Ok true, results["non empty"])

    [<Fact>]
    member _.``Content of the wrong shape fails a schema, rather than going unchecked``() =
        // Codex on #350: an array or a scalar made the required-field lookup throw, and
        // the catch reported a valid schema as unreadable - so the MCP validator said
        // "not verified" about content that had simply failed.
        let schema = """{"type":"object","required":["answer"]}"""

        for content in [ "[1,2]"; "42"; "\"just text\"" ] do
            Assert.Equal(Result.Ok false, Verification.verify content (Schema schema) noTools |> Async.RunSynchronously)

    // ------------------------------------------------------------ work nobody did

    [<Fact>]
    member _.``A swarm worker with no LLM says it did not work on the goal, and teaches the selector nothing``() =
        // The worker recorded "[Worker ...] Reasoning: <prompt>" as each Reason node's
        // output, reported the job a success, and recorded that success for the pattern.
        let recorded = ResizeArray<PatternOutcome>()

        let selector =
            { new IPatternSelector with
                member _.Recommend(_, _) = PatternKind.ChainOfThought
                member _.Score(_) = Map.empty
                member _.RecordOutcome(outcome) = recorded.Add outcome }

        // A plan that reasons, then runs a tool: the tool runs, the reasoning cannot.
        let echo: Tool =
            { Name = "swarm_echo"
              Description = "Echo the input back"
              Version = "1.0.0"
              ParentVersion = None
              CreatedAt = DateTime.UtcNow
              Execute = fun input -> async { return Result.Ok input } }

        let registry =
            { new IToolRegistry with
                member _.Register(_) = ()
                member _.Get(name) = if name = echo.Name then Some echo else None
                member _.GetAll() = [ echo ] }

        let think = PatternCompiler.think "Why is the sky blue?" None
        let act = PatternCompiler.act echo.Name Map.empty

        let plan: WoTPlan =
            { Id = Guid.NewGuid()
              Nodes = [ think; act ]
              Edges =
                [ { From = think.Id
                    To = act.Id
                    Label = None
                    Confidence = None } ]
              EntryNode = think.Id
              Metadata =
                ({ Kind = PatternKind.ChainOfThought
                   SourceGoal = "Explain why the sky is blue"
                   CompiledAt = DateTime.UtcNow
                   EstimatedTokens = None
                   EstimatedSteps = None }
                : PatternMetadata)
              Policy = [] }

        let compiler =
            { new IPatternCompiler with
                member _.CompileFor(_, _) = plan
                member _.CompileChainOfThought(_, _) = failwith "not used"
                member _.CompileReAct(_, _, _) = failwith "not used"
                member _.CompileGraphOfThoughts(_, _, _) = failwith "not used"
                member _.CompileTreeOfThoughts(_, _, _) = failwith "not used"
                member _.CompilePattern(_, _) = failwith "not used" }

        // Never connected: running one job does not touch the bus.
        use bus = new Tars.Connectors.Redis.SwarmBus("localhost:1")
        let worker = Tars.Connectors.Redis.SwarmWorker(bus, compiler, selector, registry)

        let result =
            worker.ExecuteJob
                { JobId = "job-1"
                  Goal = "Explain why the sky is blue"
                  PatternHint = None
                  MaxSteps = 3
                  Priority = 1
                  PostedBy = "test"
                  PostedAt = DateTime.UtcNow }

        Assert.False(result.Success, result.Output)
        Assert.Contains("1 Reason node(s) were not run", result.Output)
        Assert.Contains("1 other step(s) ran", result.Output)
        Assert.Empty(recorded)

    [<Fact>]
    member _.``A tool nobody registered fails its node, rather than the model making up what it returned``() =
        // The executor used to ask the LLM to "produce a plausible output for this tool
        // call" and record the answer as the tool's result, with the step Completed.
        let asked = ref 0

        let answer () =
            task {
                asked.Value <- asked.Value + 1

                return
                    { Text = "AAPL 191.20"
                      FinishReason = Some "stop"
                      Usage = None
                      Raw = None }
            }

        let llm =
            { new ILlmService with
                member _.CompleteAsync(_req) = answer ()
                member _.CompleteStreamAsync(_req, _onToken) = answer ()
                member _.EmbedAsync(_text) = task { return [| 0.1f |] }

                member _.RouteAsync(_) =
                    task {
                        return
                            { Backend = Ollama "mock"
                              Endpoint = Uri "http://localhost:11434"
                              ApiKey = None }
                    } }

        let registry =
            { new IToolRegistry with
                member _.Register(_) = ()
                member _.Get(_) = None
                member _.GetAll() = [] }

        let fetch = PatternCompiler.act "fetch_prices" Map.empty

        let plan: WoTPlan =
            { Id = Guid.NewGuid()
              Nodes = [ fetch ]
              Edges = []
              EntryNode = fetch.Id
              Metadata =
                ({ Kind = PatternKind.ReAct
                   SourceGoal = "today's prices"
                   CompiledAt = DateTime.UtcNow
                   EstimatedTokens = None
                   EstimatedSteps = None }
                : PatternMetadata)
              Policy = [] }

        let context: WoTExecutor.ExecutionContext =
            { Llm = llm
              Tools = registry
              Logger = ignore
              OnProgress = ignore
              CancellationToken = System.Threading.CancellationToken.None
              KnowledgeGraph = None
              Reflector = None
              Decider = None }

        let result = WoTExecutor.execute context plan |> Async.RunSynchronously

        match (result.Trace.Steps |> List.find (fun s -> s.NodeId = fetch.Id)).Status with
        | NodeStatus.Failed(error, _) -> Assert.Contains("not registered", error)
        | other -> failwith $"the step was {other}"

        Assert.False(result.Success)
        Assert.Equal(0, asked.Value)

    [<Fact>]
    member _.``Spawning a subagent says nothing was started, rather than reporting research it never did``() =
        // The MCP server's only subagent runner waited a second and reported
        // `Success = true`, "Completed research on: <goal>", for any goal at all.
        let registry =
            { new IToolRegistry with
                member _.Register(_) = ()
                member _.Get(_) = None
                member _.GetAll() = [] }

        let server = Tars.Connectors.Mcp.McpServer(registry)

        let answer =
            server.HandleRequest(
                """{"jsonrpc":"2.0","id":1,"method":"subagents/spawn","params":{"goal":"survey the literature"}}"""
            )
            |> Async.AwaitTask
            |> Async.RunSynchronously

        match answer with
        | None -> failwith "the server gave no answer"
        | Some json ->
            use doc = System.Text.Json.JsonDocument.Parse json
            let root = doc.RootElement

            // No subagent id to poll, so no later "completed" to believe.
            match root.TryGetProperty "result" with
            | true, result when result.ValueKind <> System.Text.Json.JsonValueKind.Null ->
                failwith $"spawn was accepted: {json}"
            | _ -> ()

            Assert.Contains("nothing was started", root.GetProperty("error").GetProperty("message").GetString())

    [<Fact>]
    member _.``Asking another agent says it was not asked, rather than answering for it``() =
        if not (TestHelpers.requireTools ()) then () else

        // query_agent returned text written in advance for each agent name - "Can approve
        // or request changes" - whatever the question, as that agent's reply.
        let answer =
            Tars.Tools.Standard.AgentTools.queryAgent """{"agent": "reviewer", "question": "Is this safe to merge?"}"""
            |> Async.AwaitTask
            |> Async.RunSynchronously

        match answer with
        | Result.Error message ->
            Assert.Contains("Not asked", message)
            Assert.DoesNotContain("Can approve or request changes", message)
        | Result.Ok text -> failwith $"answered for an agent nobody asked: {text}"

    [<Fact>]
    member _.``Delegating to a registered agent says nothing was started``() =
        if not (TestHelpers.requireTools ()) then () else

        // delegate_task answered "Task delegated ... Agent-to-agent execution initiated
        // via registry" and started nothing.
        let reviewer =
            { Tars.Tests.AgentWorkflowTests.createTestAgent () with
                Name = "Reviewer" }

        Tars.Tools.Standard.AgentTools.setRegistry
            { new IAgentRegistry with
                member _.GetAgent(_) = async { return Some reviewer }
                member _.FindAgents(_) = async { return [ reviewer ] }
                member _.GetAllAgents() = async { return [ reviewer ] } }

        let answer =
            Tars.Tools.Standard.AgentTools.delegateTask """{"agent": "reviewer", "task": "Review the parser"}"""
            |> Async.AwaitTask
            |> Async.RunSynchronously

        match answer with
        | Result.Error message ->
            Assert.Contains("Not delegated", message)
            Assert.Contains("nothing was started", message)
        | Result.Ok text -> failwith $"reported a delegation that started nothing: {text}"

    [<Fact>]
    member _.``A tool's failure reaches the caller as a failure, and its value rather than a type name``() =
        if not (TestHelpers.requireTools ()) then () else

        // The registry knew Task<string> and string. A tool returning
        // Task<Result<string, string>> - fsharp_compile, the refactor tools - or a
        // Task<bool> came back Ok, with the task's type name for its text.
        let registry = Tars.Tools.ToolRegistry()
        registry.RegisterAssembly(typeof<Tars.Tools.ToolRegistry>.Assembly)

        let run name (input: string) =
            match registry.Get name with
            | Some tool -> tool.Execute input |> Async.RunSynchronously
            | None -> failwith $"{name} is not registered"

        match run "fsharp_compile" """{"path": "no/such/project.fsproj"}""" with
        | Result.Error message -> Assert.Contains("Path not found", message)
        | Result.Ok text -> failwith $"compiling a project that does not exist came back Ok: {text}"

        Assert.Equal(Result.Ok "true", run "health_check" "")

        // And the agent tools, through the same path: a tool that only ever fails
        // must still be one the registry can call.
        match run "query_agent" """{"agent": "reviewer", "question": "Is this safe to merge?"}""" with
        | Result.Error message -> Assert.StartsWith("Not asked", message)
        | Result.Ok text -> failwith $"answered for an agent nobody asked: {text}"

        match run "delegate_task" """{"agent": "reviewer", "task": "Review the parser"}""" with
        | Result.Error message ->
            // Which refusal depends on whether a test has set an agent registry; each
            // of them is the tool's own answer, not a failure to call it.
            Assert.True(
                [ "Not delegated"; "Agent '"; "AgentRegistry not initialized" ]
                |> List.exists message.StartsWith,
                message
            )
        | Result.Ok text -> failwith $"reported a delegation that started nothing: {text}"

    [<Fact>]
    member _.``Tools with nothing behind them say so, rather than reporting a switch, a registry or a run``() =
        if not (TestHelpers.requireTools ()) then () else

        // Through the registry, as MCP and evolve reach them.
        let registry = Tars.Tools.ToolRegistry()
        registry.RegisterAssembly(typeof<Tars.Tools.ToolRegistry>.Assembly)

        let run name (input: string) =
            match registry.Get name with
            | Some tool -> tool.Execute input |> Async.RunSynchronously
            | None -> failwith $"{name} is not registered"

        let refused name input (expected: string) =
            match run name input with
            | Result.Error message -> Assert.StartsWith(expected, message)
            | Result.Ok text -> failwith $"{name} reported something that did not happen: {text}"

        // switch_model reported a switch nothing read; get_active_model reported a
        // default nobody configured as the model in use.
        refused "switch_model" "llama3:8b" "Not switched"
        refused "get_active_model" "" "Unknown here"

        // search_skills_registry answered from eight entries written in advance.
        refused "search_skills_registry" "payment" "Not searched"

        // run_metascript listed an EXECUTE step it did not run, then said
        // "Metascript execution complete".
        match run "run_metascript" "EXECUTE git_commit: ship it" with
        | Result.Ok text ->
            Assert.Contains("not run", text)
            Assert.DoesNotContain("execution complete", text)
        | Result.Error message -> failwith message

        // circuit_breaker called a service it had never heard of "healthy".
        match run "circuit_breaker" """{"service": "payments", "action": "check"}""" with
        | Result.Ok text -> Assert.DoesNotContain("healthy", text)
        | Result.Error message -> failwith message

    // ------------------------------------------------------------ a verdict, or a word

    [<Fact>]
    member _.``A rejection is not a verification because it contains the word``() =
        for answer in
            [ "NOT VERIFIED"
              "REJECTED - the claim is not VERIFIED"
              "This cannot be VERIFIED without a source."
              "**NOT VERIFIED**"
              // Reasoning that talks itself into a rejection is still a rejection:
              // stripping the block must not promote its contents to the verdict.
              "<thinking>\nVERIFIED, surely?\n</thinking>\nREJECTED"
              "" ] do
            Assert.False(EpistemicVerdict.saysVerified answer, $"'{answer}' was read as a verification")

    [<Fact>]
    member _.``A verdict of VERIFIED still reads as one, however it is dressed``() =
        for answer in
            [ "VERIFIED"
              "VERIFIED."
              "**VERIFIED**"
              "This statement is VERIFIED."
              "VERIFIED: the statement matches the cited source"
              // The explanation below the verdict is free to say what it could not
              // confirm; only the verdict line is read.
              "VERIFIED\nI could not check the second clause."
              // What a thinking model actually returns through OllamaClient: the
              // verdict is on the first line *of the answer*, not of the response.
              "<thinking>\nThe claim cannot be checked against a source directly, but the cited\nreference is authoritative.\n</thinking>\nVERIFIED"
              "<think>weighing it up</think>\nVERIFIED: matches the cited source" ] do
            Assert.True(EpistemicVerdict.saysVerified answer, $"'{answer}' was not read as a verification")

    // ----------------------------------------------------- a budget that stopped counting

    [<Fact>]
    member _.``Spending past the budget is still counted``() =
        // `TryConsume` does not record what it refuses, so the governor's total used
        // to freeze at the exact moment the budget was meant to start biting: every
        // call after the limit was both unrecorded and unblocked.
        let governor =
            BudgetGovernor(
                { Budget.Infinite with
                    MaxTokens = Some(Units.toTokens 100) }
            )

        let epistemic =
            EpistemicGovernor(CountingLlm(60) :> ILlmService, None, Some governor) :> IEpistemicGovernor

        epistemic.Verify("a statement").GetAwaiter().GetResult() |> ignore
        epistemic.Verify("another statement").GetAwaiter().GetResult() |> ignore

        Assert.Equal(120, int governor.Consumed.Tokens)
