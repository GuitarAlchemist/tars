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

        Assert.Contains("Not asked", answer)
        Assert.DoesNotContain("Can approve or request changes", answer)

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

        Assert.Contains("Not delegated", answer)
        Assert.Contains("nothing was started", answer)
        Assert.DoesNotContain("initiated", answer)

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
