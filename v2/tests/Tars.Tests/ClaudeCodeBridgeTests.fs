/// The metadata registry is process-wide and these tests now write to it, so this
/// module joins the collection the graph-editor tests use. xUnit runs collections in
/// parallel, and one test calling `clear()` while another counts is a race.
[<Xunit.Collection("ToolMetadata registry")>]
module Tars.Tests.ClaudeCodeBridgeTests

open System
open System.Text.Json
open Xunit
open Tars.Core
open Tars.Cortex
open Tars.Cortex.WoTTypes
open Tars.Cortex.ClaudeCodeBridge

// =========================================================================
// Helpers
// =========================================================================

/// Lightweight IToolRegistry that avoids loading Tars.Tools.dll (blocked by WDAC).
type private StubToolRegistry() =
    let mutable tools = Map.empty<string, Tars.Core.Tool>
    member _.Register(tool: Tars.Core.Tool) = tools <- tools |> Map.add tool.Name tool
    interface Tars.Core.IToolRegistry with
        member this.Register(tool) = this.Register(tool)
        member _.Get(name) = tools |> Map.tryFind name
        member _.GetAll() = tools |> Map.values |> Seq.toList

let private jsonOptions =
    JsonSerializerOptions(
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase)

/// Create a minimal tool registry with a dummy echo tool.
let private makeRegistry () =
    let reg = StubToolRegistry()
    reg.Register(
        { Name = "echo"
          Description = "Echo the input back"
          Version = "1.0.0"
          ParentVersion = None
          CreatedAt = DateTime.UtcNow
          Execute = fun input -> async { return Result.Ok $"ECHO: {input}" } })
    reg.Register(
        { Name = "search"
          Description = "Stub search tool"
          Version = "1.0.0"
          ParentVersion = None
          CreatedAt = DateTime.UtcNow
          Execute = fun input -> async { return Result.Ok $"SEARCH_RESULT: {input}" } })
    reg

let private compile goal maxSteps =
    let reg = makeRegistry ()
    let compiler = PatternCompiler.DefaultPatternCompiler() :> IPatternCompiler
    let selector = PatternSelector.HistoryAwareSelector() :> IPatternSelector
    let input = sprintf """{"goal": "%s", "max_steps": %d}""" goal maxSteps
    let result = compilePlan compiler selector (reg :> Tars.Core.IToolRegistry) input
    reg, result

// =========================================================================
// compilePlan tests
// =========================================================================

[<Fact>]
let ``compilePlan returns manifest with nodes and entry point`` () =
    let _, result = compile "Explain how photosynthesis works" 3

    match result with
    | Result.Ok json ->
        let doc = JsonDocument.Parse(json)
        let root = doc.RootElement
        Assert.False(String.IsNullOrEmpty(root.GetProperty("planId").GetString()))
        Assert.True(root.GetProperty("nodes").GetArrayLength() > 0)
        Assert.False(String.IsNullOrEmpty(root.GetProperty("entryNode").GetString()))
        Assert.False(String.IsNullOrEmpty(root.GetProperty("pattern").GetString()))
        Assert.Equal("Explain how photosynthesis works", root.GetProperty("goal").GetString())
    | Result.Error err ->
        Assert.Fail $"compilePlan failed: {err}"

[<Fact>]
let ``compilePlan fails on missing goal`` () =
    let reg = makeRegistry ()
    let compiler = PatternCompiler.DefaultPatternCompiler() :> IPatternCompiler
    let selector = PatternSelector.HistoryAwareSelector() :> IPatternSelector
    let result = compilePlan compiler selector (reg :> Tars.Core.IToolRegistry) """{"max_steps": 3}"""

    match result with
    | Result.Error msg -> Assert.Contains("goal", msg)
    | Result.Ok _ -> Assert.Fail "Should have failed with missing goal"

// =========================================================================
// executeStep tests
// =========================================================================

[<Fact>]
let ``executeStep records Reason node output`` () =
    let reg, planResult = compile "Test reasoning" 2

    match planResult with
    | Result.Error err -> Assert.Fail err
    | Result.Ok planJson ->

    let doc = JsonDocument.Parse(planJson)
    let planId = doc.RootElement.GetProperty("planId").GetString()
    let entryNode = doc.RootElement.GetProperty("entryNode").GetString()

    let stepInput =
        sprintf """{"plan_id": "%s", "node_id": "%s", "input": "Claude's reasoning output here"}"""
            planId entryNode

    let result =
        executeStep (reg :> Tars.Core.IToolRegistry) stepInput
        |> Async.RunSynchronously

    match result with
    | Result.Ok json ->
        let stepDoc = JsonDocument.Parse(json)
        let root = stepDoc.RootElement
        Assert.True(root.GetProperty("success").GetBoolean())
        Assert.Equal("Claude's reasoning output here", root.GetProperty("output").GetString())
    | Result.Error err ->
        Assert.Fail $"executeStep failed: {err}"

[<Fact>]
let ``executeStep fails on unknown plan`` () =
    let reg = makeRegistry ()
    let stepInput = """{"plan_id": "nonexistent", "node_id": "step1", "input": "test"}"""

    let result =
        executeStep (reg :> Tars.Core.IToolRegistry) stepInput
        |> Async.RunSynchronously

    match result with
    | Result.Error msg -> Assert.Contains("Plan not found", msg)
    | Result.Ok _ -> Assert.Fail "Should have failed"

// =========================================================================
// Full round-trip: compile -> execute each step -> complete
// =========================================================================

[<Fact>]
let ``full round-trip compile execute complete`` () =
    let reg, planResult = compile "Summarize a document" 3

    match planResult with
    | Result.Error err -> Assert.Fail err
    | Result.Ok planJson ->

    let doc = JsonDocument.Parse(planJson)
    let planId = doc.RootElement.GetProperty("planId").GetString()
    let nodes = doc.RootElement.GetProperty("nodes")

    // Execute every node
    let mutable executedCount = 0
    for i in 0 .. nodes.GetArrayLength() - 1 do
        let node = nodes[i]
        let nodeId = node.GetProperty("id").GetString()
        let stepInput =
            sprintf """{"plan_id": "%s", "node_id": "%s", "input": "Output for step %d"}"""
                planId nodeId (i + 1)

        let result =
            executeStep (reg :> Tars.Core.IToolRegistry) stepInput
            |> Async.RunSynchronously

        match result with
        | Result.Ok json ->
            let stepDoc = JsonDocument.Parse(json)
            Assert.True(stepDoc.RootElement.GetProperty("success").GetBoolean())
            executedCount <- executedCount + 1
        | Result.Error err ->
            Assert.Fail $"Step {nodeId} failed: {err}"

    Assert.True(executedCount > 0, "Should have executed at least one step")

    // Complete
    let completeInput =
        sprintf """{"plan_id": "%s", "final_output": "The document summary is..."}""" planId

    let selector = PatternSelector.HistoryAwareSelector() :> IPatternSelector
    let completeResult = completePlan selector completeInput

    match completeResult with
    | Result.Ok json ->
        let compDoc = JsonDocument.Parse(json)
        let root = compDoc.RootElement
        Assert.True(root.GetProperty("success").GetBoolean())
        Assert.Equal(executedCount, root.GetProperty("successfulSteps").GetInt32())
        Assert.Equal(0, root.GetProperty("failedSteps").GetInt32())
    | Result.Error err ->
        Assert.Fail $"completePlan failed: {err}"

// =========================================================================
// validateStep tests
// =========================================================================

[<Fact>]
let ``validateStep rejects non-Validate nodes`` () =
    let _, planResult = compile "Check things" 2

    match planResult with
    | Result.Error err -> Assert.Fail err
    | Result.Ok planJson ->

    let doc = JsonDocument.Parse(planJson)
    let planId = doc.RootElement.GetProperty("planId").GetString()
    let firstNodeId = doc.RootElement.GetProperty("nodes").[0].GetProperty("id").GetString()

    let input =
        sprintf """{"plan_id": "%s", "node_id": "%s", "content": "some content"}"""
            planId firstNodeId

    let result = validateStep input

    match result with
    | Result.Error msg -> Assert.Contains("not a Validate node", msg)
    | Result.Ok _ -> () // If the first node happens to be Validate, that's fine too

[<Fact>]
let ``completePlan removes plan from active list`` () =
    let _, planResult = compile "Temporary plan" 2

    match planResult with
    | Result.Error err -> Assert.Fail err
    | Result.Ok planJson ->

    let doc = JsonDocument.Parse(planJson)
    let planId = doc.RootElement.GetProperty("planId").GetString()

    // Complete it
    let selector = PatternSelector.HistoryAwareSelector() :> IPatternSelector
    let _ = completePlan selector (sprintf """{"plan_id": "%s", "final_output": "done"}""" planId)

    // Try again - should fail
    let result = completePlan selector (sprintf """{"plan_id": "%s", "final_output": "again"}""" planId)
    match result with
    | Result.Error msg -> Assert.Contains("Plan not found", msg)
    | Result.Ok _ -> Assert.Fail "Second completePlan should have failed"

[<Fact>]
let ``manifest nodes have correct structure`` () =
    let _, planResult = compile "Analyze code quality" 3

    match planResult with
    | Result.Error err -> Assert.Fail err
    | Result.Ok planJson ->

    let doc = JsonDocument.Parse(planJson)
    let nodes = doc.RootElement.GetProperty("nodes")

    for i in 0 .. nodes.GetArrayLength() - 1 do
        let node = nodes[i]
        // Every node must have id, kind, and next
        Assert.False(String.IsNullOrEmpty(node.GetProperty("id").GetString()))
        let kind = node.GetProperty("kind").GetString()
        Assert.True(
            [ "Reason"; "Tool"; "Validate"; "Memory"; "Control" ] |> List.contains kind,
            $"Unknown kind: {kind}")
        // "next" property should exist (array of next node IDs)
        let mutable nextElem = JsonElement()
        let nodeId = node.GetProperty("id").GetString()
        Assert.True(node.TryGetProperty("next", &nextElem), sprintf "Node %s missing 'next'" nodeId)

// =========================================================================
// memoryOp tests
// =========================================================================

let private makeLedger () =
    let ledger = Tars.Knowledge.KnowledgeLedger.createInMemory ()
    ledger.Initialize() |> Async.AwaitTask |> Async.RunSynchronously
    ledger

[<Fact>]
let ``memoryOp stats returns graph statistics`` () =
    let ledger = makeLedger ()
    let result = memoryOp ledger """{"operation": "stats"}""" |> Async.RunSynchronously
    match result with
    | Result.Ok json ->
        Assert.False(String.IsNullOrEmpty(json))
    | Result.Error err ->
        Assert.Fail err

[<Fact>]
let ``memoryOp assert then search round-trip`` () =
    let ledger = makeLedger ()

    // Assert a triple
    let assertResult =
        memoryOp ledger """{"operation": "assert", "subject": "photosynthesis", "predicate": "produces", "object": "oxygen"}"""
        |> Async.RunSynchronously

    match assertResult with
    | Result.Error err -> Assert.Fail err
    | Result.Ok json ->
        let doc = JsonDocument.Parse(json)
        Assert.True(doc.RootElement.GetProperty("success").GetBoolean())

    // Search for it
    let searchResult =
        memoryOp ledger """{"operation": "search", "query": "photosynthesis"}"""
        |> Async.RunSynchronously

    match searchResult with
    | Result.Error err -> Assert.Fail err
    | Result.Ok json ->
        let doc = JsonDocument.Parse(json)
        Assert.True(doc.RootElement.GetProperty("count").GetInt32() >= 1)

[<Fact>]
let ``memoryOp search returns empty for no matches`` () =
    let ledger = makeLedger ()
    let result =
        memoryOp ledger """{"operation": "search", "query": "xyznonexistent"}"""
        |> Async.RunSynchronously

    match result with
    | Result.Ok json ->
        let doc = JsonDocument.Parse(json)
        Assert.Equal(0, doc.RootElement.GetProperty("count").GetInt32())
    | Result.Error err ->
        Assert.Fail err

[<Fact>]
let ``memoryOp fails on unknown operation`` () =
    let ledger = makeLedger ()
    let result =
        memoryOp ledger """{"operation": "delete"}"""
        |> Async.RunSynchronously

    match result with
    | Result.Error msg -> Assert.Contains("Unknown operation", msg)
    | Result.Ok _ -> Assert.Fail "Should have failed"

// =========================================================================
// The gate, on this side of it
// =========================================================================

/// The first node the compiler turned into a tool call, and the tool it named.
let private firstToolNode (planJson: string) =
    let doc = JsonDocument.Parse planJson

    doc.RootElement.GetProperty("nodes").EnumerateArray()
    |> Seq.tryPick (fun n ->
        if n.GetProperty("kind").GetString() = "Tool" then
            let named = n.GetProperty("toolName")

            if named.ValueKind = JsonValueKind.String then
                Some(n.GetProperty("id").GetString(), named.GetString())
            else
                None
        else
            None)

let private planWithAToolNode () =
    let reg, planResult = compile "Search the codebase for something" 3

    match planResult with
    | Result.Error err -> failwith err
    | Result.Ok planJson ->
        let planId = JsonDocument.Parse(planJson).RootElement.GetProperty("planId").GetString()

        match firstToolNode planJson with
        | None -> failwith $"the compiler produced no tool node, so the gate is untested: {planJson}"
        | Some(nodeId, toolName) -> reg, planId, nodeId, toolName

let private step planId nodeId approve =
    let approveJson =
        approve |> List.map (sprintf "\"%s\"") |> String.concat ", "

    sprintf """{"plan_id": "%s", "node_id": "%s", "input": "x", "approve": [%s]}""" planId nodeId approveJson

[<Fact>]
let ``a tool that changes something is refused until the step names the node approved`` () =
    let reg, planId, nodeId, toolName = planWithAToolNode ()

    ToolMetadata.Testing.reset ()

    ToolMetadata.describe
        { Name = toolName
          InputSchema = ToolMetadata.objectSchema [] []
          Required = []
          Approval = ToolMetadata.Approval.mutates "writes a file to disk" }

    // Before `ToolGate`, this ran. Nothing asked, nothing said - the graph editor had a
    // gate only because that is where the gate happened to get written.
    let refused =
        executeStep (reg :> Tars.Core.IToolRegistry) (step planId nodeId [])
        |> Async.RunSynchronously

    match refused with
    | Result.Ok output -> Assert.Fail $"the tool ran unapproved: {output}"
    | Result.Error message ->
        // The refusal has to say what running it would do, not just that it was
        // refused: the person deciding is reading this sentence.
        Assert.Contains("writes a file to disk", message)
        Assert.Contains("mutates", message)
        Assert.Contains(nodeId, message)

    // And naming the node lets it through.
    let allowed =
        executeStep (reg :> Tars.Core.IToolRegistry) (step planId nodeId [ nodeId ])
        |> Async.RunSynchronously

    match allowed with
    | Result.Ok _ -> ()
    | Result.Error err -> Assert.Fail $"approving it did not let it run: {err}"

    ToolMetadata.Testing.reset ()

[<Fact>]
let ``a tool that only observes still runs unasked`` () =
    let reg, planId, nodeId, toolName = planWithAToolNode ()

    ToolMetadata.Testing.reset ()

    ToolMetadata.describe
        { Name = toolName
          InputSchema = ToolMetadata.objectSchema [] []
          Required = []
          Approval = ToolMetadata.Approval.observes "reads without changing anything" }

    let result =
        executeStep (reg :> Tars.Core.IToolRegistry) (step planId nodeId [])
        |> Async.RunSynchronously

    match result with
    | Result.Ok _ -> ()
    | Result.Error err -> Assert.Fail $"a read-only tool was refused: {err}"

    ToolMetadata.Testing.reset ()

[<Fact>]
let ``an undescribed tool still runs here, which is the opposite of the graph editor`` () =
    let reg, planId, nodeId, _ = planWithAToolNode ()

    // Nothing described at all. `GraphEditorBridge` refuses this case outright; this
    // bridge allows it, and the difference is deliberate: the editor offers a catalog
    // of six described tools, so an undescribed one in a graph means the graph was not
    // built from the catalog. This bridge drives the whole registry - some two hundred
    // tools - and failing closed would refuse every plan it runs today.
    //
    // The gate's reach here is therefore exactly the descriptor backfill's reach. This
    // test exists so that limit is stated somewhere that fails when it changes, rather
    // than living only in a comment.
    ToolMetadata.Testing.reset ()

    let result =
        executeStep (reg :> Tars.Core.IToolRegistry) (step planId nodeId [])
        |> Async.RunSynchronously

    match result with
    | Result.Ok _ -> ()
    | Result.Error err -> Assert.Fail $"an undescribed tool was refused, which this surface does not do: {err}"
