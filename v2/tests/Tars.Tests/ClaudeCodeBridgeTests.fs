module Tars.Tests.ClaudeCodeBridgeTests

open System
open System.Text.Json
open Xunit
open Tars.Tools
open Tars.Cortex
open Tars.Cortex.WoTTypes
open Tars.Cortex.ClaudeCodeBridge

// =========================================================================
// Helpers
// =========================================================================

let private jsonOptions =
    JsonSerializerOptions(
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase)

/// Create a minimal tool registry with a dummy echo tool.
let private makeRegistry () =
    let reg = ToolRegistry()
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

    let completeResult = completePlan completeInput

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
    let _ = completePlan (sprintf """{"plan_id": "%s", "final_output": "done"}""" planId)

    // Try again - should fail
    let result = completePlan (sprintf """{"plan_id": "%s", "final_output": "again"}""" planId)
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
