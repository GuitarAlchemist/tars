module Tars.Tests.WoTGoldenTraceTests

open System
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Llm
open Tars.Cortex.WoTTypes
open Tars.Cortex.PatternCompiler
open Tars.Cortex.WoTExecutor

// =============================================================================
// Golden Trace Format Types
// =============================================================================

/// Canonical representation of a WoT trace for diffing and golden testing.
type CanonicalTraceStep =
    { NodeId: string
      NodeKind: string
      Status: string
      OutputPrefix: string option
      DurationBucket: string }

type CanonicalGolden =
    { TraceId: Guid
      PatternKind: string
      Goal: string
      TotalSteps: int
      SuccessfulSteps: int
      FailedSteps: int
      Steps: CanonicalTraceStep list
      FinalStatus: string }

/// Convert a WoTTrace to canonical form for golden comparison
let toCanonical (trace: WoTTrace) : CanonicalGolden =
    let canonicalSteps =
        trace.Steps 
        |> List.map (fun step ->
            let durationMs = 
                match step.Status with
                | Completed(_, d) -> d
                | Failed(_, d) -> d
                | _ -> 0L
            
            let bucket = 
                if durationMs < 100L then "fast"
                elif durationMs < 1000L then "medium"
                else "slow"
            
            let statusStr = 
                match step.Status with
                | Completed _ -> "Completed"
                | Failed(err, _) -> sprintf "Failed: %s" (if err.Length > 50 then err.Substring(0, 50) else err)
                | Skipped r -> sprintf "Skipped: %s" r
                | Pending -> "Pending"
                | Running -> "Running"
            
            let outputPrefix = 
                step.Output 
                |> Option.map (fun o -> if o.Length > 100 then o.Substring(0, 100) else o)
            
            { NodeId = step.NodeId
              NodeKind = step.NodeType
              Status = statusStr
              OutputPrefix = outputPrefix
              DurationBucket = bucket })
    
    let successCount = 
        trace.Steps 
        |> List.filter (fun s -> match s.Status with Completed _ -> true | _ -> false) 
        |> List.length
    
    let failedCount = 
        trace.Steps 
        |> List.filter (fun s -> match s.Status with Failed _ -> true | _ -> false) 
        |> List.length
    
    { TraceId = trace.RunId
      PatternKind = trace.Plan.Metadata.Kind.ToString()
      Goal = trace.Plan.Metadata.SourceGoal
      TotalSteps = trace.Steps.Length
      SuccessfulSteps = successCount
      FailedSteps = failedCount
      Steps = canonicalSteps
      FinalStatus = trace.FinalStatus }

/// Serialize a canonical golden to JSON for storage
let serializeGolden (golden: CanonicalGolden) : string =
    System.Text.Json.JsonSerializer.Serialize(golden, 
        System.Text.Json.JsonSerializerOptions(WriteIndented = true))

/// Compare two canonical goldens, ignoring timing and IDs
let diffGoldens (expected: CanonicalGolden) (actual: CanonicalGolden) : string list =
    let mutable diffs = []
    
    if expected.PatternKind <> actual.PatternKind then
        diffs <- sprintf "PatternKind: expected '%s', got '%s'" expected.PatternKind actual.PatternKind :: diffs
    
    if expected.TotalSteps <> actual.TotalSteps then
        diffs <- sprintf "TotalSteps: expected %d, got %d" expected.TotalSteps actual.TotalSteps :: diffs
    
    if expected.SuccessfulSteps <> actual.SuccessfulSteps then
        diffs <- sprintf "SuccessfulSteps: expected %d, got %d" expected.SuccessfulSteps actual.SuccessfulSteps :: diffs
    
    if expected.FinalStatus <> actual.FinalStatus then
        diffs <- sprintf "FinalStatus: expected '%s', got '%s'" expected.FinalStatus actual.FinalStatus :: diffs
    
    let expectedKinds = expected.Steps |> List.map (fun s -> s.NodeKind)
    let actualKinds = actual.Steps |> List.map (fun s -> s.NodeKind)
    
    if expectedKinds <> actualKinds then
        diffs <- sprintf "StepKinds differ: expected %A, got %A" expectedKinds actualKinds :: diffs
    
    diffs

// =============================================================================
// Pattern Compilation Tests (No LLM needed)
// =============================================================================

[<Fact>]
let ``compileChainOfThought produces correct node structure`` () =
    let goal = "Explain how photosynthesis works"
    let steps = 3
    
    let plan = compileChainOfThought steps goal
    
    Assert.Equal(steps, plan.Nodes.Length)
    Assert.Equal(steps - 1, plan.Edges.Length)
    Assert.Equal(PatternKind.ChainOfThought, plan.Metadata.Kind)
    Assert.Equal(goal, plan.Metadata.SourceGoal)
    
    for node in plan.Nodes do
        Assert.Equal(WoTNodeKind.Reason, node.Kind)

[<Fact>]
let ``CoT plan generates valid Mermaid diagram`` () =
    let plan = compileChainOfThought 3 "Test goal"
    
    let mermaid = toMermaid plan
    
    Assert.Contains("graph TD", mermaid)
    Assert.Contains("Reason:", mermaid)
    Assert.Contains("-->", mermaid)

[<Fact>]
let ``diffGoldens detects step count changes`` () =
    let golden1 = 
        { TraceId = Guid.NewGuid()
          PatternKind = "ChainOfThought"
          Goal = "Test"
          TotalSteps = 3
          SuccessfulSteps = 3
          FailedSteps = 0
          Steps = []
          FinalStatus = "Success" }
    
    let golden2 = { golden1 with TotalSteps = 5 }
    
    let diffs = diffGoldens golden1 golden2
    
    Assert.NotEmpty(diffs)
    Assert.Contains("TotalSteps", diffs.[0])

[<Fact>]
let ``planStats returns accurate metrics`` () =
    let plan = compileChainOfThought 5 "Complex reasoning task"
    
    let stats = planStats plan
    
    Assert.Equal(5, stats.TotalNodes)
    Assert.Equal(4, stats.TotalEdges)
    Assert.Equal(PatternKind.ChainOfThought, stats.Pattern)
    Assert.True(stats.NodesByType.ContainsKey("Reason"))
    Assert.Equal(5, stats.NodesByType.["Reason"])

[<Fact>]
let ``WoTPlan has valid entry node`` () =
    let plan = compileChainOfThought 3 "Test"
    
    Assert.NotEmpty(plan.EntryNode)
    Assert.True(plan.Nodes |> List.exists (fun n -> n.Id = plan.EntryNode))

[<Fact>]
let ``toMermaid includes all nodes`` () =
    let plan = compileChainOfThought 4 "Test"
    
    let mermaid = toMermaid plan
    
    for node in plan.Nodes do
        Assert.Contains(node.Id, mermaid)

// =============================================================================
// ReAct Compiler Tests
// =============================================================================

[<Fact>]
let ``compileReAct produces multi-step workflow`` () =
    let tools = ["search_web"; "read_file"]
    let maxSteps = 2
    let goal = "Find information about AI"
    
    let plan = compileReAct tools maxSteps goal
    
    Assert.True(plan.Nodes.Length > 0)
    Assert.Equal(PatternKind.ReAct, plan.Metadata.Kind)
    Assert.Contains("require_tool_confirmation", plan.Policy)
    
    // Should have Think, Tool, and Control nodes
    let hasReason = plan.Nodes |> List.exists (fun n -> n.Kind = WoTNodeKind.Reason)
    let hasTool = plan.Nodes |> List.exists (fun n -> n.Kind = WoTNodeKind.Tool)
    let hasControl = plan.Nodes |> List.exists (fun n -> n.Kind = WoTNodeKind.Control)
    
    Assert.True(hasReason)
    Assert.True(hasTool)
    Assert.True(hasControl)

[<Fact>]
let ``compileReAct includes all tools`` () =
    let tools = ["tool_a"; "tool_b"; "tool_c"]
    let plan = compileReAct tools 1 "Test"
    
    let toolPayloads = 
        plan.Nodes 
        |> List.filter (fun n -> n.Kind = WoTNodeKind.Tool)
        |> List.choose (fun n -> 
            match n.Payload with
            | :? ToolPayload as p -> Some p.Tool
            | _ -> None)
    
    for tool in tools do
        Assert.Contains(tool, toolPayloads)

// =============================================================================
// Graph of Thoughts Compiler Tests
// =============================================================================

[<Fact>]
let ``compileGraphOfThoughts creates branching structure`` () =
    let branchingFactor = 3
    let goal = "Solve complex problem"
    
    let plan = compileGraphOfThoughts branchingFactor 2 goal
    
    Assert.Equal(PatternKind.GraphOfThoughts, plan.Metadata.Kind)
    
    // Should have: 1 decompose + N branches + N evals + 1 synthesis
    let expectedNodes = 1 + branchingFactor * 2 + 1
    Assert.Equal(expectedNodes, plan.Nodes.Length)
    
    // All nodes should be Reason type for GoT
    for node in plan.Nodes do
        Assert.Equal(WoTNodeKind.Reason, node.Kind)

[<Fact>]
let ``compileGraphOfThoughts has synthesis node`` () =
    let plan = compileGraphOfThoughts 2 1 "Test"
    
    let synthNodes = 
        plan.Nodes 
        |> List.filter (fun n -> 
            match n.Payload with
            | :? ReasonPayload as p -> p.Prompt.Contains("Synthesize")
            | _ -> false)
    
    Assert.True(synthNodes.Length > 0)

// =============================================================================
// Tree of Thoughts Compiler Tests
// =============================================================================

[<Fact>]
let ``compileTreeOfThoughts creates depth-first structure`` () =
    let beamWidth = 2
    let searchDepth = 2
    let goal = "Find optimal solution"
    
    let plan = compileTreeOfThoughts beamWidth searchDepth goal
    
    Assert.Equal(PatternKind.TreeOfThoughts, plan.Metadata.Kind)
    Assert.True(plan.Nodes.Length > beamWidth)
    Assert.True(plan.Edges.Length >= beamWidth)

[<Fact>]
let ``compileTreeOfThoughts has evaluate edges`` () =
    let plan = compileTreeOfThoughts 2 1 "Test"
    
    let evalEdges = 
        plan.Edges 
        |> List.filter (fun e -> e.Label = Some "evaluate")
    
    Assert.True(evalEdges.Length > 0)

// =============================================================================
// GoldenTraceStore Tests
// =============================================================================

open Tars.Cortex.GoldenTraceStore

[<Fact>]
let ``GoldenTraceStore serialize and deserialize roundtrip`` () =
    let golden = 
        { TraceId = Guid.NewGuid()
          PatternKind = "ChainOfThought"
          Goal = "Test goal"
          TotalSteps = 3
          SuccessfulSteps = 3
          FailedSteps = 0
          Steps = []
          FinalStatus = "Success"
          CreatedAt = DateTime.UtcNow
          Version = "1.0" }
    
    let json = serialize golden
    let result = deserialize json
    
    Assert.True(Result.isOk result)
    let deserialized = Result.defaultValue golden result
    Assert.Equal(golden.PatternKind, deserialized.PatternKind)
    Assert.Equal(golden.Goal, deserialized.Goal)
    Assert.Equal(golden.TotalSteps, deserialized.TotalSteps)

[<Fact>]
let ``GoldenTraceStore diff detects differences`` () =
    let golden1 = 
        { TraceId = Guid.NewGuid()
          PatternKind = "ChainOfThought"
          Goal = "Test"
          TotalSteps = 3
          SuccessfulSteps = 3
          FailedSteps = 0
          Steps = []
          FinalStatus = "Success"
          CreatedAt = DateTime.UtcNow
          Version = "1.0" }
    
    let golden2 = { golden1 with TotalSteps = 5; SuccessfulSteps = 4 }
    
    let diffs = diff golden1 golden2
    
    Assert.True(diffs.Length >= 2)
    Assert.True(diffs |> List.exists (fun d -> d.Field = "TotalSteps"))
    Assert.True(diffs |> List.exists (fun d -> d.Field = "SuccessfulSteps"))

[<Fact>]
let ``GoldenTraceStore diff returns empty for identical traces`` () =
    let golden = 
        { TraceId = Guid.NewGuid()
          PatternKind = "ChainOfThought"
          Goal = "Test"
          TotalSteps = 3
          SuccessfulSteps = 3
          FailedSteps = 0
          Steps = []
          FinalStatus = "Success"
          CreatedAt = DateTime.UtcNow
          Version = "1.0" }
    
    let diffs = diff golden golden
    
    Assert.Empty(diffs)
