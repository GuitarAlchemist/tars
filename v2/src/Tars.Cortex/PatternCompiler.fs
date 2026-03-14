namespace Tars.Cortex

open System
open Tars.Core
open Tars.Cortex.WoTTypes
open System.Text.Json

/// <summary>
/// Compiles reasoning patterns (ReAct, CoT, GoT, ToT) into unified WoT plans.
/// This enables all patterns to execute through a common tracing engine.
/// </summary>
module PatternCompiler =

    open ReasoningPattern

    // =========================================================================
    // Helper Functions
    // =========================================================================

    /// Generate a unique node ID with a prefix
    let private genId prefix =
        sprintf "%s_%s" prefix (Guid.NewGuid().ToString("N").Substring(0, 8))

    let private emptyMeta = { Label = None; Tags = []; Extra = Map.empty }

    /// Create a Think node
    let think prompt hint =
        { Id = genId "think"
          Kind = Reason
          Payload = { Prompt = prompt; Hint = hint }
          Metadata = emptyMeta }

    /// Create an Act node
    let act tool args =
        { Id = genId "act"
          Kind = Tool
          Payload = { ToolPayload.Tool = tool; Args = args }
          Metadata = emptyMeta }

    /// Create an Observe node
    let observe input =
        { Id = genId "observe"
          Kind = Control
          Payload = ControlPayload.Observe(input, None)
          Metadata = emptyMeta }

    /// Create a Decide node
    let decide candidates criteria =
        { Id = genId "decide"
          Kind = Control
          Payload = ControlPayload.Decide(candidates, criteria)
          Metadata = emptyMeta }

    /// Create a Sync node for memory operations
    let sync op =
        { Id = genId "sync"
          Kind = Memory
          Payload = { Operation = op }
          Metadata = emptyMeta }

    /// Create a Validate node
    let validate invariants =
        { Id = genId "validate"
          Kind = Validate
          Payload = { Invariants = invariants }
          Metadata = emptyMeta }

    /// Create metadata for a pattern
    let private metadata kind goal estimatedSteps =
        { Kind = kind
          SourceGoal = goal
          CompiledAt = DateTime.UtcNow
          EstimatedTokens = None
          EstimatedSteps = Some estimatedSteps }

    /// Create an edge between nodes
    let private edge fromId toId label =
        { From = fromId
          To = toId
          Label = label
          Confidence = None }

    /// Get the ID of a WoT node
    let nodeId (node: WoTNode) = node.Id

    // =========================================================================
    // Chain of Thought Compiler
    // =========================================================================

    /// <summary>
    /// Compiles a Chain of Thought workflow to a WoT plan.
    /// CoT is a sequence of Think nodes connected linearly.
    /// </summary>
    let compileChainOfThought (stepCount: int) (goal: string) : WoTPlan =
        // Create a series of thinking steps
        let stepPrompts =
            [ "Analyze the problem and identify key aspects: " + goal
              "Break down the problem into smaller components."
              "Reason through each component step by step."
              "Synthesize insights and identify potential solutions."
              "Formulate a clear, comprehensive final answer." ]

        let prompts =
            if stepCount <= stepPrompts.Length then
                stepPrompts |> List.take stepCount
            else
                // Add generic steps if more requested
                let extra =
                    List.init (stepCount - stepPrompts.Length) (fun i ->
                        $"Continue reasoning - Step %d{stepPrompts.Length + i + 1} of %d{stepCount}")

                stepPrompts @ extra

        let nodes = prompts |> List.map (fun p -> think p (Some Smart))
        let nodeIds = nodes |> List.map nodeId

        // Create linear edges
        let edges =
            nodeIds |> List.pairwise |> List.map (fun (a, b) -> edge a b (Some "next"))

        { Id = Guid.NewGuid()
          Nodes = nodes
          Edges = edges
          EntryNode = nodeIds.Head
          Metadata = metadata ChainOfThought goal stepCount
          Policy = [] }

    // =========================================================================
    // ReAct Compiler
    // =========================================================================

    /// <summary>
    /// Compiles a ReAct workflow to a WoT plan.
    /// ReAct alternates: Think → Tool → Observe (repeat until goal achieved)
    /// </summary>
    let compileReAct (tools: string list) (maxSteps: int) (goal: string) : WoTPlan =
        let toolList = tools |> String.concat ", "

        let mutable allNodes: WoTNode list = []
        let mutable allEdges: WoTEdge list = []
        let mutable prevNodeId: string option = None

        for step in 1..maxSteps do
            // Think: Reason about what to do
            let thinkPrompt =
                if step = 1 then
                    sprintf
                        "You are solving: %s\n\nAvailable tools: %s, Finish\n\nAnalyze the problem and decide which tool to use first."
                        goal
                        toolList
                else
                    sprintf
                        "Step %d: Based on the observations, what should we do next? Tools: %s, Finish"
                        step
                        toolList

            let thinkNode = think thinkPrompt (Some Smart)
            allNodes <- allNodes @ [ thinkNode ]

            match prevNodeId with
            | Some prev -> allEdges <- allEdges @ [ edge prev thinkNode.Id (Some "next") ]
            | None -> ()

            // Decide: Pick a tool
            let decideNode =
                decide (tools @ [ "Finish" ]) [ "Select the most appropriate tool based on reasoning" ]

            allNodes <- allNodes @ [ decideNode ]
            allEdges <- allEdges @ [ edge thinkNode.Id decideNode.Id (Some "decide") ]

            // Act: Execute chosen tool (represented as Control node with all tools as options)
            for tool in tools do
                let toolNode = act tool Map.empty
                allNodes <- allNodes @ [ toolNode ]
                allEdges <- allEdges @ [ edge decideNode.Id toolNode.Id (Some tool) ]

            // Observe: Process tool output
            let observeNode = observe ""
            allNodes <- allNodes @ [ observeNode ]

            for tool in tools do
                let toolNode =
                    allNodes
                    |> List.find (fun n ->
                        n.Kind = Tool
                        && match n.Payload with
                           | :? ToolPayload as p -> p.Tool = tool
                           | _ -> false)

                allEdges <- allEdges @ [ edge toolNode.Id observeNode.Id (Some "observe") ]

            prevNodeId <- Some observeNode.Id

        // Final synthesis
        let finishNode =
            think "Synthesize all observations into a final answer." (Some Smart)

        allNodes <- allNodes @ [ finishNode ]

        match prevNodeId with
        | Some prev -> allEdges <- allEdges @ [ edge prev finishNode.Id (Some "finish") ]
        | None -> ()

        { Id = Guid.NewGuid()
          Nodes = allNodes
          Edges = allEdges
          EntryNode = (allNodes |> List.head).Id
          Metadata = metadata ReAct goal (maxSteps * 4 + 1)
          Policy = [ "require_tool_confirmation" ] }

    // =========================================================================
    // Graph of Thoughts Compiler
    // =========================================================================

    /// <summary>
    /// Compiles a Graph of Thoughts workflow to a WoT plan.
    /// GoT branches into multiple thought paths, evaluates them, and synthesizes.
    /// </summary>
    let compileGraphOfThoughts (branchingFactor: int) (maxDepth: int) (goal: string) : WoTPlan =
        let mutable allNodes: WoTNode list = []
        let mutable allEdges: WoTEdge list = []

        // Phase 1: Decomposition
        let decomposePrompt =
            sprintf "Decompose this problem into %d distinct approaches: %s" branchingFactor goal

        let decomposeNode = think decomposePrompt (Some Smart)
        allNodes <- [ decomposeNode ]

        // Phase 2: Exploration (branching)
        let mutable branchNodes: WoTNode list = []

        for i in 1..branchingFactor do
            let prompt =
                sprintf "Explore approach %d in depth. Consider implications and edge cases." i

            let node = think prompt (Some Reasoning)
            allEdges <- allEdges @ [ edge decomposeNode.Id node.Id (Some(sprintf "branch_%d" i)) ]
            branchNodes <- branchNodes @ [ node ]

        allNodes <- allNodes @ branchNodes

        // Phase 3: Evaluation
        let mutable evalNodes: WoTNode list = []

        for (i, bn) in branchNodes |> List.mapi (fun i n -> (i + 1, n)) do
            let prompt =
                sprintf "Evaluate the quality, feasibility, and completeness of approach %d" i

            let evNode = think prompt (Some Smart)
            allEdges <- allEdges @ [ edge bn.Id evNode.Id (Some "evaluate") ]
            evalNodes <- evalNodes @ [ evNode ]

        allNodes <- allNodes @ evalNodes

        // Phase 4: Synthesis
        let synthesisNode =
            think "Synthesize the best insights from all explored approaches into a coherent answer." (Some Smart)

        for evNode in evalNodes do
            allEdges <- allEdges @ [ edge evNode.Id synthesisNode.Id (Some "synthesize") ]

        allNodes <- allNodes @ [ synthesisNode ]

        { Id = Guid.NewGuid()
          Nodes = allNodes
          Edges = allEdges
          EntryNode = decomposeNode.Id
          Metadata = metadata GraphOfThoughts goal (1 + branchingFactor * 2 + 1)
          Policy = [] }

    // =========================================================================
    // Tree of Thoughts Compiler
    // =========================================================================

    /// <summary>
    /// Compiles a Tree of Thoughts workflow to a WoT plan.
    /// ToT uses beam search with evaluation at each level.
    /// </summary>
    let compileTreeOfThoughts (beamWidth: int) (searchDepth: int) (goal: string) : WoTPlan =
        let mutable allNodes: WoTNode list = []
        let mutable allEdges: WoTEdge list = []
        let mutable currentLevel: WoTNode list = []

        // Level 0: Initial thought generation
        for i in 1..beamWidth do
            let prompt = sprintf "Generate initial approach %d for: %s" i goal
            let node = think prompt (Some Smart)
            allNodes <- allNodes @ [ node ]
            currentLevel <- currentLevel @ [ node ]

        // Iterate through search depth
        for depth in 1..searchDepth do
            // Evaluate current level
            let candidates =
                currentLevel |> List.map (fun n -> sprintf "Approach at node %s" n.Id)

            let evalNode =
                decide candidates [ "Select top candidates based on promise and feasibility" ]

            for prev in currentLevel do
                allEdges <- allEdges @ [ edge prev.Id evalNode.Id (Some "evaluate") ]

            allNodes <- allNodes @ [ evalNode ]

            // Expand best candidates
            let mutable nextLevel: WoTNode list = []

            for i in 1..beamWidth do
                let prompt =
                    sprintf "Expand and refine selected approach at depth %d, variant %d" depth i

                let node = think prompt (Some Smart)
                allEdges <- allEdges @ [ edge evalNode.Id node.Id (Some(sprintf "expand_%d" i)) ]
                nextLevel <- nextLevel @ [ node ]

            allNodes <- allNodes @ nextLevel
            currentLevel <- nextLevel

        // Final synthesis
        let synthesisNode =
            think "Select the best explored path and formulate the final answer." (Some Smart)

        for node in currentLevel do
            allEdges <- allEdges @ [ edge node.Id synthesisNode.Id (Some "finalize") ]

        allNodes <- allNodes @ [ synthesisNode ]

        let entryNode = allNodes |> List.head

        { Id = Guid.NewGuid()
          Nodes = allNodes
          Edges = allEdges
          EntryNode = entryNode.Id
          Metadata = metadata TreeOfThoughts goal (beamWidth * (searchDepth + 1) + searchDepth + 1)
          Policy = [] }

    // =========================================================================
    // Workflow of Thought Compiler (Identity)
    // =========================================================================
    // General Pattern Compiler (ReasoningPattern -> WoT Plan)
    // =========================================================================

    /// <summary>
    /// Compiles a general declarative ReasoningPattern into an executable WoTPlan.
    /// This is the key function for Neuro-Symbolic evolution.
    /// </summary>
    let compilePattern (pattern: ReasoningPattern.ReasoningPattern) (goal: string) : WoTPlan =

        // 1. Map Steps to Nodes
        let nodes =
            pattern.Steps
            |> List.map (fun step ->
                let instruction = step.InstructionTemplate |> Option.defaultValue ""

                let prompt =
                    if instruction.Contains("{goal}") then
                        instruction.Replace("{goal}", goal)
                    else if String.IsNullOrWhiteSpace instruction then
                        goal
                    else
                        instruction + "\nGoal: " + goal

                let baseNode =
                    match step.Role.ToLowerInvariant() with
                    | "reason" -> think prompt (Some Smart)
                    | "tool" ->
                        let toolName =
                            step.Parameters |> Map.tryFind "tool" |> Option.defaultValue "unknown"

                        let argsStr = step.Parameters |> Map.tryFind "args" |> Option.defaultValue "{}"

                        let args =
                            try
                                JsonSerializer.Deserialize<Map<string, obj>>(argsStr)
                            with _ ->
                                Map.empty

                        act toolName args
                    | "validate" -> validate []
                    | "critique" -> think ("Critique: " + prompt) (Some Smart)
                    | _ -> think prompt (Some Smart)

                // Preserve ID from pattern
                { baseNode with Id = step.Id })

        // 2. Map Dependencies to Edges
        let edges =
            pattern.Steps
            |> List.collect (fun step -> step.Dependencies |> List.map (fun depId -> edge depId step.Id (Some "next")))

        // 3. Determine Entry Node
        let finalNodes, finalEdges, entryId =
            if nodes.IsEmpty then
                failwith "Pattern has no steps"
            else
                let targets = edges |> List.map (fun e -> e.To) |> Set.ofList
                let roots = nodes |> List.filter (fun n -> not (Set.contains n.Id targets))

                match roots with
                | [] -> failwith "Pattern has no entry point (cycle detected)"
                | [ single ] -> nodes, edges, single.Id
                | multiple ->
                    // Create a synthetic Start node
                    let startNode = think ("Start Plan: " + goal) (Some Smart)
                    let newEdges = multiple |> List.map (fun r -> edge startNode.Id r.Id (Some "start"))
                    (startNode :: nodes), (edges @ newEdges), startNode.Id

        { Id = Guid.NewGuid()
          Nodes = finalNodes
          Edges = finalEdges
          EntryNode = entryId
          Metadata = metadata WorkflowOfThought goal nodes.Length
          Policy = [] }

    // =========================================================================


    /// <summary>
    /// Compiles a WoT DSL definition to a WoT plan.
    /// This is mostly an identity transform with validation.
    /// </summary>
    let compileWorkflowOfThought (nodes: WoTNode list) (edges: WoTEdge list) (goal: string) : WoTPlan =
        let entryNode =
            match nodes with
            | [] -> failwith "WoT plan must have at least one node"
            | h :: _ -> nodeId h

        { Id = Guid.NewGuid()
          Nodes = nodes
          Edges = edges
          EntryNode = entryNode
          Metadata = metadata WorkflowOfThought goal nodes.Length
          Policy = [] }

    // =========================================================================
    // Pattern Compiler Implementation
    // =========================================================================

    /// <summary>
    /// Default pattern compiler implementation.
    /// </summary>
    type DefaultPatternCompiler() =
        interface IPatternCompiler with
            member _.CompileChainOfThought(steps, goal) = compileChainOfThought steps goal

            member _.CompileReAct(tools, maxSteps, goal) = compileReAct tools maxSteps goal

            member _.CompileGraphOfThoughts(branchingFactor, maxDepth, goal) =
                compileGraphOfThoughts branchingFactor maxDepth goal

            member _.CompileTreeOfThoughts(beamWidth, searchDepth, goal) =
                compileTreeOfThoughts beamWidth searchDepth goal

            member _.CompilePattern(pattern, goal) = compilePattern pattern goal

    // =========================================================================
    // Visualization Helpers
    // =========================================================================

    /// Generate a Mermaid diagram for a WoT plan
    let toMermaid (plan: WoTPlan) : string =
        let sb = System.Text.StringBuilder()
        sb.AppendLine("graph TD") |> ignore

        // Add nodes
        for node in plan.Nodes do
            let id = nodeId node

            let label =
                match node.Kind with
                | Reason ->
                    match node.Payload with
                    | :? ReasonPayload as p ->
                        let preview =
                            if p.Prompt.Length > 30 then
                                p.Prompt.Substring(0, 30) + "..."
                            else
                                p.Prompt

                        $"Reason: %s{preview}"
                    | _ -> "Reason"
                | Tool ->
                    match node.Payload with
                    | :? ToolPayload as p -> $"Tool: %s{p.Tool}"
                    | _ -> "Tool"
                | Validate ->
                    match node.Payload with
                    | :? ValidatePayload as p -> $"Validate: %d{p.Invariants.Length} checks"
                    | _ -> "Validate"
                | Memory -> "Memory"
                | Control ->
                    match node.Payload with
                    | :? ControlPayload as p ->
                        match p with
                        | ControlPayload.Decide(candidates, _) -> $"Decide: %d{candidates.Length} options"
                        | ControlPayload.Observe _ -> "Observe"
                        | ControlPayload.Parallel children -> $"Parallel: %d{children.Length} children"
                        | ControlPayload.Branch(cond, _, _) -> $"Branch: %s{cond}"
                        | ControlPayload.Loop(_, until, max) -> $"Loop until: %s{until} (max %d{max})"
                    | _ -> "Control"

            sb.AppendLine $"    %s{id}[\"%s{label}\"]" |> ignore

        // Add edges
        for e in plan.Edges do
            let label = e.Label |> Option.defaultValue ""
            sb.AppendLine $"    %s{e.From} -->|%s{label}| %s{e.To}" |> ignore

        // Highlight entry node
        sb.AppendLine $"    style %s{plan.EntryNode} fill:#0f0" |> ignore

        sb.ToString()

    /// Generate statistics for a WoT plan
    let planStats (plan: WoTPlan) =
        let countByType =
            plan.Nodes
            |> List.groupBy (fun n -> n.Kind.ToString())
            |> List.map (fun (k, v) -> k, v.Length)
            |> Map.ofList

        {| TotalNodes = plan.Nodes.Length
           TotalEdges = plan.Edges.Length
           NodesByType = countByType
           Pattern = plan.Metadata.Kind
           EstimatedSteps = plan.Metadata.EstimatedSteps |}
