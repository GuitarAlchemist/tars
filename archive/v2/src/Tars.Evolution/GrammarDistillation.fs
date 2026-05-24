namespace Tars.Evolution

/// Grammar Distillation: extract reusable typed grammar rules from execution traces.
///
/// Goes beyond EBNF by producing three orthogonal rule facets:
///   1. Structural  — DAG shape (node kinds + edge topology)
///   2. Typed       — Input/output type signatures per node (categorial grammar)
///   3. Behavioral  — Post-conditions and thresholds that held during execution
///
/// Distilled rules feed into WeightedGrammar for Bayesian weight evolution.
/// Each facet can be factored and evolved independently (horizontal factorization).
///
/// Architecture:
///   Trace → Distiller → TypedProduction → WeightedGrammar → PatternSelector → Execution → Trace
///   (closed loop)

open System
open System.Security.Cryptography
open System.Text
open Tars.Core.WorkflowOfThought

// ─── Types ───────────────────────────────────────────────────────────────────

/// A typed slot in a production — carries a type signature, not just a syntactic category.
type TypedSlot = {
    /// Name of the slot (e.g., "analyse", "validate")
    Name: string
    /// Node kind (reason, work)
    Kind: string
    /// Inferred input type (e.g., "Context", "Analysis")
    InputType: string
    /// Inferred output type (e.g., "Analysis", "Validated<Analysis>")
    OutputType: string
}

/// Edge in the distilled DAG.
type DistilledEdge = {
    From: string
    To: string
}

/// Facet of a distilled grammar rule.
type DistillationFacet =
    /// DAG shape: node kinds + edges, no content
    | Structural of slots: TypedSlot list * edges: DistilledEdge list
    /// Type signatures: input/output types per node, composability check
    | Typed of slots: TypedSlot list * composable: bool
    /// Behavioral: post-conditions that held, tool constraints
    | Behavioral of conditions: string list * tools: string list

/// A typed production rule distilled from one or more execution traces.
type TypedProduction = {
    /// Unique ID (hash of structural fingerprint)
    Id: string
    /// Human-readable name
    Name: string
    /// Which facets were distilled
    Facets: DistillationFacet list
    /// How many traces contributed to this production
    TraceCount: int
    /// Average success rate across contributing traces
    SuccessRate: float
    /// Compression ratio: trace tokens / production tokens
    CompressionRatio: float
    /// Suggested promotion level based on recurrence
    SuggestedLevel: PromotionLevel
    /// When this production was last updated
    LastUpdated: DateTime
}

/// Result of a distillation run.
type DistillationResult = {
    Productions: TypedProduction list
    TracesProcessed: int
    NewRulesFound: int
    ExistingRulesReinforced: int
}

// ─── Core distillation engine ────────────────────────────────────────────────

module GrammarDistillation =

    /// Generate a stable fingerprint for a structural pattern.
    let fingerprint (slots: TypedSlot list) (edges: DistilledEdge list) : string =
        let slotSig = slots |> List.map (fun s -> $"{s.Kind}:{s.InputType}->{s.OutputType}") |> String.concat "|"
        let edgeSig = edges |> List.map (fun e -> $"{e.From}->{e.To}") |> String.concat "|"
        let combined = $"{slotSig}#{edgeSig}"
        let hash = SHA256.HashData(Encoding.UTF8.GetBytes(combined))
        Convert.ToHexString(hash).Substring(0, 12).ToLower()

    // =========================================================================
    // Facet extractors
    // =========================================================================

    /// Infer a type name from a step's role in the workflow.
    let private inferType (stepId: string) (kind: string) (toolName: string option) (outputs: string list) : string =
        match kind with
        | "reason" ->
            // Capitalize the step name as a type
            let name = stepId.Replace("_", " ").Split(' ')
                       |> Array.map (fun w -> if w.Length > 0 then string (Char.ToUpper w.[0]) + w.[1..] else w)
                       |> String.concat ""
            name
        | "work" ->
            match toolName with
            | Some tool -> $"ToolResult<{tool}>"
            | None -> "WorkOutput"
        | _ -> "Unknown"

    /// Extract structural facet: DAG shape from trace events.
    let extractStructural (events: CanonicalTraceEvent list) : DistillationFacet =
        let slots =
            events
            |> List.map (fun evt ->
                let outputType = inferType evt.StepId evt.Kind evt.ToolName evt.Outputs
                { Name = evt.StepId
                  Kind = evt.Kind
                  InputType = "Context"  // first node gets Context
                  OutputType = outputType })

        // Infer sequential edges from event order
        let edges =
            events
            |> List.pairwise
            |> List.map (fun (a, b) -> { From = a.StepId; To = b.StepId })

        // Patch input types: each node's input is the previous node's output
        let patchedSlots =
            slots
            |> List.mapi (fun i slot ->
                if i = 0 then slot
                else { slot with InputType = slots.[i - 1].OutputType })

        Structural (patchedSlots, edges)

    /// Extract typed facet: type signatures and composability.
    let extractTyped (events: CanonicalTraceEvent list) : DistillationFacet =
        let slots =
            events
            |> List.mapi (fun i evt ->
                let outputType = inferType evt.StepId evt.Kind evt.ToolName evt.Outputs
                let inputType =
                    if i = 0 then "Context"
                    else inferType events.[i-1].StepId events.[i-1].Kind events.[i-1].ToolName events.[i-1].Outputs
                { Name = evt.StepId
                  Kind = evt.Kind
                  InputType = inputType
                  OutputType = outputType })

        // Check composability: each node's input type must match previous output type
        let composable =
            slots
            |> List.pairwise
            |> List.forall (fun (a, b) -> a.OutputType = b.InputType)

        Typed (slots, composable)

    /// Extract behavioral facet: conditions and tool usage.
    let extractBehavioral (events: CanonicalTraceEvent list) : DistillationFacet =
        let tools =
            events
            |> List.choose (fun evt -> evt.ToolName)
            |> List.distinct

        let conditions =
            events
            |> List.collect (fun evt ->
                let statusCond =
                    match evt.Status with
                    | StepStatus.Ok -> [$"step '{evt.StepId}' must complete successfully"]
                    | _ -> []
                let toolCond =
                    match evt.ToolName with
                    | Some t -> [$"step '{evt.StepId}' requires tool '{t}'"]
                    | None -> []
                let outputCond =
                    if not evt.Outputs.IsEmpty then
                        [$"step '{evt.StepId}' must produce output"]
                    else []
                statusCond @ toolCond @ outputCond)

        Behavioral (conditions, tools)

    // =========================================================================
    // Full distillation
    // =========================================================================

    /// Distill a single trace into a typed production.
    let distillTrace (events: CanonicalTraceEvent list) (goal: string) : TypedProduction option =
        if events.IsEmpty then None
        else
            let structural = extractStructural events
            let typed = extractTyped events
            let behavioral = extractBehavioral events

            let slots, edges =
                match structural with
                | Structural (s, e) -> s, e
                | _ -> [], []

            let id = fingerprint slots edges

            let successCount = events |> List.filter (fun e -> e.Status = StepStatus.Ok) |> List.length
            let successRate = if events.IsEmpty then 0.0 else float successCount / float events.Length

            // Compression ratio: rough estimate of trace tokens vs production tokens
            let traceSize = events |> List.sumBy (fun e -> e.Outputs |> List.sumBy (fun o -> o.Length / 4))
            let productionSize = slots.Length * 20 // ~20 tokens per slot description
            let compression = if productionSize > 0 then float traceSize / float productionSize else 1.0

            Some {
                Id = id
                Name = goal
                Facets = [structural; typed; behavioral]
                TraceCount = 1
                SuccessRate = successRate
                CompressionRatio = max 1.0 compression
                SuggestedLevel =
                    if successRate >= 0.9 && events.Length >= 3 then Helper
                    else Implementation
                LastUpdated = DateTime.UtcNow
            }

    /// Merge two productions with the same structural fingerprint.
    let merge (a: TypedProduction) (b: TypedProduction) : TypedProduction =
        let totalTraces = a.TraceCount + b.TraceCount
        let weightedRate =
            (a.SuccessRate * float a.TraceCount + b.SuccessRate * float b.TraceCount) / float totalTraces

        // Merge behavioral facets (union conditions)
        let mergedFacets =
            a.Facets |> List.map (fun facet ->
                match facet with
                | Behavioral (aConds, aTools) ->
                    let bBehavior =
                        b.Facets |> List.tryPick (function Behavioral (c, t) -> Some (c, t) | _ -> None)
                    match bBehavior with
                    | Some (bConds, bTools) ->
                        Behavioral (
                            (aConds @ bConds) |> List.distinct,
                            (aTools @ bTools) |> List.distinct)
                    | None -> facet
                | other -> other)

        let suggestedLevel =
            if totalTraces >= 5 && weightedRate >= 0.8 then Builder
            elif totalTraces >= 3 && weightedRate >= 0.7 then Helper
            else Implementation

        { a with
            Facets = mergedFacets
            TraceCount = totalTraces
            SuccessRate = weightedRate
            SuggestedLevel = suggestedLevel
            LastUpdated = DateTime.UtcNow }

    /// Distill multiple traces into a set of typed productions, merging duplicates.
    let distillAll (traces: (CanonicalTraceEvent list * string) list) : DistillationResult =
        let raw =
            traces
            |> List.choose (fun (events, goal) -> distillTrace events goal)

        // Group by structural fingerprint and merge
        let merged =
            raw
            |> List.groupBy (fun p -> p.Id)
            |> List.map (fun (_, group) ->
                group |> List.reduce merge)

        let existingIds = raw |> List.map (fun p -> p.Id) |> List.distinct
        let newCount = merged |> List.filter (fun p -> p.TraceCount = 1) |> List.length
        let reinforcedCount = merged |> List.filter (fun p -> p.TraceCount > 1) |> List.length

        { Productions = merged
          TracesProcessed = traces.Length
          NewRulesFound = newCount
          ExistingRulesReinforced = reinforcedCount }

    // =========================================================================
    // Bridge to WeightedGrammar
    // =========================================================================

    /// Convert a TypedProduction to a WeightedRule for the Bayesian system.
    let toWeightedRule (production: TypedProduction) : WeightedGrammar.WeightedRule =
        { PatternId = production.Id
          PatternName = production.Name
          Level = production.SuggestedLevel
          RawScore =
            let base' = if production.SuccessRate >= 0.8 then 5 else 3
            min 8 (base' + min 3 production.TraceCount)
          Weight = production.SuccessRate
          Confidence = min 1.0 (float production.TraceCount / 10.0)
          SuccessRate = production.SuccessRate
          SelectionCount = production.TraceCount
          Source = WeightedGrammar.Evolved
          LastUpdated = production.LastUpdated }

    // =========================================================================
    // Rendering (for inspection and debugging)
    // =========================================================================

    /// Render a typed production as a human-readable grammar rule.
    let render (production: TypedProduction) : string =
        let lines = ResizeArray<string>()
        lines.Add $"# {production.Name} [{production.Id}]"
        lines.Add $"# traces={production.TraceCount} success={production.SuccessRate:F2} compression={production.CompressionRatio:F1}x"
        lines.Add $"# level={PromotionLevel.label production.SuggestedLevel}"
        lines.Add ""

        for facet in production.Facets do
            match facet with
            | Structural (slots, edges) ->
                lines.Add "## Structural"
                for slot in slots do
                    lines.Add $"  {slot.Name} :: {slot.Kind} : {slot.InputType} -> {slot.OutputType}"
                for edge in edges do
                    lines.Add $"  {edge.From} -> {edge.To}"
                lines.Add ""

            | Typed (slots, composable) ->
                lines.Add $"## Typed (composable={composable})"
                let chain =
                    slots
                    |> List.map (fun s -> $"{s.InputType} -> {s.OutputType}")
                    |> String.concat " >> "
                lines.Add $"  {chain}"
                lines.Add ""

            | Behavioral (conditions, tools) ->
                lines.Add "## Behavioral"
                if not tools.IsEmpty then
                    let toolList = tools |> String.concat ", "
                    lines.Add $"  tools: [{toolList}]"
                for cond in conditions do
                    lines.Add $"  require: {cond}"
                lines.Add ""

        lines |> Seq.toList |> String.concat "\n"
