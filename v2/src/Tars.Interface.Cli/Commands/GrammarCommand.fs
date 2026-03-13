module Tars.Interface.Cli.Commands.GrammarCommand

open System
open Spectre.Console
open Tars.Evolution
open Tars.Evolution.WeightedGrammar
open Tars.Evolution.ReplicatorDynamics
open Tars.Evolution.MctsTypes
open Tars.Evolution.WotMctsState
open Tars.DSL.Wot
open Tars.Core.WorkflowOfThought

module RichOutput =
    let info msg = AnsiConsole.MarkupLine($"[cyan]ℹ[/] {msg}")
    let ok msg = AnsiConsole.MarkupLine($"[green]✓[/] {msg}")
    let warn msg = AnsiConsole.MarkupLine($"[yellow]⚠[/] {msg}")

/// Display current weighted grammar rules
let private showWeights () =
    let rules = WeightedGrammar.load ()
    if rules.IsEmpty then
        RichOutput.info "No weighted rules found. Run [bold]tars grammar evolve[/] to initialize."
    else
        let table = Table()
        table.Title <- TableTitle("Probabilistic Grammar Weights")
        table.AddColumn("Pattern") |> ignore
        table.AddColumn("Level") |> ignore
        table.AddColumn("Weight") |> ignore
        table.AddColumn("Confidence") |> ignore
        table.AddColumn("Success Rate") |> ignore
        table.AddColumn("Selections") |> ignore
        table.AddColumn("Source") |> ignore

        for r in rules |> List.sortByDescending (fun r -> r.Weight) do
            let levelColor =
                match r.Level with
                | GrammarRule -> "bold magenta" | DslClause -> "bold cyan"
                | Builder -> "bold green" | Helper -> "yellow" | Implementation -> "dim"
            table.AddRow(
                r.PatternName,
                $"[{levelColor}]{PromotionLevel.label r.Level}[/]",
                $"[bold]{r.Weight:F4}[/]",
                $"{r.Confidence:F3}",
                $"{r.SuccessRate:F3}",
                string r.SelectionCount,
                match r.Source with
                | Tars -> "TARS" | GuitarAlchemist -> "GA" | MachinDeOuf -> "MDO"
                | Evolved -> "Evolved" | Manual -> "Manual"
            ) |> ignore

        AnsiConsole.Write(table)
        RichOutput.info $"Total rules: [bold]{rules.Length}[/]"

/// Run replicator dynamics on the grammar ecosystem
let private runReplicator (steps: int) =
    let rules = WeightedGrammar.load ()
    if rules.IsEmpty then
        // Bootstrap from promotion pipeline
        let records = PromotionPipeline.getRecurrenceRecords ()
        if records.IsEmpty then
            RichOutput.warn "No rules or recurrence records found. Run some workflows first."
        else
            RichOutput.info $"Bootstrapping from [bold]{records.Length}[/] recurrence records..."
            let scored = records |> List.map (fun r ->
                let score = min 8 (r.OccurrenceCount + (if r.AverageScore > 0.5 then 3 else 1))
                (r, score))
            let weighted = WeightedGrammar.fromRecurrenceRecords WeightedGrammar.defaultConfig scored
            WeightedGrammar.save weighted
            RichOutput.ok $"Created [bold]{weighted.Length}[/] weighted rules from recurrence records."
    else
        RichOutput.info $"Running replicator dynamics on [bold]{rules.Length}[/] rules for {steps} steps..."
        // Build dummy outcomes from success rates (no real outcome store yet)
        let outcomesById =
            rules |> List.map (fun r ->
                let successes = int (r.SuccessRate * float r.SelectionCount)
                let failures = r.SelectionCount - successes
                let outcomes =
                    [ for _ in 1..successes -> (true, 1000L)
                      for _ in 1..failures -> (false, 3000L) ]
                (r.PatternId, outcomes))
            |> Map.ofList

        let config = { ReplicatorDynamics.defaultConfig with Steps = steps }
        let result = ReplicatorDynamics.evolveEcosystem rules outcomesById

        // Display results
        let table = Table()
        table.Title <- TableTitle("Replicator Dynamics Results")
        table.AddColumn("Species") |> ignore
        table.AddColumn("Proportion") |> ignore
        table.AddColumn("Fitness") |> ignore
        table.AddColumn("ESS?") |> ignore
        table.AddColumn("Trend") |> ignore

        for s in result.Species |> List.sortByDescending (fun s -> s.Proportion) do
            let trend =
                match result.Trajectory |> List.tryFind (fun (id, _) -> id = s.PatternId) with
                | Some (_, history) when history.Length >= 2 ->
                    let first = history.[0]
                    let last = history.[history.Length - 1]
                    if last > first + 0.01 then "[green]↑[/]"
                    elif last < first - 0.01 then "[red]↓[/]"
                    else "[dim]→[/]"
                | _ -> "[dim]?[/]"

            table.AddRow(
                s.PatternName,
                $"[bold]{s.Proportion:F4}[/]",
                $"{s.Fitness:F3}",
                (if s.IsStable then "[green]YES[/]" else "[dim]no[/]"),
                trend
            ) |> ignore

        AnsiConsole.Write(table)

        if not result.Pruned.IsEmpty then
            let prunedNames = result.Pruned |> List.map (fun s -> s.PatternName) |> String.concat ", "
            RichOutput.warn $"Pruned {result.Pruned.Length} near-extinct species: {prunedNames}"

        if not result.Stable.IsEmpty then
            let stableNames = result.Stable |> List.map (fun s -> s.PatternName) |> String.concat ", "
            RichOutput.ok $"ESS (stable strategies): {stableNames}"

        // Update weights from replicator results
        let updatedRules =
            rules |> List.map (fun r ->
                match result.Species |> List.tryFind (fun s -> s.PatternId = r.PatternId) with
                | Some s -> { r with Weight = s.Proportion; Source = Evolved; LastUpdated = DateTime.UtcNow }
                | None -> { r with Weight = 0.0; Source = Evolved; LastUpdated = DateTime.UtcNow })
            |> List.filter (fun r -> r.Weight > 0.001)
        WeightedGrammar.save updatedRules
        RichOutput.ok $"Saved [bold]{updatedRules.Length}[/] evolved weights."

/// Run MCTS search for WoT workflow derivation
let private runSearch (maxNodes: int) (iterations: int) =
    RichOutput.info $"Searching derivation space: max [bold]{maxNodes}[/] nodes, [bold]{iterations}[/] MCTS iterations..."

    let meta : DslMeta = {
        Id = "mcts-search"
        Title = "MCTS-Derived Workflow"
        Domain = "general"
        Objective = "Find optimal workflow structure"
        Constraints = []
        SuccessCriteria = []
    }

    // Build template pool from common workflow patterns
    let templates = [
        { DslConvert.defaultNode "analyze" NodeKind.Reason with
            Goal = Some "Analyze the problem space"
            Checks = [ WotCheck.NonEmpty "${analysis}" ]
            Outputs = [ SimpleOutput "analysis" ] }
        { DslConvert.defaultNode "plan" NodeKind.Reason with
            Goal = Some "Create an execution plan"
            Checks = [ WotCheck.NonEmpty "${plan}" ]
            Outputs = [ SimpleOutput "plan" ] }
        { DslConvert.defaultNode "execute" NodeKind.Work with
            Tool = Some "code_execute"
            Checks = [ WotCheck.NonEmpty "${result}" ]
            Outputs = [ SimpleOutput "result" ] }
        { DslConvert.defaultNode "verify" NodeKind.Reason with
            Goal = Some "Verify correctness of results"
            Checks = [ WotCheck.NonEmpty "${verdict}" ]
            Outputs = [ SimpleOutput "verdict" ] }
        { DslConvert.defaultNode "refine" NodeKind.Reason with
            Goal = Some "Refine and improve the solution"
            Transformation = Some GoTTransformation.Refine
            Outputs = [ SimpleOutput "refined" ] }
    ]

    let config = { MctsTypes.defaultMctsConfig with MaxIterations = iterations; MaxRolloutDepth = 15 }
    let result = WotMctsState.searchDerivation config meta templates maxNodes

    // Display results
    AnsiConsole.MarkupLine("[bold]Search Results:[/]")
    AnsiConsole.MarkupLine($"  Iterations: [bold]{result.Iterations}[/]")
    AnsiConsole.MarkupLine($"  Avg Reward: [bold]{result.AverageReward:F4}[/]")
    AnsiConsole.MarkupLine($"  Actions found: [bold]{result.BestActions.Length}[/]")

    if not result.BestActions.IsEmpty then
        AnsiConsole.MarkupLine("")
        AnsiConsole.MarkupLine("[bold cyan]Derived Workflow:[/]")
        let mutable stepNum = 1
        for action in result.BestActions do
            match action with
            | AddNode n ->
                let kind = if n.Kind = NodeKind.Reason then "[blue]REASON[/]" else "[green]WORK[/]"
                AnsiConsole.MarkupLine($"  {stepNum}. {kind} [bold]{n.Id}[/]")
                match n.Goal with
                | Some g -> AnsiConsole.MarkupLine($"     Goal: {g}")
                | None -> ()
                match n.Tool with
                | Some t -> AnsiConsole.MarkupLine($"     Tool: {t}")
                | None -> ()
                stepNum <- stepNum + 1
            | AddEdge e ->
                AnsiConsole.MarkupLine($"  {stepNum}. [dim]EDGE[/] {e.From} → {e.To}")
                stepNum <- stepNum + 1
            | SetTransformation (id, t) ->
                AnsiConsole.MarkupLine($"  {stepNum}. [yellow]TRANSFORM[/] {id} = {t}")
                stepNum <- stepNum + 1
            | Complete ->
                AnsiConsole.MarkupLine($"  {stepNum}. [bold green]COMPLETE[/]")
                stepNum <- stepNum + 1

    RichOutput.ok "Derivation search complete."

/// Entry point for `tars grammar` command
let run (args: string list) : int =
    match args with
    | [] | [ "help" ] ->
        printfn "Usage: tars grammar <command>"
        printfn ""
        printfn "Commands:"
        printfn "  weights                Show current probabilistic grammar weights"
        printfn "  evolve [--steps N]     Run replicator dynamics on rule ecosystem"
        printfn "  search [--nodes N]     MCTS search for optimal WoT derivation"
        printfn "  help                   Show this help"
        0
    | [ "weights" ] ->
        showWeights ()
        0
    | "evolve" :: rest ->
        let mutable steps = 50
        let mutable i = 0
        while i < rest.Length do
            match rest.[i] with
            | "--steps" when i + 1 < rest.Length ->
                i <- i + 1
                steps <- try int rest.[i] with _ -> 50
            | _ -> ()
            i <- i + 1
        runReplicator steps
        0
    | "search" :: rest ->
        let mutable maxNodes = 5
        let mutable iterations = 500
        let mutable i = 0
        while i < rest.Length do
            match rest.[i] with
            | "--nodes" when i + 1 < rest.Length ->
                i <- i + 1
                maxNodes <- try int rest.[i] with _ -> 5
            | "--iterations" when i + 1 < rest.Length ->
                i <- i + 1
                iterations <- try int rest.[i] with _ -> 500
            | _ -> ()
            i <- i + 1
        runSearch maxNodes iterations
        0
    | cmd :: _ ->
        AnsiConsole.MarkupLine($"[red]Unknown grammar command:[/] {cmd}")
        1
