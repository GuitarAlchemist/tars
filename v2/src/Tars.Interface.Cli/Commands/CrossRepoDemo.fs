namespace Tars.Interface.Cli.Commands

open System
open System.IO
open System.Text.Json
open Serilog
open Spectre.Console
open Tars.Evolution
open Tars.Evolution.MctsTypes
open Tars.Evolution.WotMctsState
open Tars.DSL.Wot
open Tars.Core.WorkflowOfThought
open Tars.Interface.Cli

/// `tars demo cross-repo` — a scripted showcase of the TARS ⇄ ix ⇄ Guitar
/// Alchemist (GA) integration seams, runnable end-to-end on a dev box:
///
///   1. ix Rust grammar-guided MCTS driving a TARS workflow derivation
///   2. ix Beta-Binomial bandit (grammar.weights) behind pattern selection
///   3. the GA→TARS consistency-claim predicate registry
///   4. (opt-in) TARS evolving against GA's music-theory fitness domain
///
/// Steps 1-3 need no LLM; step 4 runs a local model when `--with-llm` is passed.
module CrossRepoDemo =

    let private rule (title: string) =
        let r = Rule($"[bold cyan]{title}[/]")
        r.Justification <- Justify.Left
        AnsiConsole.Write r

    /// Conventional sibling-repo locations.
    let private repoDir (name: string) =
        let p =
            Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "source", "repos", name)
        if Directory.Exists p then Some p else None

    let private ixBridgeConfig () : MachinBridge.MachinConfig option =
        repoDir "ix"
        |> Option.map (fun dir -> { MachinBridge.defaultConfig with WorkingDir = Some dir })

    // ── Section 0: environment ──────────────────────────────────────────────
    let private showStatus () =
        rule "Cross-repo environment"
        let ixCfg = Tars.Core.IxSkill.discover ()
        match ixCfg with
        | Some c when Tars.Core.IxSkill.isAvailable c ->
            let dir = c.RepoDir |> Option.defaultValue "?"
            AnsiConsole.MarkupLine($"  ix:  [green]available[/]  ([dim]{dir}[/])")
        | Some c ->
            let dir = c.RepoDir |> Option.defaultValue "?"
            AnsiConsole.MarkupLine($"  ix:  [yellow]found but not built[/] ([dim]{dir}[/]) — run [bold]cargo build -p ix-skill[/]")
        | None ->
            AnsiConsole.MarkupLine("  ix:  [red]not found[/] — demos fall back to built-in F#")
        match repoDir "ga" with
        | Some d -> AnsiConsole.MarkupLine($"  ga:  [green]found[/]  ([dim]{d}[/])")
        | None -> AnsiConsole.MarkupLine("  ga:  [dim]not found[/]")
        AnsiConsole.WriteLine()

    // ── Section 0b: MCP federation backends (tars mcp list) ─────────────────
    let private demoMcpList () =
        rule "0 · MCP federation backends (tars mcp list)"
        let n = McpCommand.renderConfigured ()
        if n > 0 then
            AnsiConsole.MarkupLine($"  [dim]{n} backend(s) — [bold]tars mcp server[/] re-exposes each one's tools (prefixed) to any MCP client, incl. Claude Code.[/]")
        AnsiConsole.WriteLine()

    // ── Section 1: ix Rust grammar-guided MCTS ──────────────────────────────
    let private demoMcts () =
        rule "1 · ix Rust grammar-guided MCTS → TARS workflow derivation"
        let meta : DslMeta =
            { Id = "demo-mcts"
              Title = "Cross-repo MCTS demo"
              Domain = "general"
              Objective = "Derive a workflow shape from a template pool"
              Constraints = []
              SuccessCriteria = [] }
        let templates =
            [ { DslConvert.defaultNode "analyze" NodeKind.Reason with Goal = Some "Analyze the problem" }
              { DslConvert.defaultNode "plan" NodeKind.Reason with Goal = Some "Plan an approach" }
              { DslConvert.defaultNode "execute" NodeKind.Work with Tool = Some "code_execute" }
              { DslConvert.defaultNode "verify" NodeKind.Reason with Goal = Some "Verify the result" }
              { DslConvert.defaultNode "refine" NodeKind.Reason with Goal = Some "Refine the solution" } ]

        let config = { MctsTypes.defaultMctsConfig with MaxIterations = 300; MaxRolloutDepth = 15 }
        let actions, usedIx =
            MctsBridge.searchWotDerivation (ixBridgeConfig ()) config meta templates 5

        let backend = if usedIx then "[green]ix Rust grammar-guided MCTS[/]" else "[yellow]built-in F# MCTS (ix unavailable)[/]"
        AnsiConsole.MarkupLine($"  Backend: {backend}")
        AnsiConsole.MarkupLine($"  Derived workflow ([bold]{actions.Length}[/] steps):")
        let mutable n = 1
        for a in actions do
            match a with
            | AddNode node ->
                let kind = if node.Kind = NodeKind.Reason then "[blue]REASON[/]" else "[green]WORK[/]"
                AnsiConsole.MarkupLine($"    {n}. {kind} [bold]{node.Id}[/]")
                n <- n + 1
            | Complete -> AnsiConsole.MarkupLine($"    {n}. [bold green]COMPLETE[/]"); n <- n + 1
            | _ -> AnsiConsole.MarkupLine($"    {n}. [dim]{a}[/]"); n <- n + 1
        AnsiConsole.WriteLine()

    // ── Section 2: ix Beta-Binomial bandit (grammar.weights) ────────────────
    let private demoBandit () =
        rule "2 · ix Beta-Binomial bandit → pattern selection"
        // Three reasoning patterns with different success histories. This is the
        // exact call PatternSelector makes when ranking patterns.
        let arms =
            [ "ChainOfThought", 9.0, 1.0   // 9 wins / 1 loss
              "ReAct", 2.0, 6.0            // 2 wins / 6 losses
              "WorkflowOfThought", 1.0, 1.0 ] // unexplored (Laplace prior)

        let printProbs (label: string) (probs: (string * float) list) =
            AnsiConsole.MarkupLine($"  [dim]{label}[/]")
            for (name, p) in probs |> List.sortByDescending snd do
                let bar = String('█', int (Math.Round(p * 30.0)))
                AnsiConsole.MarkupLine($"    {name,-20} [bold]{p:F3}[/] [cyan]{bar}[/]")

        // F# reference softmax over Beta means — also the selector's fallback.
        let fsharpProbs () =
            let means = arms |> List.map (fun (n, a, b) -> n, a / (a + b))
            let exps = means |> List.map (fun (n, m) -> n, exp m)
            let z = exps |> List.sumBy snd
            exps |> List.map (fun (n, e) -> n, e / z)

        match Tars.Core.IxSkill.discover () with
        | Some c when Tars.Core.IxSkill.isAvailable c ->
            let rules = arms |> List.map (fun (n, a, b) -> {| id = n; alpha = a; beta = b |})
            let input = JsonSerializer.Serialize({| rules = rules; temperature = 1.0 |})
            match (Tars.Core.IxSkill.runSkillJson c "grammar.weights" input).GetAwaiter().GetResult() with
            | Result.Ok json ->
                try
                    use d = JsonDocument.Parse json
                    let probs =
                        [ for p in d.RootElement.GetProperty("probabilities").EnumerateArray() ->
                            p.GetProperty("rule_id").GetString(), p.GetProperty("probability").GetDouble() ]
                    printProbs "via ix grammar.weights (α=wins+1, β=losses+1, softmax):" probs
                with _ -> printProbs "ix output unparseable; F# fallback:" (fsharpProbs ())
            | Result.Error e ->
                AnsiConsole.MarkupLine($"  [yellow]ix call failed ({e}); F# fallback:[/]")
                printProbs "F# Beta/softmax fallback:" (fsharpProbs ())
        | _ ->
            printProbs "ix unavailable — F# Beta/softmax fallback:" (fsharpProbs ())

        AnsiConsole.MarkupLine("  [dim]→ the evidenced arm dominates; the unexplored arm keeps non-zero mass.[/]")
        AnsiConsole.WriteLine()

    // ── Section 3: GA→TARS claim predicate registry ─────────────────────────
    let private demoContract () =
        rule "3 · GA→TARS consistency-claim predicate registry"
        let preds =
            // Read the live contract schema from the GA repo when present.
            match repoDir "ga" with
            | Some ga ->
                let schema = Path.Combine(ga, "docs", "contracts", "ga-tars-claim.schema.json")
                if File.Exists schema then
                    try
                        use d = JsonDocument.Parse(File.ReadAllText schema)
                        [ for e in d.RootElement.GetProperty("properties").GetProperty("predicate").GetProperty("enum").EnumerateArray() -> e.GetString() ]
                    with _ -> []
                else []
            | None -> []
        let preds = if List.isEmpty preds then [ "pitch_classes"; "chord_intervals"; "mode_degree" ] else preds
        let joined = String.Join(", ", preds)
        AnsiConsole.MarkupLine($"  Registered predicates ([bold]{preds.Length}[/]): {joined}")
        AnsiConsole.MarkupLine("  [dim]chord_intervals is transposition-invariant — catches quality drift across keys.[/]")
        AnsiConsole.WriteLine()

    // ── Section 4 (opt-in): GA music-theory fitness domain ──────────────────
    let private demoGaFitness (logger: ILogger) (model: string) (maxProblems: int) = task {
        rule "4 · TARS evolving against GA's music-theory fitness domain"
        AnsiConsole.MarkupLine($"  Model: [bold]{model}[/]  ·  problems: [bold]{maxProblems}[/] (GA domain)")
        let llm = LlmFactory.createWithModel logger model
        let! summary =
            BenchmarkRunner.runSuiteFromProblems
                llm (GaProblemBank.all ()) None None (Some maxProblems) true
                (fun msg -> AnsiConsole.MarkupLine($"    [dim]{Markup.Escape msg}[/]"))
        BenchmarkRunner.recordOutcomes summary
        let pct = (summary.PassRate * 100.0).ToString("F0")
        AnsiConsole.MarkupLine($"  Result: [bold]{summary.Validated}/{summary.TotalProblems}[/] passed ([bold]{pct}%%[/]) — outcomes recorded → feeds the bandit (section 2)")
        AnsiConsole.WriteLine()
    }

    let run (logger: ILogger) (args: string[]) : System.Threading.Tasks.Task<int> = task {
        let withLlm = args |> Array.exists (fun a -> a = "--with-llm")
        let model =
            args
            |> Array.tryFindIndex (fun a -> a = "--model" || a = "-m")
            |> Option.bind (fun i -> if i + 1 < args.Length then Some args.[i + 1] else None)
            |> Option.defaultValue "qwen2.5-coder:7b"

        AnsiConsole.Write(FigletText("TARS ⇄ ix ⇄ GA").Color(Color.Cyan1))
        AnsiConsole.MarkupLine("[dim]Cross-repo capability showcase[/]")
        AnsiConsole.WriteLine()

        showStatus ()
        demoMcpList ()
        demoMcts ()
        demoBandit ()
        demoContract ()

        if withLlm then
            do! demoGaFitness logger model 2
        else
            rule "4 · GA music-theory fitness domain  [dim](skipped)[/]"
            AnsiConsole.MarkupLine("  [dim]Re-run with[/] [bold]--with-llm [[--model <name>]][/] [dim]to evolve against GA's domain with a local model.[/]")
            AnsiConsole.WriteLine()

        AnsiConsole.MarkupLine("[green]✓[/] Cross-repo demo complete.")
        return 0
    }
