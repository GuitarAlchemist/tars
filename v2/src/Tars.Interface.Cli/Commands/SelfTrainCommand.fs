namespace Tars.Interface.Cli.Commands

open System
open System.Threading.Tasks
open Serilog
open Spectre.Console
open Tars.Evolution
open Tars.Interface.Cli

/// `tars self-train` — close the data→weights loop (level-4 self-improvement).
///
/// Exports the benchmark loop's verified-PASS solutions into an SFT dataset, and
/// prints the runbook for the external fine-tune + A/B re-benchmark. The dataset
/// half is fully native; the weight update is a GPU step the runbook describes.
module SelfTrainCommand =

    let private defaultOut () =
        IO.Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".tars", "self_train", "dataset.jsonl")

    let private export (out: string) (domain: string) : int =
        let filter =
            match domain.ToLowerInvariant() with
            | "ga" | "music" -> Some MusicTheory
            | _ -> None
        let stats = SelfTrain.exportDataset out filter

        AnsiConsole.MarkupLine("[bold]Self-train dataset export[/]")
        if stats.VerifiedExamples = 0 then
            AnsiConsole.MarkupLine("  [yellow]No verified examples found.[/] Run [bold]tars benchmark code run[/] first to generate PASS-verified solutions.")
        else
            AnsiConsole.MarkupLine($"  Verified examples: [bold green]{stats.VerifiedExamples}[/]  (from {stats.TotalAttempts} attempts)")
            AnsiConsole.MarkupLine($"  Unique problems:   [bold]{stats.UniqueProblems}[/]")
            if stats.FastestSelected > 0 then
                AnsiConsole.MarkupLine($"  Fastest-variant:   [bold]{stats.FastestSelected}[/] timed problem(s) (kept the quickest verified solution)")
            for (cat, n) in stats.ByCategory do
                AnsiConsole.MarkupLine($"    [dim]{Markup.Escape cat}[/]: {n}")
            AnsiConsole.MarkupLine($"  Written to:  [bold]{Markup.Escape stats.OutputPath}[/]")
            AnsiConsole.MarkupLine($"  Modelfile:   [dim]{Markup.Escape stats.ModelfilePath}[/] (for [bold]ollama create tars-coder[/])")
        0

    let private runbook () : int =
        AnsiConsole.MarkupLine("[bold]Self-train cycle (level-4 data→weights loop)[/]")
        AnsiConsole.WriteLine()
        let steps =
            [ "1. Baseline   ", "tars benchmark code run --domain all --model qwen2.5-coder:7b"
              "2. Export     ", "tars self-train export --out ~/.tars/self_train/dataset.jsonl"
              "3. Fine-tune  ", "(GPU) unsloth/llama-factory SFT on dataset.jsonl -> GGUF -> `ollama create tars-coder -f Modelfile`"
              "4. Re-bench   ", "tars benchmark code run --domain all --model tars-coder"
              "5. Compare    ", "tars benchmark code report   # delta in pass-rate = the recursion's headroom" ]
        for (label, cmd) in steps do
            AnsiConsole.MarkupLine($"  [cyan]{label}[/] [dim]{Markup.Escape cmd}[/]")
        AnsiConsole.WriteLine()
        AnsiConsole.MarkupLine("  [dim]Labels are deterministic (compile + PASS), so the loop trains only on[/]")
        AnsiConsole.MarkupLine("  [dim]ground-truth-correct solutions — the property that stops self-training collapsing.[/]")
        0

    let private sourceFor (domain: string) (idFilter: string option) : BenchmarkProblem list =
        let s =
            match domain.ToLowerInvariant() with
            | "ga" | "music" -> GaProblemBank.all ()
            | "all" -> ProblemBank.all () @ GaProblemBank.all ()
            | _ -> ProblemBank.all ()
        match idFilter with
        | Some id -> s |> List.filter (fun p -> p.Id = id)
        | None -> s

    /// `tars self-train cycle` — A/B the loop end-to-end: benchmark a baseline
    /// model, export the dataset, benchmark a candidate model, and report the
    /// pass-rate + per-problem speed deltas. In a real run the candidate is the
    /// fine-tuned baseline; here a model swap stands in so the *measurement* path
    /// is exercised without a GPU.
    let private cycle (logger: ILogger) (args: string list) : Task<int> =
        task {
            let mutable baseline = "llama3.2:1b"
            let mutable candidate = "qwen2.5-coder:7b"
            let mutable domain = "ga"
            let mutable idFilter = None
            let mutable maxN = None
            let rec parse =
                function
                | "--baseline" :: v :: tl -> baseline <- v; parse tl
                | "--candidate" :: v :: tl -> candidate <- v; parse tl
                | "--domain" :: v :: tl -> domain <- v; parse tl
                | "--id" :: v :: tl -> idFilter <- Some v; parse tl
                | "--max" :: v :: tl -> (match Int32.TryParse v with | true, n -> maxN <- Some n | _ -> ()); parse tl
                | _ :: tl -> parse tl
                | [] -> ()
            parse args

            let source = sourceFor domain idFilter
            let quietLog _ = ()

            let runWith (model: string) = task {
                let llm = LlmFactory.createWithModel logger model
                let! summary = BenchmarkRunner.runSuiteFromProblems llm source None None maxN true quietLog
                BenchmarkRunner.recordOutcomes summary
                BenchmarkRunner.saveResults summary |> ignore
                return summary
            }

            let effectiveCount = match maxN with Some n -> min n source.Length | None -> source.Length
            AnsiConsole.MarkupLine($"[bold]Self-train A/B cycle[/]  domain=[bold]{domain}[/]  problems=[bold]{effectiveCount}[/]")
            AnsiConsole.MarkupLine($"  [dim](candidate stands in for the fine-tuned baseline; this exercises the measurement path)[/]")

            AnsiConsole.MarkupLine($"  Running baseline:  [bold]{baseline}[/] ...")
            let! a = runWith baseline
            AnsiConsole.MarkupLine($"  Running candidate: [bold]{candidate}[/] ...")
            let! b = runWith candidate

            // Refresh the SFT dataset from the now-larger verified corpus.
            let st = SelfTrain.exportDataset (defaultOut ()) None

            let pct (s: BenchmarkRunSummary) = s.PassRate * 100.0
            let signI (n: int) = if n >= 0 then sprintf "+%d" n else string n
            let signF (x: float) = if x >= 0.0 then sprintf "+%.0f" x else sprintf "%.0f" x
            let passedDelta = signI (b.Validated - a.Validated)
            let rateDelta = sprintf "%s pts" (signF (pct b - pct a))
            let table = Table()
            table.AddColumn("Metric") |> ignore
            table.AddColumn(Markup.Escape baseline) |> ignore
            table.AddColumn(Markup.Escape candidate) |> ignore
            table.AddColumn("Delta") |> ignore
            table.Border(TableBorder.Rounded) |> ignore
            table.AddRow("Passed",
                sprintf "%d/%d" a.Validated a.TotalProblems,
                sprintf "%d/%d" b.Validated b.TotalProblems,
                passedDelta) |> ignore
            table.AddRow("Pass rate",
                sprintf "%.0f%%" (pct a), sprintf "%.0f%%" (pct b), rateDelta) |> ignore
            AnsiConsole.Write(table)

            // Per-problem speed delta (timed problems present in both runs).
            let timed =
                a.Attempts
                |> List.choose (fun ai ->
                    match ai.ExecutionNs, b.Attempts |> List.tryFind (fun bi -> bi.ProblemId = ai.ProblemId) with
                    | Some na, Some bi ->
                        match bi.ExecutionNs with
                        | Some nb -> Some (ai.ProblemId, na, nb)
                        | None -> None
                    | _ -> None)
            for (pid, na, nb) in timed do
                AnsiConsole.MarkupLine($"  perf [bold]{Markup.Escape pid}[/]: {float na/1_000_000.0:F2} ms -> {float nb/1_000_000.0:F2} ms")

            AnsiConsole.MarkupLine($"  [dim]Dataset refreshed: {st.VerifiedExamples} verified examples -> {Markup.Escape st.OutputPath}[/]")
            return 0
        }

    let run (logger: ILogger) (args: string list) : Task<int> =
      task {
        match args with
        | "cycle" :: rest -> return! cycle logger rest
        | "export" :: rest ->
            let mutable out = defaultOut ()
            let mutable domain = "all"
            let rec parse =
                function
                | "--out" :: v :: tl -> out <- v; parse tl
                | "--domain" :: v :: tl -> domain <- v; parse tl
                | _ :: tl -> parse tl
                | [] -> ()
            parse rest
            return export out domain
        | "runbook" :: _ -> return runbook ()
        | _ ->
            AnsiConsole.MarkupLine("Usage: [bold]tars self-train[/] <command>")
            AnsiConsole.MarkupLine("  cycle [--baseline M] [--candidate M] [--domain ga|all] [--max N]  A/B the loop end-to-end")
            AnsiConsole.MarkupLine("  export [--out <path>] [--domain all|ga]   Build SFT dataset from verified-PASS solutions")
            AnsiConsole.MarkupLine("  runbook                                   Print the full fine-tune + A/B cycle")
            return 0
      }
