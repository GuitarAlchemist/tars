namespace Tars.Interface.Cli.Commands

open System
open Serilog
open Tars.Evolution
open Tars.Interface.Cli
open Tars.Interface.Cli.SpectreUI

/// CLI command for running curated F# coding benchmarks.
/// Usage: tars benchmark code [run|status|report] [options]
module CodeBenchmark =

    let private parseDifficulty (s: string) =
        match s.ToLowerInvariant() with
        | "basic" | "beginner" -> Some Beginner
        | "intermediate" -> Some Intermediate
        | "advanced" -> Some Advanced
        | "expert" -> Some Expert
        | _ -> None

    let private parseCategory (s: string) =
        match s.ToLowerInvariant() with
        | "string" | "strings" -> Some StringManipulation
        | "algorithms" | "algo" -> Some Algorithms
        | "data" | "datastructures" -> Some DataStructures
        | "error" | "errorhandling" -> Some ErrorHandling
        | "async" -> Some AsyncPatterns
        | "type" | "types" | "typedesign" -> Some TypeDesign
        | "pattern" | "patternmatching" -> Some PatternMatching
        | _ -> None

    let private showStatus () =
        let s = ProblemBank.summary ()
        RichOutput.info $"Problem Bank: {s.Total} problems"
        printfn "  Basic:        %d" s.Basic
        printfn "  Intermediate: %d" s.Intermediate
        printfn "  Advanced:     %d" s.Advanced
        printfn "  Expert:       %d" s.Expert

        let history = BenchmarkRunner.loadHistory ()
        if history.IsEmpty then
            printfn "\nNo benchmark runs yet."
        else
            let latest = history |> List.last
            let ts = latest.Timestamp.ToString("yyyy-MM-dd HH:mm")
            RichOutput.info $"Latest run: {ts}"
            printfn "  Problems: %d  Compiled: %d  Passed: %d" latest.TotalProblems latest.Compiled latest.Validated
            printfn "  Compile rate: %.0f%%  Pass rate: %.0f%%" (latest.CompileRate * 100.0) (latest.PassRate * 100.0)

    let private showReport () =
        let history = BenchmarkRunner.loadHistory ()
        if history.IsEmpty then
            printfn "No benchmark runs yet. Run: tars benchmark code run"
        else
            RichOutput.info $"Benchmark History ({history.Length} runs)"
            printfn ""
            printfn "  %-20s  %6s  %8s  %8s  %10s" "Date" "Total" "Compiled" "Passed" "Pass Rate"
            printfn "  %-20s  %6s  %8s  %8s  %10s" "----" "-----" "--------" "------" "---------"
            for run in history do
                printfn "  %-20s  %6d  %8d  %8d  %9.0f%%" (run.Timestamp.ToString("yyyy-MM-dd HH:mm")) run.TotalProblems run.Compiled run.Validated (run.PassRate * 100.0)

    let run (logger: ILogger) (args: string[]) = task {
        match args with
        | [| "status" |] ->
            showStatus ()
            return 0

        | [| "report" |] ->
            showReport ()
            return 0

        | args when args.Length >= 1 && args.[0] = "run" ->
            let mutable difficulty = None
            let mutable category = None
            let mutable maxProblems = None
            let mutable i = 1
            while i < args.Length do
                match args.[i] with
                | "--difficulty" | "-d" when i + 1 < args.Length ->
                    difficulty <- parseDifficulty args.[i + 1]
                    i <- i + 2
                | "--category" | "-c" when i + 1 < args.Length ->
                    category <- parseCategory args.[i + 1]
                    i <- i + 2
                | "--max" | "-n" when i + 1 < args.Length ->
                    maxProblems <- Some (int args.[i + 1])
                    i <- i + 2
                | _ -> i <- i + 1

            let llm = LlmFactory.create logger

            RichOutput.info "Starting code benchmark..."
            let! summary =
                BenchmarkRunner.runSuite llm difficulty category maxProblems true
                    (fun msg -> printfn "%s" msg)

            // Record outcomes for self-improvement loop
            BenchmarkRunner.recordOutcomes summary

            // Save results
            let path = BenchmarkRunner.saveResults summary

            printfn ""
            let passPct = (summary.PassRate * 100.0).ToString("F0")
            RichOutput.info $"Results: {summary.Validated}/{summary.TotalProblems} passed ({passPct}%%)"
            printfn "  Compiled: %d/%d (%.0f%%)" summary.Compiled summary.TotalProblems (summary.CompileRate * 100.0)
            printfn "  Duration: %.1fs" (float summary.TotalDurationMs / 1000.0)
            printfn "  Saved to: %s" path

            return 0

        | _ ->
            printfn "Usage: tars benchmark code [run|status|report]"
            printfn ""
            printfn "  run [options]   Run benchmark suite"
            printfn "    --difficulty basic|intermediate|advanced|expert"
            printfn "    --category algorithms|strings|data|error|type|pattern"
            printfn "    --max N       Limit to N problems"
            printfn "  status          Show problem bank and latest results"
            printfn "  report          Show benchmark history"
            return 0
    }
