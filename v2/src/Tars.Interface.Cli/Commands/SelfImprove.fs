module Tars.Interface.Cli.Commands.SelfImprove

open Serilog
open Spectre.Console
open Tars.Evolution
open Tars.Interface.Cli

/// `tars self-improve --test <name> --file <relpath> --project <testproj.fsproj>
///                    [--repo <root>] [--model <name>]`
///
/// The self-driving self-hosting loop (ADR 0002/0003): the LLM proposes an edit
/// to make a failing test pass; the hermetic worktree gate builds + tests it in
/// isolation, and on Accept promotes the verified edit to a `self-improve/*`
/// branch (and records an SFT training win). Repo paths are relative to `--repo`
/// (default: current directory).
let run (logger: ILogger) (args: string array) : int =
    let flag name =
        args
        |> Array.tryFindIndex ((=) name)
        |> Option.bind (fun i -> if i + 1 < args.Length then Some args.[i + 1] else None)

    match flag "--test", flag "--file", flag "--project" with
    | Some test, Some file, Some proj ->
        let repo =
            flag "--repo" |> Option.defaultValue (System.IO.Directory.GetCurrentDirectory())
        // A capable coder model is required: the gate needs an exact, applicable
        // mutation. Weaker defaults (deepseek-r1:1.5b, qwen2.5-coder:7b) failed
        // live; qwen3-coder:30b landed the first autonomous Accept on TARS.
        let defaultModel = "qwen3-coder:30b"
        let llm =
            match flag "--model" with
            | Some m -> LlmFactory.createWithModel logger m
            | None -> LlmFactory.createWithModel logger defaultModel

        // Best-of-N (ADR 0002 D5): N diverse proposals, accept the first that
        // passes the gate. `--parallel` bounds concurrent dotnet-test runs (default
        // 2 — N full suites at once thrashes a single machine).
        let n = flag "--n" |> Option.bind (fun s -> match System.Int32.TryParse s with true, v -> Some v | _ -> None) |> Option.defaultValue 4
        let parallelism = flag "--parallel" |> Option.bind (fun s -> match System.Int32.TryParse s with true, v -> Some v | _ -> None) |> Option.defaultValue 2

        AnsiConsole.MarkupLine(
            "[bold]Self-improvement gate[/]: best-of-N LLM proposals → hermetic test gate (isolated worktrees)…")
        AnsiConsole.MarkupLine(
            $"  test: [bold]{Markup.Escape test}[/]   file: [bold]{Markup.Escape file}[/]   N={n} parallel={parallelism}")

        let verdict =
            SelfHostingGate.runGateBestOfN llm repo proj test file n parallelism |> Async.RunSynchronously

        match verdict with
        | SelfHostingGate.Promoted(branch, rationale) ->
            AnsiConsole.MarkupLine($"[bold green]PROMOTED[/] → {Markup.Escape branch}")
            AnsiConsole.MarkupLine($"  {Markup.Escape rationale}")
            0
        | SelfHostingGate.Rejected reason ->
            AnsiConsole.MarkupLine($"[yellow]REJECTED[/]: {Markup.Escape reason}")
            1
    | _ ->
        AnsiConsole.MarkupLine(
            "Usage: tars self-improve --test <name> --file <relpath> --project <testproj.fsproj> [--repo <root>] [--model <name>=qwen3-coder:30b] [--n 4] [--parallel 2]")
        1
