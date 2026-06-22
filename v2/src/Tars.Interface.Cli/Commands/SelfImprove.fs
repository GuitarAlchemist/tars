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
        let llm =
            match flag "--model" with
            | Some m -> LlmFactory.createWithModel logger m
            | None -> LlmFactory.create logger

        AnsiConsole.MarkupLine(
            "[bold]Self-improvement gate[/]: LLM proposes → hermetic test gate (isolated worktree)…")
        AnsiConsole.MarkupLine(
            $"  test: [bold]{Markup.Escape test}[/]   file: [bold]{Markup.Escape file}[/]")

        let verdict =
            SelfHostingGate.runGateGenerated llm repo proj test file |> Async.RunSynchronously

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
            "Usage: tars self-improve --test <name> --file <relpath> --project <testproj.fsproj> [--repo <root>] [--model <name>]")
        1
