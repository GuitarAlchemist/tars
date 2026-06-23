module Tars.Interface.Cli.Commands.SelfImprove

open Serilog
open Spectre.Console
open Tars.Evolution
open Tars.Interface.Cli

/// Drive one (test, file) gap through the hermetic best-of-N gate.
let private runSingle
    (logger: ILogger)
    (flag: string -> string option)
    (intFlag: string -> int -> int)
    : int =
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
        let n = intFlag "--n" 4
        let parallelism = intFlag "--parallel" 2

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
        AnsiConsole.MarkupLine(
            "   or: tars self-improve backlog [--backlog <path>=self-improve-backlog.json] [--run] [--repo <root>] [--model <name>] [--n 4] [--parallel 2]")
        1

/// Load the curated capability-gap backlog (ADR 0002 D5) and either list it or,
/// with `--run`, drive every entry through the same hermetic gate, reporting a
/// per-entry verdict and a final promoted/total summary.
let private runBacklog
    (logger: ILogger)
    (args: string array)
    (flag: string -> string option)
    (intFlag: string -> int -> int)
    : int =
    let backlogPath = flag "--backlog" |> Option.defaultValue "self-improve-backlog.json"
    let repo =
        flag "--repo" |> Option.defaultValue (System.IO.Directory.GetCurrentDirectory())

    match SelfImproveBacklog.load backlogPath with
    | Result.Error e ->
        AnsiConsole.MarkupLine($"[red]Backlog error[/]: {Markup.Escape e}")
        1
    | Result.Ok [] ->
        AnsiConsole.MarkupLine($"[yellow]Backlog is empty[/]: {Markup.Escape backlogPath}")
        0
    | Result.Ok entries when not (Array.contains "--run" args) ->
        let table = Table()
        table.AddColumn("id") |> ignore
        table.AddColumn("target test") |> ignore
        table.AddColumn("file") |> ignore
        for e in entries do
            table.AddRow(Markup.Escape e.Id, Markup.Escape e.TargetTest, Markup.Escape e.TargetFile)
            |> ignore
        AnsiConsole.Write(table)
        AnsiConsole.MarkupLine(
            $"{List.length entries} entries; pass [bold]--run[/] to drive them through the gate.")
        0
    | Result.Ok entries ->
        let model = flag "--model" |> Option.defaultValue "qwen3-coder:30b"
        let llm = LlmFactory.createWithModel logger model
        let n = intFlag "--n" 4
        let parallelism = intFlag "--parallel" 2
        AnsiConsole.MarkupLine(
            $"[bold]Self-improve backlog[/]: {List.length entries} entries → hermetic gate (N={n}, parallel={parallelism}, model={Markup.Escape model})")
        let results =
            entries
            |> List.map (fun e ->
                AnsiConsole.MarkupLine(
                    $"→ [bold]{Markup.Escape e.Id}[/]  ({Markup.Escape e.TargetTest})…")
                let verdict =
                    SelfHostingGate.runGateBestOfN llm repo e.TestProject e.TargetTest e.TargetFile n parallelism
                    |> Async.RunSynchronously
                match verdict with
                | SelfHostingGate.Promoted(branch, _) ->
                    AnsiConsole.MarkupLine($"  [green]PROMOTED[/] → {Markup.Escape branch}")
                | SelfHostingGate.Rejected reason ->
                    AnsiConsole.MarkupLine($"  [yellow]REJECTED[/]: {Markup.Escape reason}")
                verdict)
        let promoted =
            results
            |> List.filter (function
                | SelfHostingGate.Promoted _ -> true
                | _ -> false)
            |> List.length
        AnsiConsole.MarkupLine($"[bold]{promoted}/{List.length results}[/] promoted")
        if promoted > 0 then 0 else 1

/// `tars self-improve --test <name> --file <relpath> --project <testproj.fsproj>
///                    [--repo <root>] [--model <name>]`
/// `tars self-improve backlog [--backlog <path>] [--run] [--repo <root>] …`
///
/// The self-driving self-hosting loop (ADR 0002/0003): the LLM proposes an edit
/// to make a failing test pass; the hermetic worktree gate builds + tests it in
/// isolation, and on Accept promotes the verified edit to a `self-improve/*`
/// branch (and records an SFT training win). Repo paths are relative to `--repo`
/// (default: current directory). The `backlog` subcommand drives a curated list of
/// (test, file) capability-gap entries (ADR 0002 D5) through the same gate.
let run (logger: ILogger) (args: string array) : int =
    let flag name =
        args
        |> Array.tryFindIndex ((=) name)
        |> Option.bind (fun i -> if i + 1 < args.Length then Some args.[i + 1] else None)

    let intFlag name dflt =
        flag name
        |> Option.bind (fun s ->
            match System.Int32.TryParse s with
            | true, v -> Some v
            | _ -> None)
        |> Option.defaultValue dflt

    if Array.tryHead args = Some "backlog" then
        runBacklog logger args flag intFlag
    else
        runSingle logger flag intFlag
