/// Pipeline Command - CLI for managing project pipelines
module Tars.Interface.Cli.Commands.PipelineCommand

open System
open System.IO
open Tars.Core.Project
open Tars.Core.ProjectRegistry
open Tars.Core.PipelineExecutor
open Tars.Core.DemoGenerator

/// Create a new project
let handleNew (id: string) (name: string) (template: string) (mode: string) (path: string option) =
    let pipelineTemplate =
        match template.ToLower() with
        | "sdlc"
        | "standard" -> StandardSDLC
        | "agile"
        | "sprint" -> AgileSprint
        | "research" -> Research
        | _ -> StandardSDLC

    let executionMode =
        match mode.ToLower() with
        | "hitl"
        | "human" -> HumanInLoop
        | "continuous"
        | "auto" -> Continuous
        | "hybrid" -> Hybrid [ QualityAssurance; Demo ]
        | _ -> Continuous

    let rootPath =
        path |> Option.defaultValue (Path.Combine(Directory.GetCurrentDirectory(), id))

    match createAndRegister id name rootPath pipelineTemplate executionMode with
    | Result.Ok project ->
        printfn "✅ Created project: %s" project.Name
        printfn "   ID: %s" project.Id
        printfn "   Template: %A" project.Template
        printfn "   Mode: %A" project.ExecutionMode
        printfn "   Path: %s" project.RootPath
        printfn "   Stages: %d" (templateStages project.Template |> List.length)
        0
    | Result.Error e ->
        printfn "❌ Error: %s" e
        1

/// List all projects
let handleList () =
    let projects = listProjects ()

    if projects.IsEmpty then
        printfn "No projects registered."
    else
        printfn "📁 Projects (%d):" projects.Length
        printfn ""

        for p in projects do
            let state = getProjectState p.Id

            let status =
                match state with
                | Some s when s.CompletedAt.IsSome -> "✅ Complete"
                | Some s when s.CurrentStage.IsSome -> sprintf "🔄 %s" (stageName s.CurrentStage.Value)
                | _ -> "⏸️ Not started"

            printfn "  %s (%s)" p.Name p.Id
            printfn "    Template: %A | Mode: %A | Status: %s" p.Template p.ExecutionMode status
            printfn ""

    0

/// Get project status
let handleStatus (id: string) =
    match getProject id, getProjectState id with
    | None, _ ->
        printfn "❌ Project '%s' not found" id
        1
    | Some project, None ->
        printfn "❌ State not found for '%s'" id
        1
    | Some project, Some state ->
        printfn "📊 Project: %s (%s)" project.Name project.Id
        printfn ""
        printfn "Template: %A" project.Template
        printfn "Mode: %A" project.ExecutionMode
        printfn "Path: %s" project.RootPath
        printfn ""
        printfn "Pipeline Status:"
        let stages = templateStages project.Template

        for stage in stages do
            let isComplete = state.CompletedAt.IsSome
            let isCurrent = state.CurrentStage = Some stage

            let icon =
                if isComplete then "✅"
                elif isCurrent then "🔄"
                else "⏳"

            printfn "  %s %s" icon (stageName stage)

        printfn ""
        0

/// Run pipeline
let handleRun (id: string) =
    match getProject id with
    | None ->
        printfn "❌ Project '%s' not found" id
        1
    | Some project ->
        printfn "🚀 Running pipeline for: %s" project.Name
        printfn ""

        let executor = createExecutor ()
        let mutable lastEvent = ""

        executor.SetEventHandler(fun event ->
            match event with
            | PipelineStarted(_, stage) -> printfn "▶️ Pipeline started at: %s" (stageName stage)
            | StageStarted(_, stage) -> printfn "  📍 Stage: %s" (stageName stage)
            | StageCompleted(_, stage, _) -> printfn "  ✅ Completed: %s" (stageName stage)
            | ApprovalRequired(_, stage) ->
                printfn "  ⏸️ Approval required for: %s" (stageName stage)
                lastEvent <- "approval"
            | PipelineCompleted _ ->
                printfn ""
                printfn "✅ Pipeline completed!"
            | PipelineFailed(_, error) ->
                printfn ""
                printfn "❌ Pipeline failed: %s" error
            | _ -> ())

        let result = executor.ExecuteAsync(id) |> Async.AwaitTask |> Async.RunSynchronously

        match result with
        | Result.Ok() when lastEvent = "approval" ->
            printfn ""
            printfn "ℹ️ Pipeline paused. Use 'tars pipeline approve %s <stage>' to continue." id
            0
        | Result.Ok() -> 0
        | Result.Error e ->
            printfn "❌ Error: %s" e
            1

/// Generate demo
let handleDemo (id: string) (format: string) (output: string option) =
    match getProject id with
    | None ->
        printfn "❌ Project '%s' not found" id
        1
    | Some project ->
        let demoFormat =
            match format.ToLower() with
            | "html" -> HtmlPresentation
            | "json" -> JsonSummary
            | "text"
            | "plain" -> PlainText
            | _ -> MarkdownReport

        let builder = DemoBuilder()
        builder.SetProject(project.Id, project.Name)
        builder.SetTitle(sprintf "%s - Pipeline Demo" project.Name)
        builder.SetSubtitle(sprintf "Template: %A | Mode: %A" project.Template project.ExecutionMode)

        for stage in templateStages project.Template do
            let stageNameStr = stageName stage

            builder.AddSection(
                stageNameStr,
                sprintf "%s Phase" stageNameStr,
                sprintf "Completed %s phase with automated pipeline execution." stageNameStr
            )

        let demo = builder.Build(demoFormat)
        let content = render demo

        match output with
        | Some path ->
            match saveDemo demo path with
            | Result.Ok savedPath -> printfn "✅ Demo saved to: %s" savedPath
            | Result.Error e -> printfn "❌ Failed to save: %s" e
        | None -> printfn "%s" content

        0

/// Parse command arguments
let parseArg (args: string[]) (key: string) : string option =
    args
    |> Array.tryFindIndex (fun a -> a = key)
    |> Option.bind (fun i -> if i + 1 < args.Length then Some args.[i + 1] else None)

/// Run pipeline command
let run (args: string[]) =
    if args.Length < 1 then
        printfn "Usage: tars pipeline <command> [options]"
        printfn "Commands: new, list, status, run, demo"
        1
    else
        let subCmd = args.[0]
        let subArgs = if args.Length > 1 then args.[1..] else [||]

        match subCmd with
        | "new" when subArgs.Length >= 1 ->
            let id = subArgs.[0]

            let name =
                parseArg subArgs "-n"
                |> Option.orElse (parseArg subArgs "--name")
                |> Option.defaultValue id

            let template =
                parseArg subArgs "-t"
                |> Option.orElse (parseArg subArgs "--template")
                |> Option.defaultValue "sdlc"

            let mode =
                parseArg subArgs "-m"
                |> Option.orElse (parseArg subArgs "--mode")
                |> Option.defaultValue "continuous"

            let path = parseArg subArgs "-p" |> Option.orElse (parseArg subArgs "--path")
            handleNew id name template mode path
        | "new" ->
            printfn "Usage: tars pipeline new <id> [-n name] [-t template] [-m mode]"
            1
        | "list" -> handleList ()
        | "status" when subArgs.Length >= 1 -> handleStatus subArgs.[0]
        | "status" ->
            printfn "Usage: tars pipeline status <id>"
            1
        | "run" when subArgs.Length >= 1 -> handleRun subArgs.[0]
        | "run" ->
            printfn "Usage: tars pipeline run <id>"
            1
        | "demo" when subArgs.Length >= 1 ->
            let id = subArgs.[0]

            let format =
                parseArg subArgs "-f"
                |> Option.orElse (parseArg subArgs "--format")
                |> Option.defaultValue "markdown"

            let output = parseArg subArgs "-o" |> Option.orElse (parseArg subArgs "--output")
            handleDemo id format output
        | "demo" ->
            printfn "Usage: tars pipeline demo <id> [-f format] [-o output]"
            1
        | _ ->
            printfn "Unknown command: %s" subCmd
            printfn "Commands: new, list, status, run, demo"
            1
