namespace Tars.Interface.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open Spectre.Console
open Tars.Core
open Tars.Llm
open Tars.Cortex
open Tars.DSL.Wot
open Tars.Interface.Cli

module PlanCmd =

    type PlanOptions = { Goal: string; Model: string option }

    let parseArgs (args: string[]) : PlanOptions =
        let mutable goal = ""
        let mutable model = None
        let mutable i = 0

        while i < args.Length do
            match args.[i] with
            | "--model" when i + 1 < args.Length ->
                model <- Some args.[i + 1]
                i <- i + 2
            | arg when not (arg.StartsWith("-")) ->
                goal <- if goal = "" then arg else goal + " " + arg
                i <- i + 1
            | _ -> i <- i + 1

        if goal = "" then
            { Goal = "Task for TARS"
              Model = model }
        else
            { Goal = goal; Model = model }

    let private extractTrsx (text: string) =
        // Regex to extract ```hcl ... ``` or ```trsx ... ``` or just ``` ... ```
        let pattern = @"```(?:hcl|trsx)?\s*([\s\S]*?)```"
        let m = System.Text.RegularExpressions.Regex.Match(text, pattern)
        if m.Success then m.Groups.[1].Value.Trim() else text.Trim()

    let run (config: TarsConfig) (options: PlanOptions) : Task<int> =
        task {
            AnsiConsole.MarkupLine($"[blue]🧠 Planning workflow for goal:[/] [white]{Markup.Escape(options.Goal)}[/]")

            let log = Serilog.Log.Logger
            let llm = LlmFactory.create log

            let basePrompt =
                PlannerPrompts.generatePlanPrompt options.Goal (Some config.VariantOverlays)

            let fullPrompt = "You are a TARS Workflow Architect.\n\n" + basePrompt

            let settings: LlmRequest =
                { Model = options.Model
                  ModelHint = Some "planning"
                  SystemPrompt = None
                  MaxTokens = Some 4096
                  Temperature = Some 0.2
                  Stop = []
                  Messages =
                    [ { Role = Role.User
                        Content = fullPrompt } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = None
                  ContextWindow = None }

            try
                AnsiConsole.MarkupLine("[dim]Thinking...[/]")

                let! response = llm.CompleteAsync(settings)

                let trsxContent = extractTrsx response.Text

                let timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss")

                let safeGoal =
                    options.Goal.ToLowerInvariant()
                    |> Seq.filter (fun c -> Char.IsLetterOrDigit(c) || c = '_')
                    |> Seq.truncate 30
                    |> Seq.map string
                    |> String.concat ""

                let filename = $"plan_{timestamp}_{safeGoal}.wot.trsx"
                let dir = Path.Combine(".wot", "plans")

                if not (Directory.Exists(dir)) then
                    Directory.CreateDirectory(dir) |> ignore

                let path = Path.Combine(dir, filename)

                File.WriteAllText(path, trsxContent)
                AnsiConsole.MarkupLine($"[green]✓ Plan generated:[/] [white]{path}[/]")

                match WotParser.parseFile path with
                | Result.Ok _ ->
                    AnsiConsole.MarkupLine("[green]✓ Syntax Valid[/]")
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine("Run this plan with:")
                    AnsiConsole.MarkupLine($"[cyan]tars wot run {path} --reason llm[/]")
                    return 0
                | Result.Error errs ->
                    AnsiConsole.MarkupLine("[red]⚠ Generated plan has syntax errors:[/]")

                    for e in errs do
                        AnsiConsole.MarkupLine($"  Line {e.Line}: {Markup.Escape(e.Message)}")

                    AnsiConsole.MarkupLine("\nPlease check and fix the file manually.")
                    return 1
            with ex ->
                AnsiConsole.MarkupLine($"[red]Plan Generation Failed:[/] {Markup.Escape(ex.Message)}")
                return 1
        }
