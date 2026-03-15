/// <summary>
/// CLI Commands for Advanced Prompting & Cognitive Grounding
/// Exposes Phase 8 and Phase 11 capabilities to the command line.
/// </summary>
module Tars.Interface.Cli.Commands.GroundingCommands

open System
open Serilog
open Spectre.Console
open Tars.Core
open Tars.Cortex
open Tars.Cortex.AdvancedPrompting
open Tars.Cortex.CognitiveGrounding
open Tars.Llm
open Tars.Security
open Tars.Interface.Cli

/// Options for the prompt command
type PromptOptions = {
    Question: string
    Strategy: string option  // "self-ask", "least-to-most", "knowledge", "pal", "meta", "auto"
    Verbose: bool
    Cloud: bool
}

/// Options for the ground command
type GroundOptions = {
    Query: string
    Verbose: bool
    Cloud: bool
}

/// Run advanced prompting with selected strategy
let runPrompt (logger: ILogger) (options: PromptOptions) =
    task {
        AnsiConsole.MarkupLine("[bold cyan]⚡ Advanced Prompting[/]")
        
        // Load credentials
        let secretsPath = "secrets.json"
        match CredentialVault.loadSecretsFromDisk secretsPath with
        | Result.Ok() -> ()
        | Result.Error err -> 
            AnsiConsole.MarkupLine($"[yellow]Warning: Could not load secrets: {err}[/]")
        
        // Setup LLM
        let llmService = LlmFactory.create logger

        let model =
            if options.Cloud then "gpt-oss:120b-cloud"
            else "llama3.2:3b"

        let config = { defaultConfig with Verbose = options.Verbose }
        
        // Determine strategy
        let strategy = 
            options.Strategy 
            |> Option.map (fun s -> s.ToLowerInvariant())
            |> Option.defaultWith (fun () -> autoSelectStrategy options.Question)
        
        AnsiConsole.MarkupLine($"[dim]Strategy: {strategy}[/]")
        if options.Cloud then
            AnsiConsole.MarkupLine($"[cyan]☁️  Cloud model: {model}[/]")
        
        // Execute the selected strategy
        let! result = 
            match strategy with
            | "self-ask" | "selfask" ->
                AnsiConsole.Status().StartAsync("Self-Ask reasoning...", fun _ -> 
                    selfAsk llmService config options.Question)
            | "least-to-most" | "leasttomost" | "l2m" ->
                AnsiConsole.Status().StartAsync("Least-to-Most decomposition...", fun _ -> 
                    leastToMost llmService config options.Question)
            | "knowledge" | "generate-knowledge" | "gk" ->
                AnsiConsole.Status().StartAsync("Generating knowledge...", fun _ -> 
                    generateKnowledge llmService config options.Question)
            | "pal" | "code" | "program" ->
                AnsiConsole.Status().StartAsync("Program-Aided Language...", fun _ -> 
                    programAided llmService config options.Question)
            | "meta" | "meta-prompt" ->
                AnsiConsole.Status().StartAsync("Meta-prompting...", fun _ -> 
                    metaPrompt llmService config options.Question)
            | _ ->
                // Default: auto-select
                let autoStrategy = autoSelectStrategy options.Question
                AnsiConsole.MarkupLine($"[dim]Auto-selected: {autoStrategy}[/]")
                match autoStrategy with
                | "PAL" -> programAided llmService config options.Question
                | "Least-to-Most" -> leastToMost llmService config options.Question
                | "Generate-Knowledge" -> generateKnowledge llmService config options.Question
                | _ -> selfAsk llmService config options.Question
        
        // Display results
        AnsiConsole.WriteLine()
        let panel = Panel(Markup.Escape(result.FinalAnswer))
        panel.Header <- PanelHeader($"[green]Answer[/] ({result.Strategy})")
        panel.Border <- BoxBorder.Rounded
        AnsiConsole.Write(panel)
        
        if options.Verbose && not result.IntermediateSteps.IsEmpty then
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("[dim]Intermediate Steps:[/]")
            for step in result.IntermediateSteps do
                AnsiConsole.MarkupLine($"[dim]  • {Markup.Escape(step.Substring(0, min 100 step.Length))}...[/]")
        
        AnsiConsole.MarkupLine($"[dim]Tokens: {result.TotalTokens}[/]")
        
        return 0
    }

/// Run grounded query with verification
let runGround (logger: ILogger) (ledger: Tars.Knowledge.KnowledgeLedger) (options: GroundOptions) =
    task {
        AnsiConsole.MarkupLine("[bold cyan]🔍 Grounded Query[/]")
        
        // Load credentials
        let secretsPath = "secrets.json"
        match CredentialVault.loadSecretsFromDisk secretsPath with
        | Result.Ok() -> ()
        | Result.Error err -> 
            AnsiConsole.MarkupLine($"[yellow]Warning: Could not load secrets: {err}[/]")
        
        // Setup LLM
        let llmService = LlmFactory.create logger

        let model =
            if options.Cloud then "gpt-oss:120b-cloud"
            else "llama3.2:3b"

        if options.Cloud then
            AnsiConsole.MarkupLine($"[cyan]☁️  Cloud model: {model}[/]")
        
        // Execute grounded query
        let! result = 
            AnsiConsole.Status().StartAsync("Querying with grounding...", fun _ -> 
                groundedQuery llmService ledger options.Query)
        
        // Display answer
        AnsiConsole.WriteLine()
        let panel = Panel(Markup.Escape(result.Answer))
        panel.Header <- PanelHeader($"[green]Grounded Answer[/] (Confidence: {result.OverallConfidence:P0})")
        panel.Border <- BoxBorder.Rounded
        AnsiConsole.Write(panel)
        
        // Show warnings
        if not result.Warnings.IsEmpty then
            AnsiConsole.WriteLine()
            for warning in result.Warnings do
                AnsiConsole.MarkupLine($"[yellow]{Markup.Escape(warning)}[/]")
        
        // Show citations
        if not result.Citations.IsEmpty then
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("[dim]Citations:[/]")
            for kvp in result.Citations do
                AnsiConsole.MarkupLine($"[dim]  [{kvp.Key}] {Markup.Escape(kvp.Value)}[/]")
        
        // Show claim verification in verbose mode
        if options.Verbose && not result.Claims.IsEmpty then
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("[dim]Claim Verification:[/]")
            for vc in result.Claims do
                let statusStr = 
                    match vc.Status with
                    | Verified (_, conf) -> $"[green]✓ Verified ({conf:P0})[/]"
                    | Contradicted _ -> "[red]✗ Contradicted[/]"
                    | Unverifiable _ -> "[yellow]? Unverifiable[/]"
                    | Pending -> "[dim]⋯ Pending[/]"
                let claimPreview = 
                    if vc.Claim.Statement.Length > 50 
                    then vc.Claim.Statement.Substring(0, 50) + "..." 
                    else vc.Claim.Statement
                AnsiConsole.MarkupLine($"  {statusStr} {Markup.Escape(claimPreview)}")
        
        return 0
    }

/// Parse and run prompt command
let parseAndRunPrompt (logger: ILogger) (args: string array) =
    task {
        // Show help
        if args |> Array.exists (fun a -> a = "--help" || a = "-h") then
            printfn "Usage: tars prompt <question> [options]"
            printfn ""
            printfn "Advanced prompting with multiple reasoning strategies"
            printfn ""
            printfn "Options:"
            printfn "  --strategy <name>   Strategy: self-ask, least-to-most, knowledge, pal, meta, auto"
            printfn "  --cloud, -c         Use cloud LLM (gpt-oss:120b-cloud)"
            printfn "  --verbose, -v       Show intermediate steps"
            printfn "  --help, -h          Show this help"
            printfn ""
            printfn "Examples:"
            printfn "  tars prompt \"Why is the sky blue?\""
            printfn "  tars prompt \"Calculate the factorial of 10\" --strategy pal"
            printfn "  tars prompt \"Explain quantum entanglement\" --cloud --verbose"
            return 0
        else
            let mutable options = { 
                Question = ""
                Strategy = None
                Verbose = false
                Cloud = false
            }
            
            let mutable i = 0
            while i < args.Length do
                match args.[i] with
                | "--strategy" when i + 1 < args.Length ->
                    i <- i + 1
                    options <- { options with Strategy = Some args.[i] }
                | "--verbose" | "-v" -> 
                    options <- { options with Verbose = true }
                | "--cloud" | "-c" -> 
                    options <- { options with Cloud = true }
                | arg when not (arg.StartsWith("--")) && options.Question = "" ->
                    options <- { options with Question = arg }
                | _ -> ()
                i <- i + 1
            
            if String.IsNullOrWhiteSpace(options.Question) then
                AnsiConsole.MarkupLine("[red]Error: Question is required. Use --help for usage.[/]")
                return 1
            else
                return! runPrompt logger options
    }
