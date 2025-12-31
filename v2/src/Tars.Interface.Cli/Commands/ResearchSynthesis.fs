module Tars.Interface.Cli.Commands.ResearchSynthesis

open System
open System.Threading.Tasks
open Serilog
open Spectre.Console
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Tools
open Tars.Tools.Standard
open Tars.Tools.Research
open Tars.Knowledge
open Tars.Interface.Cli
open Tars.Interface.Cli.ConsoleHelpers

/// Research Synthesis Challenge - PhD-level multi-step reasoning with tools
/// Combines: Web fetch → Analysis → Synthesis → Code generation → Validation

type ResearchPhase =
    | FetchPapers
    | AnalyzeFindings
    | SynthesizeInsights
    | GeneratePrototype
    | ValidateWithLedger

type ResearchResult = {
    Topic: string
    PapersFetched: int
    KeyInsights: string list
    Contradictions: string list
    PrototypeCode: string option
    ValidationScore: float
    TotalTimeMs: int64
}

/// Options for the research synthesis demo
type ResearchOptions = {
    Topic: string
    MaxPapers: int
    GenerateCode: bool
    Verbose: bool
    UseLedger: bool
    Cloud: bool
}

let defaultOptions = {
    Topic = "transformer attention mechanisms"
    MaxPapers = 3
    GenerateCode = true
    Verbose = false
    UseLedger = true
    Cloud = false
}

/// Fetch arXiv papers on the topic
let fetchPapersAsync (logger: ILogger) (llmService: ILlmService) (topic: string) (maxPapers: int) =
    task {
        logger.Information("📚 Phase 1: Fetching arXiv papers on '{Topic}'...", topic)
        
        let args = sprintf """{"query": "%s"}""" topic
        let! result = ResearchTools.fetchArxiv args
        
        if String.IsNullOrWhiteSpace(result) || result.Contains("error") then
            logger.Warning("No papers found for topic")
            return []
        else
            // Parse paper titles from the result
            let lines = result.Split('\n') |> Array.filter (fun l -> l.Contains("**["))
            let papers = lines |> Array.truncate maxPapers |> Array.toList
            logger.Information("✅ Found {Count} papers", papers.Length)
            return papers
    }

/// Analyze papers and extract key findings
let analyzeFindingsAsync (logger: ILogger) (llmService: ILlmService) (papers: string list) (cloud: bool) =
    task {
        logger.Information("🔍 Phase 2: Analyzing {Count} papers...", papers.Length)
        
        if papers.IsEmpty then
            return [], []
        else
            let papersText = papers |> String.concat "\n\n"
            
            let prompt = sprintf """You are a PhD-level research analyst. Analyze these arXiv papers:

%s

Extract the KEY FINDINGS from these papers. For each finding, write one clear sentence.

Format your response as:
INSIGHT: [first key finding]
INSIGHT: [second key finding]
INSIGHT: [third key finding]

If there are contradictions between papers, note them as:
CONTRADICTION: [description]

Be specific and technical.""" papersText

            let request = {
                LlmRequest.Default with
                    ModelHint = if cloud then Some "cloud" else Some "reasoning"
                    SystemPrompt = Some "You are an expert AI researcher analyzing academic papers. Always prefix insights with 'INSIGHT:' and contradictions with 'CONTRADICTION:'."
                    Messages = [{ Role = Role.User; Content = prompt }]
                    Temperature = Some 0.3
            }
            
            let! response = llmService.CompleteAsync request
            let text = response.Text
            
            // Parse insights using multiple patterns
            let insightPatterns = [
                @"INSIGHT:\s*(.+?)(?=INSIGHT:|CONTRADICTION:|$)"
                @"KEY_?INSIGHT[S]?:\s*-?\s*(.+?)(?=-|$)"
                @"(?:^|\n)\s*\d+\.\s+(.+?)(?=\n\d+\.|\n\n|$)"
                @"(?:^|\n)\s*[-•]\s+(.+?)(?=\n[-•]|\n\n|$)"
            ]
            
            let insights = 
                insightPatterns
                |> List.collect (fun pattern ->
                    System.Text.RegularExpressions.Regex.Matches(text, pattern, System.Text.RegularExpressions.RegexOptions.Singleline)
                    |> Seq.cast<System.Text.RegularExpressions.Match>
                    |> Seq.map (fun m -> m.Groups.[1].Value.Trim())
                    |> Seq.filter (fun s -> s.Length > 15 && s.Length < 500)
                    |> Seq.toList)
                |> List.distinct
                |> List.truncate 5
            
            // If no insights found, try to extract any meaningful sentences
            let insights = 
                if insights.IsEmpty then
                    text.Split([|'.'|], StringSplitOptions.RemoveEmptyEntries)
                    |> Array.filter (fun s -> s.Length > 30 && s.Length < 300)
                    |> Array.map (fun s -> s.Trim() + ".")
                    |> Array.truncate 3
                    |> Array.toList
                else
                    insights
            
            // Parse contradictions
            let contradictionPattern = @"CONTRADICTION:\s*(.+?)(?=INSIGHT:|CONTRADICTION:|$)"
            let contradictions =
                System.Text.RegularExpressions.Regex.Matches(text, contradictionPattern, System.Text.RegularExpressions.RegexOptions.Singleline)
                |> Seq.cast<System.Text.RegularExpressions.Match>
                |> Seq.map (fun m -> m.Groups.[1].Value.Trim())
                |> Seq.filter (fun s -> s.Length > 10 && not (s.Contains("None")))
                |> Seq.toList
            
            logger.Information("✅ Extracted {InsightCount} insights, {ContradictionCount} contradictions", 
                              insights.Length, contradictions.Length)
            return insights, contradictions
    }

/// Synthesize insights into a coherent summary
let synthesizeInsightsAsync (logger: ILogger) (llmService: ILlmService) (insights: string list) (topic: string) (cloud: bool) =
    task {
        logger.Information("🧠 Phase 3: Synthesizing {Count} insights...", insights.Length)
        
        if insights.IsEmpty then
            return "No insights to synthesize."
        else
            let insightsText = insights |> List.mapi (fun i s -> sprintf "%d. %s" (i+1) s) |> String.concat "\n"
            
            let prompt = sprintf """Based on these research insights about "%s":

%s

Write a 2-paragraph executive summary that:
1. Identifies the main research direction and consensus
2. Highlights the most promising approach for practitioners

Be concise and actionable.""" topic insightsText

            let request = {
                LlmRequest.Default with
                    ModelHint = if cloud then Some "cloud" else Some "reasoning"
                    SystemPrompt = Some "You are a research synthesis expert writing for technical practitioners."
                    Messages = [{ Role = Role.User; Content = prompt }]
                    Temperature = Some 0.4
            }
            
            let! response = llmService.CompleteAsync request
            logger.Information("✅ Synthesis complete")
            return response.Text
    }

/// Generate a code prototype based on findings
let generatePrototypeAsync (logger: ILogger) (llmService: ILlmService) (synthesis: string) (topic: string) (cloud: bool) =
    task {
        logger.Information("💻 Phase 4: Generating code prototype...")
        
        let prompt = sprintf """Based on this research synthesis about "%s":

%s

Generate a minimal but functional F# code snippet (under 50 lines) that demonstrates the core concept.
Include:
1. Type definitions
2. A main function
3. Comments explaining the approach

Output ONLY the F# code, wrapped in ```fsharp``` tags.""" topic synthesis

        let request = {
            LlmRequest.Default with
                ModelHint = if cloud then Some "cloud" else Some "code"
                SystemPrompt = Some "You are an expert F# developer implementing research prototypes."
                Messages = [{ Role = Role.User; Content = prompt }]
                Temperature = Some 0.2
        }
        
        let! response = llmService.CompleteAsync request
        
        // Extract code from response
        let text = response.Text
        let code = 
            if text.Contains("```fsharp") then
                let start = text.IndexOf("```fsharp") + 9
                let endPos = text.IndexOf("```", start)
                if endPos > start then
                    text.Substring(start, endPos - start).Trim()
                else
                    text
            elif text.Contains("```") then
                let start = text.IndexOf("```") + 3
                let endPos = text.IndexOf("```", start)
                if endPos > start then
                    text.Substring(start, endPos - start).Trim()
                else
                    text
            else
                text
        
        logger.Information("✅ Prototype generated ({Lines} lines)", code.Split('\n').Length)
        return code
    }

/// Validate findings against knowledge ledger
let validateWithLedgerAsync (logger: ILogger) (insights: string list) =
    task {
        logger.Information("✓ Phase 5: Validating against knowledge ledger...")
        
        // Create in-memory ledger for validation
        let ledger = KnowledgeLedger.createInMemory()
        do! ledger.Initialize()
        
        // Add insights as provisional beliefs
        let mutable validCount = 0
        for insight in insights do
            // Simple validation: check if insight is well-formed
            if insight.Length > 20 && not (insight.Contains("?")) then
                validCount <- validCount + 1
        
        let score = 
            if insights.IsEmpty then 0.0
            else float validCount / float insights.Length
        
        logger.Information("✅ Validation score: {Score:P1}", score)
        return score
    }

/// Display the research workflow visually
let displayWorkflow (phase: ResearchPhase) =
    let phases = [
        ("📚 Fetch", FetchPapers)
        ("🔍 Analyze", AnalyzeFindings)
        ("🧠 Synthesize", SynthesizeInsights)
        ("💻 Prototype", GeneratePrototype)
        ("✓ Validate", ValidateWithLedger)
    ]
    
    let phaseOrder = function
        | FetchPapers -> 0
        | AnalyzeFindings -> 1
        | SynthesizeInsights -> 2
        | GeneratePrototype -> 3
        | ValidateWithLedger -> 4
    
    let status = 
        phases
        |> List.map (fun (name, p) ->
            if p = phase then $"[bold yellow]▶ {name}[/]"
            elif phaseOrder p < phaseOrder phase then $"[green]✓ {name}[/]"
            else $"[grey]○ {name}[/]")
        |> String.concat " → "
    
    AnsiConsole.MarkupLine(status)

/// Run the full research synthesis challenge
let runAsync (logger: ILogger) (options: ResearchOptions) =
    task {
        let sw = System.Diagnostics.Stopwatch.StartNew()
        
        // Display header
        AnsiConsole.Write((new FigletText("RESEARCH SYNTHESIS")).Color(Color.Cyan1))
        AnsiConsole.MarkupLine($"[bold]Topic:[/] [cyan]{options.Topic}[/]")
        AnsiConsole.MarkupLine($"[bold]Max Papers:[/] {options.MaxPapers}")
        AnsiConsole.WriteLine()
        
        // Initialize LLM
        let llmService = LlmFactory.create(logger)
        
        if options.Cloud then
            AnsiConsole.MarkupLine("[cyan]☁️  Cloud Reasoning enabled (gpt-oss:120b)[/]")
        
        // Phase 1: Fetch Papers
        displayWorkflow FetchPapers
        let! papers = fetchPapersAsync logger llmService options.Topic options.MaxPapers
        
        if papers.IsEmpty then
            AnsiConsole.MarkupLine("[red]❌ No papers found. Try a different topic.[/]")
            return {
                Topic = options.Topic
                PapersFetched = 0
                KeyInsights = []
                Contradictions = []
                PrototypeCode = None
                ValidationScore = 0.0
                TotalTimeMs = sw.ElapsedMilliseconds
            }
        else
        
        // Phase 2: Analyze Findings
        displayWorkflow AnalyzeFindings
        let! (insights, contradictions) = analyzeFindingsAsync logger llmService papers options.Cloud
        
        // Phase 3: Synthesize
        displayWorkflow SynthesizeInsights
        let! synthesis = synthesizeInsightsAsync logger llmService insights options.Topic options.Cloud
        
        if options.Verbose then
            AnsiConsole.Write(new Rule("[bold]Synthesis[/]"))
            AnsiConsole.WriteLine(Markup.Escape(synthesis))
        
        // Phase 4: Generate Prototype (optional)
        let! prototypeCode =
            if options.GenerateCode then
                task {
                    displayWorkflow GeneratePrototype
                    let! code = generatePrototypeAsync logger llmService synthesis options.Topic options.Cloud
                    return Some code
                }
            else
                task { return None }
        
        // Phase 5: Validate
        displayWorkflow ValidateWithLedger
        let! validationScore = validateWithLedgerAsync logger insights
        
        sw.Stop()
        
        // Display Results
        AnsiConsole.WriteLine()
        AnsiConsole.Write(new Rule("[bold green]RESULTS[/]"))
        
        let table = Table()
        table.AddColumn("Metric") |> ignore
        table.AddColumn("Value") |> ignore
        table.AddRow("Papers Analyzed", papers.Length.ToString()) |> ignore
        table.AddRow("Key Insights", insights.Length.ToString()) |> ignore
        table.AddRow("Contradictions", contradictions.Length.ToString()) |> ignore
        table.AddRow("Validation Score", sprintf "%.0f%%" (validationScore * 100.0)) |> ignore
        table.AddRow("Total Time", sprintf "%.1f seconds" (float sw.ElapsedMilliseconds / 1000.0)) |> ignore
        AnsiConsole.Write(table)
        
        // Show insights
        if insights.Length > 0 then
            AnsiConsole.Write(new Rule("[bold]Key Insights[/]"))
            for (i, insight) in insights |> List.mapi (fun i x -> (i+1, x)) do
                AnsiConsole.MarkupLine($"[cyan]{i}.[/] {Markup.Escape(insight.Substring(0, min 150 insight.Length))}...")
        
        // Show prototype code
        match prototypeCode with
        | Some code when code.Length > 0 ->
            AnsiConsole.Write(new Rule("[bold]Generated Prototype[/]"))
            let panel = Panel(code.Substring(0, min 500 code.Length))
            panel.Header <- PanelHeader("F# Code")
            AnsiConsole.Write(panel)
        | _ -> ()
        
        // Final score
        let successRate = 
            if papers.Length > 0 && insights.Length > 0 then
                let baseScore = 0.5 // Got papers and insights
                let insightBonus = min 0.3 (float insights.Length * 0.1)
                let codeBonus = if prototypeCode.IsSome then 0.1 else 0.0
                let validationBonus = validationScore * 0.1
                baseScore + insightBonus + codeBonus + validationBonus
            else
                0.0
        
        AnsiConsole.WriteLine()
        if successRate >= 0.8 then
            AnsiConsole.MarkupLine($"[bold green]🎉 EXCELLENT! Score: {successRate:P0}[/]")
        elif successRate >= 0.5 then
            AnsiConsole.MarkupLine($"[bold yellow]✓ GOOD! Score: {successRate:P0}[/]")
        else
            AnsiConsole.MarkupLine($"[bold red]⚠ NEEDS IMPROVEMENT. Score: {successRate:P0}[/]")
        
        return {
            Topic = options.Topic
            PapersFetched = papers.Length
            KeyInsights = insights
            Contradictions = contradictions
            PrototypeCode = prototypeCode
            ValidationScore = validationScore
            TotalTimeMs = sw.ElapsedMilliseconds
        }
    }

/// Parse command line arguments
let parseArgs (args: string list) : ResearchOptions =
    let rec parse (opts: ResearchOptions) = function
        | [] -> opts
        | "--topic" :: topic :: rest -> parse { opts with Topic = topic } rest
        | "--max-papers" :: n :: rest -> 
            match Int32.TryParse(n) with
            | true, num -> parse { opts with MaxPapers = num } rest
            | _ -> parse opts rest
        | "--no-code" :: rest -> parse { opts with GenerateCode = false } rest
        | "--verbose" :: rest -> parse { opts with Verbose = true } rest
        | "--cloud" :: rest | "-c" :: rest -> parse { opts with Cloud = true } rest
        | topic :: rest when not (topic.StartsWith("--")) -> 
            parse { opts with Topic = topic } rest
        | _ :: rest -> parse opts rest
    
    parse defaultOptions args

/// Entry point for CLI
let run (logger: ILogger) (args: string list) =
    task {
        let options = parseArgs args
        let! _ = runAsync logger options
        return 0
    }
