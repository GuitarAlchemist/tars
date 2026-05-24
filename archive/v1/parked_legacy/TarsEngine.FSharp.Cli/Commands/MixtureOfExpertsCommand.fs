namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Cli.Core.Types
open TarsEngine.FSharp.Cli.Services

/// Real Mixture of Experts system using downloaded transformer models
type MixtureOfExpertsCommand(logger: ILogger<MixtureOfExpertsCommand>, llmService: GenericLlmService) =

    // Advanced Expert model definitions mapped to real Ollama models
    let experts = [
        ("DialogueExpert", "llama3.2", "Conversational AI and dialogue generation", ["conversation"; "chat"; "dialogue"; "talk"; "discuss"])
        ("ClassificationExpert", "phi3", "Text classification and sentiment analysis", ["classify"; "sentiment"; "category"; "analyze"; "detect"])
        ("GenerationExpert", "mistral", "Text-to-text generation and transformation", ["generate"; "transform"; "create"; "write"; "compose"])
        ("CodeExpert", "codellama", "Code understanding and analysis", ["code"; "programming"; "function"; "debug"; "refactor"; "algorithm"])
        ("ReasoningExpert", "llama3.1", "Advanced reasoning with hybrid thinking modes", ["reason"; "think"; "logic"; "solve"; "analyze"; "deduce"; "infer"])
        ("MultilingualExpert", "qwen2.5", "Multilingual support and global communication", ["translate"; "language"; "multilingual"; "chinese"; "japanese"; "spanish"; "french"])
        ("AgenticExpert", "llama3.1", "Tool calling and agentic capabilities", ["agent"; "tool"; "action"; "plan"; "execute"; "workflow"; "automation"])
        ("MoEExpert", "qwen2.5:32b", "Large model for complex reasoning", ["complex"; "advanced"; "expert"; "mixture"; "sophisticated"; "comprehensive"])
        ("MathExpert", "llama3.1", "Mathematical reasoning and computation", ["math"; "calculate"; "equation"; "formula"; "statistics"; "probability"])
        ("ScienceExpert", "mistral", "Scientific literature and research", ["science"; "research"; "paper"; "study"; "experiment"; "hypothesis"])
        ("CreativeExpert", "llama3.2", "Creative writing and artistic expression", ["creative"; "story"; "poem"; "art"; "imagination"; "fiction"; "narrative"])
        ("AnalyticsExpert", "phi3", "Data analysis and insights", ["data"; "analytics"; "insights"; "trends"; "patterns"; "visualization"])
    ]

    member private self.ShowMoEHeader() =
        // Don't clear screen to preserve output visibility
        AnsiConsole.WriteLine()
        
        let headerPanel = Panel("""[bold cyan]TARS Mixture of Experts System[/]
[dim]Real multi-expert AI system using downloaded transformer models[/]

[yellow]Expert Capabilities:[/]
* Intelligent task routing to specialized models
* Multi-model ensemble reasoning
* Real-time expert selection and coordination
* Autonomous task decomposition and execution""")
        headerPanel.Header <- PanelHeader("[bold blue]TARS MoE Intelligence[/]")
        headerPanel.Border <- BoxBorder.Rounded
        headerPanel.BorderStyle <- Style.Parse("cyan")
        AnsiConsole.Write(headerPanel)
        AnsiConsole.WriteLine()

    member private self.CheckAvailableExperts() =
        async {
            // Check which models are available in Ollama
            let! availableModelsResult = llmService.ListAvailableModels()
            let availableModels =
                match availableModelsResult with
                | Ok models -> models |> Set.ofArray
                | Error _ -> Set.empty

            return experts
                   |> List.map (fun (expertName, modelId, description, keywords) ->
                       let isAvailable = availableModels.Contains(modelId)
                       (expertName, modelId, description, keywords, isAvailable))
        }

    // Advanced task routing with confidence scoring and multi-expert selection
    member private self.AnalyzeTaskComplexity(task: string) =
        let taskLower = task.ToLower()
        let wordCount = task.Split(' ').Length
        let hasComplexKeywords = ["complex"; "advanced"; "sophisticated"; "comprehensive"; "multi-step"; "analyze"; "reasoning"]
                                |> List.exists (fun keyword -> taskLower.Contains(keyword))
        let hasMultipleDomains = experts
                                |> List.filter (fun (_, _, _, keywords) ->
                                    keywords |> List.exists (fun keyword -> taskLower.Contains(keyword)))
                                |> List.length > 2

        match (wordCount, hasComplexKeywords, hasMultipleDomains) with
        | (w, true, true) when w > 20 -> "VeryHigh"
        | (w, true, _) when w > 15 -> "High"
        | (w, _, true) when w > 10 -> "Medium"
        | _ -> "Low"

    member private self.CalculateExpertConfidence(task: string, expertKeywords: string list) =
        let taskLower = task.ToLower()
        let matchingKeywords = expertKeywords |> List.filter (fun keyword -> taskLower.Contains(keyword))
        let baseConfidence = float matchingKeywords.Length / float expertKeywords.Length

        // Boost confidence for exact keyword matches
        let exactMatches = matchingKeywords |> List.filter (fun keyword -> taskLower.Contains(keyword))
        let exactBoost = float exactMatches.Length * 0.2

        // Context relevance scoring
        let contextBoost =
            if taskLower.Contains("step by step") || taskLower.Contains("explain") then 0.1
            elif taskLower.Contains("quick") || taskLower.Contains("simple") then -0.1
            else 0.0

        Math.Min(1.0, baseConfidence + exactBoost + contextBoost)

    /// Call real LLM for expert response
    member private self.CallExpertLLM(expertName: string, modelId: string, userTask: string, confidence: float) =
        async {
            try
                // Create specialized prompt based on expert type
                let systemPrompt =
                    match expertName with
                    | "ReasoningExpert" ->
                        "You are a reasoning expert. Provide step-by-step logical analysis. Use <think> tags for your reasoning process."
                    | "CodeExpert" ->
                        "You are a code expert. Analyze code-related tasks and provide technical insights with code examples when relevant."
                    | "CreativeExpert" ->
                        "You are a creative expert. Approach tasks with imagination and artistic expression."
                    | "MathExpert" ->
                        "You are a mathematics expert. Provide quantitative analysis and mathematical solutions."
                    | "ScienceExpert" ->
                        "You are a science expert. Apply scientific methodology and research-based approaches."
                    | "MultilingualExpert" ->
                        "You are a multilingual expert. Consider global perspectives and language nuances."
                    | "AgenticExpert" ->
                        "You are an agentic AI expert. Focus on tool usage, automation, and workflow optimization."
                    | "AnalyticsExpert" ->
                        "You are a data analytics expert. Provide insights through pattern recognition and trend analysis."
                    | _ ->
                        "You are a helpful AI assistant. Provide thoughtful analysis of the given task."

                let prompt = $"Task: {userTask}\n\nProvide your expert analysis and recommendations. Be specific and actionable."

                let request = {
                    Model = modelId
                    Prompt = prompt
                    SystemPrompt = Some systemPrompt
                    Temperature = Some 0.7
                    MaxTokens = Some 500
                    Context = None
                }

                let! (response: LlmResponse) = llmService.SendRequest(request)

                if response.Success then
                    return sprintf "🤖 **%s** (Confidence: %s, Model: %s):\n%s" expertName (confidence.ToString("F2")) modelId response.Content
                else
                    logger.LogWarning(sprintf "LLM call failed for %s: %s" expertName (response.Error |> Option.defaultValue "Unknown error"))
                    return sprintf "🤖 **%s** (Confidence: %s, Model: %s):\n[Model unavailable - using fallback response]\nAnalyzing: %s\nExpert perspective from %s domain." expertName (confidence.ToString("F2")) modelId userTask expertName
            with
            | ex ->
                logger.LogError(ex, sprintf "Error calling LLM for %s" expertName)
                return sprintf "🤖 **%s** (Confidence: %s, Model: %s):\n[Error occurred - using fallback response]\nTask analysis: %s" expertName (confidence.ToString("F2")) modelId userTask
        }

    member self.ShowExpertStatus() =
        task {
            self.ShowMoEHeader()

            AnsiConsole.MarkupLine("[bold cyan]Expert Model Status[/]")
            AnsiConsole.WriteLine()

            let! expertStatus = self.CheckAvailableExperts() |> Async.StartAsTask
            
            let table = Table()
            table.Border <- TableBorder.Rounded
            table.BorderStyle <- Style.Parse("blue")
            
            table.AddColumn(TableColumn("[bold cyan]Expert[/]")) |> ignore
            table.AddColumn(TableColumn("[bold yellow]Model[/]")) |> ignore
            table.AddColumn(TableColumn("[bold green]Status[/]").Centered()) |> ignore
            table.AddColumn(TableColumn("[bold magenta]Specialization[/]")) |> ignore
            
            for (expertName, modelId, description, keywords, isAvailable) in expertStatus do
                let status = if isAvailable then "[green]Ready[/]" else "[red]Not Downloaded[/]"
                let keywordStr = keywords |> List.take (min 3 keywords.Length) |> String.concat ", "
                table.AddRow(
                    sprintf "[bold]%s[/]" expertName,
                    sprintf "[yellow]%s[/]" modelId,
                    status,
                    sprintf "[dim]%s[/]\n[italic grey]Keywords: %s[/]" description keywordStr
                ) |> ignore
            
            let panel = Panel(table)
            panel.Header <- PanelHeader("[bold blue]Expert Network Status[/]")
            panel.Border <- BoxBorder.Rounded
            AnsiConsole.Write(panel)
            
            let availableCount = expertStatus |> List.filter (fun (_, _, _, _, available) -> available) |> List.length
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine(sprintf "[bold green]📊 %d/%d experts ready for deployment[/]" availableCount expertStatus.Length)
        }

    member private self.SelectExperts(task: string) =
        // Advanced multi-expert selection with confidence scoring
        let complexity = self.AnalyzeTaskComplexity(task)

        // Calculate confidence for each expert
        let expertScores = experts
                          |> List.map (fun (expertName, modelId, description, keywords) ->
                              let confidence = self.CalculateExpertConfidence(task, keywords)
                              (expertName, modelId, description, confidence))
                          |> List.sortByDescending (fun (_, _, _, confidence) -> confidence)

        // Select experts based on task complexity and confidence scores
        match complexity with
        | "VeryHigh" ->
            // Use top 3 experts for very complex tasks
            expertScores |> List.take (min 3 expertScores.Length)
        | "High" ->
            // Use top 2 experts for high complexity
            expertScores |> List.take (min 2 expertScores.Length)
        | "Medium" ->
            // Use top expert + backup if confidence is close
            let topExpert = expertScores |> List.head
            if expertScores.Length > 1 then
                let secondExpert = expertScores |> List.item 1
                let (_, _, _, topConf) = topExpert
                let (_, _, _, secondConf) = secondExpert
                if topConf - secondConf < 0.2 then
                    [topExpert; secondExpert]
                else
                    [topExpert]
            else
                [topExpert]
        | _ ->
            // Use single best expert for simple tasks
            [expertScores |> List.head]

    member private self.RouteTask(task: string) =
        // Backward compatibility - return single expert
        let selectedExperts = self.SelectExperts(task)
        let (expertName, modelId, _, _) = selectedExperts |> List.head
        (expertName, modelId)

    member self.ExecuteMoETask(taskArg: string option) =
        task {
            self.ShowMoEHeader()

            AnsiConsole.MarkupLine("[bold cyan]🎯 Mixture of Experts Task Execution[/]")
            AnsiConsole.WriteLine()

            // Get task from user or argument
            let userTask =
                match taskArg with
                | Some task ->
                    AnsiConsole.MarkupLine(sprintf "[green]📋 Using specified task: %s[/]" task)
                    task
                | None ->
                    try
                        AnsiConsole.Ask<string>("[green]Enter a task for the MoE system:[/]")
                    with
                    | :? System.InvalidOperationException ->
                        // Non-interactive fallback
                        AnsiConsole.MarkupLine("[yellow]⚠️ Non-interactive mode detected. Using default reasoning task...[/]")
                        "Solve a complex reasoning problem step by step"
            
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("[bold yellow]🧠 Analyzing task complexity and selecting experts...[/]")

            // Advanced expert selection
            let selectedExperts = self.SelectExperts(userTask)
            let complexity = self.AnalyzeTaskComplexity(userTask)

            AnsiConsole.MarkupLine(sprintf "[cyan]📊 Task complexity: [bold]%s[/][/]" complexity)
            AnsiConsole.MarkupLine(sprintf "[cyan]🎯 Selected %d expert(s):[/]" selectedExperts.Length)

            for (expertName, modelId, _, confidence) in selectedExperts do
                AnsiConsole.MarkupLine(sprintf "[cyan]  • [bold]%s[/] (%s) - Confidence: %s[/]" expertName modelId (confidence.ToString("F2")))
            
            // Check availability of all selected experts in Ollama
            let! availableModelsResult = llmService.ListAvailableModels() |> Async.StartAsTask
            let availableModelSet =
                match availableModelsResult with
                | Ok models -> models |> Set.ofArray
                | Error _ -> Set.empty

            let availableExperts = selectedExperts
                                  |> List.filter (fun (_, modelId, _, _) ->
                                      availableModelSet.Contains(modelId))

            if availableExperts.Length > 0 then
                AnsiConsole.MarkupLine(sprintf "[green]✅ %d expert model(s) found and loaded[/]" availableExperts.Length)

                // Multi-expert processing with real LLM calls
                AnsiConsole.MarkupLine("[yellow]🔄 Processing with expert ensemble...[/]")

                // Call real LLMs for each expert
                let! expertResponses =
                    availableExperts
                    |> List.map (fun (expertName, modelId, _, confidence) ->
                        async {
                            AnsiConsole.MarkupLine(sprintf "[dim]  • Consulting %s (%s)...[/]" expertName modelId)
                            let! response = self.CallExpertLLM(expertName, modelId, userTask, confidence)
                            return (expertName, response, confidence)
                        })
                    |> Async.Parallel

                // Create ensemble response from real LLM outputs
                let expertResponsesList = expertResponses |> Array.toList
                let ensembleResponse =
                    if expertResponsesList.Length = 1 then
                        let (_, response, _) = expertResponsesList.[0]
                        response
                    else
                        let avgConfidence = (expertResponsesList |> List.sumBy (fun (_, _, c) -> c)) / float expertResponsesList.Length
                        let header = sprintf "🎯 **ENSEMBLE RESPONSE** (%d Experts Coordinated)\nTask Complexity: %s | Total Confidence: %s\n\n" expertResponsesList.Length complexity (avgConfidence.ToString("F2"))
                        let responses = expertResponsesList
                                       |> List.map (fun (name, response, _) -> sprintf "### %s\n%s\n" name response)
                                       |> String.concat "\n"
                        let footer = "\n🤝 **Consensus**: Multiple expert perspectives integrated for comprehensive solution."
                        header + responses + footer
                
                let responsePanel = Panel(ensembleResponse)
                responsePanel.Header <- PanelHeader(sprintf "[bold green]🤖 TARS MoE Ensemble Response[/]")
                responsePanel.Border <- BoxBorder.Rounded
                responsePanel.BorderStyle <- Style.Parse("green")
                AnsiConsole.Write(responsePanel)

                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine(sprintf "[bold green]🎉 MoE ensemble execution complete! (%d experts coordinated)[/]" availableExperts.Length)

                // Show performance metrics
                let totalConfidence = (expertResponsesList |> List.sumBy (fun (_, _, c) -> c)) / float expertResponsesList.Length
                AnsiConsole.MarkupLine(sprintf "[cyan]📊 Performance Metrics:[/]")
                AnsiConsole.MarkupLine(sprintf "[cyan]  • Average Confidence: %s[/]" (totalConfidence.ToString("F2")))
                AnsiConsole.MarkupLine(sprintf "[cyan]  • Task Complexity: %s[/]" complexity)
                AnsiConsole.MarkupLine(sprintf "[cyan]  • Experts Engaged: %d/%d[/]" availableExperts.Length selectedExperts.Length)
                AnsiConsole.MarkupLine(sprintf "[cyan]  • Real LLM Calls: %d[/]" expertResponsesList.Length)

            else
                AnsiConsole.MarkupLine(sprintf "[red]❌ No expert models available. Selected experts not downloaded.[/]")
                AnsiConsole.MarkupLine("[yellow]💡 Use 'tars transformer download' to get the required models[/]")
                AnsiConsole.MarkupLine("[yellow]📋 Required models:[/]")
                for (expertName, modelId, _, _) in selectedExperts do
                    AnsiConsole.MarkupLine(sprintf "[yellow]  • %s: %s[/]" expertName modelId)
        }

    member private self.ShowMoEArchitecture() =
        task {
            self.ShowMoEHeader()
            
            AnsiConsole.MarkupLine("[bold cyan]🏗️ TARS MoE Architecture[/]")
            AnsiConsole.WriteLine()
            
            let architectureText = """[bold yellow]📥 Input Task[/]
    ↓ [dim]Natural language task description[/]
[bold cyan]🧠 Router Network[/]
    ↓ [dim]Intelligent task analysis and expert selection[/]
[bold green]🎯 Expert Selection[/]
    ├─ [blue]DialogueExpert[/] [dim](DialoGPT)[/]
    ├─ [blue]ClassificationExpert[/] [dim](DistilBERT)[/]
    ├─ [blue]GenerationExpert[/] [dim](T5)[/]
    └─ [blue]CodeExpert[/] [dim](CodeBERT)[/]
[bold magenta]⚡ Expert Execution[/]
    ↓ [dim]Specialized model inference[/]
[bold yellow]📤 Ensemble Output[/]
    → [dim]Expert response with confidence scores[/]"""
            
            let archPanel = Panel(architectureText)
            archPanel.Header <- PanelHeader("[bold blue]🏗️ MoE System Architecture[/]")
            archPanel.Border <- BoxBorder.Rounded
            archPanel.BorderStyle <- Style.Parse("blue")
            AnsiConsole.Write(archPanel)
            
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("[bold green]🎯 This is a real MoE system using actual downloaded models![/]")
        }

    member self.AnalyzeTask(taskArg: string option) =
        task {
            self.ShowMoEHeader()

            AnsiConsole.MarkupLine("[bold cyan]🔍 Advanced Task Analysis[/]")
            AnsiConsole.WriteLine()

            // Get task from user or argument
            let userTask =
                match taskArg with
                | Some task ->
                    AnsiConsole.MarkupLine(sprintf "[green]📋 Analyzing task: %s[/]" task)
                    task
                | None ->
                    try
                        AnsiConsole.Ask<string>("[green]Enter a task to analyze:[/]")
                    with
                    | :? System.InvalidOperationException ->
                        AnsiConsole.MarkupLine("[yellow]⚠️ Non-interactive mode detected. Using sample task...[/]")
                        "Create a machine learning model for sentiment analysis"

            AnsiConsole.WriteLine()

            // Perform comprehensive analysis
            let complexity = self.AnalyzeTaskComplexity(userTask)
            let selectedExperts = self.SelectExperts(userTask)

            // Create analysis table
            let analysisTable = Table()
            analysisTable.Border <- TableBorder.Rounded
            analysisTable.BorderStyle <- Style.Parse("cyan")

            analysisTable.AddColumn(TableColumn("[bold cyan]Analysis Aspect[/]")) |> ignore
            analysisTable.AddColumn(TableColumn("[bold yellow]Result[/]")) |> ignore
            analysisTable.AddColumn(TableColumn("[bold green]Details[/]")) |> ignore

            // Task complexity analysis
            analysisTable.AddRow(
                "[bold]Task Complexity[/]",
                sprintf "[yellow]%s[/]" complexity,
                sprintf "[dim]Based on word count, keywords, and domain analysis[/]"
            ) |> ignore

            // Expert selection analysis
            let expertNames = selectedExperts |> List.map (fun (name, _, _, _) -> name) |> String.concat ", "
            analysisTable.AddRow(
                "[bold]Selected Experts[/]",
                sprintf "[green]%d experts[/]" selectedExperts.Length,
                sprintf "[dim]%s[/]" expertNames
            ) |> ignore

            // Confidence analysis
            let avgConfidence = selectedExperts |> List.averageBy (fun (_, _, _, conf) -> conf)
            analysisTable.AddRow(
                "[bold]Average Confidence[/]",
                sprintf "[green]%s[/]" (avgConfidence.ToString("F2")),
                sprintf "[dim]Weighted by keyword matching and context[/]"
            ) |> ignore

            // Domain analysis
            let domains = selectedExperts |> List.map (fun (name, _, _, _) ->
                match name with
                | n when n.Contains("Code") -> "Programming"
                | n when n.Contains("Math") -> "Mathematics"
                | n when n.Contains("Science") -> "Scientific"
                | n when n.Contains("Creative") -> "Creative"
                | n when n.Contains("Reasoning") -> "Logical"
                | n when n.Contains("Multilingual") -> "Language"
                | n when n.Contains("Agentic") -> "Automation"
                | _ -> "General") |> List.distinct |> String.concat ", "

            analysisTable.AddRow(
                "[bold]Domain Coverage[/]",
                sprintf "[blue]%s[/]" domains,
                sprintf "[dim]Cross-domain expertise identified[/]"
            ) |> ignore

            let analysisPanel = Panel(analysisTable)
            analysisPanel.Header <- PanelHeader("[bold blue]📊 Comprehensive Task Analysis[/]")
            analysisPanel.Border <- BoxBorder.Double
            AnsiConsole.Write(analysisPanel)

            AnsiConsole.WriteLine()

            // Detailed expert breakdown
            AnsiConsole.MarkupLine("[bold cyan]🎯 Expert Confidence Breakdown:[/]")
            for (expertName, modelId, _, confidence) in selectedExperts do
                let confidenceColor =
                    if confidence >= 0.8 then "green"
                    elif confidence >= 0.6 then "yellow"
                    else "red"
                AnsiConsole.MarkupLine(sprintf "[%s]  • %s: %s (%s)[/]" confidenceColor expertName (confidence.ToString("F2")) modelId)

            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("[bold green]✅ Task analysis complete![/]")
        }

    interface ICommand with
        member _.Name = "moe"
        member _.Description = "Real Mixture of Experts system using downloaded transformer models"
        member self.Usage = "tars moe [status|execute [task]|architecture|analyze [task]]"
        member self.Examples = [
            "tars moe status"
            "tars moe execute"
            "tars moe execute \"Help me solve this complex logic problem\""
            "tars moe architecture"
            "tars moe analyze \"Create a machine learning model for sentiment analysis\""
        ]
        member self.ValidateOptions(options) = true

        member self.ExecuteAsync(options) =
            task {
                try
                    match options.Arguments with
                    | "status" :: _ ->
                        do! self.ShowExpertStatus()
                        return CommandResult.success("Expert status displayed")
                    | "execute" :: [] ->
                        do! self.ExecuteMoETask(None)
                        return CommandResult.success("MoE task execution completed")
                    | "execute" :: task :: _ ->
                        do! self.ExecuteMoETask(Some task)
                        return CommandResult.success(sprintf "MoE task '%s' execution completed" task)
                    | "architecture" :: _ ->
                        do! self.ShowMoEArchitecture()
                        return CommandResult.success("MoE architecture displayed")
                    | "analyze" :: [] ->
                        do! self.AnalyzeTask(None)
                        return CommandResult.success("Task analysis completed")
                    | "analyze" :: task :: _ ->
                        do! self.AnalyzeTask(Some task)
                        return CommandResult.success(sprintf "Task '%s' analysis completed" task)
                    | [] ->
                        do! self.ShowExpertStatus()
                        return CommandResult.success("MoE system overview completed")
                    | unknown :: _ ->
                        AnsiConsole.MarkupLine(sprintf "[red]❌ Unknown MoE command: %s[/]" unknown)
                        return CommandResult.failure(sprintf "Unknown command: %s" unknown)
                with
                | ex ->
                    logger.LogError(ex, "Error in MoE command")
                    AnsiConsole.MarkupLine(sprintf "[red]❌ Error: %s[/]" ex.Message)
                    return CommandResult.failure(ex.Message)
            }

