namespace TarsEngine.FSharp.Core

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Core.TarsInferenceIntegration

/// TARS Inference CLI Commands
module TarsInferenceCLI =

    /// CLI command types
    type TarsInferenceCommand =
        | Infer of prompt: string * maxTokens: int option * temperature: float option
        | Chat of message: string * context: string option
        | Generate of prompt: string * useCase: string option
        | Analyze of data: string
        | Reason of problem: string
        | Research of topic: string * agentType: string option
        | Diagnose of systemData: string
        | Interactive

    /// Parse CLI arguments into commands
    let parseInferenceCommand (args: string[]) : TarsInferenceCommand option =
        match args with
        | [| "infer"; prompt |] -> Some(Infer(prompt, None, None))
        | [| "infer"; prompt; maxTokens |] -> 
            match Int32.TryParse(maxTokens) with
            | true, tokens -> Some(Infer(prompt, Some(tokens), None))
            | false, _ -> Some(Infer(prompt, None, None))
        | [| "infer"; prompt; maxTokens; temperature |] ->
            match Int32.TryParse(maxTokens), Double.TryParse(temperature) with
            | (true, tokens), (true, temp) -> Some(Infer(prompt, Some(tokens), Some(temp)))
            | (true, tokens), _ -> Some(Infer(prompt, Some(tokens), None))
            | _ -> Some(Infer(prompt, None, None))
        
        | [| "chat"; message |] -> Some(Chat(message, None))
        | [| "chat"; message; context |] -> Some(Chat(message, Some(context)))
        
        | [| "generate"; prompt |] -> Some(Generate(prompt, None))
        | [| "generate"; prompt; useCase |] -> Some(Generate(prompt, Some(useCase)))
        
        | [| "analyze"; data |] -> Some(Analyze(data))
        | [| "reason"; problem |] -> Some(Reason(problem))
        
        | [| "research"; topic |] -> Some(Research(topic, None))
        | [| "research"; topic; agentType |] -> Some(Research(topic, Some(agentType)))
        
        | [| "diagnose"; systemData |] -> Some(Diagnose(systemData))
        | [| "interactive" |] -> Some(Interactive)
        
        | _ -> None

    /// Execute TARS inference command
    let executeInferenceCommand (command: TarsInferenceCommand) : Task<int> =
        task {
            try
                let inferenceService = getTarsInference()
                
                match command with
                | Infer(prompt, maxTokens, temperature) ->
                    printfn "🧠 TARS Inference"
                    printfn "================"
                    printfn "Prompt: %s" prompt
                    
                    let request = {
                        Prompt = prompt
                        MaxTokens = maxTokens |> Option.defaultValue 512
                        Temperature = temperature |> Option.defaultValue 0.7
                        Context = None
                        AgentType = None
                        UseCase = "general"
                    }
                    
                    let! result = inferenceService.InferAsync(request)
                    match result with
                    | Ok(response) ->
                        printfn ""
                        printfn "Response:"
                        printfn "%s" response.GeneratedText
                        printfn ""
                        printfn "Tokens: %d | Time: %dms | CUDA: %b | Confidence: %.1f%%" 
                            response.TokenCount response.InferenceTimeMs response.UsedCuda (response.Confidence * 100.0)
                        return 0
                    | Error(msg) ->
                        printfn "❌ Inference failed: %s" msg
                        return 1
                
                | Chat(message, context) ->
                    printfn "💬 TARS Chat"
                    printfn "============"
                    printfn "You: %s" message
                    
                    let! result = inferenceService.ChatAsync message (context |> Option.defaultValue "")
                    match result with
                    | Ok(response) ->
                        printfn "TARS: %s" response
                        return 0
                    | Error(msg) ->
                        printfn "❌ Chat failed: %s" msg
                        return 1
                
                | Generate(prompt, useCase) ->
                    printfn "✨ TARS Generate"
                    printfn "================"
                    printfn "Prompt: %s" prompt
                    printfn "Use Case: %s" (useCase |> Option.defaultValue "general")
                    
                    let request = {
                        Prompt = prompt
                        MaxTokens = 1024
                        Temperature = 0.8
                        Context = None
                        AgentType = None
                        UseCase = useCase |> Option.defaultValue "general"
                    }
                    
                    let! result = inferenceService.InferAsync(request)
                    match result with
                    | Ok(response) ->
                        printfn ""
                        printfn "Generated Content:"
                        printfn "%s" response.GeneratedText
                        return 0
                    | Error(msg) ->
                        printfn "❌ Generation failed: %s" msg
                        return 1
                
                | Analyze(data) ->
                    printfn "🔍 TARS Analysis"
                    printfn "================"
                    printfn "Analyzing data..."
                    
                    let! result = inferenceService.AnalyzeAsync(data)
                    match result with
                    | Ok(analysis) ->
                        printfn ""
                        printfn "Analysis Results:"
                        printfn "%s" analysis
                        return 0
                    | Error(msg) ->
                        printfn "❌ Analysis failed: %s" msg
                        return 1
                
                | Reason(problem) ->
                    printfn "🤔 TARS Reasoning"
                    printfn "================="
                    printfn "Problem: %s" problem
                    
                    let! result = inferenceService.ReasonAsync(problem)
                    match result with
                    | Ok(reasoning) ->
                        printfn ""
                        printfn "Reasoning:"
                        printfn "%s" reasoning
                        return 0
                    | Error(msg) ->
                        printfn "❌ Reasoning failed: %s" msg
                        return 1
                
                | Research(topic, agentType) ->
                    printfn "🔬 TARS Research"
                    printfn "================"
                    printfn "Topic: %s" topic
                    printfn "Agent Type: %s" (agentType |> Option.defaultValue "general")
                    
                    let researchService = TarsEnabledJanusResearchService(inferenceService)
                    let! result = researchService.CoordinateResearchWithTarsInference(topic)
                    
                    printfn ""
                    printfn "Research Results:"
                    printfn "%s" result
                    return 0
                
                | Diagnose(systemData) ->
                    printfn "🏥 TARS Diagnostics"
                    printfn "==================="
                    printfn "Diagnosing system..."
                    
                    let diagnosticService = TarsDiagnosticService(inferenceService)
                    let! result = diagnosticService.DiagnoseSystem(systemData)
                    
                    printfn ""
                    printfn "Diagnostic Results:"
                    printfn "%s" result
                    return 0
                
                | Interactive ->
                    return! runInteractiveMode(inferenceService)
                    
            with
            | ex ->
                printfn "💥 Command execution failed: %s" ex.Message
                return 1
        }

    /// Run interactive TARS inference mode
    and runInteractiveMode (inferenceService: ITarsInferenceService) : Task<int> =
        task {
            printfn "🤖 TARS Interactive Inference Mode"
            printfn "=================================="
            printfn "Type 'exit' to quit, 'help' for commands"
            printfn ""
            
            let mutable continue = true
            let mutable context = ""
            
            while continue do
                printf "TARS> "
                let input = Console.ReadLine()
                
                match input.ToLower().Trim() with
                | "exit" | "quit" -> 
                    continue <- false
                    printfn "Goodbye!"
                
                | "help" ->
                    printfn ""
                    printfn "Available commands:"
                    printfn "  chat <message>     - Chat with TARS"
                    printfn "  analyze <data>     - Analyze data"
                    printfn "  reason <problem>   - Reason about a problem"
                    printfn "  research <topic>   - Conduct research"
                    printfn "  diagnose <data>    - System diagnostics"
                    printfn "  clear              - Clear context"
                    printfn "  exit               - Exit interactive mode"
                    printfn ""
                
                | "clear" ->
                    context <- ""
                    printfn "Context cleared."
                
                | _ when input.StartsWith("chat ") ->
                    let message = input.Substring(5)
                    let! result = inferenceService.ChatAsync message context
                    match result with
                    | Ok(response) ->
                        printfn "TARS: %s" response
                        context <- context + $"\nUser: {message}\nTARS: {response}"
                    | Error(msg) ->
                        printfn "❌ Chat failed: %s" msg
                
                | _ when input.StartsWith("analyze ") ->
                    let data = input.Substring(8)
                    let! result = inferenceService.AnalyzeAsync(data)
                    match result with
                    | Ok(analysis) ->
                        printfn "Analysis: %s" analysis
                    | Error(msg) ->
                        printfn "❌ Analysis failed: %s" msg
                
                | _ when input.StartsWith("reason ") ->
                    let problem = input.Substring(7)
                    let! result = inferenceService.ReasonAsync(problem)
                    match result with
                    | Ok(reasoning) ->
                        printfn "Reasoning: %s" reasoning
                    | Error(msg) ->
                        printfn "❌ Reasoning failed: %s" msg
                
                | _ when input.StartsWith("research ") ->
                    let topic = input.Substring(9)
                    let researchService = TarsEnabledJanusResearchService(inferenceService)
                    let! result = researchService.CoordinateResearchWithTarsInference(topic)
                    printfn "Research: %s" result
                
                | _ when input.StartsWith("diagnose ") ->
                    let systemData = input.Substring(9)
                    let diagnosticService = TarsDiagnosticService(inferenceService)
                    let! result = diagnosticService.DiagnoseSystem(systemData)
                    printfn "Diagnosis: %s" result
                
                | _ when not (String.IsNullOrWhiteSpace(input)) ->
                    // Default to chat
                    let! result = inferenceService.ChatAsync input context
                    match result with
                    | Ok(response) ->
                        printfn "TARS: %s" response
                        context <- context + $"\nUser: {input}\nTARS: {response}"
                    | Error(msg) ->
                        printfn "❌ Chat failed: %s" msg
                
                | _ -> ()
            
            return 0
        }

    /// Show TARS inference help
    let showInferenceHelp () =
        printfn "🧠 TARS Inference Engine Commands"
        printfn "=================================="
        printfn ""
        printfn "Usage: tars <command> [options]"
        printfn ""
        printfn "Commands:"
        printfn "  infer <prompt> [maxTokens] [temperature]  - General inference"
        printfn "  chat <message> [context]                  - Chat with TARS"
        printfn "  generate <prompt> [useCase]               - Generate content"
        printfn "  analyze <data>                            - Analyze data"
        printfn "  reason <problem>                          - Reason about problem"
        printfn "  research <topic> [agentType]              - Conduct research"
        printfn "  diagnose <systemData>                     - System diagnostics"
        printfn "  interactive                               - Interactive mode"
        printfn ""
        printfn "Examples:"
        printfn "  tars infer \"Explain quantum computing\""
        printfn "  tars chat \"Hello TARS\" \"Previous conversation context\""
        printfn "  tars generate \"Write a story about AI\" \"creative\""
        printfn "  tars analyze \"CPU: 85%, Memory: 2.1GB, Disk: 45%\""
        printfn "  tars reason \"How to optimize database performance\""
        printfn "  tars research \"Janus cosmological model\" \"cosmologist\""
        printfn "  tars diagnose \"System logs and metrics data\""
        printfn "  tars interactive"
        printfn ""
        printfn "Agent Types for Research:"
        printfn "  cosmologist, data-scientist, mathematician, research-director"
        printfn ""
        printfn "Use Cases for Generate:"
        printfn "  creative, technical, research, analysis, documentation"
