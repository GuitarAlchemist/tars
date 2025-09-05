namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Spectre.Console
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Commands

/// LLM integration module for the chatbot
module ChatbotLLM =

    /// Send a request to the LLM service
    let sendLLMRequest (llmService: IUnifiedLLMService) (request: ChatbotLLMRequest) : Task<ChatbotLLMResponse> =
        task {
            try
                let llmRequest = {
                    Model = request.Model
                    Prompt = request.Prompt
                    SystemPrompt = request.SystemPrompt
                    Temperature = request.Temperature
                    MaxTokens = request.MaxTokens
                    Context = request.Context
                }

                let! response = llmService.SendRequest(llmRequest)

                if response.Success then
                    return ResultUtils.llmSuccess response.Content request.Model
                else
                    return ResultUtils.llmFailure (response.Error |> Option.defaultValue "Unknown error") request.Model
            with
            | ex ->
                return ResultUtils.llmFailure ex.Message request.Model
        }

    /// Get available models from the LLM service
    let getAvailableModels (llmService: IUnifiedLLMService) : Task<Result<string list, string>> =
        task {
            try
                let! models = llmService.GetAvailableModels()
                return Ok models
            with
            | ex ->
                return Error ex.Message
        }

    /// Process a chat message using LLM
    let processChatMessage (llmService: IUnifiedLLMService) (message: string) (session: ChatbotSession) : Task<ChatbotResult> =
        task {
            try
                // Get available models
                let! modelsResult = getAvailableModels llmService

                match modelsResult with
                | Ok models when models.Length > 0 ->
                    // Use the first available model or the session's current model
                    let model = 
                        match session.CurrentModel with
                        | Some currentModel when List.contains currentModel models -> currentModel
                        | _ -> models.[0]

                    AnsiConsole.MarkupLine($"[green]🤖 Using model: {model}[/]")

                    let request = {
                        Model = model
                        Prompt = message
                        SystemPrompt = Some "You are TARS, a helpful AI assistant. Provide thoughtful and concise responses."
                        Temperature = Some 0.7
                        MaxTokens = Some 300
                        Context = None
                    }

                    let! response = sendLLMRequest llmService request

                    if response.Success then
                        let responsePanel = Panel(response.Content)
                        responsePanel.Header <- PanelHeader($"[bold green]🤖 {model} Response[/]")
                        responsePanel.Border <- BoxBorder.Rounded
                        responsePanel.BorderStyle <- Style.Parse("green")
                        AnsiConsole.Write(responsePanel)
                        AnsiConsole.MarkupLine("[green]✅ Real LLM response generated[/]")

                        let updatedSession = 
                            session
                            |> ChatbotSessionUtils.addToHistory message response.Content
                            |> ChatbotSessionUtils.setModel model

                        return ResultUtils.success "LLM response generated successfully" (Some updatedSession)
                    else
                        let errorMsg = response.Error |> Option.defaultValue "Unknown error"
                        AnsiConsole.MarkupLine($"[red]❌ LLM Error: {errorMsg}[/]")
                        AnsiConsole.MarkupLine("[yellow]💡 Try downloading a model with 'tars transformer download'[/]")
                        return ResultUtils.failure "LLM request failed"

                | Ok _ ->
                    // No models available - use single statement to avoid FS0597
                    AnsiConsole.MarkupLine("[yellow]⚠️ No LLM models available. Use 'tars transformer download' to get models[/]")
                    return ResultUtils.failure "No LLM models available"

                | Error err ->
                    // Error getting models - use single statement to avoid FS0597
                    AnsiConsole.MarkupLine($"[red]❌ Failed to check available models: {err}. Make sure Ollama is running[/]")
                    return ResultUtils.failure "Failed to check available models"

            with
            | :? System.InvalidCastException ->
                AnsiConsole.MarkupLine("[red]❌ LLM service configuration error[/]")
                return ResultUtils.failure "LLM service configuration error"
            | ex ->
                AnsiConsole.MarkupLine($"[red]❌ Unexpected error: {ex.Message}[/]")
                return ResultUtils.failure $"Unexpected error: {ex.Message}"
        }

    /// Check if LLM service is available
    let checkLLMAvailability (llmService: IUnifiedLLMService) : Task<bool> =
        task {
            try
                let! modelsResult = getAvailableModels llmService
                match modelsResult with
                | Ok models -> return models.Length > 0
                | Error _ -> return false
            with
            | _ -> return false
        }

    /// Get LLM service status
    let getLLMStatus (llmService: IUnifiedLLMService) : Task<string> =
        task {
            try
                let! isAvailable = checkLLMAvailability llmService
                if isAvailable then
                    let! modelsResult = getAvailableModels llmService
                    match modelsResult with
                    | Ok models ->
                        let modelList = String.Join(", ", models)
                        return $"✅ LLM Service Available - {models.Length} models: {modelList}"
                    | Error err ->
                        return $"⚠️ LLM Service Error: {err}"
                else
                    return "❌ LLM Service Unavailable"
            with
            | ex ->
                return $"❌ LLM Service Error: {ex.Message}"
        }

    /// Create a simple LLM request for testing
    let createSimpleRequest (model: string) (prompt: string) : ChatbotLLMRequest =
        {
            Model = model
            Prompt = prompt
            SystemPrompt = Some "You are TARS, a helpful AI assistant."
            Temperature = Some 0.7
            MaxTokens = Some 150
            Context = None
        }

    /// Display LLM response in a formatted panel
    let displayLLMResponse (response: ChatbotLLMResponse) : unit =
        if response.Success then
            let responsePanel = Panel(response.Content)
            responsePanel.Header <- PanelHeader($"[bold green]🤖 {response.Model} Response[/]")
            responsePanel.Border <- BoxBorder.Rounded
            responsePanel.BorderStyle <- Style.Parse("green")
            AnsiConsole.Write(responsePanel)
        else
            let errorMessage = response.Error |> Option.defaultValue "Unknown error"
            AnsiConsole.MarkupLine($"[red]❌ LLM Error: {errorMessage}[/]")

    /// Process multiple messages in batch
    let processBatchMessages (llmService: IUnifiedLLMService) (messages: string list) (session: ChatbotSession) : Task<ChatbotResult list> =
        task {
            let mutable currentSession = session
            let results = ResizeArray<ChatbotResult>()

            for message in messages do
                let! result = processChatMessage llmService message currentSession
                results.Add(result)
                
                // Update session if successful
                match result.Session with
                | Some updatedSession -> currentSession <- updatedSession
                | None -> ()

            return results |> List.ofSeq
        }
