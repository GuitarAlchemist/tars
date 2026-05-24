namespace TarsEngineFSharp

open System
open System.Net.Http
open System.Text
open System.Text.Json
open System.Threading.Tasks

module LlmService =
    type LlmConfig = {
        Endpoint: string
        Model: string
        MaxTokens: int
        Temperature: float
        TopP: float
    }

    type CompletionRequest = {
        model: string
        prompt: string
        max_tokens: int
        temperature: float
        top_p: float
    }

    type CompletionChoice = {
        text: string
        index: int
        logprobs: obj option
        finish_reason: string
    }

    type CompletionResponse = {
        id: string
        ``object``: string
        created: int64
        model: string
        choices: CompletionChoice[]
    }

    type FunctionParameter = {
        Name: string
        Description: string
        Required: bool
        Type: string
    }

    type Function = {
        Name: string
        Description: string
        Parameters: FunctionParameter list
        Handler: Map<string, obj> -> Async<string>
    }

    let processWithFunctions (message: string) (functions: Function list) = async {
        // TODO: Implement actual LLM logic here
        // For now, we'll just simulate processing the weather function
        if message.ToLower().Contains("weather") then
            let weatherFunction = functions |> List.find (fun f -> f.Name = "get_weather")
            let args = Map.ofList [("location", "London" :> obj)]
            let! response = weatherFunction.Handler args
            return response
        else
            return "I'm not sure how to help with that. Try asking about the weather!"
    }

    type LlmClient(config: LlmConfig) =
        let client = new HttpClient()
        
        member _.GenerateCompletion(prompt: string) = async {
            let request = {
                model = config.Model
                prompt = prompt
                max_tokens = config.MaxTokens
                temperature = config.Temperature
                top_p = config.TopP
            }
            
            let jsonContent = 
                JsonSerializer.Serialize(request)
                |> fun json -> new StringContent(json, Encoding.UTF8, "application/json")
            
            try
                let! response = 
                    client.PostAsync(
                        $"%s{config.Endpoint}/v1/completions",
                        jsonContent)
                    |> Async.AwaitTask
                
                let! content = 
                    response.Content.ReadAsStringAsync()
                    |> Async.AwaitTask
                
                let result = 
                    JsonSerializer.Deserialize<CompletionResponse>(content)
                
                return 
                    if result.choices.Length > 0 then
                        Ok result.choices.[0].text
                    else
                        Error "No completion generated"
            with ex ->
                return Error ex.Message
        }
        
        interface IDisposable with
            member _.Dispose() = 
                client.Dispose()

    // Create default client with vLLM configuration
    let createDefaultClient() =
        let config = {
            Endpoint = "http://localhost:8000"
            Model = "mistral-7b"
            MaxTokens = 100
            Temperature = 0.7
            TopP = 0.9
        }
        new LlmClient(config)