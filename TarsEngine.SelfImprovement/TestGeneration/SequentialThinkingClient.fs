namespace TarsEngine.SelfImprovement.TestGeneration

open System
open System.Net.Http
open System.Text
open System.Text.Json
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.SelfImprovement.Common

/// Request model for the Sequential Thinking Server
type SequentialStepRequest = {
    /// The current step in the reasoning process
    step: string
    /// The context for the current step
    context: string list
}

/// Response model from the Sequential Thinking Server
type SequentialStepResponse = {
    /// The next step in the reasoning process
    next_step: string
    /// The updated context after the current step
    updated_context: string list
}

/// Configuration options for the Sequential Thinking client
type SequentialThinkingOptions = {
    /// The base URL of the Sequential Thinking Server
    BaseUrl: string
    /// The timeout for requests to the Sequential Thinking Server in seconds
    TimeoutSeconds: int
    /// The maximum number of retries for failed requests
    MaxRetries: int
    /// The delay between retries in milliseconds
    RetryDelayMilliseconds: int
}

/// Default Sequential Thinking options
module SequentialThinkingDefaults =
    let options = {
        BaseUrl = "http://localhost:3000"
        TimeoutSeconds = 30
        MaxRetries = 3
        RetryDelayMilliseconds = 1000
    }

/// Interface for the Sequential Thinking client
type ISequentialThinkingClient =
    /// Execute a step in the Sequential Thinking process
    abstract member ExecuteStep: step: string * context: string list -> Task<Result<SequentialStepResponse, string>>
    
    /// Execute a cognitive loop with multiple steps
    abstract member ExecuteCognitiveLoop: initialStep: string * initialContext: string list * maxDepth: int -> Task<Result<string list * string list, string>>

/// Implementation of the Sequential Thinking client
type SequentialThinkingClient(logger: ILogger<SequentialThinkingClient>, options: SequentialThinkingOptions) =
    
    let httpClient = new HttpClient()
    
    do
        httpClient.BaseAddress <- Uri(options.BaseUrl)
        httpClient.Timeout <- TimeSpan.FromSeconds(float options.TimeoutSeconds)
    
    let serializeRequest (request: SequentialStepRequest) =
        JsonSerializer.Serialize(request)
    
    let deserializeResponse (responseContent: string) =
        try
            let response = JsonSerializer.Deserialize<SequentialStepResponse>(responseContent)
            Ok response
        with
        | ex ->
            logger.LogError(ex, "SequentialThinkingClient: Error deserializing response: {ResponseContent}", responseContent)
            Error $"Error deserializing response: {ex.Message}"
    
    let rec executeWithRetry (action: unit -> Task<Result<'T, string>>) (retryCount: int) =
        task {
            try
                let! result = action()
                return result
            with
            | ex ->
                if retryCount > 0 then
                    logger.LogWarning(ex, "SequentialThinkingClient: Request failed, retrying ({RetryCount} attempts left)", retryCount)
                    do! Task.Delay(options.RetryDelayMilliseconds)
                    return! executeWithRetry action (retryCount - 1)
                else
                    logger.LogError(ex, "SequentialThinkingClient: Request failed after all retries")
                    return Error $"Request failed after all retries: {ex.Message}"
        }
    
    interface ISequentialThinkingClient with
        member this.ExecuteStep(step, context) =
            let executeStep() =
                task {
                    try
                        let request = {
                            step = step
                            context = context
                        }
                        
                        let json = serializeRequest request
                        let content = new StringContent(json, Encoding.UTF8, "application/json")
                        
                        logger.LogInformation("SequentialThinkingClient: Executing step: {Step}", step)
                        let! response = httpClient.PostAsync("/step", content)
                        
                        if response.IsSuccessStatusCode then
                            let! responseContent = response.Content.ReadAsStringAsync()
                            logger.LogInformation("SequentialThinkingClient: Step executed successfully")
                            return deserializeResponse responseContent
                        else
                            let! errorContent = response.Content.ReadAsStringAsync()
                            logger.LogError("SequentialThinkingClient: Error executing step: {StatusCode} - {ErrorContent}", response.StatusCode, errorContent)
                            return Error $"Error executing step: {response.StatusCode} - {errorContent}"
                    with
                    | ex ->
                        logger.LogError(ex, "SequentialThinkingClient: Error executing step")
                        return Error $"Error executing step: {ex.Message}"
                }
            
            executeWithRetry executeStep options.MaxRetries
        
        member this.ExecuteCognitiveLoop(initialStep, initialContext, maxDepth) =
            let rec loop currentStep context depth steps =
                task {
                    if depth = 0 then
                        return Ok (List.rev steps, context)
                    else
                        let! result = (this :> ISequentialThinkingClient).ExecuteStep(currentStep, context)
                        
                        match result with
                        | Ok response ->
                            logger.LogInformation("SequentialThinkingClient: Cognitive loop step {Depth}: {Step} -> {NextStep}", maxDepth - depth + 1, currentStep, response.next_step)
                            return! loop response.next_step response.updated_context (depth - 1) (response.next_step :: steps)
                        | Error error ->
                            logger.LogError("SequentialThinkingClient: Error in cognitive loop: {Error}", error)
                            return Error $"Error in cognitive loop: {error}"
                }
            
            task {
                logger.LogInformation("SequentialThinkingClient: Starting cognitive loop with initial step: {InitialStep}", initialStep)
                return! loop initialStep initialContext maxDepth [initialStep]
            }
    
    interface IDisposable with
        member this.Dispose() =
            httpClient.Dispose()

/// Factory for creating Sequential Thinking clients
type SequentialThinkingClientFactory(loggerFactory: ILoggerFactory) =
    
    /// Create a new Sequential Thinking client with the specified options
    member this.CreateSequentialThinkingClient(options: SequentialThinkingOptions) =
        let logger = loggerFactory.CreateLogger<SequentialThinkingClient>()
        new SequentialThinkingClient(logger, options)
    
    /// Create a new Sequential Thinking client with default options
    member this.CreateSequentialThinkingClient() =
        this.CreateSequentialThinkingClient(SequentialThinkingDefaults.options)
