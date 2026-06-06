namespace TarsEngine

open System
open System.IO
open System.Net
open System.Text
open System.Text.Json
open System.Threading.Tasks
open TarsEngine.TarsProductionAiEngine
open TarsEngine.TarsModelLoader

/// TARS REST API Server - Ollama-compatible API server
module TarsApiServer =
    
    // ============================================================================
    // API TYPES (OLLAMA COMPATIBLE)
    // ============================================================================
    
    type GenerateRequest = {
        model: string
        prompt: string
        stream: bool option
        raw: bool option
        format: string option
        options: GenerateOptions option
    }
    
    and GenerateOptions = {
        temperature: float32 option
        top_p: float32 option
        top_k: int option
        repeat_penalty: float32 option
        seed: int option
        num_predict: int option
        stop: string[] option
    }
    
    type GenerateResponse = {
        model: string
        created_at: string
        response: string
        ``done``: bool
        context: int[] option
        total_duration: int64 option
        load_duration: int64 option
        prompt_eval_count: int option
        prompt_eval_duration: int64 option
        eval_count: int option
        eval_duration: int64 option
    }
    
    type ChatMessage = {
        role: string
        content: string
    }
    
    type ChatRequest = {
        model: string
        messages: ChatMessage[]
        stream: bool option
        options: GenerateOptions option
    }
    
    type ChatResponse = {
        model: string
        created_at: string
        message: ChatMessage
        ``done``: bool
        total_duration: int64 option
        load_duration: int64 option
        prompt_eval_count: int option
        prompt_eval_duration: int64 option
        eval_count: int option
        eval_duration: int64 option
    }
    
    type ModelInfo = {
        name: string
        modified_at: string
        size: int64
        digest: string
        details: ModelDetails
    }
    
    and ModelDetails = {
        format: string
        family: string
        families: string[] option
        parameter_size: string
        quantization_level: string
    }
    
    type ListModelsResponse = {
        models: ModelInfo[]
    }
    
    type ShowModelRequest = {
        name: string
    }
    
    type ShowModelResponse = {
        license: string option
        modelfile: string option
        parameters: string option
        template: string option
        details: ModelDetails
    }
    
    // ============================================================================
    // TARS API SERVER
    // ============================================================================
    
    type TarsApiServer(port: int, aiEngine: TarsProductionAiEngine) =
        let listener = new HttpListener()
        let mutable isRunning = false
        let loadedModels = System.Collections.Concurrent.ConcurrentDictionary<string, LoadedModel>()
        
        /// JSON serialization options
        let jsonOptions = JsonSerializerOptions()
        do jsonOptions.PropertyNamingPolicy <- JsonNamingPolicy.CamelCase
        do jsonOptions.WriteIndented <- true
        
        /// Serialize object to JSON
        let toJson (obj: 'T) = JsonSerializer.Serialize(obj, jsonOptions)
        
        /// Deserialize JSON to object
        let fromJson<'T> (json: string) = JsonSerializer.Deserialize<'T>(json, jsonOptions)
        
        /// Send JSON response
        let sendJsonResponse (response: HttpListenerResponse) (statusCode: HttpStatusCode) (data: 'T) =
            let json = toJson data
            let buffer = Encoding.UTF8.GetBytes(json)
            
            response.StatusCode <- int statusCode
            response.ContentType <- "application/json"
            response.ContentLength64 <- int64 buffer.Length
            response.Headers.Add("Access-Control-Allow-Origin", "*")
            response.Headers.Add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            response.Headers.Add("Access-Control-Allow-Headers", "Content-Type")
            
            response.OutputStream.Write(buffer, 0, buffer.Length)
            response.OutputStream.Close()
        
        /// Send error response
        let sendErrorResponse (response: HttpListenerResponse) (statusCode: HttpStatusCode) (message: string) =
            let errorObj = {| error = message |}
            sendJsonResponse response statusCode errorObj
        
        /// Read request body
        let readRequestBody (request: HttpListenerRequest) = async {
            use reader = new StreamReader(request.InputStream)
            return! reader.ReadToEndAsync() |> Async.AwaitTask
        }
        
        /// Handle /api/generate endpoint
        let handleGenerate (request: HttpListenerRequest) (response: HttpListenerResponse) = async {
            try
                let! body = readRequestBody request
                let generateReq = fromJson<GenerateRequest> body
                
                printfn $"🤖 Generate request for model: {generateReq.model}"
                printfn $"📝 Prompt: \"{generateReq.prompt.[..Math.Min(50, generateReq.prompt.Length-1)]}...\""
                
                let startTime = DateTime.UtcNow
                
                // Convert to TARS API request
                let apiRequest = {
                    Model = generateReq.model
                    Prompt = generateReq.prompt
                    MaxTokens = generateReq.options |> Option.bind (fun o -> o.num_predict)
                    Temperature = generateReq.options |> Option.bind (fun o -> o.temperature)
                    TopP = generateReq.options |> Option.bind (fun o -> o.top_p)
                    TopK = generateReq.options |> Option.bind (fun o -> o.top_k)
                    Stop = generateReq.options |> Option.bind (fun o -> o.stop)
                    Stream = generateReq.stream
                    Seed = generateReq.options |> Option.bind (fun o -> o.seed)
                }
                
                // Process with TARS AI engine
                let! aiResponse = aiEngine.ProcessApiRequest(apiRequest)
                
                let endTime = DateTime.UtcNow
                let totalDuration = int64 ((endTime - startTime).TotalNanoseconds)
                
                // Convert to Ollama format
                let ollamaResponse = {
                    model = generateReq.model
                    created_at = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                    response = aiResponse.Choices.[0].Text
                    ``done`` = true
                    context = None
                    total_duration = Some totalDuration
                    load_duration = Some 1000000L // 1ms in nanoseconds
                    prompt_eval_count = Some aiResponse.Usage.PromptTokens
                    prompt_eval_duration = Some (totalDuration / 2L)
                    eval_count = Some aiResponse.Usage.CompletionTokens
                    eval_duration = Some (totalDuration / 2L)
                }
                
                sendJsonResponse response HttpStatusCode.OK ollamaResponse
                
                printfn $"✅ Generated {aiResponse.Usage.CompletionTokens} tokens"
                
            with
            | ex ->
                printfn $"❌ Generate error: {ex.Message}"
                sendErrorResponse response HttpStatusCode.InternalServerError ex.Message
        }
        
        /// Handle /api/chat endpoint
        let handleChat (request: HttpListenerRequest) (response: HttpListenerResponse) = async {
            try
                let! body = readRequestBody request
                let chatReq = fromJson<ChatRequest> body
                
                printfn $"💬 Chat request for model: {chatReq.model}"
                
                // Convert messages to prompt
                let prompt = 
                    chatReq.messages 
                    |> Array.map (fun msg -> $"{msg.role}: {msg.content}")
                    |> String.concat "\n"
                
                let startTime = DateTime.UtcNow
                
                // Convert to TARS API request
                let apiRequest = {
                    Model = chatReq.model
                    Prompt = prompt
                    MaxTokens = chatReq.options |> Option.bind (fun o -> o.num_predict)
                    Temperature = chatReq.options |> Option.bind (fun o -> o.temperature)
                    TopP = chatReq.options |> Option.bind (fun o -> o.top_p)
                    TopK = chatReq.options |> Option.bind (fun o -> o.top_k)
                    Stop = chatReq.options |> Option.bind (fun o -> o.stop)
                    Stream = chatReq.stream
                    Seed = chatReq.options |> Option.bind (fun o -> o.seed)
                }
                
                // Process with TARS AI engine
                let! aiResponse = aiEngine.ProcessApiRequest(apiRequest)
                
                let endTime = DateTime.UtcNow
                let totalDuration = int64 ((endTime - startTime).TotalNanoseconds)
                
                // Convert to Ollama chat format
                let chatResponse = {
                    model = chatReq.model
                    created_at = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                    message = { role = "assistant"; content = aiResponse.Choices.[0].Text }
                    ``done`` = true
                    total_duration = Some totalDuration
                    load_duration = Some 1000000L
                    prompt_eval_count = Some aiResponse.Usage.PromptTokens
                    prompt_eval_duration = Some (totalDuration / 2L)
                    eval_count = Some aiResponse.Usage.CompletionTokens
                    eval_duration = Some (totalDuration / 2L)
                }
                
                sendJsonResponse response HttpStatusCode.OK chatResponse
                
                printfn $"✅ Chat response generated"
                
            with
            | ex ->
                printfn $"❌ Chat error: {ex.Message}"
                sendErrorResponse response HttpStatusCode.InternalServerError ex.Message
        }
        
        /// Handle /api/tags endpoint (list models)
        let handleListModels (response: HttpListenerResponse) = async {
            try
                let models = [|
                    {
                        name = "tars-tiny-1b"
                        modified_at = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                        size = 1_000_000_000L
                        digest = "sha256:tars-tiny-1b"
                        details = {
                            format = "tars"
                            family = "llama"
                            families = Some [| "llama" |]
                            parameter_size = "1B"
                            quantization_level = "F32"
                        }
                    }
                    {
                        name = "tars-small-3b"
                        modified_at = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                        size = 3_000_000_000L
                        digest = "sha256:tars-small-3b"
                        details = {
                            format = "tars"
                            family = "llama"
                            families = Some [| "llama" |]
                            parameter_size = "3B"
                            quantization_level = "F32"
                        }
                    }
                    {
                        name = "tars-medium-7b"
                        modified_at = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                        size = 7_000_000_000L
                        digest = "sha256:tars-medium-7b"
                        details = {
                            format = "tars"
                            family = "llama"
                            families = Some [| "llama" |]
                            parameter_size = "7B"
                            quantization_level = "F32"
                        }
                    }
                |]
                
                let listResponse = { models = models }
                sendJsonResponse response HttpStatusCode.OK listResponse
                
                printfn $"📋 Listed {models.Length} available models"
                
            with
            | ex ->
                printfn $"❌ List models error: {ex.Message}"
                sendErrorResponse response HttpStatusCode.InternalServerError ex.Message
        }
        
        /// Handle /api/show endpoint
        let handleShowModel (request: HttpListenerRequest) (response: HttpListenerResponse) = async {
            try
                let! body = readRequestBody request
                let showReq = fromJson<ShowModelRequest> body
                
                let showResponse = {
                    license = Some "Apache 2.0"
                    modelfile = Some $"FROM {showReq.name}\nPARAMETER temperature 0.7"
                    parameters = Some "temperature 0.7\ntop_p 0.9\ntop_k 40"
                    template = Some "{{ .Prompt }}"
                    details = {
                        format = "tars"
                        family = "llama"
                        families = Some [| "llama" |]
                        parameter_size = "7B"
                        quantization_level = "F32"
                    }
                }
                
                sendJsonResponse response HttpStatusCode.OK showResponse
                
                printfn $"📊 Showed model info for: {showReq.name}"
                
            with
            | ex ->
                printfn $"❌ Show model error: {ex.Message}"
                sendErrorResponse response HttpStatusCode.InternalServerError ex.Message
        }
        
        /// Handle CORS preflight requests
        let handleOptions (response: HttpListenerResponse) =
            response.StatusCode <- 200
            response.Headers.Add("Access-Control-Allow-Origin", "*")
            response.Headers.Add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            response.Headers.Add("Access-Control-Allow-Headers", "Content-Type")
            response.Close()
        
        /// Process HTTP request
        let processRequest (context: HttpListenerContext) = async {
            let request = context.Request
            let response = context.Response
            
            try
                printfn $"📡 {request.HttpMethod} {request.Url.AbsolutePath}"
                
                match request.HttpMethod, request.Url.AbsolutePath with
                | "OPTIONS", _ -> 
                    handleOptions response
                | "POST", "/api/generate" -> 
                    do! handleGenerate request response
                | "POST", "/api/chat" -> 
                    do! handleChat request response
                | "GET", "/api/tags" -> 
                    do! handleListModels response
                | "POST", "/api/show" -> 
                    do! handleShowModel request response
                | "GET", "/" -> 
                    let html = """
                    <!DOCTYPE html>
                    <html>
                    <head><title>TARS AI Server</title></head>
                    <body>
                        <h1>🚀 TARS AI Server</h1>
                        <p>Ollama-compatible API server running!</p>
                        <h2>Available Endpoints:</h2>
                        <ul>
                            <li>POST /api/generate - Generate text</li>
                            <li>POST /api/chat - Chat completion</li>
                            <li>GET /api/tags - List models</li>
                            <li>POST /api/show - Show model info</li>
                        </ul>
                        <p>🌟 TARS AI: The next generation of AI inference!</p>
                    </body>
                    </html>
                    """
                    let buffer = Encoding.UTF8.GetBytes(html)
                    response.ContentType <- "text/html"
                    response.ContentLength64 <- int64 buffer.Length
                    response.OutputStream.Write(buffer, 0, buffer.Length)
                    response.Close()
                | _ -> 
                    sendErrorResponse response HttpStatusCode.NotFound "Endpoint not found"
                    
            with
            | ex ->
                printfn $"❌ Request processing error: {ex.Message}"
                try
                    sendErrorResponse response HttpStatusCode.InternalServerError ex.Message
                with
                | _ -> response.Close()
        }
        
        /// Start the API server
        member _.Start() = async {
            listener.Prefixes.Add($"http://localhost:{port}/")
            listener.Prefixes.Add($"http://127.0.0.1:{port}/")
            listener.Prefixes.Add($"http://+:{port}/")
            
            listener.Start()
            isRunning <- true
            
            printfn ""
            printfn "🚀 TARS API Server Started!"
            printfn "=========================="
            printfn $"🌐 Listening on port: {port}"
            printfn $"📡 Base URL: http://localhost:{port}"
            printfn $"🔗 Web UI: http://localhost:{port}"
            printfn $"📋 Models: http://localhost:{port}/api/tags"
            printfn ""
            printfn "🔥 Ollama-compatible endpoints:"
            printfn $"   POST http://localhost:{port}/api/generate"
            printfn $"   POST http://localhost:{port}/api/chat"
            printfn $"   GET  http://localhost:{port}/api/tags"
            printfn $"   POST http://localhost:{port}/api/show"
            printfn ""
            printfn "💡 Use with any Ollama client!"
            printfn "   curl -X POST http://localhost:11434/api/generate \\"
            printfn "        -H \"Content-Type: application/json\" \\"
            printfn "        -d '{\"model\":\"tars-medium-7b\",\"prompt\":\"Hello!\"}'"
            printfn ""
            
            // Process requests in background
            while isRunning do
                try
                    let! context = listener.GetContextAsync() |> Async.AwaitTask
                    Async.Start(processRequest context)
                with
                | ex when isRunning ->
                    printfn $"❌ Server error: {ex.Message}"
                | _ -> ()
        }
        
        /// Stop the API server
        member _.Stop() =
            isRunning <- false
            listener.Stop()
            listener.Close()
            printfn "🛑 TARS API Server stopped"
        
        interface IDisposable with
            member this.Dispose() = this.Stop()
