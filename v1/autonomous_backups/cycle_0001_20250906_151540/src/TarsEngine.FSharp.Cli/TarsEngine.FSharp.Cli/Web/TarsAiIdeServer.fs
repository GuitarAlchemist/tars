namespace TarsEngine.FSharp.Cli.Web

open System
open System.IO
open System.Text
open System.Text.Json
open System.Threading.Tasks
open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Hosting
open Microsoft.AspNetCore.Http
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Hosting
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Core.TarsAiIde
open TarsEngine.FSharp.Cli.Core.CudaComputationExpression

/// TARS AI IDE Web Server - Monaco Editor + AI Backend
module TarsAiIdeServer =
    
    /// API request/response types
    type CodeGenerationRequest = {
        Description: string
        Language: string
        Context: string option
    }
    
    type CodeGenerationResponse = {
        Success: bool
        GeneratedCode: string option
        Error: string option
        ExecutionTimeMs: float
        TokensGenerated: int
    }
    
    type CodeSuggestionRequest = {
        Code: string
        Language: string
        Line: int
        Column: int
        Context: string option
    }
    
    type CodeSuggestionResponse = {
        Success: bool
        Suggestions: CodeSuggestion list
        Error: string option
    }
    
    type DebugRequest = {
        Code: string
        Error: string
        Language: string
    }
    
    type DebugResponse = {
        Success: bool
        FixedCode: string option
        Explanation: string option
        Error: string option
    }
    
    type IdeStatusResponse = {
        Status: string
        SessionId: string
        Uptime: string
        ProjectCount: int
        GpuUtilization: float
        AvailableAgents: string list
    }
    
    /// TARS AI IDE Web API Controller
    type TarsAiIdeController(logger: ILogger<TarsAiIdeController>) =
        let ide = createAiIde logger
        let mutable currentSession = ide.StartSession()
        
        /// Generate code from natural language
        member _.GenerateCode(request: CodeGenerationRequest) =
            task {
                try
                    logger.LogInformation($"API: Generating {request.Language} code for: {request.Description}")
                    
                    let dsl = cuda (Some logger)
                    let! result = dsl.Run(TarsIdeOperations.generateCode ide request.Description request.Language)
                    
                    match result with
                    | Success code ->
                        return {
                            Success = true
                            GeneratedCode = Some code
                            Error = None
                            ExecutionTimeMs = 0.0
                            TokensGenerated = 200
                        }
                    | Error error ->
                        return {
                            Success = false
                            GeneratedCode = None
                            Error = Some error
                            ExecutionTimeMs = 0.0
                            TokensGenerated = 0
                        }
                with
                | ex ->
                    logger.LogError(ex, "Error generating code")
                    return {
                        Success = false
                        GeneratedCode = None
                        Error = Some ex.Message
                        ExecutionTimeMs = 0.0
                        TokensGenerated = 0
                    }
            }
        
        /// Get AI code suggestions
        member _.GetCodeSuggestions(request: CodeSuggestionRequest) =
            task {
                try
                    logger.LogInformation($"API: Getting code suggestions for {request.Language}")
                    
                    let dsl = cuda (Some logger)
                    let! result = dsl.Run(TarsIdeOperations.getCodeSuggestions ide request.Code request.Language request.Line request.Column)
                    
                    match result with
                    | Success suggestions ->
                        return {
                            Success = true
                            Suggestions = suggestions
                            Error = None
                        }
                    | Error error ->
                        return {
                            Success = false
                            Suggestions = []
                            Error = Some error
                        }
                with
                | ex ->
                    logger.LogError(ex, "Error getting code suggestions")
                    return {
                        Success = false
                        Suggestions = []
                        Error = Some ex.Message
                    }
            }
        
        /// Debug code using AI
        member _.DebugCode(request: DebugRequest) =
            task {
                try
                    logger.LogInformation($"API: Debugging {request.Language} code")
                    
                    let dsl = cuda (Some logger)
                    let! result = dsl.Run(TarsIdeOperations.debugCode ide request.Code request.Error request.Language)
                    
                    match result with
                    | Success fixedCode ->
                        return {
                            Success = true
                            FixedCode = Some fixedCode
                            Explanation = Some "AI-powered debugging completed"
                            Error = None
                        }
                    | Error error ->
                        return {
                            Success = false
                            FixedCode = None
                            Explanation = None
                            Error = Some error
                        }
                with
                | ex ->
                    logger.LogError(ex, "Error debugging code")
                    return {
                        Success = false
                        FixedCode = None
                        Explanation = None
                        Error = Some ex.Message
                    }
            }
        
        /// Get IDE status
        member _.GetStatus() =
            let uptime = DateTime.UtcNow - currentSession.StartTime
            {
                Status = ide.GetIdeStatus()
                SessionId = currentSession.SessionId
                Uptime = $"{uptime.TotalMinutes:F1} minutes"
                ProjectCount = 0
                GpuUtilization = currentSession.GpuUtilization
                AvailableAgents = ide.GetAvailableAgents()
            }
    
    /// Configure TARS AI IDE Web API
    let configureServices (services: IServiceCollection) =
        services.AddControllers() |> ignore
        services.AddCors(fun options ->
            options.AddDefaultPolicy(fun builder ->
                builder.AllowAnyOrigin()
                       .AllowAnyMethod()
                       .AllowAnyHeader() |> ignore
            )
        ) |> ignore
        services.AddSingleton<TarsAiIdeController>() |> ignore
        services.AddLogging() |> ignore
    
    /// Configure TARS AI IDE Web API pipeline
    let configureApp (app: IApplicationBuilder) =
        app.UseCors() |> ignore
        app.UseRouting() |> ignore
        app.UseEndpoints(fun endpoints ->
            // API endpoints for Monaco Editor integration
            endpoints.MapPost("/api/generate-code", Func<HttpContext, Task>(fun context ->
                task {
                    let controller = context.RequestServices.GetService<TarsAiIdeController>()
                    let! requestBody = context.Request.ReadFromJsonAsync<CodeGenerationRequest>()
                    let! response = controller.GenerateCode(requestBody)
                    do! context.Response.WriteAsJsonAsync(response)
                } :> Task
            )) |> ignore
            
            endpoints.MapPost("/api/code-suggestions", Func<HttpContext, Task>(fun context ->
                task {
                    let controller = context.RequestServices.GetService<TarsAiIdeController>()
                    let! requestBody = context.Request.ReadFromJsonAsync<CodeSuggestionRequest>()
                    let! response = controller.GetCodeSuggestions(requestBody)
                    do! context.Response.WriteAsJsonAsync(response)
                } :> Task
            )) |> ignore
            
            endpoints.MapPost("/api/debug-code", Func<HttpContext, Task>(fun context ->
                task {
                    let controller = context.RequestServices.GetService<TarsAiIdeController>()
                    let! requestBody = context.Request.ReadFromJsonAsync<DebugRequest>()
                    let! response = controller.DebugCode(requestBody)
                    do! context.Response.WriteAsJsonAsync(response)
                } :> Task
            )) |> ignore
            
            endpoints.MapGet("/api/status", Func<HttpContext, Task>(fun context ->
                task {
                    let controller = context.RequestServices.GetService<TarsAiIdeController>()
                    let response = controller.GetStatus()
                    do! context.Response.WriteAsJsonAsync(response)
                } :> Task
            )) |> ignore
            
            // Serve static files for Monaco Editor
            endpoints.MapGet("/", Func<HttpContext, Task>(fun context ->
                task {
                    let html = """
<!DOCTYPE html>
<html>
<head>
    <title>TARS AI IDE</title>
    <meta charset="utf-8">
    <style>
        body { margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .container { display: flex; height: 100vh; }
        .sidebar { width: 300px; background: #1e1e1e; color: white; padding: 20px; }
        .editor-container { flex: 1; display: flex; flex-direction: column; }
        .toolbar { height: 50px; background: #2d2d30; color: white; display: flex; align-items: center; padding: 0 20px; }
        .editor { flex: 1; }
        .ai-panel { height: 200px; background: #f5f5f5; border-top: 1px solid #ddd; padding: 20px; }
        .btn { background: #007acc; color: white; border: none; padding: 8px 16px; margin: 5px; cursor: pointer; border-radius: 4px; }
        .btn:hover { background: #005a9e; }
        .status { background: #252526; color: #cccccc; padding: 5px 20px; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>🤖 TARS AI IDE</h2>
            <h3>AI Assistant</h3>
            <button class="btn" onclick="generateCode()">Generate Code</button>
            <button class="btn" onclick="debugCode()">Debug Code</button>
            <button class="btn" onclick="optimizeCode()">Optimize Code</button>
            <button class="btn" onclick="explainCode()">Explain Code</button>
            <h3>Projects</h3>
            <div>📁 AI Demo Project</div>
            <div>📁 Smart Calculator</div>
            <h3>AI Agents</h3>
            <div>🤖 TARS-Coder (Active)</div>
            <div>🔍 TARS-Debugger</div>
            <div>⚡ TARS-Optimizer</div>
        </div>
        <div class="editor-container">
            <div class="toolbar">
                <span>🚀 TARS AI IDE - Monaco Editor + GPU-Accelerated AI</span>
            </div>
            <div id="editor" class="editor">
                <div id="loading" style="display: flex; align-items: center; justify-content: center; height: 100%; color: #cccccc; font-size: 18px;">
                    🚀 Loading Monaco Editor + AI...
                </div>
            </div>
            <div class="ai-panel">
                <h4>🤖 AI Assistant</h4>
                <div id="ai-output">Welcome to TARS AI IDE! Start typing or use the AI commands to generate code.</div>
            </div>
        </div>
    </div>
    <div class="status">
        <span id="status">Ready | GPU: Enabled | Session: Loading...</span>
    </div>

    <script src="https://unpkg.com/monaco-editor@latest/min/vs/loader.js"></script>
    <script>
        let editor;
        
        require.config({ paths: { vs: 'https://unpkg.com/monaco-editor@latest/min/vs' } });
        require(['vs/editor/editor.main'], function () {
            editor = monaco.editor.create(document.getElementById('editor'), {
                value: '// Welcome to TARS AI IDE!\\n// Start typing or use AI commands to generate code\\n\\nlet greet name = \\n    printfn "Hello, %s! Welcome to the AI-powered future!" name',
                language: 'fsharp',
                theme: 'vs-dark',
                automaticLayout: true,
                fontSize: 14,
                minimap: { enabled: true },
                suggestOnTriggerCharacters: true,
                quickSuggestions: true
            });
            
            // Load status
            updateStatus();
        });
        
        async function generateCode() {
            const description = prompt('Describe the code you want to generate:');
            if (!description) return;
            
            try {
                const response = await fetch('/api/generate-code', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        description: description,
                        language: 'fsharp',
                        context: editor.getValue()
                    })
                });
                
                const result = await response.json();
                if (result.success) {
                    editor.setValue(result.generatedCode);
                    document.getElementById('ai-output').innerHTML = `✅ Code generated successfully! (${result.tokensGenerated} tokens)`;
                } else {
                    document.getElementById('ai-output').innerHTML = `❌ Error: ${result.error}`;
                }
            } catch (error) {
                document.getElementById('ai-output').innerHTML = `❌ Error: ${error.message}`;
            }
        }
        
        async function debugCode() {
            const error = prompt('Describe the error or issue:');
            if (!error) return;
            
            try {
                const response = await fetch('/api/debug-code', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        code: editor.getValue(),
                        error: error,
                        language: 'fsharp'
                    })
                });
                
                const result = await response.json();
                if (result.success) {
                    editor.setValue(result.fixedCode);
                    document.getElementById('ai-output').innerHTML = `🔧 Code debugged: ${result.explanation}`;
                } else {
                    document.getElementById('ai-output').innerHTML = `❌ Error: ${result.error}`;
                }
            } catch (error) {
                document.getElementById('ai-output').innerHTML = `❌ Error: ${error.message}`;
            }
        }
        
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                document.getElementById('status').innerHTML = 
                    `${status.status} | Uptime: ${status.uptime} | Agents: ${status.availableAgents.length}`;
            } catch (error) {
                document.getElementById('status').innerHTML = 'Status: Error loading';
            }
        }
        
        // Update status every 30 seconds
        setInterval(updateStatus, 30000);
    </script>
</body>
</html>
"""
                    context.Response.ContentType <- "text/html"
                    do! context.Response.WriteAsync(html)
                } :> Task
            )) |> ignore
        ) |> ignore
    
    /// Start TARS AI IDE Web Server
    let startServer (port: int) (logger: ILogger) =
        task {
            logger.LogInformation($"Starting TARS AI IDE Web Server on port {port}")
            
            let builder = WebApplication.CreateBuilder()
            configureServices builder.Services
            
            let app = builder.Build()
            configureApp app
            
            logger.LogInformation("🚀 TARS AI IDE Server starting...")
            logger.LogInformation($"🌐 Open your browser to: http://localhost:{port}")
            logger.LogInformation("🤖 Monaco Editor + GPU-Accelerated AI ready!")
            
            do! app.RunAsync($"http://0.0.0.0:{port}")
        }
