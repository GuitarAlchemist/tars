namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open Spectre.Console
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.AI.HuggingFaceTypes
open TarsEngine.FSharp.Cli.AI.HuggingFaceInferenceTypes
open TarsEngine.FSharp.Cli.AI.HuggingFaceModelLoader
open TarsEngine.FSharp.Cli.AI.SimpleHuggingFaceInference
open Microsoft.Extensions.Logging

/// Hugging Face CUDA Command - Demonstrates HF model integration with CUDA
module HuggingFaceCudaCommand =
    
    /// Command options
    type HuggingFaceCudaOptions = {
        ListModels: bool
        DownloadModel: string option
        LoadModel: string option
        GenerateText: string option
        ClassifyText: string option
        GetEmbeddings: string option
        AnswerQuestion: (string * string) option // (question, context)
        ShowCapabilities: bool
        Verbose: bool
    }
    
    /// Parse command arguments
    let parseArguments (args: string[]) : HuggingFaceCudaOptions =
        let mutable options = {
            ListModels = false
            DownloadModel = None
            LoadModel = None
            GenerateText = None
            ClassifyText = None
            GetEmbeddings = None
            AnswerQuestion = None
            ShowCapabilities = false
            Verbose = false
        }
        
        let mutable i = 0
        while i < args.Length do
            let normalizedArg = args.[i].ToLower()
            match normalizedArg with
            | "--list-models" | "--list" | "--list-models=true" | "--list=true" -> 
                options <- { options with ListModels = true }
            | "--capabilities" | "--capabilities=true" -> 
                options <- { options with ShowCapabilities = true }
            | "--verbose" | "--verbose=true" | "-v" -> 
                options <- { options with Verbose = true }
            | arg when arg.StartsWith("--download=") -> 
                options <- { options with DownloadModel = Some (arg.Substring(11)) }
            | arg when arg.StartsWith("--load=") -> 
                options <- { options with LoadModel = Some (arg.Substring(7)) }
            | arg when arg.StartsWith("--generate=") -> 
                options <- { options with GenerateText = Some (arg.Substring(11)) }
            | arg when arg.StartsWith("--classify=") -> 
                options <- { options with ClassifyText = Some (arg.Substring(11)) }
            | arg when arg.StartsWith("--embeddings=") -> 
                options <- { options with GetEmbeddings = Some (arg.Substring(13)) }
            | _ -> ()
            i <- i + 1
        
        options
    
    /// Show popular models for different tasks
    let showPopularModels (console: IAnsiConsole) (modelLoader: HuggingFaceModelLoader) =
        console.MarkupLine("[bold blue]🤖 Popular Hugging Face Models[/]")
        console.WriteLine()
        
        let tasks = [
            ("Text Generation", "text-generation")
            ("Text Classification", "text-classification")
            ("Question Answering", "question-answering")
            ("Sentence Similarity", "sentence-similarity")
            ("Summarization", "summarization")
        ]
        
        for (taskName, taskId) in tasks do
            console.MarkupLine($"[bold yellow]{taskName}:[/]")
            let models = modelLoader.GetPopularModels(taskId)
            for model in models do
                console.MarkupLine($"  • [cyan]{model}[/]")
            console.WriteLine()
    
    /// Download and load a model
    let downloadAndLoadModel (console: IAnsiConsole) (logger: ILogger) (modelLoader: HuggingFaceModelLoader) (modelId: string) =
        task {
            console.MarkupLine($"[bold yellow]📦 Processing model: {modelId}[/]")
            console.WriteLine()
            
            // Get model info
            console.MarkupLine("🔍 [yellow]Fetching model information...[/]")
            let! modelInfoResult = modelLoader.GetModelInfoAsync(modelId, System.Threading.CancellationToken.None)
            
            match modelInfoResult with
            | Success (modelInfo, _) ->
                // Display model info
                let table = Table()
                table.AddColumn("Property") |> ignore
                table.AddColumn("Value") |> ignore
                
                table.AddRow("Model ID", modelInfo.ModelId) |> ignore
                table.AddRow("Author", modelInfo.Author) |> ignore
                table.AddRow("Task", modelInfo.Task) |> ignore
                table.AddRow("Architecture", modelInfo.Architecture) |> ignore
                table.AddRow("Framework", modelInfo.Framework) |> ignore
                table.AddRow("Size", $"{modelInfo.ModelSize / 1024L / 1024L}MB") |> ignore
                table.AddRow("Downloads", $"{modelInfo.Downloads:N0}") |> ignore
                table.AddRow("License", modelInfo.License |> Option.defaultValue "Unknown") |> ignore
                
                console.Write(table)
                console.WriteLine()
                
                // Download model
                if not (modelLoader.IsModelCached(modelId)) then
                    console.MarkupLine("📥 [yellow]Downloading model...[/]")
                    let! downloadResult = modelLoader.DownloadModelAsync(modelInfo, System.Threading.CancellationToken.None)
                    
                    match downloadResult with
                    | Success (cacheEntry, metadata) ->
                        let downloadedBytes = metadata.TryFind("downloadedBytes") |> Option.map unbox<int64> |> Option.defaultValue 0L
                        console.MarkupLine($"[green]✅ Model downloaded: {downloadedBytes} bytes[/]")
                    | Failure (error, _) ->
                        console.MarkupLine($"[red]❌ Download failed: {error}[/]")
                else
                    console.MarkupLine("[green]✅ Model already cached[/]")

                // Load model
                if not (modelLoader.IsModelLoaded(modelId)) then
                    console.MarkupLine("🧠 [yellow]Loading model into memory...[/]")
                    let! loadResult = modelLoader.LoadModelAsync(modelId, System.Threading.CancellationToken.None)

                    match loadResult with
                    | Success (cacheEntry, _) ->
                        console.MarkupLine($"[green]✅ Model loaded[/]")
                    | Failure (error, _) ->
                        console.MarkupLine($"[red]❌ Load failed: {error}[/]")
                else
                    console.MarkupLine("[green]✅ Model already loaded[/]")

            | Failure (error, _) ->
                console.MarkupLine($"[red]❌ Failed to fetch model info: {error}[/]")
        }
    
    /// Run inference demo
    let runInferenceDemo (console: IAnsiConsole) (logger: ILogger) (inferenceEngine: SimpleHuggingFaceInferenceEngine) (modelId: string) (task: string) (input: string) =
        task {
            console.MarkupLine($"[bold yellow]🚀 Running {task} inference[/]")
            console.MarkupLine($"[cyan]Model: {modelId}[/]")
            console.MarkupLine($"[cyan]Input: {input}[/]")
            console.WriteLine()
            
            let correlationId = generateCorrelationId()
            
            match task.ToLower() with
            | "text-generation" ->
                let request = {
                    RequestId = Guid.NewGuid().ToString("N").[..15]
                    ModelId = modelId
                    Task = TextGeneration (100, 0.7f, 0.9f)
                    InputText = input
                    BatchInputs = None
                    Parameters = Map.empty
                    UseCache = true
                    ReturnTokens = true
                    ReturnAttentions = false
                    ReturnHiddenStates = false
                    CorrelationId = correlationId
                }
                
                let! result = inferenceEngine.GenerateTextAsync(request, System.Threading.CancellationToken.None)

                match result with
                | Success (inferenceResult, _) ->
                    console.MarkupLine($"[green]✅ Generated text:[/]")
                    console.MarkupLine($"[white]{inferenceResult.Result.GeneratedText}[/]")
                    console.MarkupLine($"[dim]Inference time: {inferenceResult.Metrics.InferenceTime.TotalMilliseconds:F2}ms[/]")
                | Failure (error, _) ->
                    console.MarkupLine($"[red]❌ Generation failed: {error}[/]")
            
            | "text-classification" ->
                let request = {
                    RequestId = Guid.NewGuid().ToString("N").[..15]
                    ModelId = modelId
                    Task = TextClassification [| "positive"; "negative"; "neutral" |]
                    InputText = input
                    BatchInputs = None
                    Parameters = Map.empty
                    UseCache = true
                    ReturnTokens = false
                    ReturnAttentions = false
                    ReturnHiddenStates = false
                    CorrelationId = correlationId
                }
                
                let! result = inferenceEngine.ClassifyTextAsync(request, System.Threading.CancellationToken.None)

                match result with
                | Success (inferenceResult, _) ->
                    console.MarkupLine($"[green]✅ Classification results:[/]")
                    for (label, score) in inferenceResult.Result.Classifications do
                        console.MarkupLine($"  • [cyan]{label}[/]: {score:F3}")
                    console.MarkupLine($"[dim]Inference time: {inferenceResult.Metrics.InferenceTime.TotalMilliseconds:F2}ms[/]")
                | Failure (error, _) ->
                    console.MarkupLine($"[red]❌ Classification failed: {error}[/]")
            
            | "embeddings" ->
                let request = {
                    RequestId = Guid.NewGuid().ToString("N").[..15]
                    ModelId = modelId
                    Task = SentenceEmbeddings
                    InputText = input
                    BatchInputs = None
                    Parameters = Map.empty
                    UseCache = true
                    ReturnTokens = false
                    ReturnAttentions = false
                    ReturnHiddenStates = false
                    CorrelationId = correlationId
                }
                
                let! result = inferenceEngine.GenerateEmbeddingsAsync(request, System.Threading.CancellationToken.None)

                match result with
                | Success (inferenceResult, _) ->
                    let embeddings = inferenceResult.Result.Embeddings
                    let magnitude = inferenceResult.Result.Magnitude
                    console.MarkupLine($"[green]✅ Embeddings generated:[/]")
                    console.MarkupLine($"  • Dimensions: {embeddings.Length}")
                    console.MarkupLine($"  • Magnitude: {magnitude:F3}")
                    console.MarkupLine($"  • Sample values: [[{embeddings.[0]:F3}, {embeddings.[1]:F3}, {embeddings.[2]:F3}, ...]]")
                    console.MarkupLine($"[dim]Inference time: {inferenceResult.Metrics.InferenceTime.TotalMilliseconds:F2}ms[/]")
                | Failure (error, _) ->
                    console.MarkupLine($"[red]❌ Embedding generation failed: {error}[/]")
            
            | _ ->
                console.MarkupLine($"[red]❌ Unsupported task: {task}[/]")
        }
    
    /// Show help information
    let showHelp (console: IAnsiConsole) =
        console.MarkupLine("[bold blue]🤖 TARS Hugging Face CUDA Integration[/]")
        console.WriteLine()
        console.MarkupLine("[bold]Usage:[/]")
        console.MarkupLine("  tars hf-cuda [[options]]")
        console.WriteLine()
        console.MarkupLine("[bold]Options:[/]")
        console.MarkupLine("  --list-models           Show popular models")
        console.MarkupLine("  --download=MODEL        Download a model")
        console.MarkupLine("  --load=MODEL            Load a model")
        console.MarkupLine("  --generate=TEXT         Generate text")
        console.MarkupLine("  --classify=TEXT         Classify text")
        console.MarkupLine("  --embeddings=TEXT       Generate embeddings")
        console.MarkupLine("  --capabilities          Show engine capabilities")
        console.MarkupLine("  --verbose, -v           Verbose output")
        console.WriteLine()
        console.MarkupLine("[bold]Examples:[/]")
        console.MarkupLine("  tars hf-cuda --list-models")
        console.MarkupLine("  tars hf-cuda --download=gpt2")
        console.MarkupLine("  tars hf-cuda --generate=\"Hello world\"")
        console.MarkupLine("  tars hf-cuda --classify=\"I love this product!\"")
    
    /// Execute the Hugging Face CUDA command
    let executeCommand (args: string[]) (logger: ILogger) =
        task {
            try
                let options = parseArguments args
                let console = AnsiConsole.Create(AnsiConsoleSettings())
                
                console.MarkupLine("[bold blue]🤖 TARS Hugging Face CUDA Integration[/]")
                console.WriteLine()
                
                // Create configuration
                let config = {
                    ApiToken = None
                    CacheDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars", "models")
                    MaxCacheSize = 10L * 1024L * 1024L * 1024L // 10GB
                    AutoCleanup = true
                    CleanupThreshold = 0.8
                    DefaultTimeout = TimeSpan.FromMinutes(5.0)
                    MaxConcurrentDownloads = 3
                    PreferredFormat = "onnx"
                    EnableTelemetry = false
                }
                
                // Create model loader and inference engine
                use modelLoader = HuggingFaceModelLoader.createHuggingFaceModelLoader logger config
                use inferenceEngine = SimpleHuggingFaceInference.createSimpleHuggingFaceInferenceEngine logger modelLoader
                
                // Initialize inference engine
                let! initResult = inferenceEngine.InitializeAsync(System.Threading.CancellationToken.None)
                match initResult with
                | Success (_, _) ->
                    if options.Verbose then
                        logger.LogInformation("Hugging Face CUDA inference engine initialized")
                | Failure (error, _) ->
                    console.MarkupLine($"[red]❌ Failed to initialize inference engine: {error}[/]")
                    return 1
                
                // Execute based on options
                if options.ListModels then
                    showPopularModels console modelLoader
                
                if options.ShowCapabilities then
                    console.MarkupLine("[bold yellow]🚀 Engine Capabilities:[/]")
                    for capability in inferenceEngine.GetCapabilities() do
                        console.MarkupLine($"  • {capability}")
                    console.WriteLine()
                
                // Handle model operations
                if options.DownloadModel.IsSome || options.LoadModel.IsSome then
                    let modelId = options.DownloadModel |> Option.orElse options.LoadModel |> Option.get
                    do! downloadAndLoadModel console logger modelLoader modelId

                // Handle inference operations (using default model for demo)
                let defaultModelId = "gpt2"

                if options.GenerateText.IsSome then
                    do! runInferenceDemo console logger inferenceEngine defaultModelId "text-generation" options.GenerateText.Value

                if options.ClassifyText.IsSome then
                    do! runInferenceDemo console logger inferenceEngine defaultModelId "text-classification" options.ClassifyText.Value

                if options.GetEmbeddings.IsSome then
                    do! runInferenceDemo console logger inferenceEngine defaultModelId "embeddings" options.GetEmbeddings.Value
                
                if not options.ListModels && not options.ShowCapabilities && options.DownloadModel.IsNone && options.LoadModel.IsNone then
                    showHelp console
                
                return 0
            
            with
            | ex ->
                logger.LogError(ex, "Hugging Face CUDA command failed")
                AnsiConsole.MarkupLine($"[red]❌ Command execution failed: {ex.Message}[/]")
                return 1
        }
