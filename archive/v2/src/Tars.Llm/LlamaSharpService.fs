namespace Tars.Llm

open System
open System.Collections.Generic
open System.IO
open System.Text
open System.Threading
open LLama
open LLama.Common
open LLama.Sampling
open LLama.Native
open Microsoft.Extensions.Logging.Abstractions

/// <summary>
/// In-process LLM service using LLamaSharp with GPU backends.
/// </summary>
type LlamaSharpService(_config: LlmServiceConfig, modelPath: string) =
    static let mutable nativeInitialised = false

    do
        if not nativeInitialised then
            if NativeLibraryConfig.LLama.LibraryHasLoaded then
                nativeInitialised <- true
            elif OperatingSystem.IsWindows() then
                let baseDir = AppContext.BaseDirectory
                let vulkanPath = Path.Combine(baseDir, "runtimes", "win-x64", "native", "vulkan")

                if Directory.Exists vulkanPath then
                    NativeLibraryConfig.All
                        .WithAutoFallback(true)
                        .WithCuda(false)
                        .WithVulkan(true)
                        .WithSearchDirectory(vulkanPath)
                    |> ignore
                else
                    // Fallback to searching for llama.dll in the base directory or other locations
                    ()

                nativeInitialised <- true
            else
                nativeInitialised <- true

    let parameters = ModelParams(modelPath)

    do
        let contextWindow =
            let raw = Environment.GetEnvironmentVariable("TARS_LLAMASHARP_CONTEXT_WINDOW")

            match UInt32.TryParse(raw) with
            | true, value when value > 0u -> value
            | _ -> 4096u

        // Force a check after config
        if not (NativeApi.llama_supports_gpu_offload ()) then
            // Try one more time with default search logic if specific path failed
            ()

        parameters.ContextSize <- contextWindow
        parameters.GpuLayerCount <- 35 // Leave some layers for CPU to save VRAM
        parameters.MainGpu <- 0
        parameters.SplitMode <- GPUSplitMode.None
        parameters.Embeddings <- true

    // Load model once
    let model = LLamaWeights.LoadFromFile(parameters)
    let context = model.CreateContext(parameters)
    let executor = new InteractiveExecutor(context, NullLogger.Instance)
    let embedder = new LLamaEmbedder(model, parameters, NullLogger.Instance)

    // We need a lock because the context is not thread-safe for concurrent generation
    let semaphore = new SemaphoreSlim(1, 1)

    let buildHistory (request: LlmRequest) =
        let history = ChatHistory()

        match request.SystemPrompt with
        | Some sys -> history.AddMessage(AuthorRole.System, sys)
        | None -> ()

        for msg in request.Messages do
            let role =
                match msg.Role with
                | Role.System -> AuthorRole.System
                | Role.User -> AuthorRole.User
                | Role.Assistant -> AuthorRole.Assistant

            history.AddMessage(role, msg.Content)

        history

    let buildInferenceParams (request: LlmRequest) =
        let inferenceParams = InferenceParams()
        inferenceParams.MaxTokens <- request.MaxTokens |> Option.defaultValue 1024
        inferenceParams.AntiPrompts <- request.Stop |> List.distinct |> ResizeArray

        let temp = request.Temperature |> Option.map float32 |> Option.defaultValue 0.7f

        let sampling =
            match request.Seed with
            | Some seed when seed >= 0 -> new DefaultSamplingPipeline(Temperature = temp, Seed = uint32 seed)
            | _ -> new DefaultSamplingPipeline(Temperature = temp)

        inferenceParams.SamplingPipeline <- sampling
        inferenceParams

    let collectTokens (stream: IAsyncEnumerable<string>) (onToken: (string -> unit) option) =
        task {
            let sb = StringBuilder()
            let enumerator = stream.GetAsyncEnumerator()

            let mutable captured: exn option = None

            try
                let mutable loop = true

                while loop do
                    let! hasNext = enumerator.MoveNextAsync().AsTask()

                    if hasNext then
                        let token = enumerator.Current
                        sb.Append(token) |> ignore

                        match onToken with
                        | Some emit -> emit token
                        | None -> ()
                    else
                        loop <- false
            with ex ->
                captured <- Some ex

            do! enumerator.DisposeAsync().AsTask()

            match captured with
            | Some ex -> return raise ex
            | None -> return sb.ToString()
        }

    interface ILlmService with
        member _.CompleteAsync(request: LlmRequest) =
            task {
                do! semaphore.WaitAsync()

                try
                    let history = buildHistory request
                    let inferenceParams = buildInferenceParams request
                    let session = ChatSession(executor)
                    let stream = session.ChatAsync(history, inferenceParams, CancellationToken.None)
                    let! text = collectTokens stream None

                    return { Text = text; FinishReason = Some "stop"; Usage = None; Raw = None }
                finally
                    semaphore.Release() |> ignore
            }

        member _.EmbedAsync(text: string) =
            task {
                do! semaphore.WaitAsync()

                try
                    let! embeddings = embedder.GetEmbeddings(text, CancellationToken.None)

                    if embeddings.Count = 0 then
                        return Array.empty
                    else
                        return embeddings.[0]
                finally
                    semaphore.Release() |> ignore
            }

        member _.CompleteStreamAsync(request: LlmRequest, onToken: string -> unit) =
            task {
                do! semaphore.WaitAsync()

                try
                    let history = buildHistory request
                    let inferenceParams = buildInferenceParams request
                    let session = ChatSession(executor)
                    let stream = session.ChatAsync(history, inferenceParams, CancellationToken.None)
                    let! text = collectTokens stream (Some onToken)

                    return { Text = text; FinishReason = Some "stop"; Usage = None; Raw = None }
                finally
                    semaphore.Release() |> ignore
            }

        member _.RouteAsync(request: LlmRequest) =
            task {
                return { Backend = LlamaSharp modelPath; Endpoint = Uri("local://llama.sharp"); ApiKey = None }
            }
