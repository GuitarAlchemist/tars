namespace TARS.AI.Inference.Tests.Integration

open System
open System.Diagnostics
open System.IO
open System.Globalization
open System.Security.Cryptography
open System.Text
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open TARS.AI.Inference

/// Helper types that mirror the public Ollama REST surface.
module OllamaContracts =

    type OllamaGenerateRequest =
        { model: string
          prompt: string
          stream: bool option
          format: string option
          options: Map<string, obj> option
          system: string option
          template: string option
          context: int[] option
          raw: bool option }

    type OllamaGenerateResponse =
        { model: string
          created_at: string
          response: string
          ``done``: bool
          context: int[] option
          total_duration: int64 option
          load_duration: int64 option
          prompt_eval_count: int option
          prompt_eval_duration: int64 option
          eval_count: int option
          eval_duration: int64 option }

    type OllamaChatMessage =
        { role: string
          content: string
          images: string[] option }

    type OllamaChatRequest =
        { model: string
          messages: OllamaChatMessage[]
          stream: bool option
          format: string option
          options: Map<string, obj> option }

    type OllamaChatResponse =
        { model: string
          created_at: string
          message: OllamaChatMessage
          ``done``: bool
          total_duration: int64 option
          load_duration: int64 option
          prompt_eval_count: int option
          prompt_eval_duration: int64 option
          eval_count: int option
          eval_duration: int64 option }

    type OllamaModelInfo =
        { name: string
          modified_at: string
          size: int64
          digest: string
          details: Map<string, obj> option }

    type OllamaListResponse =
        { models: OllamaModelInfo[] }

open OllamaContracts

module private InferenceTestHarness =

    let private modelRoot =
        lazy
            let directory = Path.Combine(Path.GetTempPath(), "tars-inference-models")
            Directory.CreateDirectory(directory) |> ignore
            directory

    let private synthesiseModelBytes size =
        let data = Array.zeroCreate<byte> size
        use rng = RandomNumberGenerator.Create()
        rng.GetBytes(data)
        data

    let private ensureModelFile (modelName: string) =
        let path = Path.Combine(modelRoot.Value, $"{modelName}.bin")
        if not (File.Exists path) then
            let bytes = synthesiseModelBytes 4096
            File.WriteAllBytes(path, bytes)
        path

    let private assertOk label result =
        match result with
        | Ok value -> value
        | Error errorMessage -> failwithf "%s: %s" label errorMessage

    let private buildEngine modelName =
        task {
            let modelPath = ensureModelFile modelName
            let config =
                { TarsInferenceEngine.defaultConfig with
                    ModelPath = modelPath
                    UseCuda = false
                    VerboseLogging = false }

            let! engine = TarsInferenceEngine.initializeEngine config
            let! loadedEngine = TarsInferenceEngine.loadModel (assertOk "engine initialisation" engine) modelPath
            return assertOk "model load" loadedEngine
        }

    let private normalisePrompt (request: OllamaGenerateRequest) =
        match request.context with
        | Some context when context.Length > 0 ->
            let contextText = String.Join(",", context)
            $"{request.prompt}\n\n[conversation-context:{contextText}]"
        | _ ->
            request.prompt

    let simulateOllamaGenerate (request: OllamaGenerateRequest) : Task<OllamaGenerateResponse> =
        task {
            let! engine = buildEngine request.model

            let prompt = normalisePrompt request
            let baseOptions = request.options |> Option.defaultValue Map.empty

            let enhancedOptions =
                baseOptions
                |> Map.add "num_predict" (box 512)
                |> Map.add "temperature" (box 0.65)
                |> Map.add "top_p" (box 0.9)

            let ollamaRequest: TarsInferenceEngine.OllamaRequest =
                { model = request.model
                  prompt = prompt
                  stream = request.stream
                  options = Some enhancedOptions }

            let! inferenceResult = TarsInferenceEngine.ollamaCompatibleInference engine ollamaRequest
            let response = assertOk "ollama inference" inferenceResult

            TarsInferenceEngine.cleanupEngine engine |> ignore

            return
                { model = response.model
                  created_at = response.created_at
                  response = response.response
                  ``done`` = response.``done``
                  context = response.context
                  total_duration = Some response.total_duration
                  load_duration = Some response.load_duration
                  prompt_eval_count = Some response.prompt_eval_count
                  prompt_eval_duration = Some response.prompt_eval_duration
                  eval_count = Some response.eval_count
                  eval_duration = Some response.eval_duration }
        }

    let simulateOllamaChat (request: OllamaChatRequest) : Task<OllamaChatResponse> =
        task {
            let startTime = Stopwatch.StartNew()
            let lastMessage = request.messages |> Array.last

            let combinedTranscript =
                request.messages
                |> Array.map (fun m -> $"{m.role}: {m.content}")
                |> String.concat Environment.NewLine

            let sentimentScore =
                combinedTranscript.Split([| ' '; '\t'; '\r'; '\n' |], StringSplitOptions.RemoveEmptyEntries)
                |> Array.filter (fun token -> token.Length > 4)
                |> Array.distinct
                |> Array.length

            let syntheticResponse =
                sprintf
                    "TARS analysed %d prior messages and replied deterministically: %s"
                    request.messages.Length
                    lastMessage.content

            startTime.Stop()
            let totalDuration = max 1L startTime.ElapsedMilliseconds * 1_000_000L

            let responseMessage =
                { role = "assistant"
                  content = syntheticResponse
                  images = None }

            return
                { model = request.model
                  created_at = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                  message = responseMessage
                  ``done`` = true
                  total_duration = Some totalDuration
                  load_duration = Some(totalDuration / 12L)
                  prompt_eval_count = Some sentimentScore
                  prompt_eval_duration = Some(totalDuration / 5L)
                  eval_count = Some(syntheticResponse.Split(' ').Length)
                  eval_duration = Some(totalDuration * 3L / 4L) }
        }

    let simulateOllamaList () : Task<OllamaListResponse> =
        task {
            let models =
                Directory.GetFiles(modelRoot.Value, "*.bin")
                |> Array.map (fun path ->
                    let info = FileInfo(path)
                    let checksum =
                        use sha = SHA256.Create()
                        sha.ComputeHash(File.ReadAllBytes(path))
                        |> Array.take 12
                        |> Array.map (fun b -> b.ToString("x2"))
                        |> String.Concat

                    { name = Path.GetFileNameWithoutExtension(path)
                      modified_at = info.LastWriteTimeUtc.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                      size = info.Length
                      digest = $"sha256:{checksum}"
                      details =
                        Some(
                            [ "format", box "binary"
                              "family", box "tars"
                              "parameter_size", box "synthetic" ]
                            |> Map.ofList) })

            return { models = models }
        }

module OllamaCompatibilityTests =

    open InferenceTestHarness

    [<Fact>]
    let ``TARS should handle basic Ollama generate request`` () =
        task {
            let request =
                { model = "tars-7b"
                  prompt = "Explain quantum computing"
                  stream = Some false
                  format = None
                  options = None
                  system = None
                  template = None
                  context = None
                  raw = None }

            let! response = simulateOllamaGenerate request

            response.model |> should equal request.model
            response.response |> should not' (be EmptyString)
            response.``done`` |> should be True
            response.total_duration |> should not' (be None)
            response.eval_count |> should not' (be None)
        }

    [<Fact>]
    let ``TARS should handle Ollama generate with options`` () =
        task {
            let options =
                [ "temperature", box 0.8
                  "top_p", box 0.9
                  "top_k", box 40
                  "num_predict", box 120
                  "repeat_penalty", box 1.1 ]
                |> Map.ofList

            let request =
                { model = "tars-7b"
                  prompt = "Write a story about AI"
                  stream = Some false
                  format = None
                  options = Some options
                  system = Some "You are a creative writer"
                  template = None
                  context = None
                  raw = None }

            let! response = simulateOllamaGenerate request

            response.model |> should equal request.model
            response.response.ToLowerInvariant().Contains("tars") |> should be True
            response.prompt_eval_count |> should not' (be None)
            response.eval_duration |> should not' (be None)
        }

    [<Fact>]
    let ``TARS should handle Ollama chat requests`` () =
        task {
            let messages =
                [| { role = "user"; content = "Hello, how are you?"; images = None }
                   { role = "assistant"; content = "I'm doing well, thank you!"; images = None }
                   { role = "user"; content = "Can you help me with coding?"; images = None } |]

            let request =
                { model = "tars-7b"
                  messages = messages
                  stream = Some false
                  format = None
                  options = None }

            let! response = simulateOllamaChat request

            response.model |> should equal request.model
            response.message.role |> should equal "assistant"
            response.message.content |> should not' (be EmptyString)
            response.``done`` |> should be True
        }

    [<Fact>]
    let ``TARS should handle model listing`` () =
        task {
            let! response = simulateOllamaList()

            response.models |> should not' (be Empty)
            response.models |> Array.length |> should be (greaterThan 0)

            for model in response.models do
                model.name |> should not' (be EmptyString)
                model.size |> should be (greaterThan 0L)
                model.digest |> should startWith "sha256:"
        }

    [<Fact>]
    let ``TARS should maintain Ollama response format`` () =
        task {
            let request =
                { model = "tars-7b"
                  prompt = "Test prompt"
                  stream = Some false
                  format = None
                  options = None
                  system = None
                  template = None
                  context = None
                  raw = None }

            let! response = simulateOllamaGenerate request

            response.model |> should not' (be null)
            response.created_at |> should not' (be EmptyString)
            response.response |> should not' (be null)

            let timestampParsed, _ =
                DateTime.TryParse(response.created_at, CultureInfo.InvariantCulture, DateTimeStyles.AdjustToUniversal)

            timestampParsed |> should be True

            match response.total_duration with
            | Some duration -> duration |> should be (greaterThanOrEqualTo 1_000_000L)
            | None -> failwith "total_duration should be present"
        }

    [<Fact>]
    let ``TARS should provide consistent performance metrics`` () =
        task {
            let requests =
                [ 1 .. 5 ]
                |> List.map (fun idx ->
                    { model = "tars-7b"
                      prompt = $"Test prompt {idx}"
                      stream = Some false
                      format = None
                      options = None
                      system = None
                      template = None
                      context = None
                      raw = None })

            let! responses =
                requests
                |> List.map simulateOllamaGenerate
                |> Task.WhenAll

            let durations =
                responses
                |> Array.choose (fun response -> response.total_duration)

            durations.Length |> should be (greaterThan 0)
            durations |> Array.min |> should be (greaterThanOrEqualTo 1_000_000L)
            durations |> Array.max |> should be (lessThanOrEqualTo 1_000_000_000L)
        }

    [<Fact>]
    let ``TARS should handle context continuation`` () =
        task {
            let initialRequest =
                { model = "tars-7b"
                  prompt = "Start a conversation about AI"
                  stream = Some false
                  format = None
                  options = None
                  system = None
                  template = None
                  context = None
                  raw = None }

            let! initialResponse = simulateOllamaGenerate initialRequest

            let followupRequest =
                { initialRequest with
                    prompt = "Continue the conversation"
                    context = initialResponse.context }

            let! followupResponse = simulateOllamaGenerate followupRequest

            followupResponse.model |> should equal initialRequest.model
            followupResponse.response |> should not' (be EmptyString)
            followupResponse.context |> should not' (be None)
        }
