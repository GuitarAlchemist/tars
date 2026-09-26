namespace Tars.Tests

open System
open System.Net
open System.Net.Http
open System.Text
open Xunit
open Tars.Llm
open Tars.Llm.Routing

type LlmServiceTests(output: Xunit.Abstractions.ITestOutputHelper) =

    let getFreePort () =
        let l = new System.Net.Sockets.TcpListener(IPAddress.Loopback, 0)
        l.Start()
        let port = (l.LocalEndpoint :?> IPEndPoint).Port
        l.Stop()
        port

    [<Fact>]
    member this.``Routes to Ollama for code hint``() =
        let routingCfg =
            { RoutingConfig.Default with
                OllamaBaseUri = Uri("http://localhost:11434/")
                VllmBaseUri = Uri("http://localhost:8000/")
                OpenAIBaseUri = Uri("https://api.openai.com/")
                GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com/")
                AnthropicBaseUri = Uri("https://api.anthropic.com/")
                DefaultOllamaModel = "ollama-model"
                DefaultVllmModel = "vllm-model" }

        let req =
            { ModelHint = Some "code"
              Model = None
              SystemPrompt = None
              MaxTokens = None
              Temperature = None
              Stop = []
              Messages = []
              Tools = []
              ToolChoice = None
              ResponseFormat = None
              Stream = false
              JsonMode = false
              Seed = None

              ContextWindow = None }

        let routed = chooseBackend routingCfg req

        match routed.Backend with
        | Ollama m -> Assert.Equal("ollama-model", m)
        | _ -> Assert.Fail("Should have routed to Ollama")

    [<Fact>]
    member this.``Routes reasoning hint to configured reasoning model``() =
        let routingCfg =
            { RoutingConfig.Default with
                OllamaBaseUri = Uri("http://localhost:11434/")
                VllmBaseUri = Uri("http://localhost:8000/")
                OpenAIBaseUri = Uri("https://api.openai.com/")
                GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com/")
                AnthropicBaseUri = Uri("https://api.anthropic.com/")
                DefaultOllamaModel = "ollama-model"
                DefaultVllmModel = "vllm-model"
                ReasoningModel = Some "reasoning-model" }

        let req =
            { ModelHint = Some "reasoning"
              Model = None
              SystemPrompt = None
              MaxTokens = None
              Temperature = None
              Stop = []
              Messages = []
              Tools = []
              ToolChoice = None
              ResponseFormat = None
              Stream = false
              JsonMode = false
              Seed = None

              ContextWindow = None }

        let routed = chooseBackend routingCfg req

        // With PreferredProvider = "Ollama" (default), reasoning routes through localRoute
        // which uses the ReasoningModel on the Ollama backend
        match routed.Backend with
        | Ollama m -> Assert.Equal("reasoning-model", m)
        | _ -> Assert.Fail("Should have routed to Ollama with reasoning model")

    [<Fact>]
    member this.``Routes to Docker Model Runner for docker hint``() =
        let routingCfg =
            { RoutingConfig.Default with
                OllamaBaseUri = Uri("http://localhost:11434/")
                VllmBaseUri = Uri("http://localhost:8000/")
                OpenAIBaseUri = Uri("https://api.openai.com/")
                GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com/")
                AnthropicBaseUri = Uri("https://api.anthropic.com/")
                DefaultOllamaModel = "ollama-model"
                DefaultVllmModel = "vllm-model"
                DockerModelRunnerBaseUri = Some(Uri("http://localhost:12434/"))
                DefaultDockerModelRunnerModel = Some "docker-model" }

        let req =
            { ModelHint = Some "docker"
              Model = None
              SystemPrompt = None
              MaxTokens = None
              Temperature = None
              Stop = []
              Messages = []
              Tools = []
              ToolChoice = None
              ResponseFormat = None
              Stream = false
              JsonMode = false
              Seed = None

              ContextWindow = None }

        let routed = chooseBackend routingCfg req

        match routed.Backend with
        | DockerModelRunner m -> Assert.Equal("docker-model", m)
        | _ -> Assert.Fail("Should have routed to Docker Model Runner")

    [<Fact>]
    member this.``Routes to LlamaCpp for llamacpp hint``() =
        let routingCfg =
            { RoutingConfig.Default with
                OllamaBaseUri = Uri("http://localhost:11434/")
                VllmBaseUri = Uri("http://localhost:8000/")
                OpenAIBaseUri = Uri("https://api.openai.com/")
                GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com/")
                AnthropicBaseUri = Uri("https://api.anthropic.com/")
                DefaultOllamaModel = "ollama-model"
                DefaultVllmModel = "vllm-model"
                LlamaCppBaseUri = Some(Uri("http://localhost:8080/"))
                DefaultLlamaCppModel = Some "llama-model.gguf" }

        let req =
            { ModelHint = Some "llamacpp"
              Model = None
              SystemPrompt = None
              MaxTokens = None
              Temperature = None
              Stop = []
              Messages = []
              Tools = []
              ToolChoice = None
              ResponseFormat = None
              Stream = false
              JsonMode = false
              Seed = None

              ContextWindow = None }

        let routed = chooseBackend routingCfg req

        match routed.Backend with
        | LlamaCpp(m, _) -> Assert.Equal("llama-model.gguf", m)
        | _ -> Assert.Fail("Should have routed to LlamaCpp")

    [<Fact>]
    member this.``Falls back to Ollama when Docker Model Runner not configured``() =
        let routingCfg =
            { RoutingConfig.Default with
                OllamaBaseUri = Uri("http://localhost:11434/")
                VllmBaseUri = Uri("http://localhost:8000/")
                OpenAIBaseUri = Uri("https://api.openai.com/")
                GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com/")
                AnthropicBaseUri = Uri("https://api.anthropic.com/")
                DefaultOllamaModel = "ollama-fallback"
                DefaultVllmModel = "vllm-model"
                DockerModelRunnerBaseUri = None // Not configured
                DefaultDockerModelRunnerModel = None }

        let req =
            { ModelHint = Some "docker"
              Model = None
              SystemPrompt = None
              MaxTokens = None
              Temperature = None
              Stop = []
              Messages = []
              Tools = []
              ToolChoice = None
              ResponseFormat = None
              Stream = false
              JsonMode = false
              Seed = None

              ContextWindow = None }

        let routed = chooseBackend routingCfg req

        // Should fall back to Ollama when Docker Model Runner is not configured
        match routed.Backend with
        | Ollama m -> Assert.Equal("ollama-fallback", m)
        | _ -> Assert.Fail("Should have fallen back to Ollama")

    [<Fact>]
    member this.``Falls back to Ollama when LlamaCpp not configured``() =
        let routingCfg =
            { RoutingConfig.Default with
                OllamaBaseUri = Uri("http://localhost:11434/")
                VllmBaseUri = Uri("http://localhost:8000/")
                OpenAIBaseUri = Uri("https://api.openai.com/")
                GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com/")
                AnthropicBaseUri = Uri("https://api.anthropic.com/")
                DefaultOllamaModel = "ollama-fallback"
                DefaultVllmModel = "vllm-model"
                LlamaCppBaseUri = None // Not configured
                DefaultLlamaCppModel = None }

        let req =
            { ModelHint = Some "perf" // perf hint routes to LlamaCpp
              Model = None
              SystemPrompt = None
              MaxTokens = None
              Temperature = None
              Stop = []
              Messages = []
              Tools = []
              ToolChoice = None
              ResponseFormat = None
              Stream = false
              JsonMode = false
              Seed = None

              ContextWindow = None }

        let routed = chooseBackend routingCfg req

        // Should fall back to Ollama when LlamaCpp is not configured
        match routed.Backend with
        | Ollama m -> Assert.Equal("ollama-fallback", m)
        | _ -> Assert.Fail("Should have fallen back to Ollama")

    [<Fact>]
    member this.``Ollama Client sends correct request``() =
        task {
            let port = getFreePort ()
            let baseUri = Uri($"http://localhost:{port}/")
            use listener = new HttpListener()
            listener.Prefixes.Add(baseUri.ToString())
            listener.Start()

            let serverLoop =
                task {
                    while listener.IsListening do
                        try
                            let! context = listener.GetContextAsync()
                            let req = context.Request
                            let resp = context.Response

                            if req.Url.AbsolutePath = "/api/chat" && req.HttpMethod = "POST" then
                                use reader = new System.IO.StreamReader(req.InputStream)
                                let! body = reader.ReadToEndAsync()

                                // Verify request body
                                Assert.Contains("test-model", body)
                                Assert.Contains("hello", body)

                                let json =
                                    """{ "model": "test-model", "message": { "role": "assistant", "content": "world" }, "done": true }"""

                                let bytes = Encoding.UTF8.GetBytes(json)
                                resp.ContentType <- "application/json"
                                resp.ContentLength64 <- int64 bytes.Length
                                resp.OutputStream.Write(bytes, 0, bytes.Length)
                                resp.OutputStream.Close()
                            else
                                resp.StatusCode <- 404
                                resp.Close()
                        with _ ->
                            ()
                }

            try
                use httpClient = new HttpClient()

                let req =
                    { ModelHint = None
                      Model = None
                      SystemPrompt = None
                      MaxTokens = None
                      Temperature = None
                      Stop = []
                      Messages = [ { Role = Role.User; Content = "hello" } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = false
                      Seed = None

                      ContextWindow = None }

                let! response = OllamaClient.sendChatAsync httpClient baseUri "test-model" None req

                Assert.Equal("world", response.Text)
                Assert.Equal(Some "done", response.FinishReason)
            finally
                listener.Stop()
        }

    [<Fact>]
    member this.``vLLM Client sends correct request``() =
        task {
            let port = getFreePort ()
            let baseUri = Uri($"http://localhost:{port}/")
            use listener = new HttpListener()
            listener.Prefixes.Add(baseUri.ToString())
            listener.Start()

            let serverLoop =
                task {
                    while listener.IsListening do
                        try
                            let! context = listener.GetContextAsync()
                            let req = context.Request
                            let resp = context.Response

                            if req.Url.AbsolutePath = "/v1/chat/completions" && req.HttpMethod = "POST" then
                                use reader = new System.IO.StreamReader(req.InputStream)
                                let! body = reader.ReadToEndAsync()

                                // Verify request body
                                Assert.Contains("vllm-model", body)
                                Assert.Contains("hello", body)

                                let json =
                                    """{
                                "id": "chatcmpl-123",
                                "choices": [{
                                    "index": 0,
                                    "message": { "role": "assistant", "content": "world" },
                                    "finish_reason": "stop"
                                }]
                            }"""

                                let bytes = Encoding.UTF8.GetBytes(json)
                                resp.ContentType <- "application/json"
                                resp.ContentLength64 <- int64 bytes.Length
                                resp.OutputStream.Write(bytes, 0, bytes.Length)
                                resp.OutputStream.Close()
                            else
                                resp.StatusCode <- 404
                                resp.Close()
                        with _ ->
                            ()
                }

            try
                use httpClient = new HttpClient()

                let req =
                    { ModelHint = None
                      Model = None
                      SystemPrompt = None
                      MaxTokens = None
                      Temperature = None
                      Stop = []
                      Messages = [ { Role = Role.User; Content = "hello" } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = false
                      Seed = None

                      ContextWindow = None }

                let! response = OpenAiCompatibleClient.sendChatAsync httpClient baseUri "vllm-model" None req

                Assert.Equal("world", response.Text)
                Assert.Equal(Some "stop", response.FinishReason)
            finally
                listener.Stop()
        }

    [<Fact>]
    member this.``Ollama Client streaming collects all tokens``() =
        task {
            let port = getFreePort ()
            let baseUri = Uri($"http://localhost:{port}/")
            use listener = new HttpListener()
            listener.Prefixes.Add(baseUri.ToString())
            listener.Start()

            let serverLoop =
                task {
                    while listener.IsListening do
                        try
                            let! context = listener.GetContextAsync()
                            let req = context.Request
                            let resp = context.Response

                            if req.Url.AbsolutePath = "/api/chat" && req.HttpMethod = "POST" then
                                resp.ContentType <- "application/x-ndjson"

                                // Simulate streaming NDJSON chunks
                                let chunks =
                                    [| """{"model":"test","message":{"role":"assistant","content":"Hello"},"done":false}"""
                                       """{"model":"test","message":{"role":"assistant","content":" world"},"done":false}"""
                                       """{"model":"test","message":{"role":"assistant","content":"!"},"done":true}""" |]

                                for chunk in chunks do
                                    let bytes = Encoding.UTF8.GetBytes(chunk + "\n")
                                    resp.OutputStream.Write(bytes, 0, bytes.Length)
                                    resp.OutputStream.Flush()

                                resp.OutputStream.Close()
                            else
                                resp.StatusCode <- 404
                                resp.Close()
                        with _ ->
                            ()
                }

            try
                use httpClient = new HttpClient()
                let tokens = ResizeArray<string>()

                let req =
                    { ModelHint = None
                      Model = None
                      SystemPrompt = None
                      MaxTokens = None
                      Temperature = None
                      Stop = []
                      Messages = [ { Role = Role.User; Content = "hello" } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = false
                      Seed = None

                      ContextWindow = None }

                let! response =
                    OllamaClient.sendChatStreamAsync httpClient baseUri "test" None req (fun t -> tokens.Add(t))

                // Verify all tokens were collected
                Assert.Equal(3, tokens.Count)
                Assert.Equal("Hello", tokens.[0])
                Assert.Equal(" world", tokens.[1])
                Assert.Equal("!", tokens.[2])

                // Verify final response
                Assert.Equal("Hello world!", response.Text)
                let allTokens = String.Join("", tokens)
                output.WriteLine($"Streaming collected {tokens.Count} tokens: {allTokens}")
            finally
                listener.Stop()
        }

    [<Fact>]
    member this.``OpenAI Client streaming handles SSE format``() =
        task {
            let port = getFreePort ()
            let baseUri = Uri($"http://localhost:{port}/")
            use listener = new HttpListener()
            listener.Prefixes.Add(baseUri.ToString())
            listener.Start()

            let serverLoop =
                task {
                    while listener.IsListening do
                        try
                            let! context = listener.GetContextAsync()
                            let req = context.Request
                            let resp = context.Response

                            if req.Url.AbsolutePath = "/v1/chat/completions" && req.HttpMethod = "POST" then
                                resp.ContentType <- "text/event-stream"

                                // Simulate SSE streaming chunks
                                let chunks =
                                    [| """data: {"id":"1","choices":[{"delta":{"content":"Hi"}}]}"""
                                       """data: {"id":"1","choices":[{"delta":{"content":" there"}}]}"""
                                       """data: {"id":"1","choices":[{"delta":{"content":"!"}}]}"""
                                       """data: [DONE]""" |]

                                for chunk in chunks do
                                    let bytes = Encoding.UTF8.GetBytes(chunk + "\n\n")
                                    resp.OutputStream.Write(bytes, 0, bytes.Length)
                                    resp.OutputStream.Flush()

                                resp.OutputStream.Close()
                            else
                                resp.StatusCode <- 404
                                resp.Close()
                        with _ ->
                            ()
                }

            try
                use httpClient = new HttpClient()
                let tokens = ResizeArray<string>()

                let req =
                    { ModelHint = None
                      Model = None
                      SystemPrompt = None
                      MaxTokens = None
                      Temperature = None
                      Stop = []
                      Messages = [ { Role = Role.User; Content = "hello" } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = false
                      Seed = None

                      ContextWindow = None }

                let! response =
                    OpenAiCompatibleClient.sendChatStreamAsync httpClient baseUri "test" None req (fun t ->
                        tokens.Add(t))

                // Verify all tokens were collected
                Assert.Equal(3, tokens.Count)
                Assert.Equal("Hi", tokens.[0])
                Assert.Equal(" there", tokens.[1])
                Assert.Equal("!", tokens.[2])

                // Verify final response
                Assert.Equal("Hi there!", response.Text)
                let allTokens = String.Join("", tokens)
                output.WriteLine($"SSE streaming collected {tokens.Count} tokens: {allTokens}")
            finally
                listener.Stop()
        }

    [<Fact>]
    member this.``ILlmService CompleteStreamAsync interface works``() =
        task {
            // Create a mock service that implements streaming
            let tokens = ResizeArray<string>()

            let mockService =
                { new ILlmService with
                    member _.CompleteAsync(req) =
                        task {
                            return
                                { Text = "Hello"
                                  FinishReason = Some "stop"
                                  Usage = None
                                  Raw = None }
                        }

                    member _.EmbedAsync(text) = task { return [| 0.1f; 0.2f; 0.3f |] }

                    member _.RouteAsync(_) =
                        task {
                            return
                                { Backend = Ollama "mock"
                                  Endpoint = Uri "http://localhost:11434"
                                  ApiKey = None }
                        }

                    member _.CompleteStreamAsync(req, onToken) =
                        task {
                            onToken "Hello"
                            onToken " "
                            onToken "World"

                            return
                                { Text = "Hello World"
                                  FinishReason = Some "stop"
                                  Usage = None
                                  Raw = None }
                        } }

            let req =
                { ModelHint = None
                  Model = None
                  SystemPrompt = None
                  MaxTokens = None
                  Temperature = None
                  Stop = []
                  Messages = [ { Role = Role.User; Content = "test" } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = None

                  ContextWindow = None }

            let! response = mockService.CompleteStreamAsync(req, fun t -> tokens.Add(t))

            Assert.Equal(3, tokens.Count)
            Assert.Equal("Hello World", response.Text)
            output.WriteLine("ILlmService streaming interface works correctly")
        }

// ------------------------------------------------------- where embeddings go

/// The routing that decides whether a caller's text leaves the machine.
module EmbeddingRoutingTests =

    let private routing model provider key =
        { RoutingConfig.Default with
            DefaultEmbeddingModel = model
            PreferredProvider = provider
            OpenAIKey = key }

    let private declaring embeddingProvider model key =
        { routing model "Ollama" key with
            EmbeddingProvider = Some embeddingProvider }

    [<Fact>]
    let ``a local embedding model stays local, whatever it is called`` () =
        // The four names the old substring list knew, and four it did not. All of
        // these are ordinary `ollama pull` models.
        for model in
            [ "nomic-embed-text"
              "mxbai-embed-large"
              "bge-m3"
              "all-minilm"
              "snowflake-arctic-embed"
              "granite-embedding" ] do
            Assert.False(Embedder.usesOpenAi (routing model "Ollama" None), $"{model} would have been sent to OpenAI")

    [<Fact>]
    let ``a key alone does not send local text to OpenAI`` () =
        Assert.False(Embedder.usesOpenAi (routing "bge-m3" "Ollama" (Some "sk-test")))

    [<Fact>]
    let ``OpenAI is used when it is asked for and can be called`` () =
        Assert.True(Embedder.usesOpenAi (routing "text-embedding-3-small" "Ollama" (Some "sk-test")))
        Assert.True(Embedder.usesOpenAi (routing "text-embedding-3-large" "OpenAI" (Some "sk-test")))

    [<Fact>]
    let ``without a key, OpenAI is not called even when it is asked for`` () =
        // The request could only 401, and the text would have left anyway.
        Assert.False(Embedder.usesOpenAi (routing "text-embedding-3-small" "OpenAI" None))

    [<Fact>]
    let ``a blank key is no key`` () =
        // Configuration stores an empty OPENAI_API_KEY as `Some ""`, and
        // `getEmbeddingsAsync` sends no authorization header for one — so this would
        // be the text leaving the machine purely to collect a 401.
        for key in [ ""; "   "; "\t" ] do
            Assert.False(Embedder.usesOpenAi (routing "text-embedding-3-small" "OpenAI" (Some key)))

    [<Fact>]
    let ``a declared embedding provider decides, whatever the model is called`` () =
        // The point of the setting: a local alias named `text-embedding-nomic` is not
        // evidence of anything, and saying so should be enough to keep it local.
        Assert.False(Embedder.usesOpenAi (declaring "Ollama" "text-embedding-nomic" (Some "sk-test")))
        Assert.False(Embedder.usesOpenAi (declaring "Ollama" "text-embedding-3-small" (Some "sk-test")))
        Assert.True(Embedder.usesOpenAi (declaring "OpenAI" "our-embedding-alias" (Some "sk-test")))
        Assert.True(Embedder.usesOpenAi (declaring "openai" "text-embedding-3-small" (Some "sk-test")))

    [<Fact>]
    let ``declaring OpenAI without a key still does not send`` () =
        Assert.False(Embedder.usesOpenAi (declaring "OpenAI" "text-embedding-3-small" None))
        Assert.False(Embedder.usesOpenAi (declaring "OpenAI" "text-embedding-3-small" (Some " ")))

    [<Fact>]
    let ``choosing OpenAI for chat does not move local embeddings`` () =
        // `Llm.Provider` and `Llm.EmbeddingModel` are separate settings, and the
        // embedding default is local. Chat on OpenAI with embeddings on Ollama is an
        // ordinary setup, not an instruction to post `nomic-embed-text` to OpenAI.
        for model in [ "nomic-embed-text"; "bge-m3"; "some-gateway-model" ] do
            Assert.False(
                Embedder.usesOpenAi (routing model "OpenAI" (Some "sk-test")),
                $"{model} would have been sent to OpenAI because chat uses it"
            )
