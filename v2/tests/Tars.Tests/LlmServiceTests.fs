namespace Tars.Tests

open System
open System.Net
open System.Net.Http
open System.Threading.Tasks
open System.Text
open System.Text.Json
open Xunit
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService

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
            { OllamaBaseUri = Uri("http://localhost:11434/")
              VllmBaseUri = Uri("http://localhost:8000/")
              OpenAIBaseUri = Uri("https://api.openai.com/")
              GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com/")
              AnthropicBaseUri = Uri("https://api.anthropic.com/")
              DefaultOllamaModel = "ollama-model"
              DefaultVllmModel = "vllm-model"
              DefaultOpenAIModel = "gpt-4o"
              DefaultGoogleGeminiModel = "gemini-pro"
              DefaultAnthropicModel = "claude-3-opus-20240229"
              DefaultEmbeddingModel = "nomic-embed-text"
              OllamaKey = None
              VllmKey = None
              OpenAIKey = None
              GoogleGeminiKey = None
              AnthropicKey = None }

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
              Seed = None }

        let routed = chooseBackend routingCfg req

        match routed.Backend with
        | Ollama m -> Assert.Equal("ollama-model", m)
        | _ -> Assert.Fail("Should have routed to Ollama")

    [<Fact>]
    member this.``Routes to vLLM for reasoning hint``() =
        let routingCfg =
            { OllamaBaseUri = Uri("http://localhost:11434/")
              VllmBaseUri = Uri("http://localhost:8000/")
              OpenAIBaseUri = Uri("https://api.openai.com/")
              GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com/")
              AnthropicBaseUri = Uri("https://api.anthropic.com/")
              DefaultOllamaModel = "ollama-model"
              DefaultVllmModel = "vllm-model"
              DefaultOpenAIModel = "gpt-4o"
              DefaultGoogleGeminiModel = "gemini-pro"
              DefaultAnthropicModel = "claude-3-opus-20240229"
              DefaultEmbeddingModel = "nomic-embed-text"
              OllamaKey = None
              VllmKey = None
              OpenAIKey = None
              GoogleGeminiKey = None
              AnthropicKey = None }

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
              Seed = None }

        let routed = chooseBackend routingCfg req

        match routed.Backend with
        | Vllm m -> Assert.Equal("vllm-model", m)
        | _ -> Assert.Fail("Should have routed to vLLM")

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
                      Seed = None }

                let! response = OllamaClient.sendChatAsync httpClient baseUri "test-model" req

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
                      Seed = None }

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
                      Seed = None }

                let! response = OllamaClient.sendChatStreamAsync httpClient baseUri "test" req (fun t -> tokens.Add(t))

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
                      Seed = None }

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
                  Seed = None }

            let! response = mockService.CompleteStreamAsync(req, fun t -> tokens.Add(t))

            Assert.Equal(3, tokens.Count)
            Assert.Equal("Hello World", response.Text)
            output.WriteLine("ILlmService streaming interface works correctly")
        }
