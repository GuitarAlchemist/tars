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
              DefaultEmbeddingModel = "nomic-embed-text" }

        let req =
            { ModelHint = Some "code"
              MaxTokens = None
              Temperature = None
              Messages = [] }

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
              DefaultEmbeddingModel = "nomic-embed-text" }

        let req =
            { ModelHint = Some "reasoning"
              MaxTokens = None
              Temperature = None
              Messages = [] }

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
                      MaxTokens = None
                      Temperature = None
                      Messages = [ { Role = Role.User; Content = "hello" } ] }

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
                      MaxTokens = None
                      Temperature = None
                      Messages = [ { Role = Role.User; Content = "hello" } ] }

                let! response = OpenAiCompatibleClient.sendChatAsync httpClient baseUri "vllm-model" req

                Assert.Equal("world", response.Text)
                Assert.Equal(Some "stop", response.FinishReason)
            finally
                listener.Stop()
        }
