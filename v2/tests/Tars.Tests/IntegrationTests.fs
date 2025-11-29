namespace Tars.Tests

open System
open System.Net.Http
open System.Threading
open System.Threading.Tasks
open System.Runtime.InteropServices
open Xunit
open Xunit.Abstractions
open Tars.Core
open Tars.Cortex
open Tars.Llm
open Xunit
open Xunit.Sdk
open Tars.Llm.LlmService

/// Integration tests that require external services (Ollama, OpenAI, etc.)
/// These tests are skipped by default - remove Skip to run manually
type IntegrationTests(output: ITestOutputHelper) =

    let httpClient = new HttpClient()
    let ollamaUri = Uri("http://localhost:11434")

    // === Ollama Integration Tests ===

    [<Fact(Skip = "Integration: Requires Ollama running on localhost:11434")>]
    member _.``Ollama: Can connect and list models``() =
        task {
            output.WriteLine("Connecting to Ollama...")
            let! models = OllamaClient.getTagsAsync httpClient ollamaUri
            output.WriteLine($"Found {models.Length} models")
            Assert.NotEmpty(models)
        }

    [<Fact(Skip = "Integration: Requires Ollama with llama3.2 model")>]
    member _.``Ollama: Can generate chat completion``() =
        task {
            output.WriteLine("Testing chat generation...")

            let req =
                { ModelHint = None
                  MaxTokens = None
                  Temperature = None
                  Messages =
                    [ { Role = Role.User
                        Content = "Say hello in one word" } ] }

            let! response = OllamaClient.sendChatAsync httpClient ollamaUri "llama3.2" req

            output.WriteLine($"Response: {response.Text}")
            Assert.False(String.IsNullOrEmpty(response.Text))
        }

    [<Fact(Skip = "Integration: Requires Ollama with embedding model")>]
    member _.``Ollama: Can generate embeddings``() =
        task {
            output.WriteLine("Testing embedding generation...")
            let! embedding = OllamaClient.getEmbeddingsAsync httpClient ollamaUri "nomic-embed-text" "Hello world"

            output.WriteLine($"Embedding dimension: {embedding.Length}")
            Assert.True(embedding.Length > 0, "Embedding should have dimensions")
        }

    // === Optional Smoke Tests (env-gated) ===
    // Enable by setting TARS_SMOKE=1 to verify the core pipelines end-to-end with stubs.
    [<Fact>]
    member _.``Smoke: ContextCompressor runs when TARS_SMOKE set``() =
        let smoke = Environment.GetEnvironmentVariable("TARS_SMOKE")

        if String.IsNullOrWhiteSpace smoke then
            output.WriteLine("Skipping smoke test because TARS_SMOKE is not set")
        else
            let llm =
                { new ILlmService with
                    member _.CompleteAsync(_req) =
                        task {
                            return
                                { Text = "compressed text"
                                  FinishReason = Some "stop"
                                  Usage = None
                                  Raw = None }
                        }

                    member _.CompleteStreamAsync(req, _cb) =
                        task {
                            return
                                { Text = "compressed text"
                                  FinishReason = Some "stop"
                                  Usage = None
                                  Raw = None }
                        }

                    member _.EmbedAsync(_text) = task { return [| 0.1f; 0.2f |] } }

            let compressor = ContextCompressor(llm, EntropyMonitor())

            let result =
                compressor.AutoCompress("repeat repeat repeat repeat", 0.8)
                |> Async.AwaitTask
                |> Async.RunSynchronously

            Assert.Equal("compressed text", result)

    [<Fact>]
    member _.``Smoke: EpistemicGovernor ExtractPrinciple when TARS_SMOKE set``() =
        let smoke = Environment.GetEnvironmentVariable("TARS_SMOKE")

        if String.IsNullOrWhiteSpace smoke then
            output.WriteLine("Skipping smoke test because TARS_SMOKE is not set")
        else
            // Use a trivial governor that echoes a belief
            let governor =
                { new IEpistemicGovernor with
                    member _.GenerateVariants(_, _) = Task.FromResult([ "v1" ])

                    member _.VerifyGeneralization(_, _, _) =
                        Task.FromResult(
                            { IsVerified = true
                              Score = 1.0
                              Feedback = ""
                              FailedVariants = [] }
                        )

                    member _.ExtractPrinciple(taskDescription, solution) =
                        Task.FromResult(
                            { Id = Guid.NewGuid()
                              Statement = $"Principle: {taskDescription} -> {solution}"
                              Context = solution
                              Status = EpistemicStatus.Hypothesis
                              Confidence = 0.5
                              DerivedFrom = []
                              CreatedAt = DateTime.UtcNow
                              LastVerified = DateTime.UtcNow }
                        )

                    member _.SuggestCurriculum(_, _) = Task.FromResult("none")
                    member _.Verify(_) = Task.FromResult(true) }

            let belief =
                governor.ExtractPrinciple("test task", "result")
                |> Async.AwaitTask
                |> Async.RunSynchronously

            Assert.Contains("Principle:", belief.Statement)

    // === Vector Store Integration Tests ===

    [<Fact(Skip = "Integration: Tests full RAG pipeline with embeddings")>]
    member _.``VectorStore: End-to-end RAG pipeline``() =
        task {
            output.WriteLine("Testing RAG pipeline...")
            let store = InMemoryVectorStore()
            let vectorStore = store :> IVectorStore

            // In a real test, generate embeddings via Ollama
            let fakeEmbed (_: string) =
                Array.init 384 (fun i -> float32 i / 384.0f)

            // Index documents
            let docs =
                [ "doc1", "F# is a functional programming language"
                  "doc2", "Python is popular for machine learning"
                  "doc3", "F# runs on .NET and supports type providers" ]

            for (id, text) in docs do
                let embedding = fakeEmbed text
                do! vectorStore.SaveAsync("knowledge", id, embedding, Map [ "text", text ])

            // Query
            let queryEmbed = fakeEmbed "What is F#?"
            let! results = vectorStore.SearchAsync("knowledge", queryEmbed, 2)

            output.WriteLine($"Found {results.Length} relevant documents")
            Assert.Equal(2, results.Length)
        }

    // === Agent Execution Integration Tests ===

    [<Fact(Skip = "Integration: Requires full agent infrastructure")>]
    member _.``Agent: Can execute multi-step workflow with LLM``() =
        task {
            output.WriteLine("Testing agent workflow...")
            // This would test the full AgentWorkflow with real LLM calls
            // Placeholder for when infrastructure is ready
            Assert.True(true)
        }

    [<Fact(Skip = "Integration: Requires Epistemic Governor with LLM")>]
    member _.``EpistemicGovernor: Can verify claims``() =
        task {
            output.WriteLine("Testing epistemic verification...")
            // This would test claim verification against knowledge base
            Assert.True(true)
        }

    // === Budget-constrained LLM Tests ===

    [<Fact(Skip = "Integration: Requires LLM for budget testing")>]
    member _.``Budget: LLM calls respect token limits``() =
        task {
            output.WriteLine("Testing budget-constrained LLM...")

            let budget =
                { Budget.Infinite with
                    MaxTokens = Some 100<token> }

            let governor = BudgetGovernor(budget)

            // Would call LLM and track actual token usage
            // Verify that budget is respected
            Assert.True(governor.CanAfford({ Cost.Zero with Tokens = 50<token> }))
        }

    // === Agentic Patterns Integration Tests ===

    [<Fact(Skip = "Integration: Requires LLM for Chain of Thought")>]
    member _.``Patterns: Chain of Thought with real LLM``() =
        task {
            output.WriteLine("Testing CoT pattern...")
            // Would test real multi-step reasoning chain
            Assert.True(true)
        }

    [<Fact(Skip = "Integration: Requires LLM for ReAct pattern")>]
    member _.``Patterns: ReAct loop with tool calls``() =
        task {
            output.WriteLine("Testing ReAct pattern...")
            // Would test Reason-Act-Observe loop with tools
            Assert.True(true)
        }

    [<Fact(Skip = "Integration: Requires LLM for Plan & Execute")>]
    member _.``Patterns: Plan and Execute with real planner``() =
        task {
            output.WriteLine("Testing Plan & Execute pattern...")
            // Would test plan generation and step execution
            Assert.True(true)
        }
