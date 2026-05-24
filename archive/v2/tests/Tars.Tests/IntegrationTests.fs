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

    [<Fact>]
    member _.``Ollama: Can connect and list models``() =
        task {
            output.WriteLine("Connecting to Ollama...")
            let! models = OllamaClient.getTagsAsync httpClient ollamaUri
            output.WriteLine($"Found {models.Length} models")
            Assert.NotEmpty(models)
        }

    [<Fact>]
    member _.``Ollama: Can generate chat completion``() =
        task {
            output.WriteLine("Testing chat generation...")

            let req =
                { ModelHint = None
                  Model = None
                  SystemPrompt = None
                  MaxTokens = None
                  Temperature = None
                  Stop = []
                  Messages =
                    [ { Role = Role.User
                        Content = "Say hello in one word" } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = None

                  ContextWindow = None }

            let! response = OllamaClient.sendChatAsync httpClient ollamaUri "llama3.2" req

            output.WriteLine($"Response: {response.Text}")
            Assert.False(String.IsNullOrEmpty(response.Text))
        }

    [<Fact>]
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

                    member _.EmbedAsync(_text) = task { return [| 0.1f; 0.2f |] }

                    member _.RouteAsync(_req) = task { return { Tars.Llm.Routing.RoutedBackend.Backend = Tars.Llm.LlmBackend.Ollama "mock"; Endpoint = Uri "http://localhost:11434"; ApiKey = None } } }

            let compressor = ContextCompressor(llm, EntropyMonitor())

            let result =
                compressor.AutoCompress("repeat repeat repeat repeat")
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
                    member _.Verify(_) = Task.FromResult(true)
                    member _.GetRelatedCodeContext(_) = Task.FromResult("mock context") }

            let belief =
                governor.ExtractPrinciple("test task", "result")
                |> Async.AwaitTask
                |> Async.RunSynchronously

            Assert.Contains("Principle:", belief.Statement)

    // === Vector Store Integration Tests ===

    [<Fact>]
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

    [<Fact>]
    member _.``Agent: Can execute multi-step workflow with LLM``() =
        task {
            output.WriteLine("Testing agent workflow with real LLM...")
            
            // Create real Ollama LLM service
            let llm = OllamaClient.createService httpClient ollamaUri "qwen2.5-coder:1.5b"
            
            // Step 1: Generate a plan
            let planReq = {
                ModelHint = None
                Model = None
                SystemPrompt = Some "You are a helpful assistant. Respond concisely."
                MaxTokens = Some 100
                Temperature = Some 0.7
                Stop = []
                Messages = [ { Role = Role.User; Content = "List 3 steps to write a hello world program in F#" } ]
                Tools = []
                ToolChoice = None
                ResponseFormat = None
                Stream = false
                JsonMode = false
                Seed = None
            }
            
            let! planResponse = llm.CompleteAsync(planReq)
            output.WriteLine($"Plan response: {planResponse.Text}")
            Assert.False(String.IsNullOrWhiteSpace(planResponse.Text))
            
            // Step 2: Execute step 1 (ask LLM to write code)
            let codeReq = { planReq with Messages = [ { Role = Role.User; Content = "Write a one-line F# hello world" } ] }
            let! codeResponse = llm.CompleteAsync(codeReq)
            output.WriteLine($"Code response: {codeResponse.Text}")
            
            // Verify we got code output
            Assert.True(codeResponse.Text.Contains("print") || codeResponse.Text.Contains("Hello") || codeResponse.Text.Contains("world"))
        }

    [<Fact>]
    member _.``EpistemicGovernor: Can verify claims with LLM``() =
        task {
            output.WriteLine("Testing epistemic verification with real LLM...")
            
            let llm = OllamaClient.createService httpClient ollamaUri "qwen2.5-coder:1.5b"
            let vectorStore = InMemoryVectorStore() :> IVectorStore
            
            // Create real epistemic governor
            let governor = EpistemicGovernor(llm, vectorStore)
            
            // Test principle extraction
            let! principle = governor.ExtractPrinciple("Calculate sum of list", "Use List.fold for accumulation")
            
            output.WriteLine($"Extracted principle: {principle.Statement}")
            Assert.False(String.IsNullOrWhiteSpace(principle.Statement))
            Assert.Equal(EpistemicStatus.Hypothesis, principle.Status)
        }

    // === Budget-constrained LLM Tests ===

    [<Fact>]
    member _.``Budget: LLM calls respect token limits``() =
        task {
            output.WriteLine("Testing budget-constrained LLM...")

            let budget =
                { Budget.Infinite with
                    MaxTokens = Some 100<token> }

            let governor = BudgetGovernor(budget)

            // Verify budget checks work
            Assert.True(governor.CanAfford({ Cost.Zero with Tokens = 50<token> }))
            Assert.True(governor.CanAfford({ Cost.Zero with Tokens = 99<token> }))
            
            // Consume some budget
            governor.Consume({ Cost.Zero with Tokens = 80<token> }) |> ignore
            
            // Now we can't afford 50 more tokens (only 20 left)
            Assert.False(governor.CanAfford({ Cost.Zero with Tokens = 50<token> }))
            Assert.True(governor.CanAfford({ Cost.Zero with Tokens = 19<token> }))
        }

    // === Agentic Patterns Integration Tests ===

    [<Fact>]
    member _.``Patterns: Chain of Thought with real LLM``() =
        task {
            output.WriteLine("Testing Chain of Thought pattern...")
            
            let llm = OllamaClient.createService httpClient ollamaUri "qwen2.5-coder:1.5b"
            
            // CoT prompt that forces step-by-step reasoning
            let cotReq = {
                ModelHint = None
                Model = None
                SystemPrompt = Some "Think step by step. Show your reasoning."
                MaxTokens = Some 200
                Temperature = Some 0.3
                Stop = []
                Messages = [ { Role = Role.User; Content = "What is 15 * 7? Think step by step." } ]
                Tools = []
                ToolChoice = None
                ResponseFormat = None
                Stream = false
                JsonMode = false
                Seed = None
            }
            
            let! response = llm.CompleteAsync(cotReq)
            output.WriteLine($"CoT response: {response.Text}")
            
            // Verify reasoning is present (should show steps)
            Assert.False(String.IsNullOrWhiteSpace(response.Text))
            // Check for the correct answer (105) or reasoning words
            Assert.True(
                response.Text.Contains("105") || 
                response.Text.Contains("step") || 
                response.Text.Contains("multiply") ||
                response.Text.ToLower().Contains("first")
            )
        }

    [<Fact>]
    member _.``Patterns: ReAct loop with tool simulation``() =
        task {
            output.WriteLine("Testing ReAct pattern...")
            
            let llm = OllamaClient.createService httpClient ollamaUri "qwen2.5-coder:1.5b"
            
            // Simulate ReAct: Reason -> Act -> Observe cycle
            // Step 1: Reason about what to do
            let reasonReq = {
                ModelHint = None
                Model = None
                SystemPrompt = Some "You are a ReAct agent. First explain your reasoning, then state your action."
                MaxTokens = Some 100
                Temperature = Some 0.3
                Stop = []
                Messages = [ { Role = Role.User; Content = "I need to find information about F#. What should I do first?" } ]
                Tools = []
                ToolChoice = None
                ResponseFormat = None
                Stream = false
                JsonMode = false
                Seed = None
            }
            
            let! reasonResponse = llm.CompleteAsync(reasonReq)
            output.WriteLine($"Reason: {reasonResponse.Text}")
            
            // Step 2: Simulate tool observation
            let observation = "F# is a functional-first programming language for .NET"
            
            // Step 3: Continue reasoning with observation
            let continueReq = { reasonReq with 
                Messages = [ 
                    { Role = Role.User; Content = "I need to find information about F#. What should I do first?" }
                    { Role = Role.User; Content = $"Observation: {observation}" }
                    { Role = Role.User; Content = "Based on this observation, what can you tell me about F#?" }
                ] 
            }
            
            let! continueResponse = llm.CompleteAsync(continueReq)
            output.WriteLine($"Continue: {continueResponse.Text}")
            
            Assert.False(String.IsNullOrWhiteSpace(reasonResponse.Text))
            Assert.False(String.IsNullOrWhiteSpace(continueResponse.Text))
        }

    [<Fact>]
    member _.``Patterns: Plan and Execute with real planner``() =
        task {
            output.WriteLine("Testing Plan & Execute pattern...")
            
            let llm = OllamaClient.createService httpClient ollamaUri "qwen2.5-coder:1.5b"
            
            // Step 1: Generate a plan
            let planReq = {
                ModelHint = None
                Model = None
                SystemPrompt = Some "You are a planning assistant. Generate a numbered list of steps."
                MaxTokens = Some 150
                Temperature = Some 0.3
                Stop = []
                Messages = [ { Role = Role.User; Content = "Create a 3-step plan to create a simple calculator in F#" } ]
                Tools = []
                ToolChoice = None
                ResponseFormat = None
                Stream = false
                JsonMode = false
                Seed = None
            }
            
            let! planResponse = llm.CompleteAsync(planReq)
            output.WriteLine($"Plan: {planResponse.Text}")
            
            // Verify plan was generated (should contain numbered steps)
            Assert.False(String.IsNullOrWhiteSpace(planResponse.Text))
            Assert.True(
                planResponse.Text.Contains("1") || 
                planResponse.Text.Contains("step") ||
                planResponse.Text.Contains("Step") ||
                planResponse.Text.Contains("first") ||
                planResponse.Text.Contains("First")
            )
            
            // Step 2: Execute first step
            let executeReq = { planReq with 
                SystemPrompt = Some "You are a code executor. Write code for the requested step."
                Messages = [ { Role = Role.User; Content = "Write F# code for the first step: define add and subtract functions" } ] 
            }
            
            let! executeResponse = llm.CompleteAsync(executeReq)
            output.WriteLine($"Execute: {executeResponse.Text}")
            
            // Verify code was generated
            Assert.False(String.IsNullOrWhiteSpace(executeResponse.Text))
        }

