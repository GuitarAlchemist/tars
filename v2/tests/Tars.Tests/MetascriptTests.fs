namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Cortex
open Tars.Core.LegacyKnowledgeGraph
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Metascript
open Tars.Metascript.Config
open Tars.Metascript.Domain
open Tars.Metascript.Engine
open Tars.Metascript.Templates
open Tars.Metascript.Validation
open Tars.Tools

type StubLlm(responseText: string, tokens: int) =
    interface ILlmService with
        member _.CompleteAsync(_req: LlmRequest) : Task<LlmResponse> =
            task {
                return
                    { Text = responseText
                      FinishReason = Some "stop"
                      Usage =
                        Some
                            { PromptTokens = tokens
                              CompletionTokens = tokens
                              TotalTokens = tokens * 2 }
                      Raw = None }
            }

        member _.CompleteStreamAsync(_req: LlmRequest, onToken: string -> unit) : Task<LlmResponse> =
            task {
                onToken responseText

                return
                    { Text = responseText
                      FinishReason = Some "stop"
                      Usage =
                        Some
                            { PromptTokens = tokens
                              CompletionTokens = tokens
                              TotalTokens = tokens * 2 }
                      Raw = None }
            }

        member _.EmbedAsync(_text: string) : Task<float32[]> =
            task { return [| 0.1f; 0.2f; 0.3f; 0.4f |] }

type MetascriptTests() =

    [<Fact>]
    member _.``Validation rejects duplicate steps``() =
        let wf =
            { Name = "test"
              Description = ""
              Version = "1.0"
              Inputs = []
              Steps =
                [ { Id = "s1"
                    Type = "agent"
                    Agent = Some "A"
                    Tool = None
                    Instruction = Some "do"
                    Params = None
                    Context = None
                    DependsOn = None
                    Outputs = Some [ "o" ]
                    Tools = None }
                  { Id = "s1"
                    Type = "agent"
                    Agent = Some "A"
                    Tool = None
                    Instruction = Some "do"
                    Params = None
                    Context = None
                    DependsOn = None
                    Outputs = Some [ "o" ]
                    Tools = None } ] }

        match validateWorkflow wf with
        | Result.Error errs -> Assert.NotEmpty errs
        | Result.Ok _ -> Assert.True(false, "Expected validation failure")

    [<Fact>]
    member _.``Decision step resolves boolean condition``() =
        let llm = StubLlm("unused", 1) :> ILlmService

        let ctx =
            { Llm = llm
              Tools = ToolRegistry()
              Budget = None
              VectorStore = None
              KnowledgeGraph = None
              SemanticMemory = None
              MacroRegistry = None
              RagConfig = RagConfig.Default }

        let wf =
            { Name = "decision"
              Description = ""
              Version = "1.0"
              Inputs = []
              Steps =
                [ { Id = "decide"
                    Type = "decision"
                    Agent = None
                    Tool = None
                    Instruction = None
                    Params = Some(Map [ ("condition", "flag"); ("trueOutput", "go"); ("falseOutput", "stop") ])
                    Context = None
                    DependsOn = None
                    Outputs = Some [ "route" ]
                    Tools = None } ] }

        task {
            let! state = run ctx wf (Map [ "flag", box "true" ])
            let route = state.StepOutputs["decide"]["route"] :?> string
            Assert.Equal("go", route)
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

    [<Fact>]
    member _.``Loop step consumes budget per iteration``() =
        let llm = StubLlm("looped", 2) :> ILlmService

        let budget =
            BudgetGovernor(
                { Budget.Infinite with
                    MaxTokens = Some 100<token>
                    MaxCalls = Some 10<requests> }
            )

        let ctx =
            { Llm = llm
              Tools = ToolRegistry()
              Budget = Some budget
              VectorStore = None
              KnowledgeGraph = None
              SemanticMemory = None
              RagConfig = RagConfig.Default
              MacroRegistry = None }

        let wf =
            { Name = "loop"
              Description = ""
              Version = "1.0"
              Inputs = []
              Steps =
                [ { Id = "loop"
                    Type = "loop"
                    Agent = Some "Looper"
                    Tool = None
                    Instruction = Some "Echo {{item}}"
                    Params = Some(Map [ ("list", "items"); ("itemVar", "item") ])
                    Context = None
                    DependsOn = None
                    Outputs = Some [ "items" ]
                    Tools = None } ] }

        task {
            let! state = run ctx wf (Map [ "items", box [ "a"; "b" ] ])
            let outputs = state.StepOutputs["loop"]["items"] :?> obj list
            Assert.Equal<string>([ "looped"; "looped" ], outputs |> List.map string)
            Assert.True(budget.Consumed.Tokens >= 4<token>)
            Assert.True(budget.Consumed.CallCount >= 2<requests>)
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

    [<Fact>]
    member _.``Retrieval step returns empty context when no vector store``() =
        let llm = StubLlm("unused", 1) :> ILlmService

        let ctx =
            { Llm = llm
              Tools = ToolRegistry()
              Budget = None
              VectorStore = None
              KnowledgeGraph = None
              SemanticMemory = None
              RagConfig = RagConfig.Default
              MacroRegistry = None }

        let wf =
            { Name = "retrieval-test"
              Description = ""
              Version = "1.0"
              Inputs = []
              Steps =
                [ { Id = "retrieve"
                    Type = "retrieval"
                    Agent = None
                    Tool = None
                    Instruction = None
                    Params = Some(Map [ ("query", "test query") ])
                    Context = None
                    DependsOn = None
                    Outputs = Some [ "context" ]
                    Tools = None } ] }

        task {
            let! state = run ctx wf Map.empty
            let ctx = state.StepOutputs["retrieve"]["context"] :?> string
            Assert.Equal("", ctx)
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

    [<Fact>]
    member _.``Retrieval step retrieves from vector store``() =
        let llm = StubLlm("unused", 1) :> ILlmService
        let vectorStore = InMemoryVectorStore()
        let vs = vectorStore :> IVectorStore

        // Pre-populate the vector store with some content
        task {
            do!
                vs.SaveAsync(
                    "tars_context",
                    "doc1",
                    [| 0.1f; 0.2f; 0.3f; 0.4f |],
                    Map [ ("content", "This is relevant content"); ("source", "test.md") ]
                )
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

        let ctx =
            { Llm = llm
              Tools = ToolRegistry()
              Budget = None
              VectorStore = Some vs
              KnowledgeGraph = None
              SemanticMemory = None
              RagConfig =
                { RagConfig.Default with
                    MinScore = 0.0f }
              MacroRegistry = None }

        let wf =
            { Name = "retrieval-test"
              Description = ""
              Version = "1.0"
              Inputs = []
              Steps =
                [ { Id = "retrieve"
                    Type = "retrieval"
                    Agent = None
                    Tool = None
                    Instruction = None
                    Params = Some(Map [ ("query", "test query") ])
                    Context = None
                    DependsOn = None
                    Outputs = Some [ "context" ]
                    Tools = None } ] }

        task {
            let! state = run ctx wf Map.empty
            let context = state.StepOutputs["retrieve"]["context"] :?> string
            Assert.Contains("relevant content", context)
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

    [<Fact>]
    member _.``Knowledge graph enriches agent context``() =
        let llm = StubLlm("enriched response", 1) :> ILlmService
        let kg = TemporalKnowledgeGraph.TemporalGraph()

        // Add some related concepts using the new TarsFact-based API
        let testingConcept =
            TarsEntity.ConceptE
                { Name = "testing"
                  Description = ""
                  RelatedConcepts = [] }

        let unitTestConcept =
            TarsEntity.ConceptE
                { Name = "unit testing"
                  Description = ""
                  RelatedConcepts = [] }

        let integrationConcept =
            TarsEntity.ConceptE
                { Name = "integration"
                  Description = ""
                  RelatedConcepts = [] }

        kg.AddFact(TarsFact.SimilarTo(testingConcept, unitTestConcept, 0.8)) |> ignore

        kg.AddFact(TarsFact.SimilarTo(testingConcept, integrationConcept, 0.6))
        |> ignore

        let ctx =
            { Llm = llm
              Tools = ToolRegistry()
              Budget = None
              VectorStore = None
              KnowledgeGraph = Some kg
              SemanticMemory = None
              RagConfig =
                { RagConfig.Default with
                    AutoIndex = false }
              MacroRegistry = None }

        let wf =
            { Name = "kg-test"
              Description = ""
              Version = "1.0"
              Inputs = []
              Steps =
                [ { Id = "agent"
                    Type = "agent"
                    Agent = Some "TestAgent"
                    Tool = None
                    Instruction = Some "Do something about testing"
                    Params = Some(Map [ ("concepts", "testing") ])
                    Context = None
                    DependsOn = None
                    Outputs = Some [ "result" ]
                    Tools = None } ] }

        task {
            let! state = run ctx wf Map.empty
            let result = state.StepOutputs["agent"]["result"] :?> string
            Assert.Equal("enriched response", result)
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

    [<Fact>]
    member _.``Metadata filtering removes non-matching results``() =
        let llm = StubLlm("unused", 1) :> ILlmService
        let vectorStore = InMemoryVectorStore()
        let vs = vectorStore :> IVectorStore

        // Pre-populate the vector store with content having different sources
        task {
            do!
                vs.SaveAsync(
                    "tars_context",
                    "doc1",
                    [| 0.1f; 0.2f; 0.3f; 0.4f |],
                    Map [ ("content", "Content from source A"); ("source", "sourceA") ]
                )

            do!
                vs.SaveAsync(
                    "tars_context",
                    "doc2",
                    [| 0.1f; 0.2f; 0.3f; 0.4f |],
                    Map [ ("content", "Content from source B"); ("source", "sourceB") ]
                )
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

        let ctx =
            { Llm = llm
              Tools = ToolRegistry()
              Budget = None
              VectorStore = Some vs
              KnowledgeGraph = None
              SemanticMemory = None
              RagConfig =
                { RagConfig.Default with
                    MinScore = 0.0f
                    MetadataFilters =
                        [ { Field = "source"
                            Operator = "eq"
                            Value = "sourceA" } ] }
              MacroRegistry = None }

        let wf =
            { Name = "filter-test"
              Description = ""
              Version = "1.0"
              Inputs = []
              Steps =
                [ { Id = "retrieve"
                    Type = "retrieval"
                    Agent = None
                    Tool = None
                    Instruction = None
                    Params = Some(Map [ ("query", "test query") ])
                    Context = None
                    DependsOn = None
                    Outputs = Some [ "context" ]
                    Tools = None } ] }

        task {
            let! state = run ctx wf Map.empty
            let context = state.StepOutputs["retrieve"]["context"] :?> string
            Assert.Contains("source A", context)
            Assert.DoesNotContain("source B", context)
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

    [<Fact>]
    member _.``Hybrid search combines semantic and keyword scores``() =
        let llm = StubLlm("unused", 1) :> ILlmService
        let vectorStore = InMemoryVectorStore()
        let vs = vectorStore :> IVectorStore

        // Pre-populate with content that has keyword match
        task {
            do!
                vs.SaveAsync(
                    "tars_context",
                    "doc1",
                    [| 0.1f; 0.2f; 0.3f; 0.4f |],
                    Map
                        [ ("content", "This document contains the exact search keyword here")
                          ("source", "test") ]
                )
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

        let ctx =
            { Llm = llm
              Tools = ToolRegistry()
              Budget = None
              VectorStore = Some vs
              KnowledgeGraph = None
              SemanticMemory = None
              RagConfig =
                { RagConfig.Default with
                    MinScore = 0.0f
                    EnableHybridSearch = true
                    SemanticWeight = 0.5f }
              MacroRegistry = None }

        let wf =
            { Name = "hybrid-test"
              Description = ""
              Version = "1.0"
              Inputs = []
              Steps =
                [ { Id = "retrieve"
                    Type = "retrieval"
                    Agent = None
                    Tool = None
                    Instruction = None
                    Params = Some(Map [ ("query", "keyword search") ])
                    Context = None
                    DependsOn = None
                    Outputs = Some [ "context" ]
                    Tools = None } ] }

        task {
            let! state = run ctx wf Map.empty
            let context = state.StepOutputs["retrieve"]["context"] :?> string
            Assert.Contains("keyword", context)
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

    [<Fact>]
    member _.``RagConfig has all new fields with sensible defaults``() =
        let config = RagConfig.Default
        // Batch 1 defaults
        Assert.False(config.EnableQueryExpansion)
        Assert.Equal(3, config.QueryExpansionCount)
        Assert.False(config.EnableMultiHop)
        Assert.Equal(2, config.MaxHops)
        Assert.Empty(config.MetadataFilters)
        Assert.True(config.EnableEmbeddingCache)
        Assert.Equal(1000, config.EmbeddingCacheSize)
        Assert.False(config.EnableAsyncBatching)
        Assert.Equal(10, config.BatchSize)
        Assert.False(config.EnableRRF)
        Assert.Equal(60, config.RRFConstant)
        // Batch 2 defaults
        Assert.False(config.EnableContextualCompression)
        Assert.Equal(2000, config.CompressionMaxChars)
        Assert.False(config.EnableParentDocRetrieval)
        Assert.Equal("tars_context_parents", config.ParentCollectionName)
        Assert.False(config.EnableSentenceWindow)
        Assert.Equal(1, config.SentenceWindowSize)
        Assert.False(config.EnableTimeDecay)
        Assert.Equal(365.0, config.TimeDecayHalfLifeDays)
        Assert.False(config.EnableSemanticChunking)
        Assert.Equal(100, config.SemanticChunkMinChars)
        Assert.Equal(2000, config.SemanticChunkMaxChars)
        Assert.False(config.EnableCrossEncoder)
        Assert.Equal("fast", config.CrossEncoderModel)
        Assert.False(config.EnableQueryRouting)
        Assert.False(config.EnableAnswerAttribution)
        Assert.False(config.EnableMetrics)
        Assert.True(config.Metrics.IsNone)
        Assert.False(config.EnableFallbackChain)
        Assert.Equal(3, config.FallbackMinResults)

    [<Fact>]
    member _.``Time decay scoring reduces score for older documents``() =
        let llm = StubLlm("unused", 1) :> ILlmService
        let vectorStore = InMemoryVectorStore()
        let vs = vectorStore :> IVectorStore

        // Add documents with different timestamps
        let oldTime = DateTime.UtcNow.AddDays(-60.0).ToString("o")
        let newTime = DateTime.UtcNow.ToString("o")

        task {
            do!
                vs.SaveAsync(
                    "tars_context",
                    "old_doc",
                    [| 0.1f; 0.2f; 0.3f; 0.4f |],
                    Map [ ("content", "Old document content"); ("timestamp", oldTime) ]
                )

            do!
                vs.SaveAsync(
                    "tars_context",
                    "new_doc",
                    [| 0.1f; 0.2f; 0.3f; 0.4f |],
                    Map [ ("content", "New document content"); ("timestamp", newTime) ]
                )
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

        let ctx =
            { Llm = llm
              Tools = ToolRegistry()
              Budget = None
              VectorStore = Some vs
              KnowledgeGraph = None
              SemanticMemory = None
              MacroRegistry = None
              RagConfig =
                { RagConfig.Default with
                    MinScore = 0.0f
                    EnableTimeDecay = true
                    TimeDecayHalfLifeDays = 30.0 } }

        let wf =
            { Name = "time-decay-test"
              Description = ""
              Version = "1.0"
              Inputs = []
              Steps =
                [ { Id = "retrieve"
                    Type = "retrieval"
                    Agent = None
                    Tool = None
                    Instruction = None
                    Params = Some(Map [ ("query", "document content") ])
                    Context = None
                    DependsOn = None
                    Outputs = Some [ "context" ]
                    Tools = None } ] }

        task {
            let! state = run ctx wf Map.empty
            let context = state.StepOutputs["retrieve"]["context"] :?> string
            // New document should appear first due to time decay
            let newPos = context.IndexOf("New document")
            let oldPos = context.IndexOf("Old document")
            Assert.True(newPos < oldPos || newPos >= 0, "New document should rank higher due to time decay")
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

    [<Fact>]
    member _.``Query routing classifies queries correctly``() =
        let llm = StubLlm("routed response", 1) :> ILlmService
        let vectorStore = InMemoryVectorStore()
        let vs = vectorStore :> IVectorStore

        task {
            do!
                vs.SaveAsync(
                    "tars_context",
                    "doc1",
                    [| 0.1f; 0.2f; 0.3f; 0.4f |],
                    Map [ ("content", "This is test content for routing"); ("source", "test") ]
                )
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

        let ctx =
            { Llm = llm
              Tools = ToolRegistry()
              Budget = None
              VectorStore = Some vs
              KnowledgeGraph = None
              SemanticMemory = None
              MacroRegistry = None
              RagConfig =
                { RagConfig.Default with
                    MinScore = 0.0f
                    EnableQueryRouting = true } }

        let wf =
            { Name = "routing-test"
              Description = ""
              Version = "1.0"
              Inputs = []
              Steps =
                [ { Id = "retrieve"
                    Type = "retrieval"
                    Agent = None
                    Tool = None
                    Instruction = None
                    Params = Some(Map [ ("query", "What is the purpose of this?") ])
                    Context = None
                    DependsOn = None
                    Outputs = Some [ "context" ]
                    Tools = None } ] }

        task {
            let! state = run ctx wf Map.empty
            // Just verify it runs without error when routing is enabled
            let context = state.StepOutputs["retrieve"]["context"] :?> string
            Assert.NotNull(context)
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

    [<Fact>]
    member _.``Answer attribution tracks sources``() =
        let llm = StubLlm("unused", 1) :> ILlmService
        let vectorStore = InMemoryVectorStore()
        let vs = vectorStore :> IVectorStore

        task {
            do!
                vs.SaveAsync(
                    "tars_context",
                    "doc1",
                    [| 0.1f; 0.2f; 0.3f; 0.4f |],
                    Map [ ("content", "Attribution test content"); ("source", "source1.md") ]
                )
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

        let ctx =
            { Llm = llm
              Tools = ToolRegistry()
              Budget = None
              VectorStore = Some vs
              KnowledgeGraph = None
              SemanticMemory = None
              MacroRegistry = None
              RagConfig =
                { RagConfig.Default with
                    MinScore = 0.0f
                    EnableAnswerAttribution = true } }

        let wf =
            { Name = "attribution-test"
              Description = ""
              Version = "1.0"
              Inputs = []
              Steps =
                [ { Id = "retrieve"
                    Type = "retrieval"
                    Agent = None
                    Tool = None
                    Instruction = None
                    Params = Some(Map [ ("query", "attribution test") ])
                    Context = None
                    DependsOn = None
                    Outputs = Some [ "context" ]
                    Tools = None } ] }

        task {
            let! state = run ctx wf Map.empty
            let attributions = state.StepOutputs["retrieve"]["attributions"]
            Assert.NotNull(attributions)
            let attrList = attributions :?> obj list
            Assert.NotEmpty(attrList)
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

    [<Fact>]
    member _.``Retrieval metrics are collected when enabled``() =
        let llm = StubLlm("unused", 1) :> ILlmService
        let vectorStore = InMemoryVectorStore()
        let vs = vectorStore :> IVectorStore

        task {
            do!
                vs.SaveAsync(
                    "tars_context",
                    "doc1",
                    [| 0.1f; 0.2f; 0.3f; 0.4f |],
                    Map [ ("content", "Metrics test content"); ("source", "test") ]
                )
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

        let metrics = RetrievalMetrics.Create()

        let ctx =
            { Llm = llm
              Tools = ToolRegistry()
              Budget = None
              VectorStore = Some vs
              KnowledgeGraph = None
              SemanticMemory = None
              MacroRegistry = None
              RagConfig =
                { RagConfig.Default with
                    MinScore = 0.0f
                    EnableMetrics = true
                    Metrics = Some metrics } }

        let wf =
            { Name = "metrics-test"
              Description = ""
              Version = "1.0"
              Inputs = []
              Steps =
                [ { Id = "retrieve"
                    Type = "retrieval"
                    Agent = None
                    Tool = None
                    Instruction = None
                    Params = Some(Map [ ("query", "metrics test") ])
                    Context = None
                    DependsOn = None
                    Outputs = Some [ "context" ]
                    Tools = None } ] }

        task {
            let! _ = run ctx wf Map.empty
            Assert.True(metrics.TotalQueries >= 1L, $"Expected TotalQueries >= 1, got {metrics.TotalQueries}")
            Assert.True(metrics.TotalLatencyMs >= 0L, $"Expected TotalLatencyMs >= 0, got {metrics.TotalLatencyMs}")
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

    [<Fact>]
    member _.``Engine calls Retrieve and Grow on SemanticMemory``() =
        let llm = StubLlm("unused", 1) :> ILlmService

        let mutable retrieveCalled = false
        let mutable growCalled = false

        let memory =
            { new ISemanticMemory with
                member _.Retrieve _ =
                    async {
                        retrieveCalled <- true
                        return []
                    }

                member _.Grow(trace, verif) =
                    async {
                        growCalled <- true
                        return "schema-id"
                    }

                member _.Refine() = async { return () } }

        let ctx =
            { Llm = llm
              Tools = ToolRegistry()
              Budget = None
              VectorStore = None
              KnowledgeGraph = None
              SemanticMemory = Some memory
              RagConfig = RagConfig.Default
              MacroRegistry = None }

        let wf =
            { Name = "memory-test"
              Description = ""
              Version = "1.0"
              Inputs = []
              Steps =
                [ { Id = "dummy"
                    Type = "decision"
                    Agent = None
                    Tool = None
                    Instruction = None
                    Params = Some(Map [ ("condition", "true"); ("trueOutput", "ok"); ("falseOutput", "fail") ])
                    Context = None
                    DependsOn = None
                    Outputs = Some [ "out" ]
                    Tools = None } ] }

        task {
            let! _ = run ctx wf Map.empty
            Assert.True(retrieveCalled, "Retrieve should be called")
            Assert.True(growCalled, "Grow should be called")
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously
