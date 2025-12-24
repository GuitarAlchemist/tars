namespace Tars.Tests

open Xunit
open Tars.Core
open Tars.Kernel
open Tars.Graph
open Tars.Graph.GraphRuntime
open Tars.Tools
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Cortex
open System.Threading.Tasks
open System.Collections.Generic
open System

type SelfReflectionTests() =

    [<Fact>]
    member _.``Agent uses search_docs when asked about TARS architecture``() =
        // 1. Setup Mock Vector Store
        let store = InMemoryVectorStore() :> IVectorStore
        // Pre-populate with a fake doc
        task {
            do!
                store.SaveAsync(
                    "docs",
                    "arch.md",
                    [| 0.1f; 0.2f |],
                    Map
                        [ ("title", "Architecture")
                          ("content", "TARS v2 uses a Hexagonal Architecture.")
                          ("source", "arch.md") ]
                )
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously

        // 2. Setup Mock LLM that returns a tool call
        // We simulate the LLM choosing the tool
        let mockLlm =
            { new ILlmService with
                member _.CompleteAsync req =
                    task {
                        // Return tool call unconditionally for this test
                        return
                            { Text = "TOOL:search_docs:architecture"
                              Usage = None
                              FinishReason = None
                              Raw = None }
                    }

                member _.CompleteStreamAsync(req, onToken) =
                    task {
                        return
                            { Text = ""
                              FinishReason = None
                              Usage = None
                              Raw = None }
                    }

                member _.EmbedAsync text = task { return [| 0.1f; 0.2f |] } } // Fake embedding

        // 3. Create Tool
        let searchTool =
            Tars.Core.Tool.Create(
                "search_docs",
                "Searches docs",
                fun args -> task { return Ok "Found some docs about Hexagonal Architecture" }
            )

        // 4. Setup Agent
        let tools = [ searchTool ]

        let agentBase =
            AgentFactory.create (Guid.NewGuid()) "TestAgent" "1.0" "mock-model" "System Prompt" tools []

        // Inject user message to trigger Thinking state
        let userMsg: Message =
            { Id = Guid.NewGuid()
              CorrelationId = CorrelationId(Guid.NewGuid())
              Sender = MessageEndpoint.User
              Receiver = Some(MessageEndpoint.Agent agentBase.Id)
              Performative = Performative.Query
              Intent = None
              Constraints = SemanticConstraints.Default
              Ontology = None
              Language = "text"
              Content = "Tell me about TARS architecture."
              Timestamp = DateTime.UtcNow
              Metadata = Map.empty }

        let memory = [ userMsg ]

        let agent =
            { agentBase with
                Memory = memory
                State = Thinking memory }

        let ctx =
            { Registry = AgentRegistry()
              Llm = mockLlm
              MaxSteps = 5
              BudgetGovernor = None
              OutputGuard = None
              CancellationToken = System.Threading.CancellationToken.None
              Logger = fun _ -> () }

        // 5. Run Step
        // Agent calls tool
        let outcome =
            GraphRuntime.step agent ctx |> Async.AwaitTask |> Async.RunSynchronously

        match outcome with
        | Success next ->
            match next.State with
            | Acting(t, _) when t.Name = "search_docs" -> Assert.True(true)
            | other -> Assert.Fail($"Expected Acting(search_docs), got {other}")
        | _ -> Assert.Fail("Step failed")
