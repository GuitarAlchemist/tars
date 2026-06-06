namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Llm
open Tars.Cortex
open Tars.Cortex.Patterns

module WotPersistenceTests =
    
    [<Fact>]
    let ``WoT Persists``() =
        task {
            // Mock Graph with closure
            let mutable nodes = [] : TarsEntity list
            let graph = 
                { new IGraphService with
                    member _.AddNodeAsync(e) = 
                        nodes <- e :: nodes
                        Task.FromResult("id")
                    member _.AddFactAsync(f) = Task.FromResult(Guid.NewGuid())
                    member _.AddEpisodeAsync(e) = Task.FromResult("ep")
                    member _.QueryAsync(q) = Task.FromResult([] : TarsFact list)
                    member _.PersistAsync() = Task.FromResult(()) 
                }

            // Mock LLM
            let llm = 
                { new ILlmService with 
                    member _.CompleteAsync(_) = 
                        Task.FromResult({ Text="Refined thought"; Usage=None; FinishReason=None; Raw=None })
                    member _.CompleteStreamAsync(_, _) = failwith "Not impl"
                    member _.EmbedAsync(_) = Task.FromResult([||] : float32[])
                    member _.RouteAsync(_) = 
                        Task.FromResult({ Backend = Tars.Llm.LlmBackend.Ollama("mock"); Endpoint = Uri("http://localhost"); ApiKey = None })
                }

            // Context
            let agent = 
                { Id = AgentId(Guid.NewGuid()); Name = "Test"; Version="1"; ParentVersion=None; CreatedAt=DateTime.UtcNow; Model="mock"; SystemPrompt=""; Tools=[]; Capabilities=[]; State=AgentState.Idle; Memory=[]; Fitness=1.0; Drives=Unchecked.defaultof<_>; Constitution=Unchecked.defaultof<_> }
            
            let ctx = 
                { Self = agent
                  Registry = Unchecked.defaultof<_>
                  Executor = Unchecked.defaultof<_>
                  Logger = ignore
                  Budget = None
                  Epistemic = None
                  SemanticMemory = None
                  KnowledgeGraph = Some graph
                  SymbolicReflector = None
                  CapabilityStore = None
                  Audit = None
                  CancellationToken = System.Threading.CancellationToken.None }
            
            let wotConfig = 
                { BaseConfig = 
                    { BranchingFactor = 1; MaxDepth = 1; TopK = 1; ScoreThreshold = 0.0; MinConfidence = 0.0; DiversityThreshold = 0.0; DiversityPenalty = 0.0; Constraints = []; EnableCritique = false; EnablePolicyChecks = false; EnableMemoryRecall = false; TrackEdges = false }
                  RequiredPolicies = []
                  AvailableTools = []
                  RoleAssignments = Map.empty
                  MemoryNamespace = None
                  MaxEscalations = 0
                  TimeoutMs = None }
            
            let! res = Patterns.workflowOfThought llm wotConfig "Test" ctx
            
            Assert.NotEmpty(nodes)
            let runNodes = nodes |> List.choose (function TarsEntity.RunE r -> Some r | _ -> None)
            Assert.NotEmpty(runNodes)
        }
