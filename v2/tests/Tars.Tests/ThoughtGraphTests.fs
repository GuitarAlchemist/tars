module Tars.Tests.ThoughtGraphTests

open System
open System.Threading
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Llm
open Tars.Cortex.ThoughtGraph

// Issue #280: the generate/score helpers were private to Patterns.fs and only reachable
// through a whole Graph/Tree/Workflow-of-Thought run.

/// Replies with `replies` in order; embeddings are not used by these tests.
type private ScriptedLlm(replies: string list) =
    let mutable queue = replies

    interface ILlmService with
        member _.CompleteAsync(_) =
            let next =
                match queue with
                | r :: rest ->
                    queue <- rest
                    r
                | [] -> ""

            Task.FromResult
                { Text = next
                  FinishReason = Some "stop"
                  Usage = None
                  Raw = None }

        member _.CompleteStreamAsync(_, _) = failwith "not used"
        member _.EmbedAsync(_) = failwith "not used"
        member _.RouteAsync(_) = failwith "not used"

let private context () : AgentContext =
    let agent: Agent =
        { Id = AgentId(Guid.NewGuid())
          Name = "TestAgent"
          Version = "1.0.0"
          ParentVersion = None
          CreatedAt = DateTime.UtcNow
          Model = "test-model"
          SystemPrompt = ""
          Tools = []
          Capabilities = []
          State = AgentState.Idle
          Memory = []
          Fitness = 0.0
          Drives =
            { Accuracy = 0.5
              Speed = 0.5
              Creativity = 0.5
              Safety = 0.5 }
          Constitution = AgentConstitution.Create(AgentId(Guid.NewGuid()), NeuralRole.GeneralReasoning) }

    { Self = agent
      Registry =
        { new IAgentRegistry with
            member _.GetAgent(_) = async { return None }
            member _.FindAgents(_) = async { return [] }
            member _.GetAllAgents() = async { return [] } }
      Executor =
        { new IAgentExecutor with
            member _.Execute(_, _) = async { return Success "" } }
      Logger = ignore
      Budget = None
      Epistemic = None
      SemanticMemory = None
      KnowledgeGraph = None
      SymbolicReflector = None
      CapabilityStore = None
      Audit = None
      CancellationToken = CancellationToken.None }

/// No embeddings: diversity off.
let private noDiversity =
    { defaultGoTConfig with
        BranchingFactor = 2
        DiversityPenalty = 0.0
        DiversityThreshold = 1.0 }

let private node content embedding : ThoughtNode =
    { Id = Guid.NewGuid()
      Content = content
      Score = None
      Evaluation = None
      Embedding = embedding
      ParentIds = []
      Depth = 1
      Operation = Generate
      Path = [] }

[<Fact>]
let ``generateThoughts turns headed replies into child nodes, capped at the branching factor`` () =
    let parent = Guid.NewGuid()

    let reply =
        """Here are my thoughts:
THOUGHT 1: Profile the parser before changing anything.
THOUGHT 2: Cache compiled grammars between requests.
THOUGHT 3: Rewrite the tokenizer in a single pass."""

    let nodes =
        generateThoughts (ScriptedLlm [ reply ]) (context ()) noDiversity "Speed up parsing" "" [] [ parent ] 2
        |> Async.RunSynchronously

    Assert.Equal<string>(
        [ "Profile the parser before changing anything."
          "Cache compiled grammars between requests." ],
        nodes |> List.map (fun n -> n.Content)
    )

    Assert.All(
        nodes,
        fun n ->
            Assert.Equal(2, n.Depth)
            Assert.Equal<Guid>([ parent ], n.ParentIds)
    )

[<Fact>]
let ``scoreThought reads the evaluator's JSON`` () =
    let reply =
        """{"score": 0.8, "confidence": 0.7, "reasons": ["concrete"], "risks": ["untested"]}"""

    let scored =
        scoreThought (ScriptedLlm [ reply ]) (context ()) noDiversity "goal" "" [] (node "Cache grammars" None)
        |> Async.RunSynchronously

    let evaluation = scored.Evaluation.Value
    Assert.Equal(Some 0.8, scored.Score)
    Assert.Equal(0.7, evaluation.Confidence)
    Assert.Equal<string>([ "concrete" ], evaluation.Reasons)
    Assert.Contains("untested", evaluation.Risks)

[<Fact>]
let ``scoreThought penalises a near-duplicate of a peer`` () =
    let config =
        { defaultGoTConfig with
            DiversityThreshold = 0.85
            DiversityPenalty = 0.25 }

    let reply = """{"score": 0.8, "confidence": 0.7, "reasons": [], "risks": []}"""
    let duplicate = node "Cache grammars" (Some [| 1.0f; 0.0f |])
    let peer = node "Cache the grammars" (Some [| 1.0f; 0.0f |])

    let scored =
        scoreThought (ScriptedLlm [ reply ]) (context ()) config "goal" "" [ peer ] duplicate
        |> Async.RunSynchronously

    // Identical direction: similarity 1.0 >= 0.85, so 0.8 - 0.25 * 1.0.
    Assert.Equal(0.55, scored.Score.Value, 6)
