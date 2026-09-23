module Tars.Tests.WoTDecideTests

open System
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Llm
open Tars.Cortex
open Tars.Cortex.WoTTypes

// The Decide node used to read only the first character of the model's answer and, when
// nothing matched, quietly take the first candidate — a branch the model never chose.
// These tests pin the honest behaviour, and the typed (System One) path that can now
// answer instead.

/// Answers every prompt with the same text, and records what it was asked.
type private ScriptedLlm(answer: string) =
    let asked = ResizeArray<string>()

    member _.Asked = List.ofSeq asked

    interface ILlmService with
        member _.CompleteAsync(req: LlmRequest) =
            task {
                asked.Add(
                    req.Messages
                    |> List.tryLast
                    |> Option.map (fun m -> m.Content)
                    |> Option.defaultValue ""
                )

                return
                    { Text = answer
                      FinishReason = Some "stop"
                      Usage = None
                      Raw = None }
            }

        member _.CompleteStreamAsync(_req, _onToken) = raise (NotImplementedException())
        member _.EmbedAsync(_text) = Task.FromResult(Array.empty<float32>)

        member _.RouteAsync(_) =
            task {
                return
                    { Backend = Ollama "mock"
                      Endpoint = Uri "http://localhost:11434"
                      ApiKey = None }
            }

/// A Decide node touches no tool; an empty registry keeps that honest.
type private NoTools() =
    interface IToolRegistry with
        member _.Register(_tool) = ()
        member _.Get(_name) = None
        member _.GetAll() = []

let private candidates = [ "search"; "write_file"; "finish" ]

// ------------------------------------------------------------------ matching

[<Fact>]
let ``a number picks the candidate at that position`` () =
    Assert.Equal(Result.Ok "write_file", WoTExecutor.matchCandidate candidates "2")
    Assert.Equal(Result.Ok "finish", WoTExecutor.matchCandidate candidates "3. finish")

[<Fact>]
let ``a number past the end is refused rather than truncated to its first digit`` () =
    // The old parser read '1' out of "10" and ran the first candidate.
    match WoTExecutor.matchCandidate candidates "10" with
    | Result.Ok chosen -> failwith $"expected a refusal, got {chosen}"
    | Result.Error message -> Assert.Contains("names none of the 3 candidates", message)

[<Fact>]
let ``a candidate named in words is matched`` () =
    Assert.Equal(Result.Ok "write_file", WoTExecutor.matchCandidate candidates "write_file")
    Assert.Equal(Result.Ok "finish", WoTExecutor.matchCandidate candidates "I would finish here.")

[<Fact>]
let ``an answer naming no candidate is an error, not the first candidate`` () =
    for answer in [ ""; "none of these"; "option Z" ] do
        match WoTExecutor.matchCandidate candidates answer with
        | Result.Ok chosen -> failwith $"expected a refusal for '{answer}', got {chosen}"
        | Result.Error message -> Assert.Contains("names none", message)

[<Fact>]
let ``an answer naming two candidates is ambiguous, not a decision`` () =
    match WoTExecutor.matchCandidate candidates "either search or finish" with
    | Result.Ok chosen -> failwith $"expected a refusal, got {chosen}"
    | Result.Error message -> Assert.Contains("names none", message)

// ------------------------------------------------------------- the Decide node

let private decidePlan =
    { Id = Guid.NewGuid()
      Nodes =
        [ { Id = "decide"
            Kind = Control
            Payload = Decide(candidates, [ "the evidence is already gathered" ])
            Metadata =
              { Label = Some "decide"
                Tags = []
                Extra = Map.empty } } ]
      Edges = []
      EntryNode = "decide"
      Metadata =
        { Kind = Custom "decide-only"
          SourceGoal = "pick a next step"
          CompiledAt = DateTime.UtcNow
          EstimatedTokens = None
          EstimatedSteps = Some 1 }
      Policy = [] }

let private agentContext () =
    let agent =
        { Id = AgentId(Guid.NewGuid())
          Name = "DecideTestAgent"
          Version = "1.0"
          ParentVersion = None
          CreatedAt = DateTime.UtcNow
          Model = "mock"
          SystemPrompt = "System"
          Tools = []
          Capabilities = []
          State = AgentState.Idle
          Memory = []
          Fitness = 1.0
          Drives =
            { Accuracy = 1.0
              Speed = 1.0
              Creativity = 1.0
              Safety = 1.0 }
          Constitution = AgentConstitution.Create(AgentId(Guid.NewGuid()), NeuralRole.GeneralReasoning) }

    { Self = agent
      Registry = Unchecked.defaultof<_>
      Executor = Unchecked.defaultof<_>
      Logger = ignore
      Budget = None
      Epistemic = None
      SemanticMemory = None
      KnowledgeGraph = None
      CapabilityStore = None
      Audit = None
      SymbolicReflector = None
      CancellationToken = Threading.CancellationToken.None }

/// A saved System One reply naming `choice`, with the distribution it implies.
let private savedReply (choice: string) (probabilities: (string * float) list) confidence =
    let entries =
        probabilities
        |> List.map (fun (option, p) -> $"\"{option}\": {p}")
        |> String.concat ", "

    $"""{{
      "model": "{SystemOne.PinnedModel}",
      "answers": {{
        "decision": {{
          "type": "choice",
          "choice": "{choice}",
          "confidence": {confidence},
          "probabilities": {{ {entries} }}
        }}
      }},
      "usage": {{ "input_tokens": 0, "output_tokens": 0 }}
    }}"""

let private run (decider: SystemOne.ISystemOne option) (llm: ScriptedLlm) =
    let executor =
        WoTExecutor.createExecutorWith (llm :> ILlmService) (NoTools() :> IToolRegistry) decider

    executor.Execute(decidePlan, agentContext ()) |> Async.RunSynchronously

[<Fact>]
let ``a confident typed answer decides the node without asking the model`` () =
    let llm = ScriptedLlm("1")

    let decider =
        SystemOne.ReplaySystemOne(savedReply "finish" [ "search", 0.05; "write_file", 0.05; "finish", 0.9 ] 0.86)
        :> SystemOne.ISystemOne

    let result = run (Some decider) llm

    Assert.True(result.Success, String.concat "; " result.Errors)
    Assert.Equal("finish", result.Output)
    Assert.Empty(llm.Asked) // the prose path was never taken

[<Fact>]
let ``a shared distribution falls back to the model rather than deciding`` () =
    // 0.45 against 0.40 is not a decision, whatever the confidence says.
    let llm = ScriptedLlm("1")

    let decider =
        SystemOne.ReplaySystemOne(savedReply "search" [ "search", 0.45; "write_file", 0.4; "finish", 0.15 ] 0.9)
        :> SystemOne.ISystemOne

    let result = run (Some decider) llm

    Assert.True(result.Success, String.concat "; " result.Errors)
    Assert.Equal("search", result.Output) // from the model's "1", not from the reply
    Assert.Single(llm.Asked) |> ignore

[<Fact>]
let ``a reply that fails validation leaves the prose path in charge`` () =
    let llm = ScriptedLlm("2")

    // Names an option we never offered: rejected by the contract, so it decides nothing.
    let decider =
        SystemOne.ReplaySystemOne(savedReply "deploy" [ "search", 0.05; "write_file", 0.05; "finish", 0.9 ] 0.86)
        :> SystemOne.ISystemOne

    let result = run (Some decider) llm

    Assert.True(result.Success, String.concat "; " result.Errors)
    Assert.Equal("write_file", result.Output)

[<Fact>]
let ``with no decider the node behaves exactly as before, but fails honestly`` () =
    let result = run None (ScriptedLlm("2"))
    Assert.Equal("write_file", result.Output)

    let refused = run None (ScriptedLlm("I cannot choose"))
    Assert.False(refused.Success)

// ------------------------------------------------------------------ wiring

[<Fact>]
let ``no decider is configured unless the environment asks for one`` () =
    // The live client costs money and needs a key; nothing may opt in on our behalf.
    let previous = Environment.GetEnvironmentVariable "TARS_JEV"

    try
        Environment.SetEnvironmentVariable("TARS_JEV", null)
        Assert.True((SystemOne.deciderFromEnvironment ()).IsNone)

        Environment.SetEnvironmentVariable("TARS_JEV", "0")
        Assert.True((SystemOne.deciderFromEnvironment ()).IsNone)
    finally
        Environment.SetEnvironmentVariable("TARS_JEV", previous)
