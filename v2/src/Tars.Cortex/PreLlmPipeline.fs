namespace Tars.Cortex

open System.Threading.Tasks
open Tars.Core
open Tars.Llm

/// <summary>
/// Represents the context flowing through the Pre-LLM pipeline.
/// </summary>
type PreLlmContext =
    {
        /// The original user input or system prompt
        RawInput: string

        /// The current processed prompt (rewritten/summarized)
        CurrentPrompt: string

        /// Detected intent of the prompt
        Intent: AgentDomain option

        /// Safety status (Allowed or Blocked)
        IsSafe: bool

        /// Reason for blocking if unsafe
        BlockReason: string option

        /// Metadata/Tags added by stages
        Metadata: Map<string, string>
    }

    static member Create(input: string) =
        { RawInput = input
          CurrentPrompt = input
          Intent = None
          IsSafe = true
          BlockReason = None
          Metadata = Map.empty }

/// <summary>
/// Interface for a stage in the Pre-LLM pipeline.
/// </summary>
type IPreLlmStage =
    abstract member Name: string
    abstract member ExecuteAsync: PreLlmContext -> Task<PreLlmContext>

/// <summary>
/// Gate a prompt against explicit policies.
/// </summary>
type PolicyGateStage(requiredPolicies: string list) =
    let policies = requiredPolicies |> List.distinct

    interface IPreLlmStage with
        member _.Name = "PolicyGate"

        member _.ExecuteAsync(ctx) =
            task {
                if not ctx.IsSafe then
                    return ctx
                else if policies.IsEmpty then
                    return ctx
                else
                    let input: PolicyEngine.PolicyInput =
                        { Text = ctx.CurrentPrompt
                          Metadata = ctx.Metadata }

                    let outcomes = PolicyEngine.evaluateDefault policies input

                    if PolicyEngine.anyFailed outcomes then
                        let reasons =
                            outcomes
                            |> List.filter (fun o -> not o.Passed)
                            |> List.collect (fun o -> o.Messages)
                            |> List.distinct

                        let reason =
                            match reasons with
                            | [] -> "Policy gate failed."
                            | xs -> String.concat "; " xs

                        return
                            { ctx with
                                IsSafe = false
                                BlockReason = Some reason }
                    else
                        return ctx
            }

/// <summary>
/// Safety filter mapped to explicit destructive-command policy.
/// </summary>
type SafetyFilterStage() =
    let inner = PolicyGateStage([ "no_destructive_commands" ]) :> IPreLlmStage

    interface IPreLlmStage with
        member _.Name = "SafetyFilter"
        member _.ExecuteAsync(ctx) = inner.ExecuteAsync(ctx)

type IIntentClassifier =
    abstract member ClassifyAsync: string -> Task<AgentDomain option>

type NoopIntentClassifier() =
    interface IIntentClassifier with
        member _.ClassifyAsync(_) = Task.FromResult None

type LlmIntentClassifier(llm: ILlmService) =
    let semanticClassifier: SemanticClassifier<AgentDomain> =
        SemanticClassifierFactory.createDomainClassifier llm

    interface IIntentClassifier with
        member _.ClassifyAsync(input) =
            task {
                // 1. Try fast semantic classification first (Vector Similarity)
                let! semanticResult =
                    task {
                        try
                            return! semanticClassifier.ClassifyAsync(input, 0.8) |> Async.StartAsTask
                        with _ ->
                            return None
                    }

                match semanticResult with
                | Some(domain, score) -> return Some domain
                | None ->
                    // 2. Fallback to LLM-based classification if ambiguous
                    let request: LlmRequest =
                        { ModelHint = Some "fast"
                          Model = None
                          SystemPrompt =
                            Some
                                "Classify the intent as one of: coding, planning, reasoning, chat. Respond with JSON: {\"intent\":\"...\"}."
                          MaxTokens = Some 80
                          Temperature = Some 0.0
                          Stop = []
                          Messages = [ { Role = Role.User; Content = input } ]
                          Tools = []
                          ToolChoice = None
                          ResponseFormat = Some ResponseFormat.Json
                          Stream = false
                          JsonMode = true
                          Seed = None

                          ContextWindow = None }

                    let! response =
                        task {
                            try
                                return! llm.CompleteAsync(request)
                            with _ ->
                                return
                                    { Text = ""
                                      FinishReason = Some "error"
                                      Usage = None
                                      Raw = None }
                        }

                    match JsonParsing.tryParseElement response.Text with
                    | Result.Ok elem ->
                        let mutable intentElem = Unchecked.defaultof<System.Text.Json.JsonElement>

                        if
                            elem.ValueKind = System.Text.Json.JsonValueKind.Object
                            && elem.TryGetProperty("intent", &intentElem)
                            && intentElem.ValueKind = System.Text.Json.JsonValueKind.String
                        then
                            // Also use semantic classifier here to map the "string" from LLM to AgentDomain enum
                            let! result =
                                task {
                                    try
                                        return! semanticClassifier.ClassifyAsync(intentElem.GetString(), 0.5)
                                        |> Async.StartAsTask
                                    with _ ->
                                        return None
                                }

                            return result |> Option.map fst
                        else
                            return None
                    | Result.Error _ -> return None
            }

/// <summary>
/// Classifies the intent of the user prompt using a pluggable classifier.
/// </summary>
type IntentClassifierStage(classifier: IIntentClassifier) =
    interface IPreLlmStage with
        member _.Name = "IntentClassifier"

        member _.ExecuteAsync(ctx) =
            task {
                if not ctx.IsSafe then
                    return ctx
                else
                    let! intent = classifier.ClassifyAsync(ctx.CurrentPrompt)
                    return { ctx with Intent = intent }
            }

/// <summary>
/// Compresses the context using the compressor's adaptive policy.
/// </summary>
type ContextSummarizerStage(compressor: ContextCompressor) =
    interface IPreLlmStage with
        member _.Name = "ContextSummarizer"

        member _.ExecuteAsync(ctx) =
            task {
                if not ctx.IsSafe then
                    return ctx
                else
                    let! compressed = compressor.AutoCompress(ctx.CurrentPrompt)

                    if compressed = ctx.CurrentPrompt then
                        return ctx
                    else
                        return { ctx with CurrentPrompt = compressed }
            }

/// <summary>
/// Runs the Pre-LLM pipeline.
/// </summary>
type PreLlmPipeline(stages: IPreLlmStage list) =

    member _.ExecuteAsync(input: string) =
        task {
            let mutable ctx = PreLlmContext.Create(input)

            for stage in stages do
                if ctx.IsSafe then
                    let! nextCtx = stage.ExecuteAsync(ctx)
                    ctx <- nextCtx

            return ctx
        }

/// <summary>
/// Classifies intent as a closed choice over the domains that exist, rather than asking
/// for prose and mapping the answer back by similarity.
/// </summary>
/// <remarks>
/// The options are this type's own cases, so a domain we do not have cannot come back â€”
/// which is the failure the similarity fallback could not rule out. Anything the gate
/// does not clear, and anything the contract rejects, defers to the classifier we
/// already had: a typed answer is an improvement, never a new single point of failure.
/// </remarks>
type TypedIntentClassifier(decider: SystemOne.ISystemOne, fallback: IIntentClassifier, ?gate: SystemOne.Gate) =
    let gate = defaultArg gate SystemOne.Gate.Default

    static let domains =
        [ "coding", AgentDomain.Coding, "Writing, changing, debugging or reviewing code."
          "planning", AgentDomain.Planning, "Deciding what to do next, or in what order."
          "reasoning", AgentDomain.Reasoning, "Working something out: why, whether, or how."
          "chat", AgentDomain.Chat, "Conversational, with no task behind it." ]

    static let question =
        "intent",
        SystemOne.Choice(
            "What is this request asking for?",
            domains |> List.map (fun (id, _, criterion) -> id, criterion)
        )

    /// What we say about a request, told plainly when it is only the opening of one.
    /// Callers hand over whole rendered prompts — `evolve` passes a task template of a
    /// couple of kilobytes — and the payload cap is a cost bound, not a suggestion. The
    /// opening is also where a goal is stated, so it is the part worth keeping.
    static let stateFor (input: string) (excerpt: string) =
        if excerpt.Length < input.Length then
            [ "request", SystemOne.Text excerpt
              "note", SystemOne.Text "Only the opening of the request is shown; classify from it." ]
        else
            [ "request", SystemOne.Text input ]

    /// Trim until the request fits, measuring the real payload rather than guessing at
    /// a character budget that a change to the question text would silently invalidate.
    static let fitted (input: string) =
        let budget = SystemOne.JevLimits.Default.MaxPayloadBytes

        let rec shrink (text: string) =
            let body = SystemOne.payload (stateFor input text) [ question ]

            if SystemOne.payloadBytes body <= budget || text.Length = 0 then
                text
            else
                shrink (text.Substring(0, text.Length * 3 / 4))

        shrink input

    /// The exact request this classifier would send. A caller — or a test — can check
    /// that it fits before anything is spent.
    static member RequestFor(input: string) =
        SystemOne.payload (stateFor input (fitted input)) [ question ]

    interface IIntentClassifier with
        member _.ClassifyAsync(input) =
            task {
                let! reply =
                    decider.Evaluate(stateFor input (fitted input), [ question ])
                    |> Async.StartAsTask

                let chosen =
                    reply
                    |> Result.bind (fun reply ->
                        match reply.TryAnswer "intent" with
                        | Some answer -> SystemOne.decided gate answer
                        | None -> Result.Error "no answer to the intent question")

                match chosen with
                | Result.Ok choice ->
                    return
                        domains
                        |> List.tryPick (fun (id, domain, _) -> if id = choice then Some domain else None)
                | Result.Error _ -> return! fallback.ClassifyAsync input
            }
