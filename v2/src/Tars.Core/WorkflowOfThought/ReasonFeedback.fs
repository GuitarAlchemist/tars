namespace Tars.Core.WorkflowOfThought

open System

// =============================================================================
// PHASE 17.5: REASON FEEDBACK SEAM
// =============================================================================
// The WoT executor loop used to thread FeedbackState by hand and reach directly
// into FeedbackLoop, WotController, and SymbolicMemory across ~26 sites. This
// module concentrates that policy: a stateful IReasonFeedback owns the
// FeedbackState (the loop no longer threads it) over a PURE reducer core, and an
// injected ISymbolicSink carries instrumentation. The behaviour is identical to
// the old inline logic — this is a locality/testability refactor.
// =============================================================================

/// Instrumentation sink for symbolic memory. The production sink appends to the
/// on-disk log; tests can capture in memory.
type ISymbolicSink =
    abstract member LogFact: runId: Guid * stepId: string option * content: string * confidence: float -> Async<unit>

    abstract member LogFailure:
        runId: Guid * stepId: string option * error: string * metadata: Map<string, string> -> Async<unit>

/// Production sink → the global SymbolicMemory NDJSON log.
type SymbolicMemorySink() =
    interface ISymbolicSink with
        member _.LogFact(runId, stepId, content, confidence) = SymbolicMemory.logFact runId stepId content confidence
        member _.LogFailure(runId, stepId, error, metadata) = SymbolicMemory.logFailure runId stepId error metadata

/// What the loop observed about a completed step, handed to the feedback policy.
type FeedbackObservation =
    /// A tool step produced output.
    | ToolObserved of step: Step * toolName: string
    /// A reason step completed; `stub` selects the stub-mode evidence shape.
    | ReasonObserved of step: Step * op: ReasonOperation * content: string * stub: bool

/// The feedback policy seam. Observe integrates a completed step; Decide routes a
/// scored hypothesis; the reads provide reasoner context. FeedbackState is owned
/// internally, so the executor loop no longer threads it.
type IReasonFeedback =
    /// Integrate a completed step's outcome into the feedback state.
    abstract member Observe: FeedbackObservation -> unit
    /// Routing decision for the hypothesis at a scored node (Wait if none).
    abstract member Decide: nodeId: string -> WotController.RouterDecision
    /// Evidence summary for a hypothesis (reasoner context).
    abstract member Summarize: nodeId: string -> string
    /// Aggregated evidence across nodes (reasoner context).
    abstract member Aggregate: nodeIds: string list -> string
    /// Consensus check for a protocol.
    abstract member Consensus: protocol: ConsensusProtocol -> bool * string

/// Pure reducer core: replicates exactly what the executor loop used to do inline
/// for each observation, as a referentially-transparent state transition.
module ReasonFeedbackCore =

    /// Parse a "Score: x" line from reasoner output (default 0.5).
    let parseScore (c: string) =
        let lines = c.Split('\n')
        let scoreLine = lines |> Array.tryFind (fun l -> l.StartsWith("Score:"))

        match scoreLine with
        | Some s ->
            match Double.TryParse(s.Replace("Score:", "").Trim()) with
            | true, v -> v
            | _ -> 0.5
        | None -> 0.5

    let private parentOf (sourceId: string) (state: FeedbackState) : string list =
        match FeedbackLoop.findEvidenceBySourceId sourceId state with
        | Some p -> [ p.Id ]
        | None -> []

    let reduce (state: FeedbackState) (obs: FeedbackObservation) : FeedbackState =
        match obs with
        | ToolObserved(step, toolName) ->
            let ev =
                { Id = step.Id
                  Source = ToolContribution(toolName, step.Id)
                  Content = $"Result from tool '{toolName}'"
                  Confidence = 1.0
                  Weight = 1.0
                  IsContradiction = false
                  ParentIds =
                    step.Inputs
                    |> List.choose (fun inp -> FeedbackLoop.findEvidenceBySourceId inp state |> Option.map (fun e -> e.Id))
                  Timestamp = DateTime.UtcNow }

            FeedbackLoop.addEvidence ev state

        | ReasonObserved(step, op, content, true) ->
            // Stub mode
            match op with
            | ReasonOperation.Score t ->
                let h =
                    FeedbackLoop.score (string t) "Stub Hypothesis" 0.5 0.5 0.5 0 0.5 content [] [] 0

                FeedbackLoop.registerHypothesis h state
            | ReasonOperation.Refine t ->
                let ev =
                    { Id = $"stub_ev_{step.Id}"
                      Source = ReasonerThought step.Id
                      Content = content
                      Confidence = 1.0
                      Weight = 0.5
                      IsContradiction = false
                      ParentIds = parentOf (string t) state
                      Timestamp = DateTime.UtcNow }

                fst (FeedbackLoop.updateHypothesis (string t) ev state)
            | ReasonOperation.Contradict t ->
                let ev =
                    { Id = $"stub_ev_con_{step.Id}"
                      Source = ReasonerThought step.Id
                      Content = content
                      Confidence = 1.0
                      Weight = 0.5
                      IsContradiction = true
                      ParentIds = parentOf (string t) state
                      Timestamp = DateTime.UtcNow }

                fst (FeedbackLoop.updateHypothesis (string t) ev state)
            | _ -> state

        | ReasonObserved(step, op, content, false) ->
            // Llm/Replay mode
            match op with
            | ReasonOperation.Score t ->
                let volume =
                    match FeedbackLoop.findEvidenceBySourceId (string t) state with
                    | Some _ -> 1
                    | None -> 0

                let scoreVal = parseScore content

                let h =
                    FeedbackLoop.score (string t) "Hypothesis" scoreVal scoreVal scoreVal 0 scoreVal content [] [] volume

                FeedbackLoop.registerHypothesis h state
            | ReasonOperation.Refine t ->
                let ev =
                    { Id = $"ev_{step.Id}"
                      Source = ReasonerThought step.Id
                      Content = content
                      Confidence = 0.8
                      Weight = 0.6
                      IsContradiction = false
                      ParentIds = parentOf (string t) state
                      Timestamp = DateTime.UtcNow }

                fst (FeedbackLoop.updateHypothesis (string t) ev state)
            | ReasonOperation.Contradict t ->
                let ev =
                    { Id = $"ev_con_{step.Id}"
                      Source = ReasonerThought step.Id
                      Content = content
                      Confidence = 0.9
                      Weight = 0.7
                      IsContradiction = true
                      ParentIds = parentOf (string t) state
                      Timestamp = DateTime.UtcNow }

                fst (FeedbackLoop.updateHypothesis (string t) ev state)
            | ReasonOperation.Backtrack(NodeId tid) ->
                FeedbackLoop.invalidateHypothesis tid "Reasoner signaled backtrack" state
            | ReasonOperation.Distill(NodeId tid) ->
                let ev =
                    { Id = $"distill_{step.Id}"
                      Source = ReasonerThought step.Id
                      Content = content
                      Confidence = 0.95
                      Weight = 0.8
                      IsContradiction = false
                      ParentIds = parentOf tid state
                      Timestamp = DateTime.UtcNow }

                fst (FeedbackLoop.updateHypothesis tid ev state)
            | ReasonOperation.Aggregate src ->
                let ev =
                    { Id = $"agg_{step.Id}"
                      Source = ReasonerThought step.Id
                      Content = content
                      Confidence = 0.9
                      Weight = 0.5
                      IsContradiction = false
                      ParentIds =
                        src
                        |> List.choose (function
                            | NodeId s -> FeedbackLoop.findEvidenceBySourceId s state |> Option.map (fun e -> e.Id))
                      Timestamp = DateTime.UtcNow }

                FeedbackLoop.addEvidence ev state
            | _ -> state

/// Stateful feedback policy: a thin shell over the pure reducer plus the read
/// helpers. Owns the FeedbackState for one run.
type ReasonFeedback(runId: Guid) =
    let mutable state = FeedbackLoop.create runId

    interface IReasonFeedback with
        member _.Observe obs = state <- ReasonFeedbackCore.reduce state obs

        member _.Decide(nodeId: string) =
            match state.Hypotheses |> Map.tryFind nodeId with
            | Some h -> WotController.Router.decide h 0.7 10
            | None -> WotController.Wait

        member _.Summarize(nodeId: string) = FeedbackLoop.summarizeEvidence nodeId state
        member _.Aggregate(nodeIds: string list) = FeedbackLoop.aggregateEvidence nodeIds state
        member _.Consensus(protocol: ConsensusProtocol) = FeedbackLoop.checkConsensus protocol state
