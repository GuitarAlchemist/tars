namespace Tars.Evolution

open System
open System.Threading.Tasks
open Tars.Core
open Tars.Knowledge

module LedgerIngestion =

    let private toKnowledgeAgentId (agentId: Tars.Core.AgentId) =
        let (Tars.Core.AgentId guid) = agentId
        AgentId(guid.ToString())

    let private truncateTag (value: string) =
        let trimmed = value.Trim()
        if trimmed.Length <= 120 then
            trimmed
        else
            trimmed.Substring(0, 120) + "..."

    let recordTaskResult
        (ledger: KnowledgeLedger)
        (runId: RunId option)
        (taskDef: TaskDefinition)
        (result: TaskResult)
        (logger: string -> unit)
        =
        task {
            let evaluation = result.Evaluation

            let status, rejectionReason, verifiedAt, verifiedBy =
                match evaluation with
                | Some e when e.Passed ->
                    Verified, None, Some DateTime.UtcNow, Some(AgentId "semantic-evaluator")
                | Some e ->
                    Rejected, Some e.Summary, None, None
                | None -> Pending, None, None, None

            let contentHash =
                EvidenceStore.ComputeHash($"{taskDef.Goal}\n{result.Output}")

            let segments =
                [ taskDef.Goal
                  result.Output
                  evaluation |> Option.map (fun e -> e.Summary) |> Option.defaultValue "" ]
                |> List.filter (fun s -> not (String.IsNullOrWhiteSpace s))

            let metadata =
                [ "taskId", taskDef.Id.ToString()
                  "goal", taskDef.Goal
                  "success", string result.Success
                  "durationMs", string result.Duration.TotalMilliseconds
                  "evaluationPassed",
                  evaluation |> Option.map (fun e -> string e.Passed) |> Option.defaultValue "false"
                  "evaluationConfidence",
                  evaluation |> Option.map (fun e -> e.Confidence.ToString("F2")) |> Option.defaultValue "0.0" ]
                |> Map.ofList

            let candidate =
                { Id = Guid.NewGuid()
                  SourceUrl = Uri($"tars://evolution/task/{taskDef.Id}")
                  ContentHash = contentHash
                  FetchedAt = DateTime.UtcNow
                  RawContent = result.Output
                  Segments = segments
                  ProposedAssertions = []
                  Status = status
                  Metadata = metadata
                  VerifiedAt = verifiedAt
                  VerifiedBy = verifiedBy
                  RejectionReason = rejectionReason }

            match ledger.Storage with
            | :? IEvidenceStorage as store ->
                let! saveResult = store.SaveCandidate(candidate)
                match saveResult with
                | Result.Ok () -> ()
                | Result.Error e -> logger $"[Ledger] Evidence candidate save failed: {e}"
            | _ -> ()

            match evaluation with
            | Some e when result.Success && e.Passed ->
                let agentId = toKnowledgeAgentId result.ExecutorId
                let goalHash = EvidenceStore.ComputeHash(taskDef.Goal)
                let subject = $"task:{taskDef.Id}"
                let obj = $"goal:{goalHash}"

                let provenance =
                    match runId with
                    | Some r -> { Provenance.FromRun(r, agentId) with Confidence = e.Confidence }
                    | None ->
                        { Provenance.FromUser() with
                            ExtractedBy = Some agentId
                            Confidence = e.Confidence }

                let! assertResult =
                    ledger.AssertTriple(subject, RelationType.Custom "satisfies", obj, provenance, agentId, ?runId = runId)

                match assertResult with
                | Result.Ok _ -> ()
                | Result.Error err -> logger $"[Ledger] Failed to assert task belief: {err}"
            | _ -> ()
        }

    let recordEpistemicBelief
        (ledger: KnowledgeLedger)
        (runId: RunId option)
        (belief: Tars.Core.EpistemicBelief)
        (taskId: Guid option)
        (logger: string -> unit)
        =
        task {
            let agentId = AgentId "epistemic-governor"

            let tags =
                [ "origin:epistemic"
                  if not (String.IsNullOrWhiteSpace belief.Context) then
                      $"context:{truncateTag belief.Context}"
                  if taskId.IsSome then
                      $"task:{taskId.Value}" ]

            let provenance =
                match runId with
                | Some r -> { Provenance.FromRun(r, agentId) with Confidence = belief.Confidence }
                | None ->
                    { Provenance.FromUser() with
                        ExtractedBy = Some agentId
                        Confidence = belief.Confidence }

            let knowledgeBelief =
                { Belief.create belief.Statement (RelationType.Custom "applies_to") belief.Context provenance with
                    Tags = tags }

            let! assertResult = ledger.Assert(knowledgeBelief, agentId, ?runId = runId)
            match assertResult with
            | Result.Ok _ -> ()
            | Result.Error err -> logger $"[Ledger] Failed to assert epistemic belief: {err}"
        }
