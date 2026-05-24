namespace Tars.Tests

open System
open System.IO
open Xunit
open Xunit.Abstractions
open Tars.Knowledge
open Tars.Interface.Cli.Commands
open Tars.Evolution

type KnowledgeLedgerTests(output: ITestOutputHelper) =

    [<Fact>]
    member _.``KnowledgeLedger: Weaken and strengthen update confidence``() =
        task {
            let ledger = KnowledgeLedger.createInMemory ()
            do! ledger.Initialize()

            let belief = Belief.fromTriple "TARS" Supports "Knowledge"
            let! assertResult = ledger.Assert(belief, AgentId.System)

            match assertResult with
            | Error e -> Assert.True(false, e)
            | Ok beliefId ->
                let! weakenResult = ledger.Weaken(beliefId, 0.25, "test", AgentId.System)

                match weakenResult with
                | Error e -> Assert.True(false, e)
                | Ok() -> ()

                match ledger.Get(beliefId) with
                | Some updated -> Assert.Equal(0.25, updated.Confidence, 3)
                | None -> Assert.True(false, "Belief not found after weaken")

                let! strengthenResult = ledger.Strengthen(beliefId, 0.9, "test", AgentId.System)

                match strengthenResult with
                | Error e -> Assert.True(false, e)
                | Ok() -> ()

                match ledger.Get(beliefId) with
                | Some updated -> Assert.Equal(0.9, updated.Confidence, 3)
                | None -> Assert.True(false, "Belief not found after strengthen")

                let! history = ledger.GetHistory(beliefId)
                Assert.Equal(3, history.Length)
        }

    [<Fact>]
    member _.``KnowCmd: parseArgs supports postgres and prompts``() =
        let args = [| "fetch"; "Graph theory"; "--pg"; "--show-prompt" |]
        let options = KnowCmd.parseArgs args
        Assert.Equal("fetch", options.Command)
        Assert.Equal(Some "Graph theory", options.Query)
        Assert.True(options.UsePostgres)
        Assert.True(options.ShowPrompt)

    [<Fact>]
    member _.``KnowCmd: parseArgs supports history and ingest``() =
        let historyOptions = KnowCmd.parseArgs [| "history"; "b:1234abcd" |]
        Assert.Equal(Some "b:1234abcd", historyOptions.BeliefId)

        let ingestOptions = KnowCmd.parseArgs [| "ingest"; "data.csv" |]
        Assert.Equal(Some "data.csv", ingestOptions.Path)

    [<Fact>]
    member _.``KnowCmd: tryParseBeliefId resolves short and full ids``() =
        task {
            let ledger = KnowledgeLedger.createInMemory ()
            do! ledger.Initialize()

            let belief = Belief.fromTriple "Alpha" Supports "Beta"
            let! assertResult = ledger.Assert(belief, AgentId.System)

            match assertResult with
            | Error e -> Assert.True(false, e)
            | Ok _ ->
                let shortId = belief.Id.ToString()

                match KnowCmd.tryParseBeliefId ledger shortId with
                | Ok id -> Assert.Equal(belief.Id, id)
                | Error e -> Assert.True(false, e)

                let fullId = belief.Id.Value.ToString()

                match KnowCmd.tryParseBeliefId ledger fullId with
                | Ok id -> Assert.Equal(belief.Id, id)
                | Error e -> Assert.True(false, e)
        }

    [<Fact>]
    member _.``KnowCmd: runIngest loads CSV entries``() =
        task {
            let ledger = KnowledgeLedger.createInMemory ()
            do! ledger.Initialize()

            let tempPath = Path.GetTempFileName()

            try
                File.WriteAllLines(tempPath, [| "# comment"; "Alpha,supports,Beta,0.8"; "Gamma,contradicts,Delta" |])

                do! KnowCmd.runIngest ledger tempPath

                let stats = ledger.Stats()
                Assert.Equal(2, stats.ValidBeliefs)
            finally
                if File.Exists tempPath then
                    File.Delete tempPath
        }

    [<Fact>]
    member _.``KnowledgeLedger: Retract invalidates belief``() =
        task {
            let ledger = KnowledgeLedger.createInMemory ()
            do! ledger.Initialize()

            let belief = Belief.fromTriple "A" Supports "B"
            let! assertResult = ledger.Assert(belief, AgentId.System)

            match assertResult with
            | Error e -> Assert.True(false, e)
            | Ok _ ->
                let! retractResult = ledger.Retract(belief.Id, "test", AgentId.System)

                match retractResult with
                | Error e -> Assert.True(false, e)
                | Ok() -> ()

                match ledger.Get(belief.Id) with
                | Some updated -> Assert.False(updated.IsValid)
                | None -> Assert.True(false, "Belief not found after retract")

                let results = ledger.Query(?subject = Some "A") |> Seq.toList
                Assert.Empty(results)
        }

    [<Fact>]
    member _.``KnowledgeLedger: Contradictions are tracked``() =
        task {
            let ledger = KnowledgeLedger.createInMemory ()
            do! ledger.Initialize()

            let b1 = Belief.fromTriple "A" Supports "B"
            let b2 = Belief.fromTriple "A" Contradicts "B"

            let! r1 = ledger.Assert(b1, AgentId.System)
            let! r2 = ledger.Assert(b2, AgentId.System)

            match r1, r2 with
            | Ok id1, Ok id2 ->
                let! markResult = ledger.MarkContradiction(id1, id2, "test", AgentId.System)

                match markResult with
                | Error e -> Assert.True(false, e)
                | Ok() -> ()

                let contradictions = ledger.GetContradictions() |> Seq.toList
                Assert.Single(contradictions) |> ignore
            | Error e, _ -> Assert.True(false, e)
            | _, Error e -> Assert.True(false, e)
        }

    [<Fact>]
    member _.``LedgerIngestion: records task belief when evaluation passes``() =
        task {
            let ledger = KnowledgeLedger.createInMemory ()
            do! ledger.Initialize()

            let taskDef =
                { Id = Guid.NewGuid()
                  DifficultyLevel = 1
                  Goal = "Return the sum of two integers."
                  Constraints = []
                  ValidationCriteria = "Sum is correct"
                  Timeout = TimeSpan.FromSeconds(1.0)
                  Score = 1.0 }

            let evaluation =
                { Passed = true
                  Confidence = 0.9
                  Summary = "Looks good"
                  Issues = []
                  SuggestedFixes = []
                  EvaluatedAt = DateTime.UtcNow }

            let result =
                { TaskId = taskDef.Id
                  TaskGoal = taskDef.Goal
                  ExecutorId = Tars.Core.AgentId(Guid.NewGuid())
                  Success = true
                  Output = "let add a b = a + b"
                  ExecutionTrace = []
                  Duration = TimeSpan.Zero
                  Evaluation = Some evaluation }

            do! LedgerIngestion.recordTaskResult ledger None taskDef result (fun _ -> ())

            let subject = $"task:{taskDef.Id}"
            let expectedObj = $"goal:{EvidenceStore.ComputeHash taskDef.Goal}"

            let matches =
                ledger.Query(?subject = Some subject)
                |> Seq.filter (fun b ->
                    match b.Predicate with
                    | RelationType.Custom p -> p = "satisfies"
                    | _ -> false)
                |> Seq.toList

            Assert.Contains(matches, fun b -> b.Object.Value = expectedObj)
        }

    [<Fact>]
    member _.``LedgerIngestion: skips task belief when evaluation fails``() =
        task {
            let ledger = KnowledgeLedger.createInMemory ()
            do! ledger.Initialize()

            let taskDef =
                { Id = Guid.NewGuid()
                  DifficultyLevel = 1
                  Goal = "Return the sum of two integers."
                  Constraints = []
                  ValidationCriteria = "Sum is correct"
                  Timeout = TimeSpan.FromSeconds(1.0)
                  Score = 1.0 }

            let evaluation =
                { Passed = false
                  Confidence = 0.2
                  Summary = "Incorrect"
                  Issues = [ "wrong output" ]
                  SuggestedFixes = [ "Fix the implementation" ]
                  EvaluatedAt = DateTime.UtcNow }

            let result =
                { TaskId = taskDef.Id
                  TaskGoal = taskDef.Goal
                  ExecutorId = Tars.Core.AgentId(Guid.NewGuid())
                  Success = true
                  Output = "let add a b = a - b"
                  ExecutionTrace = []
                  Duration = TimeSpan.Zero
                  Evaluation = Some evaluation }

            do! LedgerIngestion.recordTaskResult ledger None taskDef result (fun _ -> ())

            let subject = $"task:{taskDef.Id}"
            let matches = ledger.Query(?subject = Some subject) |> Seq.toList
            Assert.Empty(matches)
        }

    [<Fact>]
    member _.``InMemoryEvidenceStorage: proposals are filtered by evidence id``() =
        task {
            let ledger = KnowledgeLedger.createInMemory ()
            do! ledger.Initialize()

            let storage =
                match ledger.Storage with
                | :? IEvidenceStorage as store -> store
                | _ -> failwith "Evidence storage not available"

            let evidenceA = Guid.NewGuid()
            let evidenceB = Guid.NewGuid()

            let proposalA =
                { Id = Guid.NewGuid()
                  Subject = "Alpha"
                  Predicate = "supports"
                  Object = "Beta"
                  Confidence = 0.8
                  SourceSection = "A"
                  ExtractorAgent = AgentId.System
                  ExtractedAt = DateTime.UtcNow }

            let proposalB =
                { Id = Guid.NewGuid()
                  Subject = "Gamma"
                  Predicate = "supports"
                  Object = "Delta"
                  Confidence = 0.7
                  SourceSection = "B"
                  ExtractorAgent = AgentId.System
                  ExtractedAt = DateTime.UtcNow }

            let! saveA = storage.SaveProposal(proposalA, Some evidenceA)

            match saveA with
            | Error e -> Assert.True(false, e)
            | Ok() -> ()

            let! saveB = storage.SaveProposal(proposalB, Some evidenceB)

            match saveB with
            | Error e -> Assert.True(false, e)
            | Ok() -> ()

            let! proposalsA = storage.GetProposalsByEvidence(evidenceA)
            let! proposalsB = storage.GetProposalsByEvidence(evidenceB)

            Assert.Contains(proposalsA, fun p -> p.Id = proposalA.Id)
            Assert.DoesNotContain(proposalsA, fun p -> p.Id = proposalB.Id)
            Assert.Contains(proposalsB, fun p -> p.Id = proposalB.Id)
        }

    [<Fact>]
    member _.``RoutingConfig: maps TarsConfig defaults``() =
        let cfg = Tars.Core.ConfigurationDefaults.createDefault ()
        let routing = Tars.Llm.Routing.RoutingConfig.fromTarsConfig cfg
        Assert.Equal(Uri(cfg.Llm.BaseUrl.Value), routing.OllamaBaseUri)
        Assert.Equal(cfg.Llm.Model, routing.DefaultOllamaModel)
        Assert.Equal(cfg.Llm.EmbeddingModel, routing.DefaultEmbeddingModel)

type PostgresLedgerStorageTests(output: ITestOutputHelper) =

    [<Fact>]
    member _.``PostgresLedgerStorage: append and read events``() =
        task {
            let connStr = Environment.GetEnvironmentVariable("TARS_POSTGRES_CONNECTION")

            if String.IsNullOrWhiteSpace connStr then
                output.WriteLine("Skipping Postgres test because TARS_POSTGRES_CONNECTION is not set")
            else
                let storage = PostgresLedgerStorage.createWithConnectionString connStr
                let ledger = KnowledgeLedger(storage :> ILedgerStorage)
                do! ledger.Initialize()

                let belief = Belief.fromTriple "TARS" Supports "Persistence"
                let! assertResult = ledger.Assert(belief, AgentId.System)

                match assertResult with
                | Error e -> Assert.True(false, e)
                | Ok _ -> ()

                let! history = ledger.GetHistory(belief.Id)
                Assert.NotEmpty(history)

                let evidenceStore = storage :> IEvidenceStorage

                let candidate =
                    { Id = Guid.NewGuid()
                      SourceUrl = Uri("https://example.com")
                      ContentHash = Guid.NewGuid().ToString("N")
                      FetchedAt = DateTime.UtcNow
                      RawContent = "Example"
                      Segments = [ "Example" ]
                      ProposedAssertions = []
                      Status = Pending
                      Metadata = Map.empty
                      VerifiedAt = None
                      VerifiedBy = None
                      RejectionReason = None }

                let! candidateResult = evidenceStore.SaveCandidate(candidate)

                match candidateResult with
                | Error e -> Assert.True(false, e)
                | Ok() -> ()

                let! pending = evidenceStore.GetPendingCandidates()
                Assert.Contains(pending, fun c -> c.Id = candidate.Id)

                let proposal =
                    { Id = Guid.NewGuid()
                      Subject = "TARS"
                      Predicate = "supports"
                      Object = "Testing"
                      Confidence = 0.75
                      SourceSection = "Example"
                      ExtractorAgent = AgentId.System
                      ExtractedAt = DateTime.UtcNow }

                let! proposalResult = evidenceStore.SaveProposal(proposal, Some candidate.Id)

                match proposalResult with
                | Error e -> Assert.True(false, e)
                | Ok() -> ()

                let! proposals = evidenceStore.GetProposalsByEvidence(candidate.Id)
                Assert.Contains(proposals, fun p -> p.Id = proposal.Id)
        }
