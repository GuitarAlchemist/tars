namespace Tars.Tests

open System
open System.IO
open Xunit
open Tars.Core
open Tars.Knowledge
open Tars.Interface.Cli.Commands

module RD = Tars.Interface.Cli.Commands.ReasoningDiag

module ReasoningDiagTests =

    [<Fact>]
    let ``recordTraceToLedger persists a belief`` () =
        task {
            let ledger = KnowledgeLedger.createInMemory()
            do! ledger.Initialize()

            let trace =
                { Id = Guid.NewGuid()
                  StartTime = DateTime.UtcNow
                  EndTime = Some DateTime.UtcNow
                  Events = []
                  Tags = Map.empty }

            let! _ =
                RD.recordTraceToLedger
                    ledger
                    trace
                    RD.ReasoningMode.WorkflowOfThoughts
                    "Diagnostic goal"
                    (Some "trace.json")
                    (Success "ok")
                    None

            let subject = $"diag:wot:{trace.Id}"
            let found = ledger.Query(?subject = Some subject) |> Seq.toList

            let belief = Assert.Single(found)
            Assert.Contains(belief.Tags, fun t -> t.StartsWith("tracePath:"))
        }

    [<Fact>]
    let ``parseArgs returns defaults and ledger enabled`` () =
        match RD.parseArgs [ "wot"; "Solve reasoning" ] with
        | Result.Error err -> Assert.True(false, err)
        | Result.Ok opts ->
            Assert.Equal("Solve reasoning", opts.Goal)
            Assert.Equal(RD.ReasoningMode.WorkflowOfThoughts, opts.Mode)
            Assert.True(opts.UseLedger)
            Assert.False(opts.AutoRetry)
            Assert.EndsWith(".json", opts.EvidencePath, StringComparison.OrdinalIgnoreCase)
            Assert.Contains("diagnostics", opts.EvidencePath)

    [<Fact>]
    let ``parseArgs honors custom evidence path and ledger toggle`` () =
        let customPath = Path.Combine(Path.GetTempPath(), "diag-trace.json")

        match RD.parseArgs [ "tot"; "Explain graph"; "--no-ledger"; "--evidence"; customPath ] with
        | Result.Error err -> Assert.True(false, err)
        | Result.Ok opts ->
            Assert.Equal("Explain graph", opts.Goal)
            Assert.Equal(RD.ReasoningMode.TreeOfThoughts, opts.Mode)
            Assert.False(opts.UseLedger)
            Assert.False(opts.AutoRetry)
            Assert.Equal(customPath, opts.EvidencePath)

    [<Fact>]
    let ``parseArgs enables auto retry`` () =
        match RD.parseArgs [ "got"; "Trace reasoning"; "--auto-retry" ] with
        | Result.Error err -> Assert.True(false, err)
        | Result.Ok opts ->
            Assert.Equal(RD.ReasoningMode.GraphOfThoughts, opts.Mode)
            Assert.True(opts.AutoRetry)
