module Tars.Tests.VerifierAgentTests

open System
open Xunit
open FsUnit
open Tars.Knowledge

// Explicitly use in-memory ledger for tests
let createLedger () = KnowledgeLedger.createInMemory ()

[<Fact>]
let ``Verifier accepts consistent belief (empty context)`` () =
    async {
        // Arrange
        let ledger = createLedger ()
        let verifier = VerifierAgent(ledger)

        let proposal =
            { Id = Guid.NewGuid()
              Subject = "Sky"
              Predicate = "is"
              Object = "Blue"
              Confidence = 0.9
              SourceSection = "Test"
              ExtractorAgent = AgentId "test"
              ExtractedAt = DateTime.UtcNow }

        // Act
        let! result = verifier.Verify(proposal)

        // Assert
        match result with
        | Accepted score -> score |> should be (greaterThanOrEqualTo 0.7)
        | _ -> failwith $"Expected Accepted, but got {result}"
    }
    |> Async.RunSynchronously

[<Fact>]
let ``Verifier rejects ambiguous belief (single contradiction)`` () =
    async {
        // Arrange
        let ledger = createLedger ()

        let! _ =
            ledger.AssertTriple("Sky", Custom "is", "NOT Blue", Provenance.FromUser(), AgentId.User)
            |> Async.AwaitTask

        let verifier = VerifierAgent(ledger)

        let proposal =
            { Id = Guid.NewGuid()
              Subject = "Sky"
              Predicate = "is"
              Object = "Blue"
              Confidence = 0.9
              SourceSection = "Test"
              ExtractorAgent = AgentId "test"
              ExtractedAt = DateTime.UtcNow }

        // Act
        let! result = verifier.Verify(proposal)

        // Assert
        match result with
        | Denied reason -> reason |> should startWith "Ambiguous"
        | _ -> failwith $"Expected Denied (Ambiguous), but got {result}"
    }
    |> Async.RunSynchronously

[<Fact>]
let ``Verifier rejects inconsistent belief (many contradictions)`` () =
    async {
        // Arrange
        let ledger = createLedger ()

        let! _ =
            ledger.AssertTriple("Sky", Custom "is", "NOT Blue", Provenance.FromUser(), AgentId.User)
            |> Async.AwaitTask

        let! _ =
            ledger.AssertTriple("Sky", Custom "is", "NOT Blue", Provenance.FromUser(), AgentId.User)
            |> Async.AwaitTask

        let! _ =
            ledger.AssertTriple("Sky", Custom "is", "NOT Blue", Provenance.FromUser(), AgentId.User)
            |> Async.AwaitTask

        let! _ =
            ledger.AssertTriple("Sky", Custom "is", "NOT Blue", Provenance.FromUser(), AgentId.User)
            |> Async.AwaitTask

        let verifier = VerifierAgent(ledger)

        let proposal =
            { Id = Guid.NewGuid()
              Subject = "Sky"
              Predicate = "is"
              Object = "Blue"
              Confidence = 0.9
              SourceSection = "Test"
              ExtractorAgent = AgentId "test"
              ExtractedAt = DateTime.UtcNow }

        // Act
        let! result = verifier.Verify(proposal)

        // Assert
        match result with
        | Denied reason -> reason |> should startWith "Inconsistent"
        | _ -> failwith $"Expected Denied (Inconsistent), but got {result}"
    }
    |> Async.RunSynchronously
