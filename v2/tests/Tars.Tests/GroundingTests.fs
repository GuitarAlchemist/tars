namespace Tars.Tests

open System
open Xunit
open FsUnit
open Tars.Knowledge
open Tars.Symbolic

module GroundingTests =

    [<Fact>]
    let ``SafeAssertTriple denies contradictory belief`` () =
        task {
            // Arrange
            let ledger = KnowledgeLedger.createInMemory ()
            let provenance = Provenance.FromUser()
            
            // Assert initial truth
            let! _ = ledger.AssertTriple("Phase 11", RelationType.Custom "status", "Working", provenance, AgentId.User)
            
            // Act: Assert contradiction
            let! result = ledger.SafeAssertTriple("Phase 11", RelationType.Custom "status", "Broken", provenance, AgentId.User)
            
            // Assert
            match result with
            | Error msg -> msg |> should startWith "Verification failed"
            | Ok _ -> failwith "Should have been denied"
        }

    [<Fact>]
    let ``SafeAssertTriple accepts consistent belief`` () =
        task {
            // Arrange
            let ledger = KnowledgeLedger.createInMemory ()
            let provenance = Provenance.FromUser()
            
            // Act
            let! result = ledger.SafeAssertTriple("Sky", RelationType.Custom "is", "Blue", provenance, AgentId.User)
            
            // Assert
            result.IsOk |> should equal true
        }
