namespace Tars.Tests

open Xunit
open Tars.Symbolic
open Tars.Core
open System

module InvariantTests =

    [<Fact>]
    let ``GrammarValidity check succeeds for valid production`` () =
        // Arrange
        let rule = "Statement"
        let production = "expression | declaration"

        // Act
        let result = InvariantChecking.checkGrammarValidity rule production

        // Assert
        Assert.True(result.Satisfied)
        Assert.Equal(1.0, result.Score)
        Assert.NotEmpty(result.Evidence)

    [<Fact>]
    let ``GrammarValidity check fails for empty production`` () =
        // Arrange
        let rule = "Statement"
        let production = ""

        // Act
        let result = InvariantChecking.checkGrammarValidity rule production

        // Assert
        Assert.False(result.Satisfied)
        Assert.Equal(0.0, result.Score)

    [<Fact>]
    let ``BeliefConsistency check handles empty list`` () =
        // Arrange
        let beliefs = []

        // Act
        let result = InvariantChecking.checkBeliefConsistency beliefs

        // Assert
        Assert.True(result.Satisfied)
        Assert.Equal(1.0, result.Score)

    [<Fact>]
    let ``BeliefConsistency check handles multiple beliefs`` () =
        // Arrange
        let beliefs = [ "Belief 1"; "Belief 2"; "Belief 3" ]

        // Act
        let result = InvariantChecking.checkBeliefConsistency beliefs

        // Assert
        Assert.True(result.Satisfied)
        Assert.NotEmpty(result.Evidence)

    [<Fact>]
    let ``AlignmentThreshold check succeeds when actual exceeds threshold`` () =
        // Arrange
        let metric = "agent_alignment"
        let threshold = 0.8
        let actual = 0.95

        // Act
        let result = InvariantChecking.checkAlignmentThreshold metric threshold actual

        // Assert
        Assert.True(result.Satisfied)
        Assert.Equal(1.0, result.Score)

    [<Fact>]
    let ``AlignmentThreshold check fails when actual below threshold`` () =
        // Arrange
        let metric = "agent_alignment"
        let threshold = 0.8
        let actual = 0.5

        // Act
        let result = InvariantChecking.checkAlignmentThreshold metric threshold actual

        // Assert
        Assert.False(result.Satisfied)
        Assert.True(result.Score < 1.0 && result.Score > 0.0)

    [<Fact>]
    let ``CodeComplexityBound check succeeds when complexity within limit`` () =
        // Arrange
        let maxComplexity = 10.0
        let actual = 7.5

        // Act
        let result = InvariantChecking.checkCodeComplexityBound maxComplexity actual

        // Assert
        Assert.True(result.Satisfied)
        Assert.Equal(1.0, result.Score)

    [<Fact>]
    let ``CodeComplexityBound check fails when complexity exceeds limit`` () =
        // Arrange
        let maxComplexity = 10.0
        let actual = 15.0

        // Act
        let result = InvariantChecking.checkCodeComplexityBound maxComplexity actual

        // Assert
        Assert.False(result.Satisfied)
        Assert.True(result.Score < 1.0)

    [<Fact>]
    let ``ResourceQuota check succeeds when usage within limit`` () =
        // Arrange
        let resource = "tokens"
        let limit = 1000
        let actual = 750

        // Act
        let result = InvariantChecking.checkResourceQuota resource limit actual

        // Assert
        Assert.True(result.Satisfied)
        Assert.Equal(1.0, result.Score)

    [<Fact>]
    let ``ResourceQuota check fails when usage exceeds limit`` () =
        // Arrange
        let resource = "tokens"
        let limit = 1000
        let actual = 1500

        // Act
        let result = InvariantChecking.checkResourceQuota resource limit actual

        // Assert
        Assert.False(result.Satisfied)
        Assert.True(result.Score < 1.0)

    [<Fact>]
    let ``TemporalConstraint check succeeds when before happens first`` () =
        // Arrange
        let before = TarsEntity.FileE "file1.fs"
        let after = TarsEntity.FileE "file2.fs"
        let beforeTime = Some(DateTime.UtcNow.AddMinutes(-10.0))
        let afterTime = Some DateTime.UtcNow

        // Act
        let result =
            InvariantChecking.checkTemporalConstraint before after beforeTime afterTime

        // Assert
        Assert.True(result.Satisfied)
        Assert.Equal(1.0, result.Score)

    [<Fact>]
    let ``TemporalConstraint check fails when timestamps out of order`` () =
        // Arrange
        let before = TarsEntity.FileE "file1.fs"
        let after = TarsEntity.FileE "file2.fs"
        let beforeTime = Some DateTime.UtcNow
        let afterTime = Some(DateTime.UtcNow.AddMinutes(-10.0))

        // Act
        let result =
            InvariantChecking.checkTemporalConstraint before after beforeTime afterTime

        // Assert
        Assert.False(result.Satisfied)
        Assert.Equal(0.0, result.Score)

    [<Fact>]
    let ``CustomInvariant check succeeds when validator returns true`` () =
        // Arrange
        let name = "test_invariant"
        let validator = fun () -> true

        // Act
        let result = InvariantChecking.checkCustomInvariant name validator

        // Assert
        Assert.True(result.Satisfied)
        Assert.Equal(1.0, result.Score)

    [<Fact>]
    let ``CustomInvariant check fails when validator returns false`` () =
        // Arrange
        let name = "test_invariant"
        let validator = fun () -> false

        // Act
        let result = InvariantChecking.checkCustomInvariant name validator

        // Assert
        Assert.False(result.Satisfied)
        Assert.Equal(0.0, result.Score)

    [<Fact>]
    let ``CustomInvariant check handles exceptions`` () =
        // Arrange
        let name = "failing_invariant"
        let validator = fun () -> failwith "Test exception"

        // Act
        let result = InvariantChecking.checkCustomInvariant name validator

        // Assert
        Assert.False(result.Satisfied)
        Assert.Equal(0.0, result.Score)
        Assert.NotEmpty(result.Evidence)

    [<Fact>]
    let ``InvariantCheck isSatisfied returns true for score >=  0.5`` () =
        // Arrange
        let check =
            { Invariant = SymbolicInvariant.GrammarValidity("test", "prod")
              Satisfied = true
              Score = 0.75
              Evidence = []
              Timestamp = DateTime.UtcNow }

        // Act & Assert
        Assert.True(InvariantCheck.isSatisfied check)

    [<Fact>]
    let ``InvariantCheck isSatisfied returns false for score < 0.5`` () =
        // Arrange
        let check =
            { Invariant = SymbolicInvariant.GrammarValidity("test", "")
              Satisfied = false
              Score = 0.25
              Evidence = []
              Timestamp = DateTime.UtcNow }

        // Act & Assert
        Assert.False(InvariantCheck.isSatisfied check)

    [<Fact>]
    let ``StandardInvariants parseableGrammar creates correct invariant`` () =
        // Arrange & Act
        let invariant = StandardInvariants.parseableGrammar "Statement" "expr | decl"

        // Assert
        match invariant with
        | SymbolicInvariant.GrammarValidity(rule, prod) ->
            Assert.Equal("Statement", rule)
            Assert.Equal("expr | decl", prod)
        | _ -> Assert.True(false, "Wrong invariant type")

    [<Fact>]
    let ``StandardInvariants tokenBudget creates correct invariant`` () =
        // Arrange & Act
        let invariant = StandardInvariants.tokenBudget 1000

        // Assert
        match invariant with
        | SymbolicInvariant.ResourceQuota(resource, limit) ->
            Assert.Equal("tokens", resource)
            Assert.Equal(1000, limit)
        | _ -> Assert.True(false, "Wrong invariant type")

    [<Fact>]
    let ``Verification verify CustomOp non_empty check`` () =
        async {
            let op = Tars.Cortex.WoTTypes.VerificationOp.CustomOp "non_empty"
            let dummyExecutor _ _ = async { return Result.Ok "" }

            let! resOk = Tars.Cortex.Verification.verify "some content" op dummyExecutor
            let! resFail = Tars.Cortex.Verification.verify "   " op dummyExecutor

            match resOk, resFail with
            | Result.Ok true, Result.Ok false -> Assert.True(true)
            | _ -> failwith $"Unexpected results: %A{resOk}, %A{resFail}"
        } |> Async.StartAsTask

    [<Fact>]
    let ``Verification verify CustomOp threshold check`` () =
        async {
            let op = Tars.Cortex.WoTTypes.VerificationOp.CustomOp "threshold:metric:50.0"
            let dummyExecutor _ _ = async { return Result.Ok "" }

            let! resOk = Tars.Cortex.Verification.verify "75.0" op dummyExecutor
            let! resFail = Tars.Cortex.Verification.verify "25.0" op dummyExecutor

            match resOk, resFail with
            | Result.Ok true, Result.Ok false -> Assert.True(true)
            | _ -> failwith $"Unexpected results: %A{resOk}, %A{resFail}"
        } |> Async.StartAsTask

    [<Fact>]
    let ``Verification verify Schema match check`` () =
        async {
            let schema = """{ "type": "object", "required": ["name", "age"], "properties": { "name": { "type": "string" }, "age": { "type": "number" } } }"""
            let op = Tars.Cortex.WoTTypes.VerificationOp.Schema schema
            let dummyExecutor _ _ = async { return Result.Ok "" }

            let validJson = """{ "name": "TARS", "age": 2 }"""
            let invalidJsonType = """{ "name": "TARS", "age": "two" }"""
            let invalidJsonMissing = """{ "name": "TARS" }"""

            let! resOk = Tars.Cortex.Verification.verify validJson op dummyExecutor
            let! resFailType = Tars.Cortex.Verification.verify invalidJsonType op dummyExecutor
            let! resFailMissing = Tars.Cortex.Verification.verify invalidJsonMissing op dummyExecutor

            match resOk, resFailType, resFailMissing with
            | Result.Ok true, Result.Ok false, Result.Ok false -> Assert.True(true)
            | _ -> failwith $"Unexpected results: %A{resOk}, %A{resFailType}, %A{resFailMissing}"
        } |> Async.StartAsTask
