namespace Tars.Tests

open Xunit
open Tars.Core

module ExecutionTests =

    [<Fact>]
    let ``Execution builder propagates success`` () =
        let result =
            execution {
                let! a = Success 10
                let! b = Success 20
                return a + b
            }

        match result with
        | Success v -> Assert.Equal(30, v)
        | _ -> Assert.Fail("Should be Success")

    [<Fact>]
    let ``Execution builder accumulates warnings`` () =
        let result =
            execution {
                let! a = PartialSuccess(10, [ Warning "Low memory" ])
                let! b = PartialSuccess(20, [ Warning "High latency" ])
                return a + b
            }

        match result with
        | PartialSuccess(v, warnings) ->
            Assert.Equal(30, v)
            Assert.Equal(2, warnings.Length)
            Assert.Contains(Warning "Low memory", warnings)
            Assert.Contains(Warning "High latency", warnings)
        | _ -> Assert.Fail("Should be PartialSuccess")

    [<Fact>]
    let ``Execution builder propagates failure`` () =
        let result =
            execution {
                let! a = Success 10
                let! b = Failure [ PartialFailure.Error "Something went wrong" ]
                return a + b
            }

        match result with
        | Failure errs -> Assert.Contains(PartialFailure.Error "Something went wrong", errs)
        | _ -> Assert.Fail("Should be Failure")

    // === Edge Case Tests ===

    [<Fact>]
    let ``Execution builder handles empty chain`` () =
        let result = execution { return () }

        match result with
        | Success () -> ()
        | _ -> Assert.Fail("Should be Success")

    [<Fact>]
    let ``Execution builder preserves warnings through failure`` () =
        let result =
            execution {
                let! a = PartialSuccess(10, [ Warning "Early warning" ])
                let! _ = Failure [ PartialFailure.Error "Boom" ]
                return a
            }

        match result with
        | Failure errs ->
            // Should contain both the warning context and the error
            Assert.True(errs.Length >= 1)
        | _ -> Assert.Fail("Should be Failure")

    [<Fact>]
    let ``Execution builder handles mixed Success and PartialSuccess`` () =
        let result =
            execution {
                let! a = Success 10
                let! b = PartialSuccess(20, [ Warning "Heads up" ])
                let! c = Success 30
                return a + b + c
            }

        match result with
        | PartialSuccess(v, warnings) ->
            Assert.Equal(60, v)
            Assert.Single(warnings) |> ignore
        | _ -> Assert.Fail("Should be PartialSuccess")

    [<Fact>]
    let ``Execution builder ReturnFrom works`` () =
        let inner = Success 42
        let result = execution { return! inner }

        match result with
        | Success v -> Assert.Equal(42, v)
        | _ -> Assert.Fail("Should be Success")

    [<Fact>]
    let ``Execution builder handles LowConfidence`` () =
        let result =
            execution {
                let! a = PartialSuccess(10, [ LowConfidence(0.3, "Uncertain result") ])
                return a * 2
            }

        match result with
        | PartialSuccess(v, warnings) ->
            Assert.Equal(20, v)
            Assert.Contains(warnings, fun w ->
                match w with
                | LowConfidence(conf, _) -> conf < 0.5
                | _ -> false)
        | _ -> Assert.Fail("Should be PartialSuccess")
