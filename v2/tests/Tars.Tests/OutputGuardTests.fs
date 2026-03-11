module Tars.Tests.OutputGuardTests

open Xunit
open Tars.Core

[<Fact>]
let ``default guard flags missing required fields`` () =
    let guard = OutputGuard.defaultGuard

    let input: GuardInput =
        { ResponseText = """{"foo":1}"""
          Grammar = None
          ExpectedJsonFields = Some [ "foo"; "bar" ]
          RequireCitations = false
          Citations = None
          AllowExtraFields = false
          Metadata = Map.empty }

    let result = guard.Evaluate input |> Async.RunSynchronously

    match result.Action with
    | GuardAction.RetryWithHint _ ->
        Assert.True(result.Risk >= 0.5)
        Assert.Contains("Missing fields", result.Messages |> List.head)
    | _ -> Assert.Fail($"Expected RetryWithHint, got {result.Action}")

[<Fact>]
let ``analyzer can escalate risk and action`` () =
    let guard = OutputGuard.defaultGuard

    let analyzer =
        DelegateOutputGuardAnalyzer(fun _ ->
            async {
                return
                    Some
                        { Risk = 0.9
                          Action = GuardAction.Reject "LLM analysis flagged cargo cult"
                          Messages = [ "analysis" ] }
            })
        :> IOutputGuardAnalyzer

    let composed = OutputGuard.withAnalyzer guard (Some analyzer)

    let input: GuardInput =
        { ResponseText = """{"foo":1,"bar":2}"""
          Grammar = None
          ExpectedJsonFields = Some [ "foo"; "bar" ]
          RequireCitations = false
          Citations = None
          AllowExtraFields = false
          Metadata = Map.empty }

    let result = composed.Evaluate input |> Async.RunSynchronously

    match result.Action with
    | GuardAction.Reject msg ->
        Assert.True(result.Risk >= 0.9)
        Assert.Contains("analysis", result.Messages)
        Assert.Contains("cargo cult", msg)
    | _ -> Assert.Fail($"Expected Reject, got {result.Action}")

[<Fact>]
let ``analyzer absence falls back to base guard`` () =
    let guard = OutputGuard.defaultGuard
    let composed = OutputGuard.withAnalyzer guard None

    let input: GuardInput =
        { ResponseText = """{"foo":1,"bar":2}"""
          Grammar = None
          ExpectedJsonFields = Some [ "foo"; "bar" ]
          RequireCitations = false
          Citations = None
          AllowExtraFields = false
          Metadata = Map.empty }

    let result = composed.Evaluate input |> Async.RunSynchronously

    Assert.Equal(GuardAction.Accept, result.Action)
    Assert.True(result.Risk < 0.3)
