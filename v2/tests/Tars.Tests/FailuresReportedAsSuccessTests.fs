namespace Tars.Tests

open System
open Xunit
open Tars.Core
open Tars.Cortex
open Tars.Cortex.WoTTypes
open Tars.Llm.LlmService
open Tars.Llm

// Three places where something that had failed came back looking like a success.
// Each test is written from the failure it would have hidden.

/// Answers instantly, reporting the token usage it was given.
type CountingLlm(tokensPerCall: int) =
    interface ILlmService with
        member _.CompleteAsync(_req) =
            task {
                return
                    { Text = "VERIFIED"
                      FinishReason = Some "stop"
                      Usage =
                        Some
                            { PromptTokens = tokensPerCall
                              CompletionTokens = 0
                              TotalTokens = tokensPerCall }
                      Raw = None }
            }

        member this.CompleteStreamAsync(req, _onToken) =
            (this :> ILlmService).CompleteAsync(req)

        member _.EmbedAsync(_text) = task { return [| 0.1f |] }

        member _.RouteAsync(_) =
            task {
                return
                    { Backend = Ollama "mock"
                      Endpoint = Uri "http://localhost:11434"
                      ApiKey = None }
            }

type FailuresReportedAsSuccessTests() =

    static let noTools _name _args : Async<Result<string, string>> =
        async { return Result.Error "no tools here" }

    // -------------------------------------------------- a schema nobody could parse

    [<Fact>]
    member _.``A schema that is not valid JSON verifies nothing``() =
        // One unquoted key — a typo, or a schema a model wrote — and this used to
        // report every payload verified, including ones the schema would plainly
        // have rejected.
        let broken = "{ type: object, required: [answer] }"

        match
            Verification.verify """{"anything":1}""" (Schema broken) noTools
            |> Async.RunSynchronously
        with
        | Result.Ok verdict -> failwith $"expected an error, got {verdict}"
        | Result.Error message -> Assert.Contains("not valid JSON", message)

    [<Fact>]
    member _.``A schema that parses is still applied``() =
        let schema = """{"type":"object","required":["answer"]}"""

        let verdict (content: string) =
            match Verification.verify content (Schema schema) noTools |> Async.RunSynchronously with
            | Result.Ok value -> value
            | Result.Error message -> failwith message

        Assert.True(verdict """{"answer":"yes"}""")
        Assert.False(verdict """{"other":"no"}""")

    // ------------------------------------------------------------ a verdict, or a word

    [<Fact>]
    member _.``A rejection is not a verification because it contains the word``() =
        for answer in
            [ "NOT VERIFIED"
              "REJECTED - the claim is not VERIFIED"
              "This cannot be VERIFIED without a source."
              "**NOT VERIFIED**"
              // Reasoning that talks itself into a rejection is still a rejection:
              // stripping the block must not promote its contents to the verdict.
              "<thinking>\nVERIFIED, surely?\n</thinking>\nREJECTED"
              "" ] do
            Assert.False(EpistemicVerdict.saysVerified answer, $"'{answer}' was read as a verification")

    [<Fact>]
    member _.``A verdict of VERIFIED still reads as one, however it is dressed``() =
        for answer in
            [ "VERIFIED"
              "VERIFIED."
              "**VERIFIED**"
              "This statement is VERIFIED."
              "VERIFIED: the statement matches the cited source"
              // The explanation below the verdict is free to say what it could not
              // confirm; only the verdict line is read.
              "VERIFIED\nI could not check the second clause."
              // What a thinking model actually returns through OllamaClient: the
              // verdict is on the first line *of the answer*, not of the response.
              "<thinking>\nThe claim cannot be checked against a source directly, but the cited\nreference is authoritative.\n</thinking>\nVERIFIED"
              "<think>weighing it up</think>\nVERIFIED: matches the cited source" ] do
            Assert.True(EpistemicVerdict.saysVerified answer, $"'{answer}' was not read as a verification")

    // ----------------------------------------------------- a budget that stopped counting

    [<Fact>]
    member _.``Spending past the budget is still counted``() =
        // `TryConsume` does not record what it refuses, so the governor's total used
        // to freeze at the exact moment the budget was meant to start biting: every
        // call after the limit was both unrecorded and unblocked.
        let governor =
            BudgetGovernor(
                { Budget.Infinite with
                    MaxTokens = Some(Units.toTokens 100) }
            )

        let epistemic =
            EpistemicGovernor(CountingLlm(60) :> ILlmService, None, Some governor) :> IEpistemicGovernor

        epistemic.Verify("a statement").GetAwaiter().GetResult() |> ignore
        epistemic.Verify("another statement").GetAwaiter().GetResult() |> ignore

        Assert.Equal(120, int governor.Consumed.Tokens)
