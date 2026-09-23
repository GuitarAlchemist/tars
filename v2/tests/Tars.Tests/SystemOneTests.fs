module Tars.Tests.SystemOneTests

open Xunit
open Tars.Llm
open Tars.Llm.SystemOne

// The contract from the TypeSafe AI System One course's offline lab, ported to F#:
// a Jev reply is only a decision once it is checked against the question that asked for it.

let private questions =
    [ "next_step",
      Choice(
          "Which next step does the evidence support?",
          [ "design_review", "Evidence is incomplete."
            "bounded_implementation", "Design and authority are explicit."
            "reject", "The proposal contradicts a hard constraint." ]
      )
      "delivery_risk",
      Score(
          "How hard would the effect be to reverse?",
          [ "Low and reversible"; "Moderate or compensatable"; "High or hard to reverse" ]
      )
      "authority_present", Noul "Does the state carry explicit authority?" ]

/// The course's mock reply, whose numbers satisfy every rule.
let private validReply =
    """{
      "model": "jev-1.13.0",
      "answers": {
        "next_step": {
          "type": "choice",
          "choice": "design_review",
          "confidence": 0.82,
          "probabilities": { "design_review": 0.88, "bounded_implementation": 0.1, "reject": 0.02 }
        },
        "delivery_risk": {
          "type": "score",
          "score": 1.2,
          "confidence": 0.73,
          "legend": { "0": "Low and reversible", "1": "Moderate or compensatable", "2": "High or hard to reverse" },
          "probabilities": { "0": 0.1, "1": 0.6, "2": 0.3 }
        },
        "authority_present": { "type": "noul", "noul": 0.2 }
      },
      "usage": { "input_tokens": 0, "output_tokens": 0 }
    }"""

let private rejects (json: string) =
    match parseReply questions json with
    | Ok reply -> failwith $"expected a rejection, got {reply.Answers}"
    | Error message -> message

[<Fact>]
let ``a well-formed reply parses into typed answers`` () =
    match parseReply questions validReply with
    | Error message -> failwith message
    | Ok reply ->
        Assert.Equal(PinnedModel, reply.Model)

        match reply.TryAnswer "next_step" with
        | Some(Chose(choice, confidence, probabilities)) ->
            Assert.Equal("design_review", choice)
            Assert.Equal(0.82, confidence, 6)
            Assert.Equal(0.88, probabilities.["design_review"], 6)
        | other -> failwith $"expected a choice, got {other}"

        match reply.TryAnswer "delivery_risk" with
        | Some(Scored(score, _, _)) -> Assert.Equal(1.2, score, 6)
        | other -> failwith $"expected a score, got {other}"

        match reply.TryAnswer "authority_present" with
        | Some(Nouled noul) -> Assert.Equal(0.2, noul, 6)
        | other -> failwith $"expected a noul, got {other}"

[<Fact>]
let ``an option we never offered is not a decision`` () =
    let message =
        validReply.Replace("\"choice\": \"design_review\"", "\"choice\": \"dispatch_now\"")
        |> rejects

    Assert.Contains("not one of the options", message)

[<Fact>]
let ``a distribution that does not sum to one is rejected`` () =
    let message = validReply.Replace("\"reject\": 0.02", "\"reject\": 0.5") |> rejects

    Assert.Contains("must sum to 1", message)

[<Fact>]
let ``probabilities must cover exactly the options asked for`` () =
    let message =
        validReply.Replace("\"reject\": 0.02", "\"reject\": 0.02, \"escalate\": 0.0")
        |> rejects

    Assert.Contains("exactly the options", message)

[<Fact>]
let ``a choice the model itself considers less likely is rejected`` () =
    // Naming design_review while calling bounded_implementation more probable is a reply
    // that argues against itself; taking the named option would launder that contradiction.
    let message =
        validReply
            .Replace("\"design_review\": 0.88", "\"design_review\": 0.1")
            .Replace("\"bounded_implementation\": 0.1", "\"bounded_implementation\": 0.88")
        |> rejects

    Assert.Contains("more likely", message)

[<Fact>]
let ``a score that contradicts its own distribution is rejected`` () =
    let message = validReply.Replace("\"score\": 1.2", "\"score\": 0.4") |> rejects
    Assert.Contains("does not match its own distribution", message)

[<Fact>]
let ``a score outside the levels asked for is rejected`` () =
    let message =
        validReply
            .Replace("\"score\": 1.2", "\"score\": 3.0")
            .Replace("\"0\": 0.1, \"1\": 0.6, \"2\": 0.3", "\"0\": 0.0, \"1\": 0.0, \"2\": 1.0")
        |> rejects

    Assert.Contains("between 0 and 2", message)

[<Fact>]
let ``a legend that rewrites the levels is rejected`` () =
    let message =
        validReply.Replace("\"2\": \"High or hard to reverse\"", "\"2\": \"Catastrophic\"")
        |> rejects

    Assert.Contains("legend", message)

[<Fact>]
let ``a noul outside zero to one is rejected`` () =
    let message = validReply.Replace("\"noul\": 0.2", "\"noul\": 1.4") |> rejects
    Assert.Contains("between 0 and 1", message)

[<Fact>]
let ``answers must cover exactly the questions asked`` () =
    let message =
        validReply.Replace(
            "\"authority_present\": { \"type\": \"noul\", \"noul\": 0.2 }",
            "\"mood\": { \"type\": \"noul\", \"noul\": 0.2 }"
        )
        |> rejects

    Assert.Contains("exactly the question ids", message)

[<Fact>]
let ``usage must be exactly two non-negative counts`` () =
    Assert.Contains("usage", validReply.Replace("\"output_tokens\": 0", "\"output_tokens\": -1") |> rejects)

    Assert.Contains(
        "usage",
        validReply.Replace("\"output_tokens\": 0", "\"output_tokens\": 0, \"cached_tokens\": 3")
        |> rejects
    )

[<Fact>]
let ``a reply from an unpinned model is refused where the pin matters`` () =
    let fromAnotherModel = validReply.Replace("jev-1.13.0", "jev-latest")

    // It is a valid reply...
    Assert.True((parseReply questions fromAnotherModel).IsOk)

    // ...but thresholds tuned on the pinned version do not transfer to a moving alias.
    match parsePinnedReply questions fromAnotherModel with
    | Ok _ -> failwith "expected the pin to be enforced"
    | Error message -> Assert.Contains(PinnedModel, message)

[<Fact>]
let ``the request pins the model, keeps the options closed, and hashes stably`` () =
    let state =
        [ "goal", Text "disambiguate a parse"
          "candidates", Items [ "plan_then_act"; "act_directly" ] ]

    let body = payload state questions

    Assert.Contains($"\"model\":\"{PinnedModel}\"", body)
    Assert.Contains("\"bounded_implementation\":", body)
    Assert.Contains("\"plan_then_act\"", body)
    Assert.Equal(digest body, digest (payload state questions))
    Assert.Equal(payloadBytes body, body.Length) // ASCII here, so bytes and chars agree
    Assert.NotEqual<string>(digest body, digest (payload [ "goal", Text "something else" ] questions))

[<Fact>]
let ``the replay adapter answers from a saved reply, with no key and no network`` () =
    let port = ReplaySystemOne(validReply) :> ISystemOne

    match port.Evaluate([ "goal", Text "anything" ], questions) |> Async.RunSynchronously with
    | Error message -> failwith message
    | Ok reply -> Assert.Equal(PinnedModel, reply.Model)
