module Tars.Tests.PolicyEngineTests

open Xunit
open Tars.Core

[<Fact>]
let ``unknown policy fails`` () =
    let input: PolicyEngine.PolicyInput =
        { Text = "safe text"
          Metadata = Map.empty }

    let outcomes = PolicyEngine.evaluateDefault [ "unknown_policy" ] input

    Assert.Single outcomes |> ignore
    Assert.False outcomes.Head.Passed
    Assert.Contains("Unknown policy", outcomes.Head.Messages |> List.head)

[<Fact>]
let ``no_placeholders fails on TODO`` () =
    let input: PolicyEngine.PolicyInput =
        { Text = "TODO: replace this logic"
          Metadata = Map.empty }

    let outcomes = PolicyEngine.evaluateDefault [ "no_placeholders" ] input

    Assert.Single outcomes |> ignore
    Assert.False outcomes.Head.Passed

[<Fact>]
let ``no_destructive_commands fails on rm -rf`` () =
    let input: PolicyEngine.PolicyInput =
        { Text = "rm -rf /tmp"
          Metadata = Map.empty }

    let outcomes = PolicyEngine.evaluateDefault [ "no_destructive_commands" ] input

    Assert.Single outcomes |> ignore
    Assert.False outcomes.Head.Passed

[<Fact>]
let ``require_citations enforces citations`` () =
    let input: PolicyEngine.PolicyInput =
        { Text = "This is an answer without additional detail."
          Metadata = Map.empty }

    let outcomes = PolicyEngine.evaluateDefault [ "require_citations" ] input

    Assert.False outcomes.Head.Passed

    let citedInput: PolicyEngine.PolicyInput =
        { Text = "See [1] for the reference."
          Metadata = Map.empty }

    let cited = PolicyEngine.evaluateDefault [ "require_citations" ] citedInput

    Assert.True cited.Head.Passed

[<Fact>]
let ``schema_required enforces metadata`` () =
    let input: PolicyEngine.PolicyInput =
        { Text = "Answer without schema metadata."
          Metadata = Map.empty }

    let outcomes = PolicyEngine.evaluateDefault [ "schema_required" ] input

    Assert.False outcomes.Head.Passed

    let withSchema: PolicyEngine.PolicyInput =
        { Text = "Structured response."
          Metadata = Map.ofList [ "schema", "response-schema-v1" ] }

    let schemaOutcome = PolicyEngine.evaluateDefault [ "schema_required" ] withSchema

    Assert.True schemaOutcome.Head.Passed
