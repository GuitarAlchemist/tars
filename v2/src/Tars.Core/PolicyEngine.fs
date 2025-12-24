namespace Tars.Core

open System
open System.Text.RegularExpressions

/// Explicit policy checks for guarding prompts/outputs.
module PolicyEngine =

    type PolicyInput =
        { Text: string
          Metadata: Map<string, string> }

    type PolicyOutcome =
        { Name: string
          Passed: bool
          Messages: string list }

    type Policy = PolicyInput -> PolicyOutcome

    let private pass name =
        { Name = name
          Passed = true
          Messages = [] }

    let private fail name message =
        { Name = name
          Passed = false
          Messages = [ message ] }

    let private hasMatch (pattern: string) (text: string) =
        Regex.IsMatch(text, pattern, RegexOptions.IgnoreCase ||| RegexOptions.CultureInvariant)

    let private noPlaceholders : Policy =
        fun input ->
            let text = input.Text
            let patterns =
                [ @"\bTODO\b"
                  @"\bTBD\b"
                  @"\bFIXME\b"
                  @"\blorem ipsum\b"
                  @"\bplaceholder\b"
                  @"\bfill in\b"
                  @"\breplace me\b"
                  @"\bchangeme\b"
                  @"\bto be decided\b" ]

            if patterns |> List.exists (fun p -> hasMatch p text) then
                fail "no_placeholders" "Placeholder content detected."
            else
                pass "no_placeholders"

    let private noDestructiveCommands : Policy =
        fun input ->
            let text = input.Text
            let patterns =
                [ @"\brm\s+-rf\b"
                  @"\bformat\s+c:\b"
                  @"\bdrop\s+database\b"
                  @"\bsudo\s+rm\b"
                  @"\bmkfs\b"
                  @":\(\)\s*\{\s*:\|\:&\s*;\s*\};" ]

            if patterns |> List.exists (fun p -> hasMatch p text) then
                fail "no_destructive_commands" "Destructive command pattern detected."
            else
                pass "no_destructive_commands"

    let private requireCitations : Policy =
        fun input ->
            let text = input.Text.ToLowerInvariant()

            let hasBracketed =
                Regex.IsMatch(text, @"\[[^\]]+\]", RegexOptions.CultureInvariant)

            if hasBracketed
               || text.Contains("source")
               || text.Contains("cite")
               || text.Contains("reference") then
                pass "require_citations"
            else
                fail "require_citations" "No citation or source detected."

    let private schemaProvided : Policy =
        fun input ->
            match input.Metadata |> Map.tryFind "schema" with
            | Some value when not (String.IsNullOrWhiteSpace value) -> pass "schema_required"
            | _ -> fail "schema_required" "Schema metadata missing."

    let defaultPolicies : Map<string, Policy> =
        [ "no_placeholders", noPlaceholders
          "no_destructive_commands", noDestructiveCommands
          "require_citations", requireCitations
          "schema_required", schemaProvided ]
        |> Map.ofList

    let evaluate (policies: Map<string, Policy>) (names: string list) (input: PolicyInput) =
        let unique =
            names
            |> List.map (fun n -> n.Trim())
            |> List.filter (fun n -> not (String.IsNullOrWhiteSpace n))
            |> List.distinct

        unique
        |> List.map (fun name ->
            match policies |> Map.tryFind name with
            | Some policy -> policy input
            | None -> fail name $"Unknown policy: {name}")

    let evaluateDefault (names: string list) (input: PolicyInput) =
        evaluate defaultPolicies names input

    let anyFailed (outcomes: PolicyOutcome list) =
        outcomes |> List.exists (fun o -> not o.Passed)
