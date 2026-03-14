namespace Tars.Core

open System
open System.Text.Json

/// <summary>
/// Minimal grammar distillation helpers for structured outputs (JSON-ish) to encourage LLM conformance.
/// Distils field shapes from examples and produces prompt hints plus a parser-based validator (no regex).
/// </summary>
module GrammarDistill =

    type GrammarSpec =
        { Fields: string list
          Required: string list
          Example: string
          PromptHint: string
          Validator: string -> bool }

    /// Interface for pluggable grammar distillers (so we can swap implementations, e.g., JSON Schema/GBNF-backed).
    type IGrammarDistiller =
        abstract FromJsonExamples : examples: string list -> GrammarSpec

    let private tryParseObject (json: string) =
        try
            use doc = JsonDocument.Parse(json)
            let root = doc.RootElement
            if root.ValueKind <> JsonValueKind.Object then None
            else
                let names =
                    root.EnumerateObject()
                    |> Seq.map (fun p -> p.Name)
                    |> Seq.toList
                Some(names |> Set.ofList)
        with _ ->
            None

    /// <summary>
    /// Distil a simple JSON-object-like grammar from examples (expects flat key/value pairs).
    /// Uses JsonDocument parsing (no regex) and enforces required + no-extra fields.
    /// </summary>
    let fromJsonExamplesInternal (examples: string list) : GrammarSpec =
        let parsed =
            examples
            |> List.choose tryParseObject

        let fields =
            parsed
            |> List.collect Set.toList
            |> Set.ofList
            |> Set.toList
            |> List.sort

        let required =
            match parsed with
            | [] -> fields
            | sets ->
                sets
                |> List.reduce Set.intersect
                |> Set.toList
                |> List.sort

        let example =
            examples |> List.tryHead |> Option.defaultValue "{}"

        let promptHint =
            let fieldList = String.Join(", ", fields)
            let requiredList = String.Join(", ", required)
            $"Respond ONLY with a JSON object. Required fields: [{requiredList}]. Allowed fields: [{fieldList}]. No extra fields. No prose."

        let validator (text: string) =
            match tryParseObject text with
            | None -> false
            | Some names ->
                let hasRequired = required |> List.forall (fun r -> names.Contains r)
                let onlyKnown = names |> Set.forall (fun n -> fields |> List.contains n)
                hasRequired && onlyKnown

        { Fields = fields
          Required = required
          Example = example
          PromptHint = promptHint
          Validator = validator }

    /// Default distiller instance (parser-based).
    type DefaultDistiller() =
        interface IGrammarDistiller with
            member _.FromJsonExamples(examples) = fromJsonExamplesInternal examples

    /// Default distiller singleton.
    let defaultDistiller : IGrammarDistiller = DefaultDistiller() :> IGrammarDistiller

    /// Backward-compatible helper that uses the default distiller.
    let fromJsonExamples (examples: string list) : GrammarSpec =
        defaultDistiller.FromJsonExamples examples

    /// <summary>
    /// Build a Metascript-oriented prompt hint (simple placeholder).
    /// </summary>
    let metascriptHint =
        "Respond ONLY with a Metascript JSON node: {\"id\":\"...\",\"type\":\"agent|tool|decision\",\"instruction\":\"...\",\"outputs\":[...]} with no prose."
