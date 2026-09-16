/// Pure parsing helpers shared by the Graph-, Tree- and Workflow-of-Thought patterns.
/// Split out of Patterns.fs so they can be tested without running a workflow (#280).
module Tars.Cortex.ThoughtParsing

open System
open System.Text
open System.Text.Json
open Tars.Llm

/// Property `name` of a JSON object, matched case-insensitively.
let tryGetPropertyInsensitive (name: string) (elem: JsonElement) =
    if elem.ValueKind = JsonValueKind.Object then
        elem.EnumerateObject()
        |> Seq.tryFind (fun p -> p.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
        |> Option.map (fun p -> p.Value)
    else
        None

/// First of `names` holding a number (or a numeric string), else `defaultValue`.
let getDoubleWithDefault (names: string list) (defaultValue: float) (elem: JsonElement) =
    names
    |> List.tryPick (fun name ->
        match tryGetPropertyInsensitive name elem with
        | Some prop ->
            match prop.ValueKind with
            | JsonValueKind.Number -> Some(prop.GetDouble())
            | JsonValueKind.String ->
                match Double.TryParse(prop.GetString()) with
                | true, v -> Some v
                | _ -> None
            | _ -> None
        | None -> None)
    |> Option.defaultValue defaultValue

/// Property `name` as a string list; a single string becomes a one-item list.
let getStringList (name: string) (elem: JsonElement) =
    match tryGetPropertyInsensitive name elem with
    | Some prop ->
        match prop.ValueKind with
        | JsonValueKind.Array ->
            prop.EnumerateArray()
            |> Seq.choose (fun item ->
                if item.ValueKind = JsonValueKind.String then
                    Some(item.GetString())
                else
                    None)
            |> Seq.toList
        | JsonValueKind.String -> [ prop.GetString() ]
        | _ -> []
    | None -> []

/// Clamp to [0, 1].
let clamp01 (value: float) = value |> max 0.0 |> min 1.0

/// Remove a surrounding ``` fence (with optional language tag), if any.
let stripCodeFence (value: string) =
    let trimmed = value.Trim()

    if trimmed.StartsWith("```", StringComparison.Ordinal) then
        let withoutTicks = trimmed.Substring(3)
        let newLineIdx = withoutTicks.IndexOfAny([| '\n'; '\r' |])

        let body =
            if newLineIdx >= 0 then
                withoutTicks.Substring(newLineIdx + 1).Trim()
            else
                withoutTicks.Trim()

        if body.EndsWith("```", StringComparison.Ordinal) then
            body.Substring(0, body.Length - 3).Trim()
        else
            body
    else
        trimmed

/// Parse an LLM reply as JSON, retrying on the outermost {...} when prose surrounds it.
let tryParseJsonWithFallback (text: string) =
    let cleaned = stripCodeFence text

    if String.IsNullOrWhiteSpace cleaned then
        Result.Error "empty response"
    else
        match JsonParsing.tryParseElement cleaned with
        | Result.Ok elem -> Result.Ok elem
        | Result.Error firstError ->
            let startIdx = cleaned.IndexOf('{')
            let endIdx = cleaned.LastIndexOf('}')

            if startIdx >= 0 && endIdx > startIdx then
                let slice = cleaned.Substring(startIdx, endIdx - startIdx + 1)

                match JsonParsing.tryParseElement slice with
                | Result.Ok elem -> Result.Ok elem
                | Result.Error secondError -> Result.Error($"{firstError}; {secondError}")
            else
                Result.Error firstError

/// Split an LLM reply into thoughts. `THOUGHT n:` headers delimit thoughts when present;
/// otherwise lines longer than 20 characters are joined. Fragments of 10 characters or
/// fewer are dropped.
let parseThoughts (text: string) =
    // Models often emphasise the header (`**THOUGHT 1:**`) and open with a preamble
    // ("Here are three thoughts:"), which must not become a thought itself (#263).
    let isHeader (line: string) =
        line.TrimStart('*', '#', ' ').StartsWith("THOUGHT", StringComparison.OrdinalIgnoreCase)

    let lines = text.Split([| '\n' |], StringSplitOptions.RemoveEmptyEntries)
    let hasHeaders = lines |> Array.exists (fun l -> isHeader (l.Trim()))
    let mutable seenHeader = false
    let mutable thoughts = []
    let mutable currentThought = StringBuilder()

    for line in lines do
        let trimmed = line.Trim()

        if isHeader trimmed then
            seenHeader <- true

            if currentThought.Length > 0 then
                thoughts <- currentThought.ToString().Trim() :: thoughts
                currentThought.Clear() |> ignore

            let colonIdx = trimmed.IndexOf(':')

            if colonIdx > 0 && colonIdx < trimmed.Length - 1 then
                currentThought.Append(trimmed.Substring(colonIdx + 1).Trim().TrimStart('*').Trim()) |> ignore
        elif hasHeaders && not seenHeader then
            ()
        else if currentThought.Length > 0 || trimmed.Length > 20 then
            if currentThought.Length > 0 then
                currentThought.Append(" ") |> ignore

            currentThought.Append(trimmed) |> ignore

    if currentThought.Length > 0 then
        thoughts <- currentThought.ToString().Trim() :: thoughts

    thoughts |> List.rev |> List.filter (fun s -> s.Length > 10)
