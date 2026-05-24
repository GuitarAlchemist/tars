namespace Tars.Core

open System
open System.Text.RegularExpressions

/// <summary>
/// Grammar-based validation for LLM outputs.
/// Validates that outputs match expected patterns and provides helpful errors.
/// </summary>
module GrammarValidation =

    /// Result of validating an LLM output against a grammar
    type ValidationResult =
        | Valid of parsed: Map<string, string>
        | Invalid of ValidationError
    
    and ValidationError = {
        Expected: string
        Actual: string
        Error: string
        Suggestions: string list
    }

    /// A simple grammar pattern for validation
    /// Supports: {name} for captures, literal text, and basic regex
    type GrammarPattern = {
        Original: string
        Regex: Regex
        Captures: string list
    }

    /// Parse a simple grammar into a regex pattern
    /// Grammar syntax:
    ///   {name}     - Capture named group (non-greedy)
    ///   {name*}    - Capture named group (greedy, multi-line)
    ///   literal    - Match literal text
    ///   \n         - Match newline
    let parseGrammar (grammar: string) : GrammarPattern =
        let captures = ResizeArray<string>()

        // First, extract and replace our placeholders with temporary markers
        let mutable temp = grammar
        let mutable placeholders = []

        // Replace {name*} with greedy capture marker
        temp <- Regex.Replace(temp, @"\{(\w+)\*\}", fun m ->
            let name = m.Groups.[1].Value
            captures.Add(name)
            let marker = $"__GREEDY_{name}__"
            placeholders <- (marker, $"(?<{name}>[\\s\\S]*)") :: placeholders
            marker
        )

        // Replace {name} with non-greedy capture marker
        temp <- Regex.Replace(temp, @"\{(\w+)\}", fun m ->
            let name = m.Groups.[1].Value
            captures.Add(name)
            let marker = $"__CAPTURE_{name}__"
            placeholders <- (marker, $"(?<{name}>.*?)") :: placeholders
            marker
        )

        // Now escape the rest for regex
        let escaped = Regex.Escape(temp)

        // Replace markers back with regex patterns
        let mutable pattern = escaped
        for (marker, replacement) in placeholders do
            pattern <- pattern.Replace(Regex.Escape(marker), replacement)

        // Handle newlines - match any whitespace
        pattern <- pattern.Replace("\\n", @"\s*")

        {
            Original = grammar
            Regex = Regex(pattern, RegexOptions.Singleline)
            Captures = captures |> Seq.toList
        }

    /// Validate output against a grammar pattern
    let validate (grammar: string) (output: string) : ValidationResult =
        try
            let pattern = parseGrammar grammar
            let m = pattern.Regex.Match(output.Trim())
            
            if m.Success then
                let captures =
                    pattern.Captures
                    |> List.map (fun name -> name, m.Groups.[name].Value.Trim())
                    |> Map.ofList
                Valid captures
            else
                Invalid {
                    Expected = grammar
                    Actual = output
                    Error = "Output does not match expected pattern"
                    Suggestions = [
                        $"Expected format: {grammar}"
                        "Check that output contains all required elements"
                    ]
                }
        with ex ->
            Invalid {
                Expected = grammar
                Actual = output
                Error = $"Grammar parsing failed: {ex.Message}"
                Suggestions = ["Check grammar syntax"]
            }

    /// Check if output matches a grammar (simple bool)
    let matches (grammar: string) (output: string) : bool =
        match validate grammar output with
        | Valid _ -> true
        | Invalid _ -> false

    /// Extract a specific capture from validated output
    let extract (captureName: string) (result: ValidationResult) : string option =
        match result with
        | Valid captures -> Map.tryFind captureName captures
        | Invalid _ -> None

    /// Validate with multiple grammars, return first match
    let validateAny (grammars: string list) (output: string) : ValidationResult =
        grammars
        |> List.tryPick (fun g ->
            match validate g output with
            | Valid _ as v -> Some v
            | Invalid _ -> None)
        |> Option.defaultWith (fun () ->
            Invalid {
                Expected = String.Join(" OR ", grammars)
                Actual = output
                Error = "Output does not match any expected pattern"
                Suggestions = grammars |> List.map (sprintf "Try: %s")
            })

    /// Build a grammar hint to add to prompts
    let buildHint (grammar: string) : string =
        $"""Please respond in this exact format:
{grammar}

Replace placeholders like {{name}} with actual values."""

    /// Common grammar patterns
    module Patterns =
        let answer = "<answer>{text}</answer>"
        let json = "{\"result\": {value}}"
        let yesNo = "{answer}" // expects "yes" or "no"
        let codeBlock = "```{lang}\n{code*}\n```"
        let thinking = "<think>{reasoning*}</think>\n<answer>{answer}</answer>"
        let toolCall = """{"name": "{name}", "arguments": {args}}"""

