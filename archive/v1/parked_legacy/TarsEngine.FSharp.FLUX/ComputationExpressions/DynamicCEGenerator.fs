namespace TarsEngine.FSharp.FLUX.ComputationExpressions

open System
open System.Text.RegularExpressions

/// Dynamic Computation Expression Generator
/// Generates F# computation expressions from EBNF grammars
module DynamicCEGenerator =

    /// EBNF Grammar Rule
    type GrammarRule = {
        Name: string
        Definition: string
        IsTerminal: bool
        Dependencies: string list
    }

    /// Parse EBNF grammar content into rules
    let private parseGrammar (content: string) : GrammarRule list =
        let lines = content.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
        let rulePattern = Regex(@"^\s*(\w+)\s*::=\s*(.+)$", RegexOptions.Compiled)

        lines
        |> Array.choose (fun line ->
            let line = line.Trim()
            if String.IsNullOrEmpty(line) || line.StartsWith("//") then None
            else
                let m = rulePattern.Match(line)
                if m.Success then
                    let name = m.Groups.[1].Value
                    let definition = m.Groups.[2].Value
                    let isTerminal = not (definition.Contains("::=") || definition.Contains("|"))
                    let dependencies =
                        Regex.Matches(definition, @"\b[A-Z][a-zA-Z0-9_]*\b")
                        |> Seq.cast<Match>
                        |> Seq.map (fun m -> m.Value)
                        |> Seq.filter (fun dep -> dep <> name)
                        |> Seq.distinct
                        |> Seq.toList
                    Some { Name = name; Definition = definition; IsTerminal = isTerminal; Dependencies = dependencies }
                else None)
        |> Array.toList

    /// Generate computation expression methods from grammar rules
    let private generateMethods (rules: GrammarRule list) : string =
        let bindMethods =
            rules
            |> List.filter (not << _.IsTerminal)
            |> List.map (fun rule ->
                sprintf """    member this.%s(value, continuation) =
        match value with
        | Some result -> continuation result
        | None -> this.Zero()""" rule.Name)
            |> String.concat "\n\n"

        let combineMethods =
            rules
            |> List.filter (fun rule -> rule.Dependencies.Length > 1)
            |> List.map (fun rule ->
                let parameters = rule.Dependencies |> List.mapi (fun i dep -> sprintf "v%d" i) |> String.concat ", "
                sprintf "    member this.Combine%s(%s) = \n        (%s)" rule.Name parameters parameters)
            |> String.concat "\n\n"

        [bindMethods; combineMethods]
        |> List.filter (not << String.IsNullOrEmpty)
        |> String.concat "\n\n"

    /// Generate computation expression from grammar
    let generateComputationExpression (grammarName: string) (grammarContent: string) : string =
        if String.IsNullOrWhiteSpace(grammarContent) then
            // Fallback for empty grammar
            sprintf """// Generated Computation Expression for %s (Empty Grammar)
type %sBuilder() =
    member this.Bind(x, f) =
        match x with
        | Some value -> f value
        | None -> None
    member this.Return(x) = Some x
    member this.Zero() = None
    member this.Combine(a, b) =
        match a with
        | Some _ -> a
        | None -> b

let %s = %sBuilder()""" grammarName grammarName (grammarName.ToLowerInvariant()) grammarName
        else
            try
                let rules = parseGrammar grammarContent
                let methods = generateMethods rules
                let builderName = sprintf "%sBuilder" grammarName
                let instanceName = grammarName.ToLowerInvariant()

                sprintf """// Generated Computation Expression for %s
// Parsed %d grammar rules from EBNF content
type %s() =
    member this.Bind(x, f) =
        match x with
        | Some value -> f value
        | None -> None
    member this.Return(x) = Some x
    member this.Zero() = None
    member this.Combine(a, b) =
        match a with
        | Some _ -> a
        | None -> b
    member this.Delay(f) = f()
    member this.Run(f) = f

%s

let %s = %s()

// Grammar Rules:
%s"""
                    grammarName
                    rules.Length
                    builderName
                    methods
                    instanceName
                    builderName
                    (rules |> List.map (fun r -> sprintf "// %s ::= %s" r.Name r.Definition) |> String.concat "\n")
            with
            | ex ->
                // Fallback on parse error
                sprintf """// Generated Computation Expression for %s (Parse Error: %s)
type %sBuilder() =
    member this.Bind(x, f) =
        match x with
        | Some value -> f value
        | None -> None
    member this.Return(x) = Some x
    member this.Zero() = None

let %s = %sBuilder()""" grammarName ex.Message grammarName (grammarName.ToLowerInvariant()) grammarName

    printfn "🧬 Real Dynamic CE Generator Loaded - Parses EBNF Grammar"
