namespace TarsEngine.FSharp.Core.Grammar

open System
open System.Text
open System.Text.RegularExpressions

/// Tokens recognised when parsing grammar production expressions.
type GrammarToken =
    | NonTerminal of string
    | Terminal of string
    | Identifier of string
    | Symbol of char

/// Parsed grammar production record.
type GrammarProduction =
    { Head: string
      OriginalHead: string
      Expression: string
      Tokens: GrammarToken list
      SourceLine: int }

/// Complete grammar specification parsed from a source.
type GrammarSpecification =
    { Name: string
      SourceText: string
      Productions: GrammarProduction list
      NonTerminals: Set<string>
      Terminals: Set<string> }

/// Detailed parse error raised when the grammar text is invalid.
type GrammarParseError =
    { Line: int
      Message: string }

exception GrammarParseException of GrammarParseError

[<RequireQualifiedAccess>]
module private Internal =

    let private removeBlockComments (text: string) =
        Regex.Replace(text, @"\(\*.*?\*\)|/\*.*?\*/", "", RegexOptions.Singleline)

    let private removeLineComments (text: string) =
        Regex.Replace(text, @"(?m)^\s*(#|//).*$", "")

    let sanitiseHeading (heading: string) =
        let trimmed = heading.Trim()
        if trimmed.StartsWith("<", StringComparison.Ordinal) && trimmed.EndsWith(">", StringComparison.Ordinal) then
            trimmed.Substring(1, trimmed.Length - 2).Trim()
        else
            trimmed

    let private isIdentifierChar (c: char) =
        Char.IsLetterOrDigit(c) || c = '_' || c = '-' || c = '.'

    let private isSymbolChar (c: char) =
        match c with
        | '{' | '}' | '[' | ']' | '(' | ')' | '|' | ',' | '*' | '+' | '?' -> true
        | _ -> false

    let private hasTerminatingSemicolonOutsideQuotes (text: string) =
        let mutable inSingle = false
        let mutable inDouble = false
        let mutable escaped = false

        let mutable found = false
        for ch in text do
            if escaped then
                escaped <- false
            else
                match ch with
                | '\\' when inSingle || inDouble -> escaped <- true
                | '\'' when not inDouble -> inSingle <- not inSingle
                | '"' when not inSingle -> inDouble <- not inDouble
                | ';' when not inSingle && not inDouble -> found <- true
                | _ -> ()

        found

    let private tokenizeExpression (expression: string) (lineNumber: int) =
        let tokens = ResizeArray<GrammarToken>()
        let stack = Stack<char>()
        let mutable index = 0
        let length = expression.Length

        let push expected = stack.Push(expected)
        let pop expected =
            if stack.Count = 0 || stack.Peek() <> expected then
                let msg = $"Mismatched delimiter. Expected '{expected}' before end of rule."
                raise (GrammarParseException { Line = lineNumber; Message = msg })
            else
                stack.Pop() |> ignore

        let rec readString (quote: char) (start: int) (builder: StringBuilder) =
            if index >= length then
                let msg = $"Unterminated string literal starting at position {start}."
                raise (GrammarParseException { Line = lineNumber; Message = msg })
            else
                let current = expression[index]
                index <- index + 1
                match current with
                | '\\' when index < length ->
                    builder.Append(expression[index]) |> ignore
                    index <- index + 1
                    readString quote start builder
                | ch when ch = quote ->
                    builder.ToString()
                | ch ->
                    builder.Append(ch) |> ignore
                    readString quote start builder

        while index < length do
            let mutable current = expression[index]

            if Char.IsWhiteSpace(current) then
                index <- index + 1
            else
                match current with
                | '<' ->
                    let start = index
                    let closing = expression.IndexOf('>', index + 1)
                    if closing = -1 then
                        let msg = "Unterminated non-terminal reference."
                        raise (GrammarParseException { Line = lineNumber; Message = msg })
                    else
                        let name = expression.Substring(start + 1, closing - start - 1).Trim()
                        if String.IsNullOrWhiteSpace(name) then
                            let msg = "Non-terminal reference cannot be empty."
                            raise (GrammarParseException { Line = lineNumber; Message = msg })
                        tokens.Add(NonTerminal name)
                        index <- closing + 1
                | '"' | '\'' as quote ->
                    index <- index + 1
                    let literal = readString quote (index - 1) (StringBuilder())
                    tokens.Add(Terminal literal)
                | '{' ->
                    tokens.Add(Symbol current)
                    push '}'
                    index <- index + 1
                | '[' ->
                    tokens.Add(Symbol current)
                    push ']'
                    index <- index + 1
                | '(' ->
                    tokens.Add(Symbol current)
                    push ')'
                    index <- index + 1
                | '}' | ']' | ')' as closing ->
                    pop closing
                    tokens.Add(Symbol closing)
                    index <- index + 1
                | '|' | ',' | '*' | '+' | '?' as symbol ->
                    tokens.Add(Symbol symbol)
                    index <- index + 1
                | _ when isIdentifierChar current ->
                    let start = index
                    while index < length
                          && isIdentifierChar expression[index]
                          && not (isSymbolChar expression[index]) do
                        index <- index + 1

                    let identifier = expression.Substring(start, index - start)
                    tokens.Add(Identifier identifier)
                | unexpected ->
                    let msg = $"Unexpected character '{unexpected}' in production expression."
                    raise (GrammarParseException { Line = lineNumber; Message = msg })

        if stack.Count <> 0 then
            let expected = stack.Peek()
            let msg = $"Missing closing delimiter '{expected}'."
            raise (GrammarParseException { Line = lineNumber; Message = msg })

        tokens |> Seq.toList

    let parseProduction (lineNumber: int) (text: string) =
        let trimmed = text.Trim()
        if String.IsNullOrWhiteSpace(trimmed) then
            let msg = "Empty grammar production encountered."
            raise (GrammarParseException { Line = lineNumber; Message = msg })

        let pattern = @"^(?<lhs><[^>]+>|[A-Za-z][A-Za-z0-9_\-]*)\s*(?<delim>::=|=)\s*(?<rhs>.+)$"
        let matchResult = Regex.Match(trimmed, pattern, RegexOptions.Singleline)
        if not matchResult.Success then
            let msg = "Grammar production must contain either ::= or = as a delimiter."
            raise (GrammarParseException { Line = lineNumber; Message = msg })

        let originalHead = matchResult.Groups["lhs"].Value.Trim()
        let head = sanitiseHeading originalHead
        let delimiter = matchResult.Groups["delim"].Value
        let rhs = matchResult.Groups["rhs"].Value.Trim()

        let expression =
            if delimiter = "=" then
                let cleaned = rhs.TrimEnd()
                let mutable endIndex = cleaned.Length - 1
                while endIndex >= 0 && Char.IsWhiteSpace(cleaned[endIndex]) do
                    endIndex <- endIndex - 1
                if endIndex >= 0 && cleaned[endIndex] = ';' then
                    cleaned.Substring(0, endIndex).TrimEnd()
                else
                    let msg = "EBNF productions must terminate with ';'."
                    raise (GrammarParseException { Line = lineNumber; Message = msg })
            else
                rhs

        let tokens = tokenizeExpression expression lineNumber
        if List.isEmpty tokens then
            let msg = "Grammar production must define at least one symbol."
            raise (GrammarParseException { Line = lineNumber; Message = msg })

        { Head = head
          OriginalHead = originalHead
          Expression = expression
          Tokens = tokens
          SourceLine = lineNumber }

    let parseProductions (sourceName: string) (text: string) =
        let normalised =
            text
            |> removeBlockComments
            |> removeLineComments
            |> fun content -> content.Replace("\r\n", "\n").Replace("\r", "\n")

        let lines = normalised.Split('\n')
        let productions = ResizeArray<GrammarProduction>()

        let buffer = StringBuilder()
        let mutable currentLine = 0
        let mutable collecting = false
        let mutable delimiter = ""
        let mutable expectContinuation = false

        let flush () =
            if buffer.Length > 0 then
                let productionText = buffer.ToString()
                let production = parseProduction currentLine productionText
                productions.Add(production)
                buffer.Clear() |> ignore
                collecting <- false
                delimiter <- ""
                expectContinuation <- false

        let mutable index = 0
        while index < lines.Length do
            let raw = lines[index].TrimEnd()
            let trimmed = raw.Trim()

            if String.IsNullOrWhiteSpace(trimmed) then
                if collecting && delimiter = "::=" then
                    flush()
                index <- index + 1
            elif not collecting then
                let hasBnf = trimmed.Contains("::=")
                let hasEbnf = trimmed.Contains("=")

                if not hasBnf && not hasEbnf then
                    index <- index + 1
                else
                    currentLine <- index + 1
                    buffer.Clear() |> ignore
                    buffer.Append(trimmed) |> ignore
                    if hasBnf then
                        collecting <- true
                        delimiter <- "::="
                        expectContinuation <- trimmed.EndsWith("|", StringComparison.Ordinal)
                        index <- index + 1
                    else
                        delimiter <- "="
                        if hasTerminatingSemicolonOutsideQuotes trimmed then
                            flush()
                            index <- index + 1
                        else
                            collecting <- true
                            index <- index + 1
            else
                if delimiter = "=" then
                    buffer.Append(' ').Append(trimmed) |> ignore
                    if hasTerminatingSemicolonOutsideQuotes(buffer.ToString()) then
                        flush()
                    index <- index + 1
                else
                    let trimmedLeading = raw.TrimStart()
                    if expectContinuation || trimmedLeading.StartsWith("|", StringComparison.Ordinal) then
                        buffer.Append(' ').Append(trimmedLeading) |> ignore
                        expectContinuation <- trimmedLeading.EndsWith("|", StringComparison.Ordinal)
                        index <- index + 1
                    elif trimmed.Contains("::=") then
                        flush()
                    else
                        flush()
                        // reprocess same line as a new potential production
            done

        if collecting then
            flush()

        if productions.Count = 0 then
            let msg = $"No productions were parsed from '{sourceName}'."
            raise (GrammarParseException { Line = 0; Message = msg })

        productions |> Seq.toList

open Internal

/// Parser utilities for EBNF/BNF grammar definitions.
module EbnfParser =

    /// Try to parse a single production, returning either the parsed production or an error.
    let tryParseProduction (lineNumber: int) (text: string) : Result<GrammarProduction, GrammarParseError> =
        try
            parseProduction lineNumber text |> Ok
        with
        | :? GrammarParseException as ex ->
            let error = ex.Data0 :?> GrammarParseError
            Error error

    /// Parse a single grammar production; raises GrammarParseException on failure.
    let parseProductionOrFail (lineNumber: int) (text: string) : GrammarProduction =
        match tryParseProduction lineNumber text with
        | Ok production -> production
        | Error error -> raise (GrammarParseException error)

    /// Parse a complete grammar specification from the provided text.
    let fromText (specName: string) (text: string) : GrammarSpecification =
        let productions = parseProductions specName text

        let nonTerminals =
            productions
            |> List.collect (fun production ->
                let tokens =
                    production.Tokens
                    |> List.choose (function
                        | NonTerminal v -> Some v
                        | Identifier v -> Some v
                        | _ -> None)
                production.Head :: tokens)
            |> Set.ofList

        let terminals =
            productions
            |> List.collect (fun production ->
                production.Tokens
                |> List.choose (function
                    | Terminal v -> Some v
                    | _ -> None))
            |> Set.ofList

        { Name = specName
          SourceText = text
          Productions = productions
          NonTerminals = nonTerminals
          Terminals = terminals }
