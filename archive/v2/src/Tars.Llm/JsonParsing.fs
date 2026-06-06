namespace Tars.Llm

open System
open System.Text.Json

module JsonParsing =

    let private tryParseRaw (text: string) =
        try
            use doc = JsonDocument.Parse(text)
            Ok(doc.RootElement.Clone())
        with ex ->
            Error ex.Message

    let private tryExtractFenced (text: string) =
        let start = text.IndexOf("```", StringComparison.Ordinal)
        if start < 0 then
            None
        else
            let lineBreak = text.IndexOf('\n', start + 3)
            let contentStart = if lineBreak >= 0 then lineBreak + 1 else start + 3
            let endFence = text.LastIndexOf("```", StringComparison.Ordinal)
            if endFence > contentStart then
                Some(text.Substring(contentStart, endFence - contentStart))
            else
                None

    let private tryFindJsonSpan (text: string) =
        let mutable start = -1
        let mutable depth = 0
        let mutable inString = false
        let mutable escape = false
        let mutable result: (int * int) option = None
        let mutable idx = 0

        while idx < text.Length && result.IsNone do
            let c = text[idx]

            if start = -1 then
                if c = '{' || c = '[' then
                    start <- idx
                    depth <- 1
            else if inString then
                if escape then
                    escape <- false
                elif c = '\\' then
                    escape <- true
                elif c = '"' then
                    inString <- false
            else
                match c with
                | '"' -> inString <- true
                | '{'
                | '[' -> depth <- depth + 1
                | '}'
                | ']' ->
                    depth <- depth - 1
                    if depth = 0 then
                        result <- Some(start, idx)
                | _ -> ()

            idx <- idx + 1

        result

    let tryParseElement (text: string) =
        let trimmed = text.Trim()

        match tryParseRaw trimmed with
        | Ok elem -> Ok elem
        | Error firstError ->
            let fenced =
                match tryExtractFenced trimmed with
                | Some inner -> inner.Trim()
                | None -> trimmed

            match tryParseRaw fenced with
            | Ok elem -> Ok elem
            | Error _ ->
                match tryFindJsonSpan trimmed with
                | Some(startIdx, endIdx) ->
                    let json = trimmed.Substring(startIdx, endIdx - startIdx + 1)
                    match tryParseRaw json with
                    | Ok elem -> Ok elem
                    | Error secondError -> Error secondError
                | None -> Error firstError
