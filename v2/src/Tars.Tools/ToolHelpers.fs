namespace Tars.Tools

open System.Text.Json

module ToolHelpers =

    /// Parses a string argument that might be wrapped in JSON.
    /// If args is a JSON object { "propName": "value" }, returns "value".
    /// If args is a JSON string "value", returns "value".
    /// Otherwise returns args trimmed.
    let parseStringArg (args: string) (propName: string) =
        try
            let doc = JsonDocument.Parse(args)
            let root = doc.RootElement

            match root.ValueKind with
            | JsonValueKind.Object ->
                let mutable prop = Unchecked.defaultof<JsonElement>

                if root.TryGetProperty(propName, &prop) then
                    if prop.ValueKind = JsonValueKind.String then
                        prop.GetString()
                    else
                        prop.ToString()
                else
                    // Property not found.
                    // robustness fix: if the object has exactly one property, assume that's the argument
                    // regardless of the name (LLMs often hallucinate parameter names like "tool" vs "tool_name")
                    let props = root.EnumerateObject() |> Seq.toList

                    if props.Length = 1 && props.[0].Value.ValueKind = JsonValueKind.String then
                        props.[0].Value.GetString()
                    else
                        args.Trim()
            | JsonValueKind.String -> root.GetString()
            | _ -> args.Trim()
        with _ ->
            // JSON Parse failed -> Treat as raw string
            args.Trim()

    /// Tries to parse a string argument from JSON, returning None if not found or not JSON.
    let tryParseStringArg (args: string) (propName: string) =
        try
            let doc = JsonDocument.Parse(args)
            let root = doc.RootElement
            if root.ValueKind = JsonValueKind.Object then
                let mutable prop = Unchecked.defaultof<JsonElement>
                if root.TryGetProperty(propName, &prop) then
                    if prop.ValueKind = JsonValueKind.String then Some (prop.GetString())
                    else Some (prop.ToString())
                else
                    // If object has 1 prop, maybe it's the one?
                    let props = root.EnumerateObject() |> Seq.toList
                    if props.Length = 1 && props.[0].Value.ValueKind = JsonValueKind.String then
                        Some (props.[0].Value.GetString())
                    else None
            elif root.ValueKind = JsonValueKind.String then
                Some (root.GetString())
            else None
        with _ -> None

    /// Tries to parse an integer argument from JSON.
    let tryParseIntArg (args: string) (propName: string) =
        try
            let doc = JsonDocument.Parse(args)
            let root = doc.RootElement
            if root.ValueKind = JsonValueKind.Object then
                let mutable prop = Unchecked.defaultof<JsonElement>
                if root.TryGetProperty(propName, &prop) then
                    let mutable v = 0
                    if prop.TryGetInt32(&v) then Some v else None
                else None
            elif root.ValueKind = JsonValueKind.Number then
                Some (root.GetInt32())
            else None
        with _ ->
            // Try raw parse
            match System.Int32.TryParse(args.Trim()) with
            | true, v -> Some v
            | _ -> None
