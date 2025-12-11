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
