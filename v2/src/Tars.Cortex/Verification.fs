namespace Tars.Cortex

open System
open System.Text.RegularExpressions
open System.Text.Json
open Tars.Cortex.WoTTypes

/// <summary>
/// Implements structured verification logic (Phase 15.1 Verifier++).
/// </summary>
module Verification =

    /// <summary>
    /// Verify content against a structured operation.
    /// </summary>
    let verify
        (content: string)
        (op: VerificationOp)
        (executeTool: string -> Map<string, obj> -> Async<Result<string, string>>)
        : Async<Result<bool, string>> =
        async {
            try
                match op with
                | Contains substring ->
                    let passed = content.Contains(substring, StringComparison.OrdinalIgnoreCase)
                    return Result.Ok passed

                | Regex pattern ->
                    let passed = Regex.IsMatch(content, pattern, RegexOptions.IgnoreCase)
                    return Result.Ok passed

                | JsonPath path ->
                    // Heuristic check for now. Will be enhanced with real JSON parsing later.
                    let passed = content.Contains(path) || content.Contains($"\"{path}\"")
                    return Result.Ok passed

                | Schema schema ->
                    try
                        use doc = JsonDocument.Parse(content)
                        let root = doc.RootElement

                        try
                            use schemaParsed = JsonDocument.Parse(schema)
                            let schemaRoot = schemaParsed.RootElement
                            let mutable errors = []

                            // Check "type" constraint
                            match schemaRoot.TryGetProperty("type") with
                            | true, typeProp ->
                                let expectedType = typeProp.GetString()
                                let actualKind = root.ValueKind
                                let typeMatch =
                                    match expectedType with
                                    | "object" -> actualKind = JsonValueKind.Object
                                    | "array" -> actualKind = JsonValueKind.Array
                                    | "string" -> actualKind = JsonValueKind.String
                                    | "number" | "integer" -> actualKind = JsonValueKind.Number
                                    | "boolean" -> actualKind = JsonValueKind.True || actualKind = JsonValueKind.False
                                    | _ -> true
                                if not typeMatch then
                                    errors <- $"Expected type '%s{expectedType}' but got '%A{actualKind}'" :: errors
                            | false, _ -> ()

                            // Check "required" fields
                            match schemaRoot.TryGetProperty("required") with
                            | true, reqProp when reqProp.ValueKind = JsonValueKind.Array ->
                                for reqField in reqProp.EnumerateArray() do
                                    let fieldName = reqField.GetString()
                                    match root.TryGetProperty(fieldName) with
                                    | true, _ -> ()
                                    | false, _ -> errors <- $"Missing required field '%s{fieldName}'" :: errors
                            | _ -> ()

                            // Check "properties" field types
                            match schemaRoot.TryGetProperty("properties") with
                            | true, propsDef when root.ValueKind = JsonValueKind.Object ->
                                for prop in propsDef.EnumerateObject() do
                                    match root.TryGetProperty(prop.Name) with
                                    | true, actualVal ->
                                        match prop.Value.TryGetProperty("type") with
                                        | true, fieldType ->
                                            let ft = fieldType.GetString()
                                            let ok =
                                                match ft with
                                                | "string" -> actualVal.ValueKind = JsonValueKind.String
                                                | "number" | "integer" -> actualVal.ValueKind = JsonValueKind.Number
                                                | "boolean" -> actualVal.ValueKind = JsonValueKind.True || actualVal.ValueKind = JsonValueKind.False
                                                | "array" -> actualVal.ValueKind = JsonValueKind.Array
                                                | "object" -> actualVal.ValueKind = JsonValueKind.Object
                                                | _ -> true
                                            if not ok then
                                                errors <- $"Field '%s{prop.Name}' expected type '%s{ft}' but got '%A{actualVal.ValueKind}'" :: errors
                                        | false, _ -> ()
                                    | false, _ -> () // Not required, skip
                            | _ -> ()

                            if errors.IsEmpty then
                                return Result.Ok true
                            else
                                return Result.Ok false
                        with _ ->
                            // If schema isn't valid JSON, fallback to checking if content is valid JSON
                            return Result.Ok true
                    with _ ->
                        return Result.Ok false

                | ToolCheck(toolName, args) ->
                    let! result = executeTool toolName args

                    match result with
                    | Result.Ok _ -> return Result.Ok true
                    | Result.Error err -> return Result.Ok false

                | CustomOp name ->
                    match name.ToLowerInvariant() with
                    | "non_empty" ->
                        let passed = not (String.IsNullOrWhiteSpace content)
                        return Result.Ok passed
                    | _ when name.StartsWith("threshold:", StringComparison.OrdinalIgnoreCase) ->
                        let parts = name.Split(':')
                        if parts.Length >= 3 then
                            let thresholdValStr = parts.[2]
                            match Double.TryParse(thresholdValStr), Double.TryParse(content) with
                            | (true, limit), (true, actual) ->
                                return Result.Ok (actual >= limit)
                            | _ -> return Result.Ok false
                        else
                            return Result.Ok false
                    | _ ->
                        return Result.Ok true
            with ex ->
                return Result.Error ex.Message
        }
