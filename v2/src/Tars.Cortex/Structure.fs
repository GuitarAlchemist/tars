namespace Tars.Cortex

open System
open System.Text.Json
open System.Text.Json.Serialization
open Microsoft.FSharp.Reflection

/// <summary>
/// Utilities for generating structure constraints (JSON Schemas) from F# types.
/// </summary>
module Structure =

    type private JsonSchemaNode =
        { [<JsonPropertyName("type")>]
          Type: string option
          [<JsonPropertyName("properties")>]
          Properties: Map<string, JsonSchemaNode> option
          [<JsonPropertyName("required")>]
          Required: string list option
          [<JsonPropertyName("items")>]
          Items: JsonSchemaNode option
          [<JsonPropertyName("enum")>]
          Enum: string list option
          [<JsonPropertyName("description")>]
          Description: string option
          [<JsonPropertyName("anyOf")>]
          AnyOf: JsonSchemaNode list option }

    let private emptyNode =
        { Type = None
          Properties = None
          Required = None
          Items = None
          Enum = None
          Description = None
          AnyOf = None }

    let rec private generateNode (t: Type) : JsonSchemaNode =
        if t = typeof<string> || t = typeof<Guid> || t = typeof<DateTime> then
            { emptyNode with Type = Some "string" }
        elif t = typeof<int> || t = typeof<int64> then
            { emptyNode with Type = Some "integer" }
        elif t = typeof<float> || t = typeof<double> || t = typeof<decimal> then
            { emptyNode with Type = Some "number" }
        elif t = typeof<bool> then
            { emptyNode with Type = Some "boolean" }
        elif t.IsArray then
            { emptyNode with
                Type = Some "array"
                Items = Some(generateNode (t.GetElementType())) }
        elif
            t.IsGenericType
            && (t.GetGenericTypeDefinition() = typeof<list<_>>
                || t.GetGenericTypeDefinition() = typeof<seq<_>>)
        then
            { emptyNode with
                Type = Some "array"
                Items = Some(generateNode (t.GetGenericArguments().[0])) }
        elif t.IsGenericType && t.GetGenericTypeDefinition() = typeof<Option<_>> then
            let inner = generateNode (t.GetGenericArguments().[0])

            { emptyNode with
                AnyOf = Some [ inner; { emptyNode with Type = Some "null" } ] }
        elif FSharpType.IsRecord(t) then
            let props = FSharpType.GetRecordFields(t)

            let propMap =
                props
                |> Array.map (fun p ->
                    let name = p.Name
                    // Basic camelCase conversion usually preferred for JSON
                    let jsonName = Char.ToLowerInvariant(name.[0]).ToString() + name.Substring(1)
                    jsonName, generateNode p.PropertyType)
                |> Map.ofArray

            let required =
                props
                |> Array.filter (fun p ->
                    let pt = p.PropertyType
                    not (pt.IsGenericType && pt.GetGenericTypeDefinition() = typeof<Option<_>>))
                |> Array.map (fun p ->
                    let name = p.Name
                    Char.ToLowerInvariant(name.[0]).ToString() + name.Substring(1))
                |> Array.toList

            { emptyNode with
                Type = Some "object"
                Properties = Some propMap
                Required = Some required
            // OpenAI strict mode requires additionalProperties: false, but we'll stick to basic schema first
            }
        elif t.IsEnum then
            let names = Enum.GetNames(t) |> Array.toList

            { emptyNode with
                Type = Some "string"
                Enum = Some names }
        else
            // Fallback to string for unknown types to avoid crashing
            { emptyNode with
                Type = Some "string"
                Description = Some $"Unknown type: {t.Name}" }

    /// <summary>
    /// Generates a JSON Schema string for the given type 'T.
    /// </summary>
    let generateJsonSchema<'T> () : string =
        let node = generateNode typeof<'T>

        let options =
            JsonSerializerOptions(WriteIndented = false, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)

        options.Converters.Add(JsonFSharpConverter())
        JsonSerializer.Serialize(node, options)
