namespace TarsEngine.FSharp.Core.Mcp

open System
open System.IO
open System.Text.Json
open System.Text.Json.Nodes
open YamlDotNet.Serialization
open YamlDotNet.Serialization.NamingConventions

type McpRequest =
    { Body: string
      Headers: Map<string, string> }

module McpRequestParser =

    let private jsonOptions = JsonSerializerOptions(WriteIndented = true)

    let private yamlDeserializer =
        DeserializerBuilder()
            .WithNamingConvention(CamelCaseNamingConvention.Instance)
            .IgnoreUnmatchedProperties()
            .Build()

    let private yamlSerializer =
        SerializerBuilder()
            .JsonCompatible()
            .Build()

    let private toHeaderMap (node: JsonNode) =
        match node with
        | :? JsonObject as obj ->
            obj
            |> Seq.choose (fun kv ->
                if isNull kv.Value then None
                else
                    try
                        match kv.Value with
                        | :? JsonValue as jsonValue ->
                            let mutable value = ""
                            if jsonValue.TryGetValue<string>(&value) then
                                Some (kv.Key, value)
                            else
                                Some (kv.Key, jsonValue.ToJsonString())
                        | _ ->
                            Some (kv.Key, kv.Value.ToJsonString())
                    with _ ->
                        None)
            |> Map.ofSeq
        | _ -> Map.empty

    let private deepClone (node: JsonNode) =
        if isNull node then null else node.DeepClone()

    let private extractFromJsonNode (node: JsonNode) =
        match node with
        | :? JsonObject as obj ->
            let headers =
                let mutable headerNode: JsonNode = null
                if obj.TryGetPropertyValue("headers", &headerNode) && not (isNull headerNode) then
                    toHeaderMap headerNode
                else
                    Map.empty

            let body =
                let mutable bodyNode: JsonNode = null
                if obj.TryGetPropertyValue("body", &bodyNode) && not (isNull bodyNode) then
                    deepClone bodyNode
                else
                    let bodyObj = JsonObject()
                    for kv in obj do
                        if not (kv.Key.Equals("headers", StringComparison.OrdinalIgnoreCase)) then
                            bodyObj[kv.Key] <- deepClone kv.Value
                    bodyObj :> JsonNode

            { Body = body.ToJsonString(jsonOptions)
              Headers = headers }
        | _ ->
            { Body = node.ToJsonString(jsonOptions)
              Headers = Map.empty }

    let private tryParseJson (content: string) =
        try
            let node = JsonNode.Parse(content)
            if isNull node then
                Error "Unable to parse MCP block content."
            else
                Ok (extractFromJsonNode node)
        with _ ->
            Error "json"

    let private tryParseYaml (content: string) =
        try
            use reader = new StringReader(content)
            let yamlObject = yamlDeserializer.Deserialize(reader)
            let json = yamlSerializer.Serialize(yamlObject)
            let node = JsonNode.Parse(json)
            if isNull node then
                Error "Unable to parse MCP block content."
            else
                Ok (extractFromJsonNode node)
        with ex ->
            Error ex.Message

    let parse (content: string) =
        if String.IsNullOrWhiteSpace content then
            Error "MCP block cannot be empty."
        else
            match tryParseJson content with
            | Ok request -> Ok request
            | Error "json" ->
                match tryParseYaml content with
                | Ok request -> Ok request
                | Error err -> Error $"Failed to parse MCP block as JSON or YAML: {err}"
            | Error err -> Error err
