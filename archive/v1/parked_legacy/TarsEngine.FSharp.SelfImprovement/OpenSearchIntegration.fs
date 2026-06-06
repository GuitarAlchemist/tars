namespace TarsEngine.FSharp.SelfImprovement

open System
open System.Net.Http
open System.Net.Http.Headers
open System.Text
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// Integration helpers for sending knowledge items to OpenSearch.
module OpenSearchIntegration =

    type OpenSearchAuth =
        | NoAuth
        | Basic of username: string * password: string
        | ApiKey of string

    type OpenSearchConfig = {
        BaseUri: Uri
        IndexName: string
        Auth: OpenSearchAuth
        Enabled: bool
    }

    type IndexOutcome = {
        Indexed: int
        Failed: (KnowledgeItem * string) list
    }

    type ClusterHealth = {
        Status: string
        NodeCount: int option
        ActiveShards: int option
    }

    let private readEnv name =
        match Environment.GetEnvironmentVariable(name) with
        | null
        | "" -> None
        | value -> Some value

    let tryLoadConfig () =
        match readEnv "OPENSEARCH_URL" with
        | None -> None
        | Some url ->
            let indexName =
                readEnv "OPENSEARCH_INDEX"
                |> Option.filter (String.IsNullOrWhiteSpace >> not)
                |> Option.defaultValue "tars-knowledge"

            let auth =
                match readEnv "OPENSEARCH_API_KEY" with
                | Some key -> ApiKey key
                | None ->
                    match readEnv "OPENSEARCH_USERNAME", readEnv "OPENSEARCH_PASSWORD" with
                    | Some user, Some pass when not (String.IsNullOrWhiteSpace(user)) && not (String.IsNullOrWhiteSpace(pass)) ->
                        Basic(user, pass)
                    | _ -> NoAuth

            let enabled =
                readEnv "OPENSEARCH_ENABLED"
                |> Option.map (fun v -> v.Equals("true", StringComparison.OrdinalIgnoreCase) || v = "1")
                |> Option.defaultValue true

            Some {
                BaseUri = Uri(url, UriKind.Absolute)
                IndexName = indexName
                Auth = auth
                Enabled = enabled
            }

    let private createClient (config: OpenSearchConfig) =
        let client = new HttpClient()
        client.BaseAddress <- config.BaseUri
        client.DefaultRequestHeaders.Accept.Add(MediaTypeWithQualityHeaderValue("application/json"))

        match config.Auth with
        | NoAuth -> ()
        | Basic(user, pass) ->
            let token = Convert.ToBase64String(Encoding.UTF8.GetBytes($"{user}:{pass}"))
            client.DefaultRequestHeaders.Authorization <- AuthenticationHeaderValue("Basic", token)
        | ApiKey key ->
            client.DefaultRequestHeaders.Add("Authorization", $"ApiKey {key}")

        client

    let private openSearchJsonOptions =
        JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)

    let private indexDefinition =
        """
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 1
  },
  "mappings": {
    "properties": {
      "id": {"type": "keyword"},
      "itemType": {"type": "keyword"},
      "content": {"type": "text"},
      "source": {"type": "keyword"},
      "sourceType": {"type": "keyword"},
      "confidence": {"type": "float"},
      "tags": {"type": "keyword"},
      "relatedItems": {"type": "keyword"},
      "extractedAt": {"type": "date"}
    }
  }
}
"""

    let getClusterHealthAsync (logger: ILogger) (config: OpenSearchConfig) =
        task {
            try
                use client = createClient config
                use! response = client.GetAsync("/_cluster/health")
                if response.IsSuccessStatusCode then
                    let! payload = response.Content.ReadAsStringAsync()
                    try
                        use json = JsonDocument.Parse(payload)
                        let root = json.RootElement
                        let status =
                            let mutable statusProp = Unchecked.defaultof<JsonElement>
                            if root.TryGetProperty("status", &statusProp) && statusProp.ValueKind = JsonValueKind.String then
                                statusProp.GetString() |> Option.ofObj |> Option.defaultValue "unknown"
                            else
                                "unknown"

                        let tryGetInt (property: string) =
                            let mutable value = Unchecked.defaultof<JsonElement>
                            if root.TryGetProperty(property, &value) && value.ValueKind = JsonValueKind.Number then
                                match value.TryGetInt32() with
                                | true, v -> Some v
                                | _ -> None
                            else None

                        return Ok {
                            Status = status
                            NodeCount = tryGetInt "number_of_nodes"
                            ActiveShards = tryGetInt "active_shards"
                        }
                    with ex ->
                        logger.LogWarning(ex, "Failed to parse OpenSearch health response.")
                        return Ok {
                            Status = "unknown"
                            NodeCount = None
                            ActiveShards = None
                        }
                else
                    let! error = response.Content.ReadAsStringAsync()
                    return Error $"StatusCode={response.StatusCode}; Error={error}"
            with ex ->
                return Error ex.Message
        }

    let private ensureIndexExistsAsync (client: HttpClient) (logger: ILogger) (config: OpenSearchConfig) =
        task {
            try
                use! headResponse = client.SendAsync(new HttpRequestMessage(HttpMethod.Head, $"/{config.IndexName}"))
                if headResponse.IsSuccessStatusCode then
                    return true
                else
                    use content = new StringContent(indexDefinition, Encoding.UTF8, "application/json")
                    use! putResponse = client.PutAsync($"/{config.IndexName}", content)
                    if putResponse.IsSuccessStatusCode then
                        logger.LogInformation("Created OpenSearch index {IndexName}", config.IndexName)
                        return true
                    else
                        let! error = putResponse.Content.ReadAsStringAsync()
                        logger.LogError("Failed to create OpenSearch index {IndexName}: {Status} - {Error}", config.IndexName, putResponse.StatusCode, error)
                        return false
            with ex ->
                logger.LogError(ex, "Failed to ensure OpenSearch index {IndexName}", config.IndexName)
                return false
        }

    let private knowledgeItemToDocument (item: KnowledgeItem) =
        let sourceType =
            match item.SourceType with
            | KnowledgeSourceType.Chat -> "chat"
            | KnowledgeSourceType.Reflection -> "reflection"
            | KnowledgeSourceType.Documentation -> "documentation"
            | KnowledgeSourceType.Feature -> "feature"
            | KnowledgeSourceType.Architecture -> "architecture"
            | KnowledgeSourceType.Tutorial -> "tutorial"

        dict [
            "id", box item.Id
            "itemType", box item.Type
            "content", box item.Content
            "source", box item.Source
            "sourceType", box sourceType
            "confidence", box item.Confidence
            "tags", box item.Tags
            "relatedItems", box item.RelatedItems
            "extractedAt", box item.ExtractedAt
        ]

    let indexKnowledgeItemsAsync (logger: ILogger) (config: OpenSearchConfig) (items: KnowledgeItem list) =
        task {
            if not config.Enabled then
                logger.LogInformation("OpenSearch integration disabled via configuration; skipping indexing.")
                return { Indexed = 0; Failed = [] }
            elif List.isEmpty items then
                return { Indexed = 0; Failed = [] }
            else
                use client = createClient config

                let! ensured = ensureIndexExistsAsync client logger config
                if not ensured then
                    return { Indexed = 0; Failed = items |> List.map (fun item -> item, "Failed to ensure index exists") }
                else
                    let mutable indexed = 0
                    let failures = ResizeArray<KnowledgeItem * string>()

                    for item in items do
                        try
                            let document = knowledgeItemToDocument item
                            let json = JsonSerializer.Serialize(document, openSearchJsonOptions)
                            use content = new StringContent(json, Encoding.UTF8, "application/json")
                            let encodedId = Uri.EscapeDataString(item.Id)
                            use! response = client.PutAsync($"/{config.IndexName}/_doc/{encodedId}", content)
                            if response.IsSuccessStatusCode then
                                indexed <- indexed + 1
                            else
                                let! error = response.Content.ReadAsStringAsync()
                                logger.LogWarning("Failed to index knowledge item {Id}: {Status} - {Error}", item.Id, response.StatusCode, error)
                                failures.Add(item, error)
                        with ex ->
                            logger.LogError(ex, "Unexpected failure when indexing knowledge item {Id}", item.Id)
                            failures.Add(item, ex.Message)

                    return {
                        Indexed = indexed
                        Failed = failures |> Seq.toList
                    }
        }
