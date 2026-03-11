namespace Tars.Cortex

open System
open System.Net.Http
open System.Text
open System.Text.Json
open Tars.Core

type ChromaCollection = { id: string; name: string }

type ChromaAddRequest =
    { ids: string[]
      embeddings: float32[][]
      metadatas: Map<string, string>[]
      documents: string[] }

type ChromaQueryRequest =
    { query_embeddings: float32[][]
      n_results: int
      include_fields: string[] }

// Simple helper for JSON
module Json =
    let serialize obj = JsonSerializer.Serialize(obj)
    let deserialize<'T> (json: string) = JsonSerializer.Deserialize<'T>(json)

type ChromaVectorStore(baseUrl: string) =
    let httpClient = new HttpClient(BaseAddress = Uri(baseUrl))

    // JSON Options for snake_case and case-insensitive
    let jsonOptions =
        let opts = JsonSerializerOptions()
        opts.PropertyNameCaseInsensitive <- true
        opts

    let getOrCreateCollection (name: string) =
        task {
            // Check if collection exists first to avoiding creating duplicates if API behavior changes
            // For now, get_or_create=true is robust enough
            let req = {| name = name; get_or_create = true |}

            let content =
                new StringContent(JsonSerializer.Serialize(req, jsonOptions), Encoding.UTF8, "application/json")

            let! response =
                httpClient.PostAsync("/api/v2/tenants/default_tenant/databases/default_database/collections", content)

            if not response.IsSuccessStatusCode then
                let! err = response.Content.ReadAsStringAsync()
                failwith $"ChromaDB Error ({response.StatusCode}): {err}"

            let! body = response.Content.ReadAsStringAsync()
            let coll = JsonSerializer.Deserialize<ChromaCollection>(body, jsonOptions)
            return coll.id
        }

    interface IVectorStore with
        member this.SaveAsync(collectionName: string, id: string, vector: float32[], payload: Map<string, string>) =
            task {
                try
                    let! collId = getOrCreateCollection collectionName

                    let req =
                        {| ids = [| id |]
                           embeddings = [| vector |]
                           metadatas = [| payload |]
                           documents = [| payload |> Map.tryFind "content" |> Option.defaultValue "" |] |}

                    let content =
                        new StringContent(JsonSerializer.Serialize(req, jsonOptions), Encoding.UTF8, "application/json")

                    let! response =
                        httpClient.PostAsync(
                            $"/api/v2/tenants/default_tenant/databases/default_database/collections/{collId}/upsert",
                            content
                        )

                    if not response.IsSuccessStatusCode then
                        let! err = response.Content.ReadAsStringAsync()
                        failwith $"ChromaDB Save Error ({response.StatusCode}): {err}"
                with ex ->
                    // Log but don't crash entire flow? Or rethrow?
                    // For persistence, we should probably throw so caller knows it failed.
                    raise ex
            }

        member this.SearchAsync(collectionName: string, vector: float32[], limit: int) =
            task {
                try
                    let! collId = getOrCreateCollection collectionName

                    let req =
                        {| query_embeddings = [| vector |]
                           n_results = limit
                        // include = [ "metadatas", "distances", "documents" ]
                        |}

                    let content =
                        new StringContent(JsonSerializer.Serialize(req, jsonOptions), Encoding.UTF8, "application/json")

                    let! response =
                        httpClient.PostAsync(
                            $"/api/v2/tenants/default_tenant/databases/default_database/collections/{collId}/query",
                            content
                        )

                    if not response.IsSuccessStatusCode then
                        let! err = response.Content.ReadAsStringAsync()
                        failwith $"ChromaDB Search Error ({response.StatusCode}): {err}"

                    let! body = response.Content.ReadAsStringAsync()
                    use doc = JsonDocument.Parse(body)
                    let root = doc.RootElement

                    // Chroma returns [[]] for single query
                    let ids = root.GetProperty("ids")

                    if ids.GetArrayLength() = 0 then
                        return []
                    else
                        let firstBatchIds = ids.[0]
                        // If no results
                        if firstBatchIds.GetArrayLength() = 0 then
                            return []
                        else
                            let distances = root.GetProperty("distances").[0]
                            let metadatas = root.GetProperty("metadatas").[0]

                            let results =
                                [ 0 .. firstBatchIds.GetArrayLength() - 1 ]
                                |> List.map (fun i ->
                                    let id = firstBatchIds[i].GetString()
                                    let dist = distances[i].GetSingle()
                                    let metaElem = metadatas[i]

                                    let meta =
                                        if metaElem.ValueKind = JsonValueKind.Object then
                                            metaElem.EnumerateObject()
                                            |> Seq.map (fun p ->
                                                // Handle ValueKind safely
                                                let valStr =
                                                    match p.Value.ValueKind with
                                                    | JsonValueKind.String -> p.Value.GetString()
                                                    | JsonValueKind.Number -> p.Value.ToString()
                                                    | JsonValueKind.True -> "true"
                                                    | JsonValueKind.False -> "false"
                                                    | _ -> p.Value.ToString()

                                                p.Name, valStr)
                                            |> Map.ofSeq
                                        else
                                            Map.empty

                                    (id, dist, meta))

                            return results
                with ex ->
                    // Return empty on error to avoid breaking chat flow, but log implies we should probably expose error
                    // For now, print to console/debug
                    System.Console.WriteLine($"[Chroma Error] {ex.Message}")
                    return []
            }
