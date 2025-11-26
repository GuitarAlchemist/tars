namespace Tars.Cortex

open System
open System.Net.Http
open System.Text
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks
open Tars.Core

type ChromaCollection = { id: string; name: string }
type ChromaAddRequest = { ids: string[]; embeddings: float32[][]; metadatas: Map<string, string>[]; documents: string[] }
type ChromaQueryRequest = { query_embeddings: float32[][]; n_results: int; include_fields: string[] }

// Simple helper for JSON
module Json =
    let serialize obj = JsonSerializer.Serialize(obj)
    let deserialize<'T> (json: string) = JsonSerializer.Deserialize<'T>(json)

type ChromaVectorStore(baseUrl: string) =
    let httpClient = new HttpClient(BaseAddress = Uri(baseUrl))

    let getOrCreateCollection (name: string) =
        task {
            let req = {| name = name; get_or_create = true |}
            let content = new StringContent(Json.serialize req, Encoding.UTF8, "application/json")
            let! response = httpClient.PostAsync("/api/v1/collections", content)
            response.EnsureSuccessStatusCode() |> ignore
            let! body = response.Content.ReadAsStringAsync()
            let coll = Json.deserialize<ChromaCollection>(body)
            return coll.id
        }

    interface IVectorStore with
        member this.SaveAsync(collectionName: string, id: string, vector: float32[], payload: Map<string, string>) =
            task {
                let! collId = getOrCreateCollection collectionName
                
                let req = {|
                    ids = [| id |]
                    embeddings = [| vector |]
                    metadatas = [| payload |]
                    // Chroma requires documents usually, but we can pass empty if we only store vectors? 
                    // Or maybe we should store something. 
                    // For now, empty string or serialized payload?
                    // Let's assume document is not strictly required if we only search by vector, 
                    // but Chroma might complain. Let's pass generic text or payload as text.
                    documents = [| "" |] 
                |}
                
                let content = new StringContent(Json.serialize req, Encoding.UTF8, "application/json")
                // Use collection ID in path
                let! response = httpClient.PostAsync($"/api/v1/collections/{collId}/add", content)
                response.EnsureSuccessStatusCode() |> ignore
            }

        member this.SearchAsync(collectionName: string, vector: float32[], limit: int) =
            task {
                let! collId = getOrCreateCollection collectionName
                
                let req = {|
                    query_embeddings = [| vector |]
                    n_results = limit
                    // include = [ "metadatas", "distances", "documents" ]
                |}
                
                let content = new StringContent(Json.serialize req, Encoding.UTF8, "application/json")
                let! response = httpClient.PostAsync($"/api/v1/collections/{collId}/query", content)
                response.EnsureSuccessStatusCode() |> ignore
                
                let! body = response.Content.ReadAsStringAsync()
                // Parsing Chroma response is complex because it returns lists of lists
                // Response format: { ids: [[id1, id2]], distances: [[d1, d2]], metadatas: [[m1, m2]] }
                
                // For now, return empty or parse basically.
                // I will do a rough parse using JsonDocument to handle dynamic structure safely
                use doc = JsonDocument.Parse(body)
                let root = doc.RootElement
                // Assuming single query vector, so index 0 of lists
                let ids = root.GetProperty("ids").[0]
                let distances = root.GetProperty("distances").[0]
                let metadatas = root.GetProperty("metadatas").[0]
                
                let results = 
                    [0 .. ids.GetArrayLength() - 1]
                    |> List.map (fun i ->
                        let id = ids.[i].GetString()
                        let dist = distances.[i].GetSingle() // Chroma returns distance, not similarity usually?
                        let metaElem = metadatas.[i]
                        let meta = 
                            if metaElem.ValueKind = JsonValueKind.Object then
                                metaElem.EnumerateObject()
                                |> Seq.map (fun p -> p.Name, p.Value.ToString()) // Simplification
                                |> Map.ofSeq
                            else
                                Map.empty
                        (id, dist, meta)
                    )
                
                return results
            }
