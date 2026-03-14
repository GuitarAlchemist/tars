namespace Tars.LinkedData

open System
open System.Net.Http
open VDS.RDF
open VDS.RDF.Query
open Tars.Core

/// SPARQL Client for querying remote endpoints (Wikidata, DBpedia, etc.)
module SparqlClient =

    /// Execute a SELECT query against a remote endpoint
    let query (endpointUri: Uri) (sparql: string) : Async<Microsoft.FSharp.Core.Result<List<Map<string, string>>, string>> =
        async {
            try
                use httpClient = new HttpClient()
                // Wikidata requires a User-Agent header
                httpClient.DefaultRequestHeaders.Add("User-Agent", "TARS/v2 (https://github.com/GuitarAlchemist/tars)")
                httpClient.Timeout <- TimeSpan.FromSeconds(30.0)

                let client = new SparqlQueryClient(httpClient, endpointUri)
                
                let! resultSet = 
                    async {
                        try
                            let! results = client.QueryWithResultSetAsync(sparql) |> Async.AwaitTask
                            return Microsoft.FSharp.Core.Result.Ok results
                        with ex ->
                            return Microsoft.FSharp.Core.Result.Error ex.Message
                    }

                match resultSet with
                | Microsoft.FSharp.Core.Result.Error err -> return Microsoft.FSharp.Core.Result.Error (sprintf "SPARQL Query Failed: %s" err)
                | Microsoft.FSharp.Core.Result.Ok results ->
                    if results.IsEmpty then
                        return Microsoft.FSharp.Core.Result.Ok []
                    else
                        // Convert SparqlResultSet to List<Map<string, string>>
                        let mappedResults =
                            results
                            |> Seq.map (fun result ->
                                results.Variables
                                |> Seq.filter (fun var -> result.HasValue(var))
                                |> Seq.map (fun var -> 
                                    let node = result.[var]
                                    let value = 
                                        match node.NodeType with
                                        | NodeType.Literal -> (node :?> ILiteralNode).Value
                                        | NodeType.Uri -> (node :?> IUriNode).Uri.ToString()
                                        | _ -> node.ToString()
                                    (var, value)
                                )
                                |> Map.ofSeq
                            )
                            |> Seq.toList
                        
                        return Microsoft.FSharp.Core.Result.Ok mappedResults

            with ex ->
                return Microsoft.FSharp.Core.Result.Error (sprintf "Unexpected SPARQL Client Error: %s" ex.Message)
        }

    /// Execute a simple count query to check endpoint availability
    let ping (endpointUri: Uri) : Async<bool> =
        async {
            // Simple ASK query
            let sparql = "ASK WHERE { ?s ?p ?o } "
            try
                use httpClient = new HttpClient()
                httpClient.DefaultRequestHeaders.Add("User-Agent", "TARS/v2 (https://github.com/GuitarAlchemist/tars)")
                httpClient.Timeout <- TimeSpan.FromSeconds(5.0)

                let client = new SparqlQueryClient(httpClient, endpointUri)
                
                let! result = client.QueryWithResultSetAsync(sparql) |> Async.AwaitTask
                return result.Result
            with _ ->
                return false
        }