namespace Tars.LinkedData

open System
open System.Net.Http
open VDS.RDF
open VDS.RDF.Query
open Tars.Core

/// SPARQL Query Runner for fetching data from remote endpoints
module SparqlQueryRunner =

    /// Execute a SPARQL Select Query against a remote endpoint
    let query
        (endpointUri: Uri)
        (authOpt: string option)
        (sparqlQuery: string)
        : Async<Result<SparqlResultSet, string>> =
        async {
            try
                use httpClient = new HttpClient()
                httpClient.DefaultRequestHeaders.Add("User-Agent", "TARS/v2 (https://github.com/GuitarAlchemist/tars)")
                httpClient.Timeout <- TimeSpan.FromSeconds(30.0)

                // Auth: Prioritize argument, then env var
                let effectiveAuth =
                    match authOpt with
                    | Some a when not (String.IsNullOrWhiteSpace a) -> Some a
                    | _ ->
                        match Environment.GetEnvironmentVariable("TARS_FUSEKI_AUTH") with
                        | null
                        | "" -> None
                        | a -> Some a

                match effectiveAuth with
                | Some auth ->
                    let bytes = System.Text.Encoding.UTF8.GetBytes(auth)
                    let base64 = Convert.ToBase64String(bytes)

                    httpClient.DefaultRequestHeaders.Authorization <-
                        System.Net.Http.Headers.AuthenticationHeaderValue("Basic", base64)
                | None -> ()

                let client = new SparqlQueryClient(httpClient, endpointUri)

                try
                    let! results = client.QueryWithResultSetAsync(sparqlQuery) |> Async.AwaitTask
                    return Result.Ok results
                with ex ->
                    return Result.Error(sprintf "SPARQL Query Failed: %s" ex.Message)

            with ex ->
                return Result.Error(sprintf "Unexpected SPARQL Query Client Error: %s" ex.Message)
        }
