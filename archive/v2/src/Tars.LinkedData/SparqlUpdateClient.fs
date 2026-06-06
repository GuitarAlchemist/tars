namespace Tars.LinkedData

open System
open System.Net.Http
open VDS.RDF
open VDS.RDF.Update
open Tars.Core

/// SPARQL Update Client for sending updates (INSERT DATA, DELETE DATA) to remote endpoints
module SparqlUpdateClient =

    /// Execute a SPARQL Update against a remote endpoint
    let update (endpointUri: Uri) (authOpt: string option) (sparqlUpdate: string) : Async<Result<unit, string>> =
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

                let client = new SparqlUpdateClient(httpClient, endpointUri)

                try
                    do! client.UpdateAsync(sparqlUpdate) |> Async.AwaitTask
                    return Result.Ok()
                with ex ->
                    return Result.Error(sprintf "SPARQL Update Failed: %s" ex.Message)

            with ex ->
                return Result.Error(sprintf "Unexpected SPARQL Update Client Error: %s" ex.Message)
        }

    /// Helper to format N-Triples for INSERT DATA
    let insertData (endpointUri: Uri) (authOpt: string option) (triples: string) : Async<Result<unit, string>> =
        let updateQuery = sprintf "INSERT DATA { %s }" triples
        update endpointUri authOpt updateQuery
