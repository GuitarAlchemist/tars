namespace Tars.Connectors

open System
open System.Threading.Tasks
open Tars.Core
open Tars.LinkedData

/// Knowledge Graph implementation backed by Apache Jena Fuseki
type FusekiGraphService(endpointUri: Uri, ?authOpt: string) =

    let updateUri = Uri(endpointUri.ToString() + "/update")
    let queryUri = Uri(endpointUri.ToString() + "/query")

    interface IGraphService with
        member this.AddNodeAsync(entity: TarsEntity) =
            task {
                let triples = RdfMapper.entityToTriples entity

                if triples.IsEmpty then
                    return TarsEntity.getId entity
                else
                    let allTriples = String.concat "\n" triples
                    let! result = SparqlUpdateClient.insertData updateUri authOpt allTriples |> Async.StartAsTask

                    match result with
                    | Result.Ok() -> return TarsEntity.getId entity
                    | Result.Error err ->
                        printfn "[Fuseki] AddNode failed: %s" err
                        return TarsEntity.getId entity
            }

        member this.AddFactAsync(fact: TarsFact) =
            task {
                // Ensure nodes exist
                let! _ = (this :> IGraphService).AddNodeAsync(TarsFact.source fact)

                match TarsFact.target fact with
                | Some t ->
                    let! _ = (this :> IGraphService).AddNodeAsync t
                    ()
                | None -> ()

                let triples = RdfMapper.factToTriples fact

                if triples.IsEmpty then
                    return Guid.NewGuid()
                else
                    let allTriples = String.concat "\n" triples
                    let! result = SparqlUpdateClient.insertData updateUri authOpt allTriples |> Async.StartAsTask

                    match result with
                    | Result.Ok() -> return Guid.NewGuid()
                    | Result.Error err ->
                        printfn "[Fuseki] AddFact failed: %s" err
                        return Guid.NewGuid()
            }

        member this.AddEpisodeAsync(episode: Episode) =
            task {
                let entity = EpisodeE episode
                let! id = (this :> IGraphService).AddNodeAsync entity
                return id
            }

        member this.QueryAsync(query: string) =
            task {
                let sparql =
                    if query = "*" then
                        "SELECT ?s ?p ?o WHERE { ?s ?p ?o . FILTER(strstarts(str(?p), 'http://tars.ai/ns#')) }"
                    else
                        // Basic entity search (case-insensitive)
                        sprintf
                            "SELECT ?s ?p ?o WHERE { ?s ?p ?o . FILTER(CONTAINS(LCASE(str(?o)), LCASE('%s'))) }"
                            (query.Replace("'", "\\'"))

                let! queryResult = SparqlQueryRunner.query queryUri authOpt sparql |> Async.StartAsTask

                match queryResult with
                | Microsoft.FSharp.Core.Error _ -> return []
                | Microsoft.FSharp.Core.Ok results ->
                    // Collect unique entity URIs from results to hydrate them
                    let uris =
                        results.Results
                        |> Seq.collect (fun row ->
                            let mutable s = ""
                            let mutable o = ""

                            if row.HasValue("s") then
                                s <- row.["s"].ToString()

                            if row.HasValue("o") then
                                o <- row.["o"].ToString()

                            [ s; o ])
                        |> Seq.filter (fun uri -> uri.StartsWith("http://tars.ai/resource/"))
                        |> Seq.distinct
                        |> Seq.toList

                    // Hydrate all unique entities in parallel
                    let! hydratedEntities =
                        uris
                        |> List.map (fun uri ->
                            async {
                                let! entity = RdfReconstructor.reconstructEntityAsync queryUri authOpt uri
                                return uri, entity
                            })
                        |> Async.Parallel
                        |> Async.StartAsTask

                    let entityMap =
                        hydratedEntities
                        |> Array.choose (fun (uri, opt) -> opt |> Option.map (fun e -> uri, e))
                        |> Map.ofArray

                    // Map triples back to TarsFacts
                    return RdfReconstructor.toFacts entityMap results
            }

        member this.PersistAsync() = task { return () } // Fuseki is already persistent
