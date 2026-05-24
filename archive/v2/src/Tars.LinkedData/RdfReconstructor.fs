namespace Tars.LinkedData

open System
open VDS.RDF
open VDS.RDF.Query
open Tars.Core

/// Reconstructs TARS entities from RDF triples/results
module RdfReconstructor =

    let NS_TARS = "http://tars.ai/ns#"
    let NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    let NS_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
    let NS_XSD = "http://www.w3.org/2001/XMLSchema#"

    /// Extract a Guid from a URI or string
    let private extractGuid (s: string) =
        // Handle format: http://tars.ai/resource/run/7353a793-7359-445b-b478-207ee1ccfc56
        let parts = s.Split('/')
        let last = parts.[parts.Length - 1]

        match Guid.TryParse(last) with
        | (true, g) -> Some g
        | _ -> None

    /// Map a SPARQL Result row to a StepEntity if possible
    let toStepEntity (results: SparqlResultSet) : StepEntity list =
        results.Results
        |> Seq.choose (fun (row: ISparqlResult) ->
            try
                // Expecting variables: ?s ?stepId ?runId ?nodeType ?timestamp ?content
                let stepId = row.["stepId"].ToString()
                let nodeType = row.["nodeType"].ToString()
                let timestampStr = row.["timestamp"].ToString()
                // Remove XSD type suffix if present in string representation
                let cleanTimestamp = timestampStr.Split('^').[0].Trim('"')
                let timestamp = DateTime.Parse(cleanTimestamp)

                let runUri = row.["runId"].ToString()
                let runId = extractGuid runUri |> Option.defaultValue Guid.Empty

                let content =
                    if row.HasValue("content") then
                        row.["content"].ToString()
                    else
                        ""

                Some(
                    { RunId = runId
                      StepId = stepId
                      NodeType = nodeType
                      Content = content
                      Timestamp = timestamp }
                    : StepEntity
                )
            with _ ->
                None)
        |> Seq.toList

    /// Map a SPARQL Result row to a RunEntity if possible
    let toRunEntity (results: SparqlResultSet) : RunEntity list =
        results.Results
        |> Seq.choose (fun (row: ISparqlResult) ->
            try
                // Expecting variables: ?run ?goal ?pattern ?timestamp
                let runUri = row.["run"].ToString()
                let id = extractGuid runUri |> Option.defaultValue Guid.Empty
                let goal = row.["goal"].ToString()
                let pattern = row.["pattern"].ToString()
                let timestampStr = row.["timestamp"].ToString()
                let cleanTimestamp = timestampStr.Split('^').[0].Trim('"')
                let timestamp = DateTime.Parse(cleanTimestamp)

                Some(
                    { Id = id
                      Goal = goal
                      Pattern = pattern
                      Timestamp = timestamp }
                    : RunEntity
                )
            with _ ->
                None)
        |> Seq.toList

    /// Reconstruct an entity from its URI by fetching all its properties
    let reconstructEntityAsync (queryUri: Uri) (authOpt: string option) (entityUri: string) =
        async {
            // SELECT ?p ?o WHERE { <uri> ?p ?o }
            let sparql = sprintf "SELECT ?p ?o WHERE { <%s> ?p ?o }" (entityUri.Trim('<', '>'))
            let! queryResult = SparqlQueryRunner.query queryUri authOpt sparql

            match queryResult with
            | Microsoft.FSharp.Core.Error _ -> return None
            | Microsoft.FSharp.Core.Ok results ->
                // Map results to properties
                let props =
                    results.Results
                    |> Seq.map (fun (row: ISparqlResult) ->
                        let p = if row.HasValue("p") then row.["p"].ToString() else ""
                        let o = if row.HasValue("o") then row.["o"].ToString() else ""
                        p, o)
                    |> Map.ofSeq

                // Determine type
                let typeUri =
                    props
                    |> Map.tryFind (NS_TARS + "type")
                    |> Option.defaultValue (props |> Map.tryFind (NS_RDF + "type") |> Option.defaultValue "")

                if typeUri.EndsWith("Run") then
                    let id = extractGuid entityUri |> Option.defaultValue Guid.Empty
                    let goal = props |> Map.tryFind (NS_TARS + "goal") |> Option.defaultValue ""
                    let pattern = props |> Map.tryFind (NS_TARS + "pattern") |> Option.defaultValue ""

                    let timestampStr =
                        props
                        |> Map.tryFind (NS_TARS + "timestamp")
                        |> Option.defaultValue (DateTime.UtcNow.ToString("o"))

                    let cleanTimestamp = timestampStr.Split('^').[0].Trim('"')

                    let timestamp =
                        match DateTime.TryParse(cleanTimestamp) with
                        | (true, t) -> t
                        | _ -> DateTime.UtcNow

                    return
                        Some(
                            TarsEntity.RunE
                                { Id = id
                                  Goal = goal
                                  Pattern = pattern
                                  Timestamp = timestamp }
                        )

                elif typeUri.EndsWith("Step") then
                    let runUri = props |> Map.tryFind (NS_TARS + "runId") |> Option.defaultValue ""
                    let runId = extractGuid runUri |> Option.defaultValue Guid.Empty
                    let stepId = props |> Map.tryFind (NS_TARS + "stepId") |> Option.defaultValue ""
                    let nodeType = props |> Map.tryFind (NS_TARS + "nodeType") |> Option.defaultValue ""
                    let content = props |> Map.tryFind (NS_TARS + "content") |> Option.defaultValue ""

                    let timestampStr =
                        props
                        |> Map.tryFind (NS_TARS + "timestamp")
                        |> Option.defaultValue (DateTime.UtcNow.ToString("o"))

                    let cleanTimestamp = timestampStr.Split('^').[0].Trim('"')

                    let timestamp =
                        match DateTime.TryParse(cleanTimestamp) with
                        | (true, t) -> t
                        | _ -> DateTime.UtcNow

                    return
                        Some(
                            TarsEntity.StepE
                                { RunId = runId
                                  StepId = stepId
                                  NodeType = nodeType
                                  Content = content
                                  Timestamp = timestamp }
                        )

                elif typeUri.EndsWith("Concept") then
                    let name = props |> Map.tryFind (NS_RDFS + "label") |> Option.defaultValue ""
                    let desc = props |> Map.tryFind (NS_RDFS + "comment") |> Option.defaultValue ""

                    return
                        Some(
                            TarsEntity.ConceptE
                                { Name = name
                                  Description = desc
                                  RelatedConcepts = [] }
                        )

                else
                    return None
        }

    /// Map a set of triple results to TarsFacts using a provided entity map
    let toFacts (entities: Map<string, TarsEntity>) (results: SparqlResultSet) : TarsFact list =
        results.Results
        |> Seq.choose (fun (row: ISparqlResult) ->
            try
                let sUri = if row.HasValue("s") then row.["s"].ToString() else ""
                let pUri = if row.HasValue("p") then row.["p"].ToString() else ""
                let oUri = if row.HasValue("o") then row.["o"].ToString() else ""

                if sUri = "" || pUri = "" || oUri = "" then
                    None
                else
                    let source = entities |> Map.tryFind sUri

                    // For target, it could be a URI (entity) or a literal
                    match source with
                    | Some s ->
                        match pUri with
                        | p when p.EndsWith("nextStep") ->
                            entities |> Map.tryFind oUri |> Option.map (fun t -> TarsFact.NextStep(s, t))
                        | p when p.EndsWith("contains") ->
                            entities |> Map.tryFind oUri |> Option.map (fun t -> TarsFact.Contains(s, t))
                        | p when p.EndsWith("implements") ->
                            entities
                            |> Map.tryFind oUri
                            |> Option.map (fun t -> TarsFact.Implements(s, t, 1.0))
                        | p when p.EndsWith("dependsOn") ->
                            entities
                            |> Map.tryFind oUri
                            |> Option.map (fun t -> TarsFact.DependsOn(s, t, 1.0))
                        | p when p.EndsWith("belongsTo") || p.EndsWith("#belongsTo") ->
                            Some(TarsFact.BelongsTo(s, oUri))
                        | _ -> None
                    | None -> None
            with _ ->
                None)
        |> Seq.toList
