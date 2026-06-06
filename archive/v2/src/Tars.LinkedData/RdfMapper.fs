namespace Tars.LinkedData

open System
open Tars.Core

/// Maps TARS domain entities and facts to RDF triples
module RdfMapper =

    let NS_TARS = "http://tars.ai/ns#"
    let NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    let NS_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
    let NS_XSD = "http://www.w3.org/2001/XMLSchema#"

    /// Get URI for a TarsEntity
    let entityToUri (entity: TarsEntity) =
        let id = TarsEntity.getId entity
        // Replace colon with slash or keep it? SPARQL handles <...> nicely.
        sprintf "<http://tars.ai/resource/%s>" (id.Replace(":", "/"))

    /// Safe string escaping for literals
    let escape (s: string) =
        s.Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "\\r")

    /// Map TarsEntity properties to triples
    let entityToTriples (entity: TarsEntity) : string list =
        let subject = entityToUri entity
        let mutable triples = [
            sprintf "%s <%s> <%s> ." subject (NS_RDF + "type") (NS_TARS + "Entity")
        ]

        match entity with
        | TarsEntity.ConceptE c ->
            triples <- triples @ [
                sprintf "%s <%s> \"%s\" ." subject (NS_RDFS + "label") (escape c.Name)
                sprintf "%s <%s> \"%s\" ." subject (NS_RDFS + "comment") (escape c.Description)
                sprintf "%s <%s> <%s> ." subject (NS_RDF + "type") (NS_TARS + "Concept")
            ]
        | TarsEntity.AgentBeliefE b ->
            triples <- triples @ [
                sprintf "%s <%s> \"%s\" ." subject (NS_TARS + "statement") (escape b.Statement)
                sprintf "%s <%s> \"%f\"^^<%s> ." subject (NS_TARS + "confidence") b.Confidence (NS_XSD + "double")
                sprintf "%s <%s> <%s> ." subject (NS_RDF + "type") (NS_TARS + "Belief")
            ]
        | TarsEntity.RunE r ->
             triples <- triples @ [
                sprintf "%s <%s> \"%s\" ." subject (NS_TARS + "goal") (escape r.Goal)
                sprintf "%s <%s> \"%s\" ." subject (NS_TARS + "pattern") (escape r.Pattern)
                sprintf "%s <%s> \"%s\"^^<%s> ." subject (NS_TARS + "timestamp") (r.Timestamp.ToString("o")) (NS_XSD + "dateTime")
                sprintf "%s <%s> <%s> ." subject (NS_RDF + "type") (NS_TARS + "Run")
            ]
        | TarsEntity.StepE s ->
            triples <- triples @ [
                sprintf "%s <%s> %s ." subject (NS_TARS + "runId") (entityToUri (TarsEntity.RunE { Id = s.RunId; Goal = ""; Pattern = ""; Timestamp = DateTime.MinValue }))
                sprintf "%s <%s> \"%s\" ." subject (NS_TARS + "stepId") (escape s.StepId)
                sprintf "%s <%s> \"%s\" ." subject (NS_TARS + "nodeType") (escape s.NodeType)
                sprintf "%s <%s> \"%s\"^^<%s> ." subject (NS_TARS + "timestamp") (s.Timestamp.ToString("o")) (NS_XSD + "dateTime")
                sprintf "%s <%s> <%s> ." subject (NS_RDF + "type") (NS_TARS + "Step")
            ]
        | TarsEntity.EpisodeE e ->
            triples <- triples @ [
                sprintf "%s <%s> \"%s\" ." subject (NS_TARS + "episodeType") (Episode.typeTag e)
                sprintf "%s <%s> \"%s\"^^<%s> ." subject (NS_TARS + "timestamp") (Episode.timestamp e |> fun t -> t.ToString("o")) (NS_XSD + "dateTime")
                sprintf "%s <%s> <%s> ." subject (NS_RDF + "type") (NS_TARS + "Episode")
            ]
        | TarsEntity.FileE p ->
            triples <- triples @ [
                sprintf "%s <%s> \"%s\" ." subject (NS_TARS + "path") (escape p)
                sprintf "%s <%s> <%s> ." subject (NS_RDF + "type") (NS_TARS + "File")
            ]
        | _ -> ()

        triples

    /// Map TarsFact to triples
    let factToTriples (fact: TarsFact) : string list =
        let sourceUri = entityToUri (TarsFact.source fact)
        
        match TarsFact.target fact with
        | Some targetEntity ->
            let targetUri = entityToUri targetEntity
            let predicate = 
                match fact with
                | TarsFact.Implements _ -> NS_TARS + "implements"
                | TarsFact.DependsOn _ -> NS_TARS + "dependsOn"
                | TarsFact.Contradicts _ -> NS_TARS + "contradicts"
                | TarsFact.EvolvedFrom _ -> NS_TARS + "evolvedFrom"
                | TarsFact.Contains _ -> NS_TARS + "contains"
                | TarsFact.NextStep _ -> NS_TARS + "nextStep"
                | TarsFact.SimilarTo _ -> NS_TARS + "similarTo"
                | TarsFact.DerivedFrom _ -> NS_TARS + "derivedFrom"
                | TarsFact.BelongsTo _ -> NS_TARS + "belongsTo"
                
            [ sprintf "%s <%s> %s ." sourceUri predicate targetUri ]
        | None -> 
            match fact with
            | TarsFact.BelongsTo(e, communityId) ->
                [ sprintf "%s <%s> \"%s\" ." sourceUri (NS_TARS + "belongsTo") (escape communityId) ]
            | _ -> []
