namespace TarsEngine.FSharp.Cli.Services.RDF

open System
open System.Net.Http
open System.Text
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Core

/// RDF store type
type RdfStoreType =
    | Fuseki
    | Virtuoso

/// RDF store configuration
type RdfStoreConfig = {
    StoreType: RdfStoreType
    BaseUrl: string
    Dataset: string option
    Username: string option
    Password: string option
}

/// SPARQL query result
type SparqlResult = {
    Success: bool
    Results: string
    Error: string option
}

/// RDF triple representation
type RdfTriple = {
    Subject: string
    Predicate: string
    Object: string
    ObjectType: string // "uri", "literal", "bnode"
}

/// Knowledge to RDF mapping
type KnowledgeRdf = {
    KnowledgeUri: string
    Topic: string
    Content: string
    Source: string
    Confidence: float
    LearnedAt: DateTime
    Tags: string list
    Triples: RdfTriple list
}

/// RDF client for TARS knowledge graph
type IRdfClient =
    abstract member InsertKnowledge: KnowledgeRdf -> Task<Result<unit, string>>
    abstract member QueryKnowledge: string -> Task<Result<SparqlResult, string>>
    abstract member SearchByTopic: string -> Task<Result<KnowledgeRdf list, string>>
    abstract member GetRelatedKnowledge: string -> Task<Result<KnowledgeRdf list, string>>

/// RDF client implementation
type RdfClient(config: RdfStoreConfig, httpClient: HttpClient, logger: ILogger<RdfClient>) =
    
    let tarsNamespace = "http://tars.ai/ontology#"
    let rdfNamespace = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    let rdfsNamespace = "http://www.w3.org/2000/01/rdf-schema#"
    let xsdNamespace = "http://www.w3.org/2001/XMLSchema#"
    
    /// Generate TARS URI for knowledge
    let generateKnowledgeUri (knowledgeId: string) =
        sprintf "%sknowledge/%s" tarsNamespace knowledgeId
    
    /// Convert knowledge to RDF triples
    let knowledgeToTriples (knowledge: KnowledgeRdf) : RdfTriple list =
        let knowledgeUri = knowledge.KnowledgeUri
        [
            // Basic knowledge properties
            { Subject = knowledgeUri; Predicate = rdfNamespace + "type"; Object = tarsNamespace + "Knowledge"; ObjectType = "uri" }
            { Subject = knowledgeUri; Predicate = tarsNamespace + "topic"; Object = knowledge.Topic; ObjectType = "literal" }
            { Subject = knowledgeUri; Predicate = tarsNamespace + "content"; Object = knowledge.Content; ObjectType = "literal" }
            { Subject = knowledgeUri; Predicate = tarsNamespace + "source"; Object = knowledge.Source; ObjectType = "literal" }
            { Subject = knowledgeUri; Predicate = tarsNamespace + "confidence"; Object = knowledge.Confidence.ToString(); ObjectType = "literal" }
            { Subject = knowledgeUri; Predicate = tarsNamespace + "learnedAt"; Object = knowledge.LearnedAt.ToString("yyyy-MM-ddTHH:mm:ssZ"); ObjectType = "literal" }
            
            // Tags as separate triples
            yield! knowledge.Tags |> List.map (fun tag ->
                { Subject = knowledgeUri; Predicate = tarsNamespace + "hasTag"; Object = tag; ObjectType = "literal" }
            )
        ]
    
    /// Convert triples to SPARQL INSERT
    let triplesToSparqlInsert (triples: RdfTriple list) : string =
        let prefixes = sprintf "PREFIX tars: <%s>\nPREFIX rdf: <%s>\nPREFIX rdfs: <%s>\nPREFIX xsd: <%s>" tarsNamespace rdfNamespace rdfsNamespace xsdNamespace

        let tripleStrings =
            triples
            |> List.map (fun triple ->
                let objectStr =
                    match triple.ObjectType with
                    | "uri" -> sprintf "<%s>" triple.Object
                    | "literal" -> sprintf "\"%s\"" (triple.Object.Replace("\"", "\\\""))
                    | _ -> sprintf "\"%s\"" triple.Object
                sprintf "  <%s> <%s> %s ." triple.Subject triple.Predicate objectStr
            )
            |> String.concat "\n"

        sprintf "%s\nINSERT DATA {\n%s\n}" prefixes tripleStrings
    
    /// Execute SPARQL update
    let executeSparqlUpdate (sparql: string) : Task<Result<unit, string>> = task {
        try
            let endpoint = 
                match config.StoreType with
                | Fuseki -> sprintf "%s/%s/update" config.BaseUrl (config.Dataset |> Option.defaultValue "tars")
                | Virtuoso -> sprintf "%s/sparql" config.BaseUrl
            
            logger.LogDebug("🔍 RDF: Executing SPARQL update on {Endpoint}", endpoint)
            logger.LogDebug("📝 RDF: SPARQL query:\n{Sparql}", sparql)
            
            let content = new StringContent(sparql, Encoding.UTF8, "application/sparql-update")
            
            // Add authentication if configured
            match config.Username, config.Password with
            | Some username, Some password ->
                let authValue = Convert.ToBase64String(Encoding.ASCII.GetBytes(sprintf "%s:%s" username password))
                httpClient.DefaultRequestHeaders.Authorization <- 
                    new System.Net.Http.Headers.AuthenticationHeaderValue("Basic", authValue)
            | _ -> ()
            
            let! response = httpClient.PostAsync(endpoint, content)
            
            if response.IsSuccessStatusCode then
                logger.LogInformation("✅ RDF: Successfully executed SPARQL update")
                return Ok()
            else
                let! errorContent = response.Content.ReadAsStringAsync()
                let error = sprintf "SPARQL update failed: %s - %s" (response.StatusCode.ToString()) errorContent
                logger.LogError("❌ RDF: {Error}", error)
                return Error error
                
        with ex ->
            let error = sprintf "RDF client error: %s" ex.Message
            logger.LogError(ex, "❌ RDF: Exception during SPARQL update")
            return Error error
    }
    
    /// Execute SPARQL query
    let executeSparqlQuery (sparql: string) : Task<Result<SparqlResult, string>> = task {
        try
            let endpoint = 
                match config.StoreType with
                | Fuseki -> sprintf "%s/%s/sparql" config.BaseUrl (config.Dataset |> Option.defaultValue "tars")
                | Virtuoso -> sprintf "%s/sparql" config.BaseUrl
            
            logger.LogDebug("🔍 RDF: Executing SPARQL query on {Endpoint}", endpoint)
            logger.LogDebug("📝 RDF: SPARQL query:\n{Sparql}", sparql)
            
            let encodedQuery = System.Web.HttpUtility.UrlEncode(sparql)
            let url = sprintf "%s?query=%s&format=application/sparql-results+json" endpoint encodedQuery
            
            let! response = httpClient.GetAsync(url)
            let! content = response.Content.ReadAsStringAsync()
            
            if response.IsSuccessStatusCode then
                logger.LogInformation("✅ RDF: Successfully executed SPARQL query")
                return Ok { Success = true; Results = content; Error = None }
            else
                let error = sprintf "SPARQL query failed: %s - %s" (response.StatusCode.ToString()) content
                logger.LogError("❌ RDF: {Error}", error)
                return Ok { Success = false; Results = ""; Error = Some error }
                
        with ex ->
            let error = sprintf "RDF client error: %s" ex.Message
            logger.LogError(ex, "❌ RDF: Exception during SPARQL query")
            return Error error
    }
    
    interface IRdfClient with
        member _.InsertKnowledge(knowledge: KnowledgeRdf) = task {
            let triples = knowledgeToTriples knowledge
            let sparql = triplesToSparqlInsert triples
            return! executeSparqlUpdate sparql
        }
        
        member _.QueryKnowledge(sparql: string) = 
            executeSparqlQuery sparql
        
        member _.SearchByTopic(topic: string) =
            task {
                let sparql = sprintf "PREFIX tars: <%s>\nSELECT ?knowledge ?topic ?content ?source ?confidence ?learnedAt WHERE {\n  ?knowledge a tars:Knowledge ;\n             tars:topic ?topic ;\n             tars:content ?content ;\n             tars:source ?source ;\n             tars:confidence ?confidence ;\n             tars:learnedAt ?learnedAt .\n  FILTER(CONTAINS(LCASE(?topic), LCASE(\"%s\")))\n}" tarsNamespace topic

                match! executeSparqlQuery sparql with
                | Ok result when result.Success ->
                    // TODO: Parse JSON results and convert back to KnowledgeRdf
                    logger.LogInformation("🔍 RDF: Found knowledge for topic: {Topic}", topic)
                    return Ok []
                | Ok result ->
                    return Error (result.Error |> Option.defaultValue "Query failed")
                | Error err ->
                    return Error err
            }
        
        member _.GetRelatedKnowledge(knowledgeUri: string) =
            task {
                let sparql = sprintf "PREFIX tars: <%s>\nSELECT ?related ?topic ?content WHERE {\n  <%s> tars:relatedTo ?related .\n  ?related tars:topic ?topic ;\n           tars:content ?content .\n}" tarsNamespace knowledgeUri

                match! executeSparqlQuery sparql with
                | Ok result when result.Success ->
                    logger.LogInformation("🔍 RDF: Found related knowledge for: {Uri}", knowledgeUri)
                    return Ok []
                | Ok result ->
                    return Error (result.Error |> Option.defaultValue "Query failed")
                | Error err ->
                    return Error err
            }
