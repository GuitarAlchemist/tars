namespace TarsEngine.FSharp.Cli.Services.RDF

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open VDS.RDF
open VDS.RDF.Storage
open VDS.RDF.Query
open VDS.RDF.Parsing

/// In-memory RDF client using dotNetRDF for local triple store
type InMemoryRdfClient(logger: ILogger<InMemoryRdfClient>) =

    let store = new TripleStore()
    let graph = new Graph()
    let tarsNamespace = "http://tars.ai/ontology#"

    do
        // Initialize the graph with TARS namespace
        graph.BaseUri <- UriFactory.Create(tarsNamespace)
        store.Add(graph)
        logger.LogInformation("🗄️ RDF: Initialized in-memory triple store")

    /// Convert knowledge to RDF triples and add to store
    let addKnowledgeToStore (knowledge: KnowledgeRdf) =
        try
            let knowledgeNode = graph.CreateUriNode(UriFactory.Create(knowledge.KnowledgeUri))
            let rdfType = graph.CreateUriNode(UriFactory.Create("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"))
            let tarsKnowledge = graph.CreateUriNode(UriFactory.Create(tarsNamespace + "Knowledge"))

            // Add type triple
            graph.Assert(knowledgeNode, rdfType, tarsKnowledge)

            // Add properties
            let topicPred = graph.CreateUriNode(UriFactory.Create(tarsNamespace + "topic"))
            let topicLit = graph.CreateLiteralNode(knowledge.Topic)
            graph.Assert(knowledgeNode, topicPred, topicLit)

            let contentPred = graph.CreateUriNode(UriFactory.Create(tarsNamespace + "content"))
            let contentLit = graph.CreateLiteralNode(knowledge.Content)
            graph.Assert(knowledgeNode, contentPred, contentLit)

            let sourcePred = graph.CreateUriNode(UriFactory.Create(tarsNamespace + "source"))
            let sourceLit = graph.CreateLiteralNode(knowledge.Source)
            graph.Assert(knowledgeNode, sourcePred, sourceLit)

            let confidencePred = graph.CreateUriNode(UriFactory.Create(tarsNamespace + "confidence"))
            let confidenceLit = graph.CreateLiteralNode(knowledge.Confidence.ToString())
            graph.Assert(knowledgeNode, confidencePred, confidenceLit)

            // Add tags
            let tagPred = graph.CreateUriNode(UriFactory.Create(tarsNamespace + "hasTag"))
            for tag in knowledge.Tags do
                let tagLit = graph.CreateLiteralNode(tag)
                graph.Assert(knowledgeNode, tagPred, tagLit)

            logger.LogDebug("✅ RDF: Added knowledge '{Topic}' to in-memory store", knowledge.Topic)
            Ok()

        with
        | ex ->
            logger.LogError(ex, "❌ RDF: Failed to add knowledge to store")
            Error ex.Message

    /// Execute SPARQL query on in-memory store (REAL implementation)
    let executeSparqlQuery (sparqlQuery: string) =
        try
            logger.LogDebug("🔍 RDF: Executing SPARQL query: {Query}", sparqlQuery)

            // REAL SPARQL processing using dotNetRDF
            let parser = new SparqlQueryParser()
            let query = parser.ParseFromString(sparqlQuery)

            // Create query processor for the triple store
            let processor = new LeviathanQueryProcessor(store)
            let results = processor.ProcessQuery(query)

            match results with
            | :? SparqlResultSet as resultSet ->
                // Convert SPARQL results to JSON
                let jsonResults = System.Collections.Generic.List<string>()

                for result in resultSet do
                    let resultObj = System.Collections.Generic.Dictionary<string, string>()
                    for var in resultSet.Variables do
                        let value =
                            match result.[var] with
                            | null -> ""
                            | node -> node.ToString()
                        resultObj.[var] <- value

                    let jsonObj = System.Text.Json.JsonSerializer.Serialize(resultObj)
                    jsonResults.Add(jsonObj)

                let finalJson = "[" + String.concat "," jsonResults + "]"
                logger.LogInformation("✅ RDF: SPARQL query returned {Count} results from {TripleCount} triples", resultSet.Count, graph.Triples.Count)
                Ok { Success = true; Results = finalJson; Error = None }

            | :? Graph as resultGraph ->
                // Handle CONSTRUCT/DESCRIBE queries
                let tripleCount = resultGraph.Triples.Count
                logger.LogInformation("✅ RDF: SPARQL query returned graph with {Count} triples", tripleCount)
                Ok { Success = true; Results = sprintf "[{\"tripleCount\":%d}]" tripleCount; Error = None }

            | _ ->
                // Handle ASK queries and other result types
                logger.LogInformation("✅ RDF: SPARQL query returned other result type")
                Ok { Success = true; Results = "[{\"result\":\"processed\"}]"; Error = None }

        with
        | ex ->
            logger.LogError(ex, "❌ RDF: SPARQL query failed: {Error}", ex.Message)
            Error ex.Message

    interface IRdfClient with
        member _.InsertKnowledge(knowledge: KnowledgeRdf) =
            task {
                match addKnowledgeToStore knowledge with
                | Ok () -> return Ok()
                | Error err -> return Error err
            }

        member _.QueryKnowledge(sparql: string) =
            task {
                match executeSparqlQuery sparql with
                | Ok result -> return Ok result
                | Error err -> return Error err
            }

        member _.SearchByTopic(topic: string) =
            task {
                logger.LogInformation("🔍 RDF: Searching for topic: {Topic}", topic)

                try
                    // Create SPARQL query to find knowledge by topic
                    let sparqlQuery = sprintf "PREFIX tars: <http://tars.ai/ontology#> SELECT ?uri ?topic ?content ?source ?confidence WHERE { ?uri a tars:Knowledge ; tars:topic ?topic ; tars:content ?content ; tars:source ?source ; tars:confidence ?confidence . FILTER(CONTAINS(LCASE(?topic), LCASE(\"%s\")) || CONTAINS(LCASE(?content), LCASE(\"%s\"))) } ORDER BY DESC(?confidence) LIMIT 20" topic topic

                    match executeSparqlQuery sparqlQuery with
                    | Ok result when result.Success ->
                        // Parse JSON results and convert to KnowledgeRdf list
                        let jsonResults = System.Text.Json.JsonDocument.Parse(result.Results)
                        let knowledgeList = System.Collections.Generic.List<KnowledgeRdf>()

                        for element in jsonResults.RootElement.EnumerateArray() do
                            let uri = element.GetProperty("uri").GetString()
                            let topicValue = element.GetProperty("topic").GetString()
                            let content = element.GetProperty("content").GetString()
                            let source = element.GetProperty("source").GetString()
                            let confidenceStr = element.GetProperty("confidence").GetString()

                            match System.Double.TryParse(confidenceStr) with
                            | true, confidence ->
                                let knowledge = {
                                    KnowledgeUri = uri
                                    Topic = topicValue
                                    Content = content
                                    Source = source
                                    Confidence = confidence
                                    LearnedAt = System.DateTime.UtcNow
                                    Tags = []
                                    Triples = []
                                }
                                knowledgeList.Add(knowledge)
                            | false, _ -> ()

                        logger.LogInformation("✅ RDF: Found {Count} knowledge entries for topic '{Topic}'", knowledgeList.Count, topic)
                        return Ok (knowledgeList |> Seq.toList)

                    | Ok result ->
                        logger.LogWarning("⚠️ RDF: Search query failed: {Error}", result.Error |> Option.defaultValue "Unknown error")
                        return Ok []

                    | Error error ->
                        logger.LogError("❌ RDF: Search failed: {Error}", error)
                        return Error error

                with
                | ex ->
                    logger.LogError(ex, "❌ RDF: Exception during topic search")
                    return Error ex.Message
            }

        member _.GetRelatedKnowledge(knowledgeUri: string) =
            task {
                logger.LogInformation("🔍 RDF: Finding related knowledge for: {Uri}", knowledgeUri)

                try
                    // Create SPARQL query to find related knowledge through shared tags and concepts
                    let sparqlQuery = sprintf "PREFIX tars: <http://tars.ai/ontology#> SELECT ?relatedUri ?topic ?content ?source ?confidence WHERE { <%s> tars:hasTag ?tag . ?relatedUri a tars:Knowledge ; tars:topic ?topic ; tars:content ?content ; tars:source ?source ; tars:confidence ?confidence ; tars:hasTag ?tag . FILTER(?relatedUri != <%s>) } ORDER BY DESC(?confidence) LIMIT 10" knowledgeUri knowledgeUri

                    match executeSparqlQuery sparqlQuery with
                    | Ok result when result.Success ->
                        // Parse JSON results and convert to KnowledgeRdf list
                        let jsonResults = System.Text.Json.JsonDocument.Parse(result.Results)
                        let relatedKnowledge = System.Collections.Generic.List<KnowledgeRdf>()

                        for element in jsonResults.RootElement.EnumerateArray() do
                            let uri = element.GetProperty("relatedUri").GetString()
                            let topic = element.GetProperty("topic").GetString()
                            let content = element.GetProperty("content").GetString()
                            let source = element.GetProperty("source").GetString()
                            let confidenceStr = element.GetProperty("confidence").GetString()

                            match System.Double.TryParse(confidenceStr) with
                            | true, confidence ->
                                let knowledge = {
                                    KnowledgeUri = uri
                                    Topic = topic
                                    Content = content
                                    Source = source
                                    Confidence = confidence
                                    LearnedAt = System.DateTime.UtcNow
                                    Tags = []
                                    Triples = []
                                }
                                relatedKnowledge.Add(knowledge)
                            | false, _ -> ()

                        logger.LogInformation("✅ RDF: Found {Count} related knowledge entries for '{Uri}'", relatedKnowledge.Count, knowledgeUri)
                        return Ok (relatedKnowledge |> Seq.toList)

                    | Ok result ->
                        logger.LogWarning("⚠️ RDF: Related knowledge query failed: {Error}", result.Error |> Option.defaultValue "Unknown error")
                        return Ok []

                    | Error error ->
                        logger.LogError("❌ RDF: Related knowledge search failed: {Error}", error)
                        return Error error

                with
                | ex ->
                    logger.LogError(ex, "❌ RDF: Exception during related knowledge search")
                    return Error ex.Message
            }

    /// Activate the RDF triple store for semantic learning
    member this.ActivateTripleStore() =
        logger.LogInformation("🗄️ RDF: Activated in-memory triple store for semantic learning")

    /// Get real statistics from the RDF store
    member this.GetRdfStatistics() =
        {|
            TripleCount = graph.Triples.Count
            KnowledgeCount =
                try
                    let countQuery = "PREFIX tars: <http://tars.ai/ontology#> SELECT (COUNT(?k) as ?count) WHERE { ?k a tars:Knowledge }"
                    match executeSparqlQuery countQuery with
                    | Ok result when result.Success ->
                        let jsonResults = System.Text.Json.JsonDocument.Parse(result.Results)
                        if jsonResults.RootElement.GetArrayLength() > 0 then
                            let firstResult = jsonResults.RootElement.[0]
                            match firstResult.TryGetProperty("count") with
                            | true, countProp ->
                                match System.Int32.TryParse(countProp.GetString()) with
                                | true, count -> count
                                | false, _ -> 0
                            | false, _ -> 0
                        else 0
                    | _ -> 0
                with
                | _ -> 0
            IsActive = true
            NamespaceCount = graph.NamespaceMap.Prefixes |> Seq.length
        |}