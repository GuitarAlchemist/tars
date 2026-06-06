namespace TarsEngine.FSharp.Cli.Services.RDF

open System
open System.Net.Http
open System.Text
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Services.RDF

/// Extension methods for registering RDF services
module RdfServiceExtensions =
    
    /// Register RDF services with dependency injection
    let addRdfServices (services: IServiceCollection) =
        
        // Register HttpClient for RDF operations
        services.AddHttpClient("RdfClient") |> ignore
        
        // Register RDF client as singleton
        services.AddSingleton<IRdfClient>(fun serviceProvider ->
            let logger = serviceProvider.GetRequiredService<ILogger<RdfClient>>()
            let httpClientFactory = serviceProvider.GetRequiredService<IHttpClientFactory>()
            let httpClient = httpClientFactory.CreateClient("RdfClient")
            
            // Configure RDF store - try Fuseki first, fallback to Virtuoso
            let fusekiConfig = {
                StoreType = Fuseki
                BaseUrl = "http://localhost:3030"
                Dataset = Some "tars"
                Username = None
                Password = None
            }
            
            let virtuosoConfig = {
                StoreType = Virtuoso
                BaseUrl = "http://localhost:8890"
                Dataset = None
                Username = Some "dba"
                Password = Some "tars123"
            }
            
            // Test which RDF store is available
            try
                // Test Fuseki first
                let fusekiTest = httpClient.GetAsync("http://localhost:3030/").Result
                if fusekiTest.IsSuccessStatusCode then
                    logger.LogInformation("🗄️ RDF: Using Apache Jena Fuseki at http://localhost:3030")
                    RdfClient(fusekiConfig, httpClient, logger) :> IRdfClient
                else
                    // Fallback to Virtuoso
                    let virtuosoTest = httpClient.GetAsync("http://localhost:8890/sparql").Result
                    if virtuosoTest.IsSuccessStatusCode then
                        logger.LogInformation("🗄️ RDF: Using OpenLink Virtuoso at http://localhost:8890")
                        RdfClient(virtuosoConfig, httpClient, logger) :> IRdfClient
                    else
                        logger.LogWarning("⚠️ RDF: No RDF store available, RDF functionality will be disabled")
                        Unchecked.defaultof<IRdfClient>
            with
            | ex ->
                logger.LogWarning(ex, "⚠️ RDF: Failed to connect to RDF stores, RDF functionality will be disabled")
                Unchecked.defaultof<IRdfClient>
        ) |> ignore
        
        services

    /// Create RDF dataset in Fuseki if it doesn't exist
    let createFusekiDataset (httpClient: HttpClient) (logger: ILogger) =
        async {
            try
                let datasetConfig = """
{
  "dbName": "tars",
  "dbType": "tdb2"
}
"""
                let content = new StringContent(datasetConfig, Encoding.UTF8, "application/json")
                let! response = httpClient.PostAsync("http://localhost:3030/$/datasets", content) |> Async.AwaitTask
                
                if response.IsSuccessStatusCode then
                    logger.LogInformation("✅ RDF: Created TARS dataset in Fuseki")
                    return Ok()
                else
                    let! errorContent = response.Content.ReadAsStringAsync() |> Async.AwaitTask
                    if errorContent.Contains("already exists") then
                        logger.LogInformation("ℹ️ RDF: TARS dataset already exists in Fuseki")
                        return Ok()
                    else
                        logger.LogWarning("⚠️ RDF: Failed to create Fuseki dataset: {Error}", errorContent)
                        return Error errorContent
            with
            | ex ->
                logger.LogWarning(ex, "⚠️ RDF: Exception creating Fuseki dataset")
                return Error ex.Message
        }

    /// Initialize RDF store with TARS ontology
    let initializeRdfStore (rdfClient: IRdfClient option) (logger: ILogger) =
        async {
            match rdfClient with
            | Some client ->
                try
                    // Define TARS ontology in SPARQL
                    let ontologySparql = """
PREFIX tars: <http://tars.ai/ontology#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

INSERT DATA {
  # TARS Ontology Classes
  tars:Knowledge rdf:type owl:Class ;
                 rdfs:label "TARS Knowledge" ;
                 rdfs:comment "Represents a piece of knowledge learned by TARS" .
  
  tars:LearningSource rdf:type owl:Class ;
                      rdfs:label "Learning Source" ;
                      rdfs:comment "Source from which knowledge was acquired" .
  
  # TARS Ontology Properties
  tars:topic rdf:type owl:DatatypeProperty ;
             rdfs:label "topic" ;
             rdfs:domain tars:Knowledge ;
             rdfs:range rdfs:Literal .
  
  tars:content rdf:type owl:DatatypeProperty ;
               rdfs:label "content" ;
               rdfs:domain tars:Knowledge ;
               rdfs:range rdfs:Literal .
  
  tars:source rdf:type owl:DatatypeProperty ;
              rdfs:label "source" ;
              rdfs:domain tars:Knowledge ;
              rdfs:range rdfs:Literal .
  
  tars:confidence rdf:type owl:DatatypeProperty ;
                  rdfs:label "confidence" ;
                  rdfs:domain tars:Knowledge ;
                  rdfs:range rdfs:Literal .
  
  tars:learnedAt rdf:type owl:DatatypeProperty ;
                 rdfs:label "learned at" ;
                 rdfs:domain tars:Knowledge ;
                 rdfs:range rdfs:Literal .
  
  tars:hasTag rdf:type owl:DatatypeProperty ;
              rdfs:label "has tag" ;
              rdfs:domain tars:Knowledge ;
              rdfs:range rdfs:Literal .
  
  tars:relatedTo rdf:type owl:ObjectProperty ;
                 rdfs:label "related to" ;
                 rdfs:domain tars:Knowledge ;
                 rdfs:range tars:Knowledge .
}
"""
                    
                    match! client.QueryKnowledge(ontologySparql) |> Async.AwaitTask with
                    | Ok result when result.Success ->
                        logger.LogInformation("✅ RDF: Successfully initialized TARS ontology")
                        return Ok()
                    | Ok result ->
                        logger.LogWarning("⚠️ RDF: Failed to initialize ontology: {Error}", (result.Error |> Option.defaultValue "Unknown error"))
                        return Error (result.Error |> Option.defaultValue "Unknown error")
                    | Error error ->
                        logger.LogWarning("⚠️ RDF: Error initializing ontology: {Error}", error)
                        return Error error
                        
                with
                | ex ->
                    logger.LogWarning(ex, "⚠️ RDF: Exception initializing RDF store")
                    return Error ex.Message
            | None ->
                logger.LogInformation("ℹ️ RDF: No RDF client available, skipping ontology initialization")
                return Ok()
        }
