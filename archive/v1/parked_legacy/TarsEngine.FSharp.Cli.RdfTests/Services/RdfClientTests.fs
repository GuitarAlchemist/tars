namespace TarsEngine.FSharp.Cli.Tests.Services

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Xunit
open TarsEngine.FSharp.Cli.Services.RDF

/// Tests for RDF client functionality
module RdfClientTests =
    
    /// Create a test logger
    let createTestLogger<'T>() =
        let loggerFactory = LoggerFactory.Create(fun builder -> builder.AddConsole() |> ignore)
        loggerFactory.CreateLogger<'T>()
    
    /// Create test knowledge RDF
    let createTestKnowledge() = {
        KnowledgeUri = "http://tars.ai/ontology#knowledge/test-123"
        Topic = "F# Programming"
        Content = "F# is a functional programming language"
        Source = "test-source"
        Confidence = 0.85
        LearnedAt = DateTime.UtcNow
        Tags = ["functional"; "programming"; "fsharp"]
        Triples = []
    }
    
    [<Fact>]
    let ``InMemoryRdfClient should initialize successfully`` () =
        // Arrange & Act
        let logger = createTestLogger<InMemoryRdfClient>()
        let client = InMemoryRdfClient(logger)
        
        // Assert - should not throw
        Assert.NotNull(client)
    
    [<Fact>]
    let ``InMemoryRdfClient should implement IRdfClient interface`` () =
        // Arrange
        let logger = createTestLogger<InMemoryRdfClient>()
        let client = InMemoryRdfClient(logger)
        
        // Act & Assert
        Assert.IsAssignableFrom<IRdfClient>(client)
    
    [<Fact>]
    let ``InsertKnowledge should succeed with valid knowledge`` () =
        task {
            // Arrange
            let logger = createTestLogger<InMemoryRdfClient>()
            let client = InMemoryRdfClient(logger) :> IRdfClient
            let knowledge = createTestKnowledge()
            
            // Act
            let! result = client.InsertKnowledge(knowledge)
            
            // Assert
            match result with
            | Ok () -> Assert.True(true)
            | Error err -> Assert.True(false, sprintf "Expected success but got error: %s" err)
        }
    
    [<Fact>]
    let ``QueryKnowledge should execute SPARQL query`` () =
        task {
            // Arrange
            let logger = createTestLogger<InMemoryRdfClient>()
            let client = InMemoryRdfClient(logger) :> IRdfClient
            let sparqlQuery = "SELECT * WHERE { ?s ?p ?o } LIMIT 10"
            
            // Act
            let! result = client.QueryKnowledge(sparqlQuery)
            
            // Assert
            match result with
            | Ok sparqlResult -> 
                Assert.True(sparqlResult.Success)
                Assert.NotNull(sparqlResult.Results)
            | Error err -> Assert.True(false, sprintf "Expected success but got error: %s" err)
        }
    
    [<Fact>]
    let ``SearchByTopic should return empty list for non-existent topic`` () =
        task {
            // Arrange
            let logger = createTestLogger<InMemoryRdfClient>()
            let client = InMemoryRdfClient(logger) :> IRdfClient
            
            // Act
            let! result = client.SearchByTopic("non-existent-topic")
            
            // Assert
            match result with
            | Ok knowledgeList -> Assert.Empty(knowledgeList)
            | Error err -> Assert.True(false, sprintf "Expected success but got error: %s" err)
        }
    
    [<Fact>]
    let ``GetRelatedKnowledge should return empty list for non-existent URI`` () =
        task {
            // Arrange
            let logger = createTestLogger<InMemoryRdfClient>()
            let client = InMemoryRdfClient(logger) :> IRdfClient
            let testUri = "http://tars.ai/ontology#knowledge/non-existent"
            
            // Act
            let! result = client.GetRelatedKnowledge(testUri)
            
            // Assert
            match result with
            | Ok relatedList -> Assert.Empty(relatedList)
            | Error err -> Assert.True(false, sprintf "Expected success but got error: %s" err)
        }
    
    [<Fact>]
    let ``Multiple knowledge insertions should succeed`` () =
        task {
            // Arrange
            let logger = createTestLogger<InMemoryRdfClient>()
            let client = InMemoryRdfClient(logger) :> IRdfClient
            
            let knowledge1 = { createTestKnowledge() with Topic = "F# Programming"; KnowledgeUri = "http://tars.ai/ontology#knowledge/fsharp" }
            let knowledge2 = { createTestKnowledge() with Topic = "Functional Programming"; KnowledgeUri = "http://tars.ai/ontology#knowledge/functional" }
            let knowledge3 = { createTestKnowledge() with Topic = "Machine Learning"; KnowledgeUri = "http://tars.ai/ontology#knowledge/ml" }
            
            // Act
            let! result1 = client.InsertKnowledge(knowledge1)
            let! result2 = client.InsertKnowledge(knowledge2)
            let! result3 = client.InsertKnowledge(knowledge3)
            
            // Assert
            match result1, result2, result3 with
            | Ok (), Ok (), Ok () -> Assert.True(true)
            | _ -> Assert.True(false, "One or more knowledge insertions failed")
        }
    
    [<Fact>]
    let ``RDF client should handle concurrent operations`` () =
        task {
            // Arrange
            let logger = createTestLogger<InMemoryRdfClient>()
            let client = InMemoryRdfClient(logger) :> IRdfClient
            
            // Act - Run multiple operations concurrently
            let tasks = [
                client.InsertKnowledge({ createTestKnowledge() with Topic = "Topic1"; KnowledgeUri = "http://tars.ai/ontology#knowledge/topic1" })
                client.InsertKnowledge({ createTestKnowledge() with Topic = "Topic2"; KnowledgeUri = "http://tars.ai/ontology#knowledge/topic2" })
                client.QueryKnowledge("SELECT * WHERE { ?s ?p ?o } LIMIT 5")
                client.SearchByTopic("programming")
            ]
            
            let! results = Task.WhenAll(tasks)
            
            // Assert - All operations should complete without throwing
            Assert.Equal(4, results.Length)
        }
