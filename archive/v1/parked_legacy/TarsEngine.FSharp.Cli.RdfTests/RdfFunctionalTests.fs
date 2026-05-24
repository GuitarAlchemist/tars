namespace TarsEngine.FSharp.Cli.Tests

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Xunit
open TarsEngine.FSharp.Cli.Services.RDF

/// Functional tests for RDF triple store activation and basic functionality
module RdfFunctionalTests =
    
    /// Create a test logger
    let createTestLogger<'T>() =
        let loggerFactory = LoggerFactory.Create(fun builder -> builder.AddConsole() |> ignore)
        loggerFactory.CreateLogger<'T>()
    
    [<Fact>]
    let ``RDF InMemoryClient should initialize without errors`` () =
        // Arrange & Act
        let logger = createTestLogger<InMemoryRdfClient>()
        let client = InMemoryRdfClient(logger)
        
        // Assert - should not throw
        Assert.NotNull(client)
    
    [<Fact>]
    let ``RDF InMemoryClient should implement IRdfClient interface`` () =
        // Arrange
        let logger = createTestLogger<InMemoryRdfClient>()
        let client = InMemoryRdfClient(logger)
        
        // Act & Assert
        Assert.IsAssignableFrom<IRdfClient>(client)
    
    [<Fact>]
    let ``RDF client should handle basic SPARQL queries`` () =
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
    let ``RDF client should handle knowledge insertion`` () =
        task {
            // Arrange
            let logger = createTestLogger<InMemoryRdfClient>()
            let client = InMemoryRdfClient(logger) :> IRdfClient
            let knowledge = {
                KnowledgeUri = "http://tars.ai/ontology#knowledge/test-123"
                Topic = "Test Topic"
                Content = "Test content for RDF functionality"
                Source = "functional-test"
                Confidence = 0.85
                LearnedAt = DateTime.UtcNow
                Tags = ["test"; "rdf"; "functional"]
                Triples = []
            }
            
            // Act
            let! result = client.InsertKnowledge(knowledge)
            
            // Assert
            match result with
            | Ok () -> Assert.True(true)
            | Error err -> Assert.True(false, sprintf "Expected success but got error: %s" err)
        }
    
    [<Fact>]
    let ``RDF client should handle topic search`` () =
        task {
            // Arrange
            let logger = createTestLogger<InMemoryRdfClient>()
            let client = InMemoryRdfClient(logger) :> IRdfClient
            
            // Act
            let! result = client.SearchByTopic("programming")
            
            // Assert
            match result with
            | Ok knowledgeList -> 
                Assert.NotNull(knowledgeList)
                // Should return empty list for non-existent topic in fresh client
                Assert.Empty(knowledgeList)
            | Error err -> Assert.True(false, sprintf "Expected success but got error: %s" err)
        }
    
    [<Fact>]
    let ``RDF client should handle related knowledge queries`` () =
        task {
            // Arrange
            let logger = createTestLogger<InMemoryRdfClient>()
            let client = InMemoryRdfClient(logger) :> IRdfClient
            let testUri = "http://tars.ai/ontology#knowledge/test"
            
            // Act
            let! result = client.GetRelatedKnowledge(testUri)
            
            // Assert
            match result with
            | Ok relatedList -> 
                Assert.NotNull(relatedList)
                // Should return empty list for non-existent URI in fresh client
                Assert.Empty(relatedList)
            | Error err -> Assert.True(false, sprintf "Expected success but got error: %s" err)
        }
    
    [<Fact>]
    let ``RDF client should handle concurrent operations`` () =
        task {
            // Arrange
            let logger = createTestLogger<InMemoryRdfClient>()
            let client = InMemoryRdfClient(logger) :> IRdfClient
            
            let knowledge1 = {
                KnowledgeUri = "http://tars.ai/ontology#knowledge/concurrent-1"
                Topic = "Concurrent Test 1"
                Content = "Testing concurrent RDF operations"
                Source = "concurrent-test"
                Confidence = 0.80
                LearnedAt = DateTime.UtcNow
                Tags = ["concurrent"; "test"]
                Triples = []
            }
            
            let knowledge2 = {
                KnowledgeUri = "http://tars.ai/ontology#knowledge/concurrent-2"
                Topic = "Concurrent Test 2"
                Content = "Another concurrent test"
                Source = "concurrent-test"
                Confidence = 0.75
                LearnedAt = DateTime.UtcNow
                Tags = ["concurrent"; "test"]
                Triples = []
            }
            
            // Act - Run multiple operations concurrently
            let insertTask1 = client.InsertKnowledge(knowledge1)
            let insertTask2 = client.InsertKnowledge(knowledge2)
            let queryTask = client.QueryKnowledge("SELECT * WHERE { ?s ?p ?o } LIMIT 5")
            let searchTask = client.SearchByTopic("concurrent")

            let! insertResult1 = insertTask1
            let! insertResult2 = insertTask2
            let! queryResult = queryTask
            let! searchResult = searchTask

            // Assert - All operations should complete successfully
            match insertResult1, insertResult2, queryResult, searchResult with
            | Ok (), Ok (), Ok _, Ok _ -> Assert.True(true)
            | _ -> Assert.True(false, "One or more concurrent operations failed")
        }
    
    [<Fact>]
    let ``RDF functionality should work with multiple knowledge entries`` () =
        task {
            // Arrange
            let logger = createTestLogger<InMemoryRdfClient>()
            let client = InMemoryRdfClient(logger) :> IRdfClient
            
            let knowledgeEntries = [
                {
                    KnowledgeUri = "http://tars.ai/ontology#knowledge/fsharp"
                    Topic = "F# Programming"
                    Content = "F# is a functional programming language"
                    Source = "multi-test"
                    Confidence = 0.90
                    LearnedAt = DateTime.UtcNow
                    Tags = ["fsharp"; "functional"; "programming"]
                    Triples = []
                }
                {
                    KnowledgeUri = "http://tars.ai/ontology#knowledge/ml"
                    Topic = "Machine Learning"
                    Content = "ML is a subset of artificial intelligence"
                    Source = "multi-test"
                    Confidence = 0.85
                    LearnedAt = DateTime.UtcNow
                    Tags = ["machine_learning"; "ai"; "algorithms"]
                    Triples = []
                }
                {
                    KnowledgeUri = "http://tars.ai/ontology#knowledge/rdf"
                    Topic = "RDF Technology"
                    Content = "RDF enables semantic web technologies"
                    Source = "multi-test"
                    Confidence = 0.95
                    LearnedAt = DateTime.UtcNow
                    Tags = ["rdf"; "semantic_web"; "ontology"]
                    Triples = []
                }
            ]
            
            // Act - Insert all knowledge entries
            let insertTasks = knowledgeEntries |> List.map client.InsertKnowledge
            let! insertResults = Task.WhenAll(insertTasks)
            
            // Verify all insertions succeeded
            for result in insertResults do
                match result with
                | Ok () -> Assert.True(true)
                | Error err -> Assert.True(false, sprintf "Knowledge insertion failed: %s" err)
            
            // Test search functionality
            let! searchResult = client.SearchByTopic("programming")
            match searchResult with
            | Ok searchList -> Assert.NotNull(searchList)
            | Error err -> Assert.True(false, sprintf "Search failed: %s" err)
        }
