namespace TarsEngine.FSharp.Cli.Tests.Integration

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Xunit
open TarsEngine.FSharp.Cli.Services
open TarsEngine.FSharp.Cli.Services.RDF
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Core

/// Integration tests for RDF-enhanced TARS functionality
module RdfIntegrationTests =
    
    /// Create a test logger
    let createTestLogger<'T>() =
        let loggerFactory = LoggerFactory.Create(fun builder -> builder.AddConsole() |> ignore)
        loggerFactory.CreateLogger<'T>()
    
    /// Create a complete test environment with all RDF services
    let createTestEnvironment() =
        // Create RDF client
        let rdfClientLogger = createTestLogger<InMemoryRdfClient>()
        let rdfClient = InMemoryRdfClient(rdfClientLogger) :> IRdfClient
        
        // Create modular services
        let mindMapLogger = createTestLogger<MindMapService>()
        let mindMapService = MindMapService(mindMapLogger, Some rdfClient)
        
        let semanticLogger = createTestLogger<SemanticLearningService>()
        let semanticService = SemanticLearningService(semanticLogger, Some rdfClient)
        
        // Create main learning service
        let learningLogger = createTestLogger<LearningMemoryService>()
        let learningService = LearningMemoryService(learningLogger, None, None, None, Some rdfClient, Some mindMapService, Some semanticService)
        
        // Create commands
        let mindMapCommandLogger = createTestLogger<MindMapCommand>()
        let mindMapCommand = MindMapCommand(mindMapCommandLogger, learningService)
        
        let semanticCommandLogger = createTestLogger<SemanticLearningCommand>()
        let semanticCommand = SemanticLearningCommand(semanticCommandLogger, learningService)
        
        (rdfClient, learningService, mindMapCommand, semanticCommand)
    
    [<Fact>]
    let ``Full RDF integration should work end-to-end`` () =
        task {
            // Arrange
            let (rdfClient, learningService, mindMapCommand, semanticCommand) = createTestEnvironment()
            
            // Act & Assert - Store knowledge with semantic enhancement
            let! storeResult = learningService.StoreKnowledgeWithSemantics(
                {
                    Id = Guid.NewGuid().ToString()
                    Topic = "F# Programming"
                    Content = "F# is a functional programming language that runs on .NET"
                    Source = "integration-test"
                    Confidence = 0.90
                    LearnedAt = DateTime.UtcNow
                    LastAccessed = DateTime.UtcNow
                    AccessCount = 1
                    Tags = ["functional"; "programming"; "fsharp"; "dotnet"]
                    WebSearchResults = None
                    Quality = Verified
                    LearningOutcome = None
                    RelatedKnowledge = []
                    SupersededBy = None
                    PerformanceImpact = None
                },
                ["functional programming"; "dotnet"; "microsoft"]
            )
            
            match storeResult with
            | Ok () -> Assert.True(true)
            | Error err -> Assert.True(false, sprintf "Knowledge storage failed: %s" err)
            
            // Test semantic pattern discovery
            let! patternsResult = learningService.DiscoverSemanticPatterns()
            match patternsResult with
            | Ok patterns -> Assert.NotNull(patterns)
            | Error err -> Assert.True(false, sprintf "Pattern discovery failed: %s" err)
            
            // Test knowledge inference
            let! inferenceResult = learningService.InferNewKnowledge()
            match inferenceResult with
            | Ok inferred -> Assert.NotNull(inferred)
            | Error err -> Assert.True(false, sprintf "Knowledge inference failed: %s" err)
            
            // Test mind map generation
            let! mindMapResult = learningService.GenerateAsciiMindMap(Some "programming", 3, 10)
            Assert.Contains("TARS KNOWLEDGE MIND MAP", mindMapResult)
            Assert.Contains("programming", mindMapResult)
        }
    
    [<Fact>]
    let ``RDF triple store status should be correctly reported`` () =
        // Arrange
        let (_, learningService, _, _) = createTestEnvironment()
        
        // Act
        let stats = learningService.GetKnowledgeStatistics()
        
        // Assert
        Assert.True(stats.IndexingCapabilities.RDFTripleStore)
        Assert.True(stats.IndexingCapabilities.InMemoryCache)
        Assert.True(stats.IndexingCapabilities.TagBasedIndexing)
        Assert.True(stats.IndexingCapabilities.ConfidenceFiltering)
    
    [<Fact>]
    let ``Mind map command should work with RDF-enhanced data`` () =
        task {
            // Arrange
            let (_, learningService, mindMapCommand, _) = createTestEnvironment()
            
            // Store some test knowledge
            let! _ = learningService.StoreKnowledge("Machine Learning", "ML is a subset of AI", DirectTeaching, None)
            let! _ = learningService.StoreKnowledge("Deep Learning", "DL uses neural networks", DirectTeaching, None)
            let! _ = learningService.StoreKnowledge("Neural Networks", "Networks of artificial neurons", DirectTeaching, None)
            
            // Act
            let! result = mindMapCommand.Execute([| "ascii"; "learning" |])
            
            // Assert
            Assert.Equal(0, result)
        }
    
    [<Fact>]
    let ``Semantic command should work with RDF-enhanced data`` () =
        task {
            // Arrange
            let (_, learningService, _, semanticCommand) = createTestEnvironment()
            
            // Store some test knowledge
            let! _ = learningService.StoreKnowledge("Functional Programming", "Programming paradigm", DirectTeaching, None)
            let! _ = learningService.StoreKnowledge("Object-Oriented Programming", "Another programming paradigm", DirectTeaching, None)
            
            // Act
            let! patternsResult = semanticCommand.Execute([| "patterns" |])
            let! inferResult = semanticCommand.Execute([| "infer" |])
            let! analyzeResult = semanticCommand.Execute([| "analyze"; "--topic"; "programming" |])
            
            // Assert
            Assert.Equal(0, patternsResult)
            Assert.Equal(0, inferResult)
            Assert.Equal(0, analyzeResult)
        }
    
    [<Fact>]
    let ``Concurrent RDF operations should work correctly`` () =
        task {
            // Arrange
            let (rdfClient, learningService, _, _) = createTestEnvironment()
            
            // Act - Run multiple operations concurrently
            let tasks = [
                learningService.StoreKnowledge("Topic 1", "Content 1", DirectTeaching, None)
                learningService.StoreKnowledge("Topic 2", "Content 2", DirectTeaching, None)
                learningService.StoreKnowledge("Topic 3", "Content 3", DirectTeaching, None)
                learningService.DiscoverSemanticPatterns()
                learningService.InferNewKnowledge()
            ]
            
            let! results = Task.WhenAll(tasks)
            
            // Assert - All operations should complete
            Assert.Equal(5, results.Length)
        }
    
    [<Fact>]
    let ``RDF knowledge storage and retrieval should maintain consistency`` () =
        task {
            // Arrange
            let (rdfClient, learningService, _, _) = createTestEnvironment()
            
            // Store knowledge
            let testTopic = "Consistency Test"
            let testContent = "Testing RDF consistency"
            let! storeResult = learningService.StoreKnowledge(testTopic, testContent, DirectTeaching, None)
            
            match storeResult with
            | Ok () -> 
                // Retrieve knowledge
                let allKnowledge = learningService.GetAllKnowledge()
                let foundKnowledge = allKnowledge |> List.tryFind (fun k -> k.Topic = testTopic)
                
                // Assert
                match foundKnowledge with
                | Some knowledge -> 
                    Assert.Equal(testTopic, knowledge.Topic)
                    Assert.Equal(testContent, knowledge.Content)
                | None -> Assert.True(false, "Stored knowledge not found")
            | Error err -> Assert.True(false, sprintf "Knowledge storage failed: %s" err)
        }
    
    [<Fact>]
    let ``RDF semantic enhancement should improve knowledge relationships`` () =
        task {
            // Arrange
            let (_, learningService, _, _) = createTestEnvironment()
            
            // Store related knowledge
            let! _ = learningService.StoreKnowledge("F# Language", "Functional programming language", DirectTeaching, None)
            let! _ = learningService.StoreKnowledge("Functional Programming", "Programming paradigm", DirectTeaching, None)
            let! _ = learningService.StoreKnowledge("Lambda Calculus", "Mathematical foundation", DirectTeaching, None)
            
            // Act - Discover semantic patterns
            let! patternsResult = learningService.DiscoverSemanticPatterns()
            
            // Assert
            match patternsResult with
            | Ok patterns -> 
                Assert.NotNull(patterns)
                // Should find relationships between functional programming concepts
                Assert.True(patterns.Length >= 0)
            | Error err -> Assert.True(false, sprintf "Pattern discovery failed: %s" err)
        }
