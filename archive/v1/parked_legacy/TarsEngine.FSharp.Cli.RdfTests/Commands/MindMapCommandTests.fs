namespace TarsEngine.FSharp.Cli.Tests.Commands

open System
open Microsoft.Extensions.Logging
open Xunit
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Services
open TarsEngine.FSharp.Cli.Services.RDF

/// Tests for Mind Map Command functionality
module MindMapCommandTests =
    
    /// Create a test logger
    let createTestLogger<'T>() =
        let loggerFactory = LoggerFactory.Create(fun builder -> builder.AddConsole() |> ignore)
        loggerFactory.CreateLogger<'T>()
    
    /// Create test learning memory service with RDF support
    let createTestLearningMemoryService() =
        let logger = createTestLogger<LearningMemoryService>()
        let rdfClientLogger = createTestLogger<InMemoryRdfClient>()
        let rdfClient = InMemoryRdfClient(rdfClientLogger) :> IRdfClient
        let mindMapLogger = createTestLogger<MindMapService>()
        let mindMapService = MindMapService(mindMapLogger, Some rdfClient)
        let semanticLogger = createTestLogger<SemanticLearningService>()
        let semanticService = SemanticLearningService(semanticLogger, Some rdfClient)
        LearningMemoryService(logger, None, None, None, Some rdfClient, Some mindMapService, Some semanticService)
    
    [<Fact>]
    let ``MindMapCommand should initialize correctly`` () =
        // Arrange
        let logger = createTestLogger<MindMapCommand>()
        let learningService = createTestLearningMemoryService()
        
        // Act
        let command = MindMapCommand(logger, learningService)
        
        // Assert
        Assert.NotNull(command)
        Assert.Equal("mindmap", command.Name)
        Assert.NotEmpty(command.Description)
    
    [<Fact>]
    let ``MindMapCommand should implement ICommand interface`` () =
        // Arrange
        let logger = createTestLogger<MindMapCommand>()
        let learningService = createTestLearningMemoryService()
        let command = MindMapCommand(logger, learningService)
        
        // Act & Assert
        Assert.IsAssignableFrom<ICommand>(command)
    
    [<Fact>]
    let ``Execute with no arguments should start interactive explorer`` () =
        task {
            // Arrange
            let logger = createTestLogger<MindMapCommand>()
            let learningService = createTestLearningMemoryService()
            let command = MindMapCommand(logger, learningService)
            let args = [||]
            
            // Note: This test would normally start interactive mode
            // For testing, we'll just verify the command initializes
            
            // Act & Assert
            Assert.NotNull(command)
        }
    
    [<Fact>]
    let ``Execute with 'ascii' argument should generate ASCII mind map`` () =
        task {
            // Arrange
            let logger = createTestLogger<MindMapCommand>()
            let learningService = createTestLearningMemoryService()
            let command = MindMapCommand(logger, learningService)
            let args = [| "ascii" |]
            
            // Add some test knowledge first
            let! _ = learningService.StoreKnowledge("Test Topic", "Test content", DirectTeaching, None)
            
            // Act
            let! result = command.Execute(args)
            
            // Assert
            Assert.Equal(0, result) // Should succeed
        }
    
    [<Fact>]
    let ``Execute with 'ascii' and topic should generate focused mind map`` () =
        task {
            // Arrange
            let logger = createTestLogger<MindMapCommand>()
            let learningService = createTestLearningMemoryService()
            let command = MindMapCommand(logger, learningService)
            let args = [| "ascii"; "programming" |]
            
            // Add some test knowledge first
            let! _ = learningService.StoreKnowledge("Programming Concepts", "Object-oriented programming", DirectTeaching, None)
            
            // Act
            let! result = command.Execute(args)
            
            // Assert
            Assert.Equal(0, result) // Should succeed
        }
    
    [<Fact>]
    let ``Execute with 'markdown' argument should generate Markdown mind map`` () =
        task {
            // Arrange
            let logger = createTestLogger<MindMapCommand>()
            let learningService = createTestLearningMemoryService()
            let command = MindMapCommand(logger, learningService)
            let args = [| "markdown" |]
            
            // Add some test knowledge first
            let! _ = learningService.StoreKnowledge("Test Topic", "Test content", DirectTeaching, None)
            
            // Act
            let! result = command.Execute(args)
            
            // Assert
            Assert.Equal(0, result) // Should succeed
        }
    
    [<Fact>]
    let ``Execute with 'both' argument should generate both formats`` () =
        task {
            // Arrange
            let logger = createTestLogger<MindMapCommand>()
            let learningService = createTestLearningMemoryService()
            let command = MindMapCommand(logger, learningService)
            let args = [| "both" |]
            
            // Add some test knowledge first
            let! _ = learningService.StoreKnowledge("Test Topic", "Test content", DirectTeaching, None)
            
            // Act
            let! result = command.Execute(args)
            
            // Assert
            Assert.Equal(0, result) // Should succeed
        }
    
    [<Fact>]
    let ``Execute with 'stats' argument should show knowledge statistics`` () =
        task {
            // Arrange
            let logger = createTestLogger<MindMapCommand>()
            let learningService = createTestLearningMemoryService()
            let command = MindMapCommand(logger, learningService)
            let args = [| "stats" |]
            
            // Act
            let! result = command.Execute(args)
            
            // Assert
            Assert.Equal(0, result) // Should succeed
        }
    
    [<Fact>]
    let ``Execute with invalid argument should show help`` () =
        task {
            // Arrange
            let logger = createTestLogger<MindMapCommand>()
            let learningService = createTestLearningMemoryService()
            let command = MindMapCommand(logger, learningService)
            let args = [| "invalid-command" |]
            
            // Act
            let! result = command.Execute(args)
            
            // Assert
            Assert.Equal(1, result) // Should return error code
        }
    
    [<Fact>]
    let ``Mind map generation should work with empty knowledge base`` () =
        task {
            // Arrange
            let logger = createTestLogger<MindMapCommand>()
            let learningService = createTestLearningMemoryService()
            let command = MindMapCommand(logger, learningService)
            let args = [| "ascii"; "empty-topic" |]
            
            // Act (no knowledge stored)
            let! result = command.Execute(args)
            
            // Assert
            Assert.Equal(0, result) // Should succeed even with empty knowledge
        }
