namespace TarsEngine.FSharp.Cli.Tests.Commands

open System
open Microsoft.Extensions.Logging
open Xunit
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Services
open TarsEngine.FSharp.Cli.Services.RDF

/// Tests for Semantic Learning Command functionality
module SemanticLearningCommandTests =
    
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
    let ``SemanticLearningCommand should initialize correctly`` () =
        // Arrange
        let logger = createTestLogger<SemanticLearningCommand>()
        let learningService = createTestLearningMemoryService()
        
        // Act
        let command = SemanticLearningCommand(logger, learningService)
        
        // Assert
        Assert.NotNull(command)
        Assert.Equal("semantic", command.Name)
        Assert.NotEmpty(command.Description)
    
    [<Fact>]
    let ``SemanticLearningCommand should implement ICommand interface`` () =
        // Arrange
        let logger = createTestLogger<SemanticLearningCommand>()
        let learningService = createTestLearningMemoryService()
        let command = SemanticLearningCommand(logger, learningService)
        
        // Act & Assert
        Assert.IsAssignableFrom<ICommand>(command)
    
    [<Fact>]
    let ``Execute with no arguments should show overview`` () =
        task {
            // Arrange
            let logger = createTestLogger<SemanticLearningCommand>()
            let learningService = createTestLearningMemoryService()
            let command = SemanticLearningCommand(logger, learningService)
            let args = [||]
            
            // Act
            let! result = command.Execute(args)
            
            // Assert
            Assert.Equal(0, result) // Should succeed
        }
    
    [<Fact>]
    let ``Execute with 'patterns' argument should discover patterns`` () =
        task {
            // Arrange
            let logger = createTestLogger<SemanticLearningCommand>()
            let learningService = createTestLearningMemoryService()
            let command = SemanticLearningCommand(logger, learningService)
            let args = [| "patterns" |]
            
            // Act
            let! result = command.Execute(args)
            
            // Assert
            Assert.Equal(0, result) // Should succeed
        }
    
    [<Fact>]
    let ``Execute with 'infer' argument should infer knowledge`` () =
        task {
            // Arrange
            let logger = createTestLogger<SemanticLearningCommand>()
            let learningService = createTestLearningMemoryService()
            let command = SemanticLearningCommand(logger, learningService)
            let args = [| "infer" |]
            
            // Act
            let! result = command.Execute(args)
            
            // Assert
            Assert.Equal(0, result) // Should succeed
        }
    
    [<Fact>]
    let ``Execute with 'train' argument should run training`` () =
        task {
            // Arrange
            let logger = createTestLogger<SemanticLearningCommand>()
            let learningService = createTestLearningMemoryService()
            let command = SemanticLearningCommand(logger, learningService)
            let args = [| "train" |]
            
            // Act
            let! result = command.Execute(args)
            
            // Assert
            Assert.Equal(0, result) // Should succeed
        }
    
    [<Fact>]
    let ``Execute with 'train' and iterations should run specified iterations`` () =
        task {
            // Arrange
            let logger = createTestLogger<SemanticLearningCommand>()
            let learningService = createTestLearningMemoryService()
            let command = SemanticLearningCommand(logger, learningService)
            let args = [| "train"; "--iterations"; "5" |]
            
            // Act
            let! result = command.Execute(args)
            
            // Assert
            Assert.Equal(0, result) // Should succeed
        }
    
    [<Fact>]
    let ``Execute with 'analyze' argument should analyze relationships`` () =
        task {
            // Arrange
            let logger = createTestLogger<SemanticLearningCommand>()
            let learningService = createTestLearningMemoryService()
            let command = SemanticLearningCommand(logger, learningService)
            let args = [| "analyze"; "--topic"; "machine learning" |]
            
            // Act
            let! result = command.Execute(args)
            
            // Assert
            Assert.Equal(0, result) // Should succeed
        }
    
    [<Fact>]
    let ``Execute with invalid argument should show help`` () =
        task {
            // Arrange
            let logger = createTestLogger<SemanticLearningCommand>()
            let learningService = createTestLearningMemoryService()
            let command = SemanticLearningCommand(logger, learningService)
            let args = [| "invalid-command" |]
            
            // Act
            let! result = command.Execute(args)
            
            // Assert
            Assert.Equal(1, result) // Should return error code
        }
    
    [<Fact>]
    let ``Execute with 'ontology' argument should manage ontology`` () =
        task {
            // Arrange
            let logger = createTestLogger<SemanticLearningCommand>()
            let learningService = createTestLearningMemoryService()
            let command = SemanticLearningCommand(logger, learningService)
            let args = [| "ontology"; "--export" |]
            
            // Act
            let! result = command.Execute(args)
            
            // Assert
            Assert.Equal(0, result) // Should succeed
        }
