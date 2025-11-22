module TarsEngine.FSharp.Core.Tests.Consciousness.Conceptual.ConceptualBlendingTests

open System
open System.Threading.Tasks
open Xunit
open Microsoft.Extensions.Logging
open Microsoft.Extensions.Logging.Abstractions
open TarsEngine.FSharp.Core.Consciousness.Conceptual

/// <summary>
/// Tests for the ConceptualBlending class.
/// </summary>
type ConceptualBlendingTests() =
    let logger = NullLogger<ConceptualBlending>() :> ILogger<ConceptualBlending>
    let conceptualBlending = ConceptualBlending(logger)
    
    /// <summary>
    /// Test that the conceptual blending level is initialized correctly.
    /// </summary>
    [<Fact>]
    member _.``ConceptualBlendingLevel is initialized correctly``() =
        // Assert
        Assert.Equal(0.5, conceptualBlending.ConceptualBlendingLevel)
    
    /// <summary>
    /// Test that Update increases the conceptual blending level.
    /// </summary>
    [<Fact>]
    member _.``Update increases the conceptual blending level``() =
        // Arrange
        let initialLevel = conceptualBlending.ConceptualBlendingLevel
        
        // Act
        let result = conceptualBlending.Update()
        
        // Assert
        Assert.True(result)
        Assert.True(conceptualBlending.ConceptualBlendingLevel >= initialLevel)
    
    /// <summary>
    /// Test that GetRandomConcepts returns the correct number of concepts.
    /// </summary>
    [<Fact>]
    member _.``GetRandomConcepts returns the correct number of concepts``() =
        // Arrange
        let count = 3
        
        // Act
        let concepts = conceptualBlending.GetRandomConcepts(count)
        
        // Assert
        Assert.Equal(count, concepts.Length)
        Assert.All(concepts, fun concept -> Assert.NotEmpty(concept))
    
    /// <summary>
    /// Test that CreateBlendSpace creates a blend space with the correct properties.
    /// </summary>
    [<Fact>]
    member _.``CreateBlendSpace creates a blend space with the correct properties``() =
        // Arrange
        let inputConcepts = ["algorithm"; "pattern"; "abstraction"]
        
        // Act
        let blendSpace = conceptualBlending.CreateBlendSpace(inputConcepts)
        
        // Assert
        Assert.Equal(inputConcepts, blendSpace.InputConcepts)
        Assert.NotEmpty(blendSpace.ConceptMappings)
        Assert.NotEqual(Guid.Empty.ToString(), blendSpace.Id)
        Assert.Equal(DateTime.UtcNow.Date, blendSpace.CreatedAt.Date)
    
    /// <summary>
    /// Test that GenerateConceptualBlendIdea generates an idea with the correct properties.
    /// </summary>
    [<Fact>]
    member _.``GenerateConceptualBlendIdea generates an idea with the correct properties``() =
        // Act
        let idea = conceptualBlending.GenerateConceptualBlendIdea()
        
        // Assert
        Assert.NotEmpty(idea.Description)
        Assert.Equal(CreativeProcessType.ConceptualBlending, idea.ProcessType)
        Assert.Equal(3, idea.Concepts.Length)
        Assert.InRange(idea.Originality, 0.0, 1.0)
        Assert.InRange(idea.Value, 0.0, 1.0)
        Assert.Equal(DateTime.UtcNow.Date, idea.Timestamp.Date)
    
    /// <summary>
    /// Test that GenerateBlendedSolution generates a solution with the correct properties.
    /// </summary>
    [<Fact>]
    member _.``GenerateBlendedSolution generates a solution with the correct properties``() =
        // Arrange
        let problem = "How to improve code quality while maintaining development speed"
        let constraints = ["Must be easy to implement"; "Should not require additional tools"]
        
        // Act
        let solution = conceptualBlending.GenerateBlendedSolution(problem, constraints)
        
        // Assert
        Assert.NotEmpty(solution.Description)
        Assert.Equal(CreativeProcessType.ConceptualBlending, solution.ProcessType)
        Assert.NotEmpty(solution.Concepts)
        Assert.Equal(problem, solution.Problem)
        Assert.Equal(constraints, solution.Constraints)
        Assert.NotEmpty(solution.ImplementationSteps)
        Assert.InRange(solution.Originality, 0.0, 1.0)
        Assert.InRange(solution.Value, 0.0, 1.0)
    
    /// <summary>
    /// Test that EvaluateEmergentStructure returns a value between 0 and 1.
    /// </summary>
    [<Fact>]
    member _.``EvaluateEmergentStructure returns a value between 0 and 1``() =
        // Arrange
        let blendSpace = conceptualBlending.CreateBlendSpace(["algorithm"; "pattern"; "abstraction"])
        
        // Act
        let score = conceptualBlending.EvaluateEmergentStructure(blendSpace.Id)
        
        // Assert
        Assert.InRange(score, 0.0, 1.0)
