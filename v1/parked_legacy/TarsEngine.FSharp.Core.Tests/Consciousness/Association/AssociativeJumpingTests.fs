module TarsEngine.FSharp.Core.Tests.Consciousness.Association.AssociativeJumpingTests

open System
open System.Threading.Tasks
open Xunit
open Microsoft.Extensions.Logging
open Microsoft.Extensions.Logging.Abstractions
open TarsEngine.FSharp.Core.Consciousness.Association

/// <summary>
/// Tests for the AssociativeJumping class.
/// </summary>
type AssociativeJumpingTests() =
    let logger = NullLogger<AssociativeJumping>() :> ILogger<AssociativeJumping>
    let associativeJumping = AssociativeJumping(logger)
    
    /// <summary>
    /// Test that the associative jumping level is initialized correctly.
    /// </summary>
    [<Fact>]
    member _.``AssociativeJumpingLevel is initialized correctly``() =
        // Assert
        Assert.Equal(0.5, associativeJumping.AssociativeJumpingLevel)
    
    /// <summary>
    /// Test that Update increases the associative jumping level.
    /// </summary>
    [<Fact>]
    member _.``Update increases the associative jumping level``() =
        // Arrange
        let initialLevel = associativeJumping.AssociativeJumpingLevel
        
        // Act
        let result = associativeJumping.Update()
        
        // Assert
        Assert.True(result)
        Assert.True(associativeJumping.AssociativeJumpingLevel >= initialLevel)
    
    /// <summary>
    /// Test that GetRandomConcept returns a non-empty string.
    /// </summary>
    [<Fact>]
    member _.``GetRandomConcept returns a non-empty string``() =
        // Act
        let concept = associativeJumping.GetRandomConcept()
        
        // Assert
        Assert.NotEmpty(concept)
    
    /// <summary>
    /// Test that GetRandomConcept with a category returns a concept from that category.
    /// </summary>
    [<Fact>]
    member _.``GetRandomConcept with a category returns a concept from that category``() =
        // Arrange
        let category = "Programming"
        
        // Act
        let concept = associativeJumping.GetRandomConcept(category)
        
        // Assert
        Assert.NotEmpty(concept)
    
    /// <summary>
    /// Test that GetAssociatedConcepts returns a non-empty map for a known concept.
    /// </summary>
    [<Fact>]
    member _.``GetAssociatedConcepts returns a non-empty map for a known concept``() =
        // Arrange
        let concept = "algorithm"
        
        // Act
        let associations = associativeJumping.GetAssociatedConcepts(concept)
        
        // Assert
        Assert.NotEmpty(associations)
    
    /// <summary>
    /// Test that GetAssociatedConcepts returns an empty map for an unknown concept.
    /// </summary>
    [<Fact>]
    member _.``GetAssociatedConcepts returns an empty map for an unknown concept``() =
        // Arrange
        let concept = "nonexistent"
        
        // Act
        let associations = associativeJumping.GetAssociatedConcepts(concept)
        
        // Assert
        Assert.Empty(associations)
    
    /// <summary>
    /// Test that PerformAssociativeJump returns a non-empty list.
    /// </summary>
    [<Fact>]
    member _.``PerformAssociativeJump returns a non-empty list``() =
        // Arrange
        let startConcept = "algorithm"
        let jumpDistance = 3
        
        // Act
        let jumpPath = associativeJumping.PerformAssociativeJump(startConcept, jumpDistance)
        
        // Assert
        Assert.NotEmpty(jumpPath)
        Assert.Equal(startConcept, jumpPath.[0])
    
    /// <summary>
    /// Test that CalculateUnexpectedness returns a value between 0 and 1.
    /// </summary>
    [<Fact>]
    member _.``CalculateUnexpectedness returns a value between 0 and 1``() =
        // Arrange
        let jumpPath = ["algorithm"; "pattern"; "abstraction"]
        
        // Act
        let unexpectedness = associativeJumping.CalculateUnexpectedness(jumpPath)
        
        // Assert
        Assert.InRange(unexpectedness, 0.0, 1.0)
    
    /// <summary>
    /// Test that GenerateAssociativeThought returns a thought with the correct method.
    /// </summary>
    [<Fact>]
    member _.``GenerateAssociativeThought returns a thought with the correct method``() =
        // Arrange
        let serendipityLevel = 0.5
        
        // Act
        let thought = associativeJumping.GenerateAssociativeThought(serendipityLevel)
        
        // Assert
        Assert.Equal(ThoughtGenerationMethod.AssociativeJumping, thought.Method)
        Assert.NotEmpty(thought.Content)
        Assert.InRange(thought.Significance, 0.0, 1.0)
        Assert.Equal("AssociativeJumping", thought.Source)
    
    /// <summary>
    /// Test that GenerateAssociativeThoughts returns the correct number of thoughts.
    /// </summary>
    [<Fact>]
    member _.``GenerateAssociativeThoughts returns the correct number of thoughts``() =
        // Arrange
        let count = 3
        let serendipityLevel = 0.5
        
        // Act
        let thoughts = associativeJumping.GenerateAssociativeThoughts(count, serendipityLevel)
        
        // Assert
        Assert.Equal(count, thoughts.Length)
        Assert.All(thoughts, fun thought -> Assert.Equal(ThoughtGenerationMethod.AssociativeJumping, thought.Method))
    
    /// <summary>
    /// Test that EvaluateThought returns a value between 0 and 1.
    /// </summary>
    [<Fact>]
    member _.``EvaluateThought returns a value between 0 and 1``() =
        // Arrange
        let thought = associativeJumping.GenerateAssociativeThought(0.5)
        
        // Act
        let score = associativeJumping.EvaluateThought(thought)
        
        // Assert
        Assert.InRange(score, 0.0, 1.0)
