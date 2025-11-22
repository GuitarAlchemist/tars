module TarsEngine.FSharp.Core.Tests.Consciousness.Core.TypesTests

open System
open Xunit
open TarsEngine.FSharp.Core.Consciousness.Core

/// <summary>
/// Tests for the Consciousness.Core.Types module.
/// </summary>
type TypesTests() =
    /// <summary>
    /// Test that a Concept can be created with valid values.
    /// </summary>
    [<Fact>]
    member _.``Concept can be created with valid values``() =
        // Arrange
        let id = Guid.NewGuid()
        let name = "Test Concept"
        let description = Some "A test concept"
        let creationTime = DateTime.UtcNow
        let lastActivationTime = Some DateTime.UtcNow
        let activationCount = 10
        let metadata = Map.empty
        
        // Act
        let concept = {
            Id = id
            Name = name
            Description = description
            CreationTime = creationTime
            LastActivationTime = lastActivationTime
            ActivationCount = activationCount
            Metadata = metadata
        }
        
        // Assert
        Assert.Equal(id, concept.Id)
        Assert.Equal(name, concept.Name)
        Assert.Equal(description, concept.Description)
        Assert.Equal(creationTime, concept.CreationTime)
        Assert.Equal(lastActivationTime, concept.LastActivationTime)
        Assert.Equal(activationCount, concept.ActivationCount)
        Assert.Equal(metadata, concept.Metadata)
    
    /// <summary>
    /// Test that a ConceptActivation can be created with valid values.
    /// </summary>
    [<Fact>]
    member _.``ConceptActivation can be created with valid values``() =
        // Arrange
        let id = Guid.NewGuid()
        let name = "Test Concept"
        let description = Some "A test concept"
        let creationTime = DateTime.UtcNow
        let lastActivationTime = Some DateTime.UtcNow
        let activationCount = 10
        let metadata = Map.empty
        
        let concept = {
            Id = id
            Name = name
            Description = description
            CreationTime = creationTime
            LastActivationTime = lastActivationTime
            ActivationCount = activationCount
            Metadata = metadata
        }
        
        let activationTime = DateTime.UtcNow
        let activationStrength = 0.8
        let context = Some "Test context"
        let trigger = Some "Test trigger"
        let activationMetadata = Map.empty
        
        // Act
        let activation = {
            Concept = concept
            ActivationTime = activationTime
            ActivationStrength = activationStrength
            Context = context
            Trigger = trigger
            Metadata = activationMetadata
        }
        
        // Assert
        Assert.Equal(concept, activation.Concept)
        Assert.Equal(activationTime, activation.ActivationTime)
        Assert.Equal(activationStrength, activation.ActivationStrength)
        Assert.Equal(context, activation.Context)
        Assert.Equal(trigger, activation.Trigger)
        Assert.Equal(activationMetadata, activation.Metadata)
