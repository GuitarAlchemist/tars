module TarsEngine.FSharp.Core.Tests.Consciousness.Core.ConsciousnessCoreTests

open System
open Xunit
open TarsEngine.FSharp.Core.Consciousness.Core

/// <summary>
/// Tests for the Consciousness.Core.ConsciousnessCore module.
/// </summary>
type ConsciousnessCoreTests() =
    /// <summary>
    /// Test that ConsciousnessCore.createConcept creates a concept with the given name and description.
    /// </summary>
    [<Fact>]
    member _.``ConsciousnessCore.createConcept creates a concept with the given name and description``() =
        // Arrange
        let name = "Test Concept"
        let description = Some "A test concept"
        
        // Act
        let concept = ConsciousnessCore.createConcept name description
        
        // Assert
        Assert.Equal(name, concept.Name)
        Assert.Equal(description, concept.Description)
        Assert.Equal(0, concept.ActivationCount)
        Assert.Equal(None, concept.LastActivationTime)
        Assert.Empty(concept.Metadata)
    
    /// <summary>
    /// Test that ConsciousnessCore.activateConcept activates a concept with the given strength, context, and trigger.
    /// </summary>
    [<Fact>]
    member _.``ConsciousnessCore.activateConcept activates a concept with the given strength, context, and trigger``() =
        // Arrange
        let name = "Test Concept"
        let description = Some "A test concept"
        let concept = ConsciousnessCore.createConcept name description
        let activationStrength = 0.8
        let context = Some "Test context"
        let trigger = Some "Test trigger"
        
        // Act
        let activation = ConsciousnessCore.activateConcept concept activationStrength context trigger
        
        // Assert
        Assert.Equal(concept.Id, activation.Concept.Id)
        Assert.Equal(name, activation.Concept.Name)
        Assert.Equal(description, activation.Concept.Description)
        Assert.Equal(1, activation.Concept.ActivationCount)
        Assert.NotEqual(None, activation.Concept.LastActivationTime)
        Assert.Equal(activationStrength, activation.ActivationStrength)
        Assert.Equal(context, activation.Context)
        Assert.Equal(trigger, activation.Trigger)
        Assert.Empty(activation.Metadata)
