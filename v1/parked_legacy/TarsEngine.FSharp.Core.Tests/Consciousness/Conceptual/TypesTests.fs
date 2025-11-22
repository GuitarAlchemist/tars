module TarsEngine.FSharp.Core.Tests.Consciousness.Conceptual.TypesTests

open System
open Xunit
open TarsEngine.FSharp.Core.Consciousness.Conceptual

/// <summary>
/// Tests for the Consciousness.Conceptual.Types module.
/// </summary>
type TypesTests() =
    /// <summary>
    /// Test that a ConceptualModel can be created with valid values.
    /// </summary>
    [<Fact>]
    member _.``ConceptualModel can be created with valid values``() =
        // Arrange
        let id = Guid.NewGuid()
        let name = "Test Model"
        let description = Some "A test conceptual model"
        let creationTime = DateTime.UtcNow
        let lastModificationTime = DateTime.UtcNow
        let concepts = []
        let relationships = []
        let metadata = Map.empty
        
        // Act
        let model = {
            Id = id
            Name = name
            Description = description
            CreationTime = creationTime
            LastModificationTime = lastModificationTime
            Concepts = concepts
            Relationships = relationships
            Metadata = metadata
        }
        
        // Assert
        Assert.Equal(id, model.Id)
        Assert.Equal(name, model.Name)
        Assert.Equal(description, model.Description)
        Assert.Equal(creationTime, model.CreationTime)
        Assert.Equal(lastModificationTime, model.LastModificationTime)
        Assert.Equal(concepts, model.Concepts)
        Assert.Equal(relationships, model.Relationships)
        Assert.Equal(metadata, model.Metadata)
    
    /// <summary>
    /// Test that a ConceptualRelationship can be created with valid values.
    /// </summary>
    [<Fact>]
    member _.``ConceptualRelationship can be created with valid values``() =
        // Arrange
        let id = Guid.NewGuid()
        let sourceId = Guid.NewGuid()
        let targetId = Guid.NewGuid()
        let relationshipType = RelationshipType.IsA
        let strength = RelationshipStrength.Strong
        let description = Some "A test relationship"
        let creationTime = DateTime.UtcNow
        let metadata = Map.empty
        
        // Act
        let relationship = {
            Id = id
            SourceId = sourceId
            TargetId = targetId
            Type = relationshipType
            Strength = strength
            Description = description
            CreationTime = creationTime
            Metadata = metadata
        }
        
        // Assert
        Assert.Equal(id, relationship.Id)
        Assert.Equal(sourceId, relationship.SourceId)
        Assert.Equal(targetId, relationship.TargetId)
        Assert.Equal(relationshipType, relationship.Type)
        Assert.Equal(strength, relationship.Strength)
        Assert.Equal(description, relationship.Description)
        Assert.Equal(creationTime, relationship.CreationTime)
        Assert.Equal(metadata, relationship.Metadata)
