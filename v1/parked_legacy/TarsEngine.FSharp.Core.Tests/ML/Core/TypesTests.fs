module TarsEngine.FSharp.Core.Tests.ML.Core.TypesTests

open System
open Xunit
open TarsEngine.FSharp.Core.ML.Core

/// <summary>
/// Tests for the ML.Core.Types module.
/// </summary>
type TypesTests() =
    /// <summary>
    /// Test that a Feature can be created with valid values.
    /// </summary>
    [<Fact>]
    member _.``Feature can be created with valid values``() =
        // Arrange
        let name = "Test Feature"
        let featureType = typeof<int>
        let importance = Some 0.8
        let isCategorical = false
        let possibleValues = None
        let metadata = Map.empty
        
        // Act
        let feature = {
            Name = name
            Type = featureType
            Importance = importance
            IsCategorical = isCategorical
            PossibleValues = possibleValues
            Metadata = metadata
        }
        
        // Assert
        Assert.Equal(name, feature.Name)
        Assert.Equal(featureType, feature.Type)
        Assert.Equal(importance, feature.Importance)
        Assert.Equal(isCategorical, feature.IsCategorical)
        Assert.Equal(possibleValues, feature.PossibleValues)
        Assert.Equal(metadata, feature.Metadata)
    
    /// <summary>
    /// Test that a Label can be created with valid values.
    /// </summary>
    [<Fact>]
    member _.``Label can be created with valid values``() =
        // Arrange
        let name = "Test Label"
        let labelType = typeof<string>
        let isCategorical = true
        let possibleValues = Some ["A"; "B"; "C"]
        let metadata = Map.empty
        
        // Act
        let label = {
            Name = name
            Type = labelType
            IsCategorical = isCategorical
            PossibleValues = possibleValues
            Metadata = metadata
        }
        
        // Assert
        Assert.Equal(name, label.Name)
        Assert.Equal(labelType, label.Type)
        Assert.Equal(isCategorical, label.IsCategorical)
        Assert.Equal(possibleValues, label.PossibleValues)
        Assert.Equal(metadata, label.Metadata)
    
    /// <summary>
    /// Test that a Model can be created with valid values.
    /// </summary>
    [<Fact>]
    member _.``Model can be created with valid values``() =
        // Arrange
        let id = Guid.NewGuid()
        let name = "Test Model"
        let modelType = ModelType.Classification
        let status = ModelStatus.Trained
        let features = []
        let label = None
        let hyperparameters = { Parameters = Map.empty; Metadata = Map.empty }
        let metrics = None
        let creationTime = DateTime.UtcNow
        let lastTrainingTime = Some DateTime.UtcNow
        let version = "1.0.0"
        let metadata = Map.empty
        
        // Act
        let model = {
            Id = id
            Name = name
            Type = modelType
            Status = status
            Features = features
            Label = label
            Hyperparameters = hyperparameters
            Metrics = metrics
            CreationTime = creationTime
            LastTrainingTime = lastTrainingTime
            Version = version
            Metadata = metadata
        }
        
        // Assert
        Assert.Equal(id, model.Id)
        Assert.Equal(name, model.Name)
        Assert.Equal(modelType, model.Type)
        Assert.Equal(status, model.Status)
        Assert.Equal(features, model.Features)
        Assert.Equal(label, model.Label)
        Assert.Equal(hyperparameters, model.Hyperparameters)
        Assert.Equal(metrics, model.Metrics)
        Assert.Equal(creationTime, model.CreationTime)
        Assert.Equal(lastTrainingTime, model.LastTrainingTime)
        Assert.Equal(version, model.Version)
        Assert.Equal(metadata, model.Metadata)
