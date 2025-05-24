module TarsEngine.FSharp.Core.Tests.ML.MLServiceTests

open System
open System.IO
open System.Collections.Generic
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Xunit
open TarsEngine.FSharp.Core.ML.Core
open TarsEngine.FSharp.Core.ML.Services

/// <summary>
/// Mock logger for testing.
/// </summary>
type MockLogger<'T>() =
    interface ILogger<'T> with
        member _.BeginScope<'TState>(state: 'TState) = { new IDisposable with member _.Dispose() = () }
        member _.IsEnabled(logLevel: LogLevel) = true
        member _.Log<'TState>(logLevel: LogLevel, eventId: EventId, state: 'TState, ex: exn, formatter: Func<'TState, exn, string>) = ()

/// <summary>
/// Tests for the MLService class.
/// </summary>
type MLServiceTests() =
    let tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString())
    let options = { 
        MLFrameworkOptionsDefaults.defaultOptions with 
            ModelBasePath = Some tempDir 
    }
    
    do
        // Create temp directory for tests
        Directory.CreateDirectory(tempDir) |> ignore
    
    interface IDisposable with
        member _.Dispose() =
            // Clean up temp directory after tests
            if Directory.Exists(tempDir) then
                Directory.Delete(tempDir, true)
    
    /// <summary>
    /// Test that MLService can be created.
    /// </summary>
    [<Fact>]
    member _.``MLService can be created``() =
        // Arrange
        let frameworkLogger = MockLogger<MLFramework>()
        let framework = new MLFramework(frameworkLogger :> ILogger<MLFramework>, options)
        let serviceLogger = MockLogger<MLService>()
        
        // Act
        let service = new MLService(serviceLogger :> ILogger<MLService>, framework)
        
        // Assert
        Assert.NotNull(service)
        Assert.IsAssignableFrom<IMLService>(service)
    
    /// <summary>
    /// Test that MLService can create a model.
    /// </summary>
    [<Fact>]
    member _.``MLService can create a model``() =
        // Arrange
        let frameworkLogger = MockLogger<MLFramework>()
        let framework = new MLFramework(frameworkLogger :> ILogger<MLFramework>, options)
        let serviceLogger = MockLogger<MLService>()
        let service = new MLService(serviceLogger :> ILogger<MLService>, framework)
        
        // Act
        let features = [
            { Name = "Feature1"; Type = typeof<float32>; Importance = None; IsCategorical = false; PossibleValues = None; Metadata = Map.empty }
            { Name = "Feature2"; Type = typeof<float32>; Importance = None; IsCategorical = false; PossibleValues = None; Metadata = Map.empty }
        ]
        let label = { Name = "Label"; Type = typeof<float32>; IsCategorical = true; PossibleValues = Some ["0"; "1"]; Metadata = Map.empty }
        let model = service.CreateModel("TestModel", ModelType.Classification, features, label).Result
        
        // Assert
        Assert.NotNull(model)
        Assert.Equal("TestModel", model.Name)
        Assert.Equal(ModelType.Classification, model.Type)
        Assert.Equal(ModelStatus.NotTrained, model.Status)
        Assert.Equal(2, model.Features.Length)
        Assert.Equal(Some label, model.Label)
    
    /// <summary>
    /// Test that MLService can get all models.
    /// </summary>
    [<Fact>]
    member _.``MLService can get all models``() =
        // Arrange
        let frameworkLogger = MockLogger<MLFramework>()
        let framework = new MLFramework(frameworkLogger :> ILogger<MLFramework>, options)
        let serviceLogger = MockLogger<MLService>()
        let service = new MLService(serviceLogger :> ILogger<MLService>, framework)
        
        // Act
        let models = service.GetAllModels().Result
        
        // Assert
        Assert.NotNull(models)
        // Initially there should be no models
        Assert.Empty(models)
    
    /// <summary>
    /// Test that MLService can make a prediction.
    /// </summary>
    [<Fact>]
    member _.``MLService can make a prediction``() =
        // Arrange
        let frameworkLogger = MockLogger<MLFramework>()
        let framework = new MLFramework(frameworkLogger :> ILogger<MLFramework>, options)
        let serviceLogger = MockLogger<MLService>()
        let service = new MLService(serviceLogger :> ILogger<MLService>, framework)
        
        // Create a model
        let features = [
            { Name = "Feature1"; Type = typeof<float32>; Importance = None; IsCategorical = false; PossibleValues = None; Metadata = Map.empty }
            { Name = "Feature2"; Type = typeof<float32>; Importance = None; IsCategorical = false; PossibleValues = None; Metadata = Map.empty }
        ]
        let label = { Name = "Label"; Type = typeof<float32>; IsCategorical = true; PossibleValues = Some ["0"; "1"]; Metadata = Map.empty }
        let model = service.CreateModel("TestModel", ModelType.Classification, features, label).Result
        
        // Act
        let request = { Features = Map.ofList [("Feature1", box 1.0f); ("Feature2", box 2.0f)]; Metadata = Map.empty }
        let result = service.Predict(model.Id, request).Result
        
        // Assert
        Assert.NotNull(result)
        Assert.True(result.IsSuccessful)
        Assert.NotNull(result.Prediction)
        Assert.NotNull(result.Prediction.Value)
    
    /// <summary>
    /// Test that MLService can make batch predictions.
    /// </summary>
    [<Fact>]
    member _.``MLService can make batch predictions``() =
        // Arrange
        let frameworkLogger = MockLogger<MLFramework>()
        let framework = new MLFramework(frameworkLogger :> ILogger<MLFramework>, options)
        let serviceLogger = MockLogger<MLService>()
        let service = new MLService(serviceLogger :> ILogger<MLService>, framework)
        
        // Create a model
        let features = [
            { Name = "Feature1"; Type = typeof<float32>; Importance = None; IsCategorical = false; PossibleValues = None; Metadata = Map.empty }
            { Name = "Feature2"; Type = typeof<float32>; Importance = None; IsCategorical = false; PossibleValues = None; Metadata = Map.empty }
        ]
        let label = { Name = "Label"; Type = typeof<float32>; IsCategorical = true; PossibleValues = Some ["0"; "1"]; Metadata = Map.empty }
        let model = service.CreateModel("TestModel", ModelType.Classification, features, label).Result
        
        // Act
        let requests = [
            { Features = Map.ofList [("Feature1", box 1.0f); ("Feature2", box 2.0f)]; Metadata = Map.empty }
            { Features = Map.ofList [("Feature1", box 3.0f); ("Feature2", box 4.0f)]; Metadata = Map.empty }
        ]
        let results = service.PredictBatch(model.Id, requests).Result
        
        // Assert
        Assert.NotNull(results)
        Assert.Equal(2, results.Length)
        Assert.All(results, fun r -> Assert.True(r.IsSuccessful))
    
    /// <summary>
    /// Test that MLService can create a dataset.
    /// </summary>
    [<Fact>]
    member _.``MLService can create a dataset``() =
        // Arrange
        let frameworkLogger = MockLogger<MLFramework>()
        let framework = new MLFramework(frameworkLogger :> ILogger<MLFramework>, options)
        let serviceLogger = MockLogger<MLService>()
        let service = new MLService(serviceLogger :> ILogger<MLService>, framework)
        
        // Act
        let features = [
            { Name = "Feature1"; Type = typeof<float32>; Importance = None; IsCategorical = false; PossibleValues = None; Metadata = Map.empty }
            { Name = "Feature2"; Type = typeof<float32>; Importance = None; IsCategorical = false; PossibleValues = None; Metadata = Map.empty }
        ]
        let label = { Name = "Label"; Type = typeof<float32>; IsCategorical = true; PossibleValues = Some ["0"; "1"]; Metadata = Map.empty }
        let dataPoints = [
            { Features = Map.ofList [("Feature1", box 1.0f); ("Feature2", box 2.0f)]; Label = Some (box 0.0f); Weight = None; Metadata = Map.empty }
            { Features = Map.ofList [("Feature1", box 3.0f); ("Feature2", box 4.0f)]; Label = Some (box 1.0f); Weight = None; Metadata = Map.empty }
        ]
        let dataset = service.CreateDataset("TestDataset", features, label, dataPoints).Result
        
        // Assert
        Assert.NotNull(dataset)
        Assert.Equal("TestDataset", dataset.Name)
        Assert.Equal(2, dataset.Features.Length)
        Assert.Equal(Some label, dataset.Label)
        Assert.Equal(2, dataset.DataPoints.Length)
    
    /// <summary>
    /// Test that MLService can split a dataset.
    /// </summary>
    [<Fact>]
    member _.``MLService can split a dataset``() =
        // Arrange
        let frameworkLogger = MockLogger<MLFramework>()
        let framework = new MLFramework(frameworkLogger :> ILogger<MLFramework>, options)
        let serviceLogger = MockLogger<MLService>()
        let service = new MLService(serviceLogger :> ILogger<MLService>, framework)
        
        // Create a dataset
        let features = [
            { Name = "Feature1"; Type = typeof<float32>; Importance = None; IsCategorical = false; PossibleValues = None; Metadata = Map.empty }
            { Name = "Feature2"; Type = typeof<float32>; Importance = None; IsCategorical = false; PossibleValues = None; Metadata = Map.empty }
        ]
        let label = { Name = "Label"; Type = typeof<float32>; IsCategorical = true; PossibleValues = Some ["0"; "1"]; Metadata = Map.empty }
        let dataPoints = [
            { Features = Map.ofList [("Feature1", box 1.0f); ("Feature2", box 2.0f)]; Label = Some (box 0.0f); Weight = None; Metadata = Map.empty }
            { Features = Map.ofList [("Feature1", box 3.0f); ("Feature2", box 4.0f)]; Label = Some (box 1.0f); Weight = None; Metadata = Map.empty }
            { Features = Map.ofList [("Feature1", box 5.0f); ("Feature2", box 6.0f)]; Label = Some (box 0.0f); Weight = None; Metadata = Map.empty }
            { Features = Map.ofList [("Feature1", box 7.0f); ("Feature2", box 8.0f)]; Label = Some (box 1.0f); Weight = None; Metadata = Map.empty }
        ]
        let dataset = service.CreateDataset("TestDataset", features, label, dataPoints).Result
        
        // Act
        let split = service.SplitDataset(dataset, 0.75, 0.25).Result
        
        // Assert
        Assert.NotNull(split)
        Assert.Equal(3, split.TrainingSet.DataPoints.Length)
        Assert.Equal(1, split.TestingSet.DataPoints.Length)
        Assert.Equal(None, split.ValidationSet)
