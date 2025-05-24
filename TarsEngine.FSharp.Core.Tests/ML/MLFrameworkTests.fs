module TarsEngine.FSharp.Core.Tests.ML.MLFrameworkTests

open System
open System.IO
open System.Collections.Generic
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Microsoft.ML
open Microsoft.ML.Data
open Xunit
open TarsEngine.FSharp.Core.ML.Core

/// <summary>
/// Mock logger for testing.
/// </summary>
type MockLogger<'T>() =
    interface ILogger<'T> with
        member _.BeginScope<'TState>(state: 'TState) = { new IDisposable with member _.Dispose() = () }
        member _.IsEnabled(logLevel: LogLevel) = true
        member _.Log<'TState>(logLevel: LogLevel, eventId: EventId, state: 'TState, ex: exn, formatter: Func<'TState, exn, string>) = ()

/// <summary>
/// Test data class for ML tests.
/// </summary>
[<CLIMutable>]
type TestData = {
    [<LoadColumn(0)>]
    Label: float32
    
    [<LoadColumn(1)>]
    Feature1: float32
    
    [<LoadColumn(2)>]
    Feature2: float32
}

/// <summary>
/// Test prediction class for ML tests.
/// </summary>
[<CLIMutable>]
type TestPrediction = {
    Label: float32
    Score: float32
}

/// <summary>
/// Tests for the MLFramework class.
/// </summary>
type MLFrameworkTests() =
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
    /// Test that MLFramework can be created with default options.
    /// </summary>
    [<Fact>]
    member _.``MLFramework can be created with default options``() =
        // Arrange & Act
        let logger = MockLogger<MLFramework>()
        let framework = new MLFramework(logger :> ILogger<MLFramework>)
        
        // Assert
        Assert.NotNull(framework)
    
    /// <summary>
    /// Test that MLFramework can be created with custom options.
    /// </summary>
    [<Fact>]
    member _.``MLFramework can be created with custom options``() =
        // Arrange & Act
        let logger = MockLogger<MLFramework>()
        let framework = new MLFramework(logger :> ILogger<MLFramework>, options)
        
        // Assert
        Assert.NotNull(framework)
    
    /// <summary>
    /// Test that MLFramework can create and train a model.
    /// </summary>
    [<Fact>]
    member _.``MLFramework can create and train a model``() =
        // Arrange
        let logger = MockLogger<MLFramework>()
        let framework = new MLFramework(logger :> ILogger<MLFramework>, options)
        
        // Create test data
        let trainingData = [|
            { Label = 0.0f; Feature1 = 1.0f; Feature2 = 1.0f }
            { Label = 0.0f; Feature1 = 1.0f; Feature2 = 2.0f }
            { Label = 0.0f; Feature1 = 2.0f; Feature2 = 1.0f }
            { Label = 1.0f; Feature1 = 3.0f; Feature2 = 3.0f }
            { Label = 1.0f; Feature1 = 4.0f; Feature2 = 3.0f }
            { Label = 1.0f; Feature1 = 3.0f; Feature2 = 4.0f }
        |]
        
        // Create pipeline function
        let createPipeline (mlContext: MLContext) =
            mlContext.Transforms.Concatenate("Features", "Feature1", "Feature2")
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression())
        
        // Act
        let result = framework.LoadOrCreateModelAsync("TestModel", createPipeline, trainingData).Result
        
        // Assert
        Assert.True(result)
        
        // Verify model exists
        let modelPath = Path.Combine(tempDir, "TestModel.zip")
        Assert.True(File.Exists(modelPath))
    
    /// <summary>
    /// Test that MLFramework can make predictions.
    /// </summary>
    [<Fact>]
    member _.``MLFramework can make predictions``() =
        // Arrange
        let logger = MockLogger<MLFramework>()
        let framework = new MLFramework(logger :> ILogger<MLFramework>, options)
        
        // Create test data
        let trainingData = [|
            { Label = 0.0f; Feature1 = 1.0f; Feature2 = 1.0f }
            { Label = 0.0f; Feature1 = 1.0f; Feature2 = 2.0f }
            { Label = 0.0f; Feature1 = 2.0f; Feature2 = 1.0f }
            { Label = 1.0f; Feature1 = 3.0f; Feature2 = 3.0f }
            { Label = 1.0f; Feature1 = 4.0f; Feature2 = 3.0f }
            { Label = 1.0f; Feature1 = 3.0f; Feature2 = 4.0f }
        |]
        
        // Create pipeline function
        let createPipeline (mlContext: MLContext) =
            mlContext.Transforms.Concatenate("Features", "Feature1", "Feature2")
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression())
        
        // Train model
        let result = framework.LoadOrCreateModelAsync("TestModel2", createPipeline, trainingData).Result
        Assert.True(result)
        
        // Act
        let testData = { Label = 0.0f; Feature1 = 1.0f; Feature2 = 1.0f }
        let prediction = framework.Predict<TestData, TestPrediction>("TestModel2", testData)
        
        // Assert
        Assert.True(prediction.IsSome)
    
    /// <summary>
    /// Test that MLFramework can get model metadata.
    /// </summary>
    [<Fact>]
    member _.``MLFramework can get model metadata``() =
        // Arrange
        let logger = MockLogger<MLFramework>()
        let framework = new MLFramework(logger :> ILogger<MLFramework>, options)
        
        // Create test data
        let trainingData = [|
            { Label = 0.0f; Feature1 = 1.0f; Feature2 = 1.0f }
            { Label = 0.0f; Feature1 = 1.0f; Feature2 = 2.0f }
            { Label = 0.0f; Feature1 = 2.0f; Feature2 = 1.0f }
            { Label = 1.0f; Feature1 = 3.0f; Feature2 = 3.0f }
            { Label = 1.0f; Feature1 = 4.0f; Feature2 = 3.0f }
            { Label = 1.0f; Feature1 = 3.0f; Feature2 = 4.0f }
        |]
        
        // Create pipeline function
        let createPipeline (mlContext: MLContext) =
            mlContext.Transforms.Concatenate("Features", "Feature1", "Feature2")
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression())
        
        // Train model
        let result = framework.LoadOrCreateModelAsync("TestModel3", createPipeline, trainingData).Result
        Assert.True(result)
        
        // Act
        let metadata = framework.GetModelMetadata("TestModel3")
        
        // Assert
        Assert.True(metadata.IsSome)
        let meta = metadata.Value
        Assert.Equal("TestModel3", meta.ModelName)
        Assert.Equal(typeof<TestData>.Name, meta.DataType)
        Assert.Equal(typeof<TestPrediction>.Name, meta.PredictionType)
        Assert.Equal(trainingData.Length, meta.TrainingExamples)
    
    /// <summary>
    /// Test that MLFramework can delete a model.
    /// </summary>
    [<Fact>]
    member _.``MLFramework can delete a model``() =
        // Arrange
        let logger = MockLogger<MLFramework>()
        let framework = new MLFramework(logger :> ILogger<MLFramework>, options)
        
        // Create test data
        let trainingData = [|
            { Label = 0.0f; Feature1 = 1.0f; Feature2 = 1.0f }
            { Label = 0.0f; Feature1 = 1.0f; Feature2 = 2.0f }
            { Label = 0.0f; Feature1 = 2.0f; Feature2 = 1.0f }
            { Label = 1.0f; Feature1 = 3.0f; Feature2 = 3.0f }
            { Label = 1.0f; Feature1 = 4.0f; Feature2 = 3.0f }
            { Label = 1.0f; Feature1 = 3.0f; Feature2 = 4.0f }
        |]
        
        // Create pipeline function
        let createPipeline (mlContext: MLContext) =
            mlContext.Transforms.Concatenate("Features", "Feature1", "Feature2")
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression())
        
        // Train model
        let result = framework.LoadOrCreateModelAsync("TestModel4", createPipeline, trainingData).Result
        Assert.True(result)
        
        // Verify model exists
        let modelPath = Path.Combine(tempDir, "TestModel4.zip")
        Assert.True(File.Exists(modelPath))
        
        // Act
        let deleteResult = framework.DeleteModelAsync("TestModel4").Result
        
        // Assert
        Assert.True(deleteResult)
        Assert.False(File.Exists(modelPath))
