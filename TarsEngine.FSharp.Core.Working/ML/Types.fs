namespace TarsEngine.FSharp.Core.Working.ML

open System
open TarsEngine.FSharp.Core.Working.Types

/// <summary>
/// Represents a machine learning model.
/// </summary>
type MLModel = {
    Id: Id
    Name: string
    ModelType: MLModelType
    Status: MLModelStatus
    Accuracy: float option
    TrainingData: string option
    CreatedAt: DateTime
    LastTrainedAt: DateTime option
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents the type of ML model.
/// </summary>
and MLModelType =
    | Classification
    | Regression
    | Clustering
    | AnomalyDetection
    | NaturalLanguageProcessing
    | Recommendation

/// <summary>
/// Represents the status of an ML model.
/// </summary>
and MLModelStatus =
    | NotTrained
    | Training
    | Trained
    | Failed
    | Deprecated

/// <summary>
/// Represents ML training configuration.
/// </summary>
type MLTrainingConfig = {
    ModelName: string
    ModelType: MLModelType
    DataPath: string
    ValidationSplit: float
    Epochs: int option
    LearningRate: float option
    BatchSize: int option
    Parameters: Map<string, obj>
}

/// <summary>
/// Represents ML prediction request.
/// </summary>
type MLPredictionRequest = {
    ModelName: string
    InputData: string
    OutputFormat: string option
}

/// <summary>
/// Represents ML prediction result.
/// </summary>
type MLPredictionResult = {
    ModelName: string
    Predictions: obj list
    Confidence: float list
    ProcessingTime: TimeSpan
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents ML model evaluation results.
/// </summary>
type MLEvaluationResult = {
    ModelName: string
    Accuracy: float
    Precision: float option
    Recall: float option
    F1Score: float option
    AUC: float option
    ConfusionMatrix: int[,] option
    EvaluationTime: TimeSpan
}

/// <summary>
/// Creates a new ML model.
/// </summary>
let createMLModel name modelType =
    {
        Id = Guid.NewGuid().ToString()
        Name = name
        ModelType = modelType
        Status = NotTrained
        Accuracy = None
        TrainingData = None
        CreatedAt = DateTime.UtcNow
        LastTrainedAt = None
        Metadata = Map.empty
    }

/// <summary>
/// Creates a new training configuration.
/// </summary>
let createTrainingConfig modelName modelType dataPath =
    {
        ModelName = modelName
        ModelType = modelType
        DataPath = dataPath
        ValidationSplit = 0.2
        Epochs = Some 10
        LearningRate = Some 0.001
        BatchSize = Some 32
        Parameters = Map.empty
    }

/// <summary>
/// Creates a new prediction request.
/// </summary>
let createPredictionRequest modelName inputData =
    {
        ModelName = modelName
        InputData = inputData
        OutputFormat = None
    }
