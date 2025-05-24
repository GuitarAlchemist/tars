namespace TarsEngine.FSharp.Core.ML.Core

open System

/// <summary>
/// Exception type for ML-related errors.
/// </summary>
type MLException(message: string, ?innerException: exn) =
    inherit Exception(message, defaultArg innerException null)
    
    /// <summary>
    /// Creates a new MLException with a message.
    /// </summary>
    new(message: string) = MLException(message, null)
    
    /// <summary>
    /// Creates a new MLException with a message and inner exception.
    /// </summary>
    new(message: string, innerException: exn) = MLException(message, innerException)

/// <summary>
/// Exception type for model not found errors.
/// </summary>
type ModelNotFoundException(modelName: string, ?innerException: exn) =
    inherit MLException($"Model not found: {modelName}", defaultArg innerException null)
    
    /// <summary>
    /// The name of the model that was not found.
    /// </summary>
    member val ModelName = modelName with get
    
    /// <summary>
    /// Creates a new ModelNotFoundException with a model name.
    /// </summary>
    new(modelName: string) = ModelNotFoundException(modelName, null)
    
    /// <summary>
    /// Creates a new ModelNotFoundException with a model name and inner exception.
    /// </summary>
    new(modelName: string, innerException: exn) = ModelNotFoundException(modelName, innerException)

/// <summary>
/// Exception type for model training errors.
/// </summary>
type ModelTrainingException(modelName: string, message: string, ?innerException: exn) =
    inherit MLException($"Error training model {modelName}: {message}", defaultArg innerException null)
    
    /// <summary>
    /// The name of the model that failed training.
    /// </summary>
    member val ModelName = modelName with get
    
    /// <summary>
    /// Creates a new ModelTrainingException with a model name and message.
    /// </summary>
    new(modelName: string, message: string) = ModelTrainingException(modelName, message, null)
    
    /// <summary>
    /// Creates a new ModelTrainingException with a model name, message, and inner exception.
    /// </summary>
    new(modelName: string, message: string, innerException: exn) = ModelTrainingException(modelName, message, innerException)

/// <summary>
/// Exception type for prediction errors.
/// </summary>
type PredictionException(modelName: string, message: string, ?innerException: exn) =
    inherit MLException($"Error making prediction with model {modelName}: {message}", defaultArg innerException null)
    
    /// <summary>
    /// The name of the model that failed prediction.
    /// </summary>
    member val ModelName = modelName with get
    
    /// <summary>
    /// Creates a new PredictionException with a model name and message.
    /// </summary>
    new(modelName: string, message: string) = PredictionException(modelName, message, null)
    
    /// <summary>
    /// Creates a new PredictionException with a model name, message, and inner exception.
    /// </summary>
    new(modelName: string, message: string, innerException: exn) = PredictionException(modelName, message, innerException)
