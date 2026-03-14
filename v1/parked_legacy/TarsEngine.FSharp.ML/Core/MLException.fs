namespace TarsEngine.FSharp.ML.Core

open System

/// <summary>
/// Exception type for ML-related errors.
/// </summary>
type MLException(message: string, ?innerException: exn) =
    inherit Exception(message, defaultArg innerException null)

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
/// Exception type for model training errors.
/// </summary>
type ModelTrainingException(modelName: string, message: string, ?innerException: exn) =
    inherit MLException($"Error training model {modelName}: {message}", defaultArg innerException null)

    /// <summary>
    /// The name of the model that failed training.
    /// </summary>
    member val ModelName = modelName with get

/// <summary>
/// Exception type for prediction errors.
/// </summary>
type PredictionException(modelName: string, message: string, ?innerException: exn) =
    inherit MLException($"Error making prediction with model {modelName}: {message}", defaultArg innerException null)

    /// <summary>
    /// The name of the model that failed prediction.
    /// </summary>
    member val ModelName = modelName with get
