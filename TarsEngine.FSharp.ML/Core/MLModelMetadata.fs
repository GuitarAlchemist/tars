namespace TarsEngine.FSharp.ML.Core

open System
open System.Collections.Generic

/// <summary>
/// Metadata for a machine learning model.
/// </summary>
type MLModelMetadata = {
    /// <summary>
    /// The model name.
    /// </summary>
    ModelName: string
    
    /// <summary>
    /// The date the model was created.
    /// </summary>
    CreatedAt: DateTime
    
    /// <summary>
    /// The date the model was last updated.
    /// </summary>
    LastUpdatedAt: DateTime
    
    /// <summary>
    /// The data type name.
    /// </summary>
    DataType: string
    
    /// <summary>
    /// The prediction type name.
    /// </summary>
    PredictionType: string
    
    /// <summary>
    /// The number of training examples.
    /// </summary>
    TrainingExamples: int
    
    /// <summary>
    /// The model path.
    /// </summary>
    ModelPath: string
    
    /// <summary>
    /// The model metrics.
    /// </summary>
    Metrics: Dictionary<string, double>
    
    /// <summary>
    /// The model hyperparameters.
    /// </summary>
    HyperParameters: Dictionary<string, string>
    
    /// <summary>
    /// The model tags.
    /// </summary>
    Tags: List<string>
}
