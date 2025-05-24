namespace TarsEngine.FSharp.Core.ML.Core

open System
open System.Collections.Generic

/// <summary>
/// Metadata for a machine learning model
/// </summary>
type ModelMetadata = {
    /// <summary>
    /// Gets or sets the model name
    /// </summary>
    ModelName: string
    
    /// <summary>
    /// Gets or sets the date the model was created
    /// </summary>
    CreatedAt: DateTime
    
    /// <summary>
    /// Gets or sets the date the model was last updated
    /// </summary>
    LastUpdatedAt: DateTime
    
    /// <summary>
    /// Gets or sets the data type name
    /// </summary>
    DataType: string
    
    /// <summary>
    /// Gets or sets the prediction type name
    /// </summary>
    PredictionType: string
    
    /// <summary>
    /// Gets or sets the number of training examples
    /// </summary>
    TrainingExamples: int
    
    /// <summary>
    /// Gets or sets the model path
    /// </summary>
    ModelPath: string
    
    /// <summary>
    /// Gets or sets the model metrics
    /// </summary>
    Metrics: Dictionary<string, double>
    
    /// <summary>
    /// Gets or sets the model hyperparameters
    /// </summary>
    HyperParameters: Dictionary<string, string>
    
    /// <summary>
    /// Gets or sets the model tags
    /// </summary>
    Tags: List<string>
}
