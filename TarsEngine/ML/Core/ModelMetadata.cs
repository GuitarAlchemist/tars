using System;
using System.Collections.Generic;

namespace TarsEngine.ML.Core;

/// <summary>
/// Metadata for a machine learning model
/// </summary>
public class ModelMetadata
{
    /// <summary>
    /// Gets or sets the model name
    /// </summary>
    public string ModelName { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the timestamp when the model was created
    /// </summary>
    public DateTime CreatedAt { get; set; }
    
    /// <summary>
    /// Gets or sets the timestamp when the model was last updated
    /// </summary>
    public DateTime LastUpdatedAt { get; set; }
    
    /// <summary>
    /// Gets or sets the data type name
    /// </summary>
    public string DataType { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the prediction type name
    /// </summary>
    public string PredictionType { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the number of training examples
    /// </summary>
    public int TrainingExamples { get; set; }
    
    /// <summary>
    /// Gets or sets the model path
    /// </summary>
    public string ModelPath { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the model metrics
    /// </summary>
    public Dictionary<string, double> Metrics { get; set; } = new Dictionary<string, double>();
    
    /// <summary>
    /// Gets or sets the model hyperparameters
    /// </summary>
    public Dictionary<string, string> HyperParameters { get; set; } = new Dictionary<string, string>();
    
    /// <summary>
    /// Gets or sets the model tags
    /// </summary>
    public List<string> Tags { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the model description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the model version
    /// </summary>
    public string Version { get; set; } = "1.0";
    
    /// <summary>
    /// Gets or sets the model author
    /// </summary>
    public string Author { get; set; } = "TARS";
    
    /// <summary>
    /// Gets or sets the model purpose
    /// </summary>
    public string Purpose { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the model capabilities
    /// </summary>
    public List<string> Capabilities { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the model limitations
    /// </summary>
    public List<string> Limitations { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the model dependencies
    /// </summary>
    public List<string> Dependencies { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the model training time in seconds
    /// </summary>
    public double TrainingTimeSeconds { get; set; }
    
    /// <summary>
    /// Gets or sets the model inference time in milliseconds
    /// </summary>
    public double InferenceTimeMs { get; set; }
    
    /// <summary>
    /// Gets or sets the model size in bytes
    /// </summary>
    public long SizeBytes { get; set; }
    
    /// <summary>
    /// Gets or sets the model memory usage in bytes
    /// </summary>
    public long MemoryUsageBytes { get; set; }
    
    /// <summary>
    /// Gets or sets the model intelligence score
    /// </summary>
    public double IntelligenceScore { get; set; }
    
    /// <summary>
    /// Gets or sets the model learning rate
    /// </summary>
    public double LearningRate { get; set; }
    
    /// <summary>
    /// Gets or sets the model improvement history
    /// </summary>
    public List<ModelImprovement> ImprovementHistory { get; set; } = new List<ModelImprovement>();
}
