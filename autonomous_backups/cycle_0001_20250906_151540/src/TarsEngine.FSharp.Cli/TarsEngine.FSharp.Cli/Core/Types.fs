namespace TarsEngine.FSharp.Cli.Core

open System

/// <summary>
/// Core types for the TARS F# CLI - consolidated from multiple projects.
/// </summary>
module Types =
    
    /// <summary>
    /// Represents a unique identifier.
    /// </summary>
    type Id = string
    
    /// <summary>
    /// Represents a timestamp.
    /// </summary>
    type Timestamp = DateTime
    
    /// <summary>
    /// Represents metadata as key-value pairs.
    /// </summary>
    type Metadata = Map<string, obj>
    
    /// <summary>
    /// Represents an error with a message and optional details.
    /// </summary>
    type TarsError = {
        Message: string
        Details: string option
        Timestamp: Timestamp
        ErrorCode: string option
        StackTrace: string option
    }
    
    /// <summary>
    /// Creates a new error.
    /// </summary>
    let createError message details =
        {
            Message = message
            Details = details
            Timestamp = DateTime.UtcNow
            ErrorCode = None
            StackTrace = None
        }
    
    /// <summary>
    /// Represents execution context for TARS operations.
    /// </summary>
    type ExecutionContext = {
        Id: Id
        StartTime: Timestamp
        WorkingDirectory: string
        Variables: Map<string, obj>
        Metadata: Metadata
        UserId: string option
        SessionId: string option
    }
    
    /// <summary>
    /// Creates a new execution context.
    /// </summary>
    let createExecutionContext workingDirectory =
        {
            Id = Guid.NewGuid().ToString()
            StartTime = DateTime.UtcNow
            WorkingDirectory = workingDirectory
            Variables = Map.empty
            Metadata = Map.empty
            UserId = None
            SessionId = None
        }
    
    /// <summary>
    /// Represents intelligence measurement data.
    /// </summary>
    type IntelligenceMeasurement = {
        Id: Id
        Timestamp: DateTime
        MetricName: string
        Value: float
        Unit: string
        Context: Map<string, obj>
    }
    
    /// <summary>
    /// Represents a machine learning model.
    /// </summary>
    type MLModel = {
        Id: Id
        Name: string
        ModelType: MLModelType
        Status: MLModelStatus
        Accuracy: float option
        CreatedAt: DateTime
        LastTrainedAt: DateTime option
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
    
    /// <summary>
    /// Represents the status of an ML model.
    /// </summary>
    and MLModelStatus =
        | NotTrained
        | Training
        | Trained
        | Failed
    
    /// <summary>
    /// Creates a new intelligence measurement.
    /// </summary>
    let createMeasurement metricName value unit context =
        {
            Id = Guid.NewGuid().ToString()
            Timestamp = DateTime.UtcNow
            MetricName = metricName
            Value = value
            Unit = unit
            Context = context
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
            CreatedAt = DateTime.UtcNow
            LastTrainedAt = None
        }
