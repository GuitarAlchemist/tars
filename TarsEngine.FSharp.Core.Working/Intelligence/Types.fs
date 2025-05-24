namespace TarsEngine.FSharp.Core.Working.Intelligence

open System
open TarsEngine.FSharp.Core.Working.Types

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
    Metadata: Metadata
}

/// <summary>
/// Represents a learning curve data point.
/// </summary>
type LearningCurvePoint = {
    Timestamp: DateTime
    Iteration: int
    Performance: float
    LearningRate: float
    Accuracy: float option
    Loss: float option
}

/// <summary>
/// Represents intelligence progression over time.
/// </summary>
type IntelligenceProgression = {
    StartTime: DateTime
    EndTime: DateTime
    Measurements: IntelligenceMeasurement list
    LearningCurve: LearningCurvePoint list
    OverallProgress: float
    TrendDirection: TrendDirection
    Insights: string list
}

/// <summary>
/// Represents the direction of a trend.
/// </summary>
and TrendDirection =
    | Improving
    | Declining
    | Stable
    | Volatile

/// <summary>
/// Represents intelligence analysis results.
/// </summary>
type IntelligenceAnalysis = {
    AnalysisId: Id
    Timestamp: DateTime
    Subject: string
    Metrics: Map<string, float>
    Recommendations: string list
    Confidence: float
    Methodology: string
}

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
        Metadata = Map.empty
    }

/// <summary>
/// Creates a new learning curve point.
/// </summary>
let createLearningCurvePoint iteration performance learningRate =
    {
        Timestamp = DateTime.UtcNow
        Iteration = iteration
        Performance = performance
        LearningRate = learningRate
        Accuracy = None
        Loss = None
    }
