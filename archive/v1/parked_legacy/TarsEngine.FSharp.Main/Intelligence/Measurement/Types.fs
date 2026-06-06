namespace TarsEngine.FSharp.Main.Intelligence.Measurement

open System

/// <summary>
/// Represents a learning curve type
/// </summary>
type LearningCurveType =
    /// <summary>
    /// Unknown curve type
    /// </summary>
    | Unknown
    /// <summary>
    /// Linear curve type
    /// </summary>
    | Linear
    /// <summary>
    /// Logarithmic curve type
    /// </summary>
    | Logarithmic
    /// <summary>
    /// Exponential curve type
    /// </summary>
    | Exponential
    /// <summary>
    /// Plateau curve type
    /// </summary>
    | Plateau
    /// <summary>
    /// Declining curve type
    /// </summary>
    | Declining

/// <summary>
/// Represents a learning data point
/// </summary>
type LearningDataPoint = {
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    Timestamp: DateTime
    /// <summary>
    /// Gets or sets the value
    /// </summary>
    Value: float
    /// <summary>
    /// Gets or sets the previous value
    /// </summary>
    PreviousValue: float
    /// <summary>
    /// Gets or sets the improvement ratio
    /// </summary>
    ImprovementRatio: float
    /// <summary>
    /// Gets or sets the logarithmic value
    /// </summary>
    LogValue: float
    /// <summary>
    /// Gets or sets the logarithmic previous value
    /// </summary>
    LogPreviousValue: float
    /// <summary>
    /// Gets or sets the logarithmic learning rate
    /// </summary>
    LogLearningRate: float
}

/// <summary>
/// Represents a learning curve analysis
/// </summary>
type LearningCurveAnalysis = {
    /// <summary>
    /// Gets or sets the metric name
    /// </summary>
    MetricName: string
    /// <summary>
    /// Gets or sets the data points
    /// </summary>
    DataPoints: LearningDataPoint list
    /// <summary>
    /// Gets or sets the learning rate
    /// </summary>
    LearningRate: float
    /// <summary>
    /// Gets or sets the logarithmic learning rate
    /// </summary>
    LogLearningRate: float
    /// <summary>
    /// Gets or sets the plateau value
    /// </summary>
    PlateauValue: float
    /// <summary>
    /// Gets or sets the time to plateau in days
    /// </summary>
    TimeToPlateauDays: float
    /// <summary>
    /// Gets or sets the curve type
    /// </summary>
    CurveType: LearningCurveType
    /// <summary>
    /// Gets or sets the efficiency score
    /// </summary>
    EfficiencyScore: float
    /// <summary>
    /// Gets or sets the forecast values
    /// </summary>
    ForecastValues: Map<DateTime, float>
}

/// <summary>
/// Represents a milestone type
/// </summary>
type MilestoneType =
    /// <summary>
    /// Learning milestone
    /// </summary>
    | Learning
    /// <summary>
    /// Problem solving milestone
    /// </summary>
    | ProblemSolving
    /// <summary>
    /// Creativity milestone
    /// </summary>
    | Creativity
    /// <summary>
    /// Efficiency milestone
    /// </summary>
    | Efficiency
    /// <summary>
    /// Adaptability milestone
    /// </summary>
    | Adaptability
    /// <summary>
    /// Other milestone
    /// </summary>
    | Other

/// <summary>
/// Represents an intelligence milestone
/// </summary>
type IntelligenceMilestone = {
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    Timestamp: DateTime
    /// <summary>
    /// Gets or sets the description
    /// </summary>
    Description: string
    /// <summary>
    /// Gets or sets the intelligence score
    /// </summary>
    IntelligenceScore: float
    /// <summary>
    /// Gets or sets the milestone type
    /// </summary>
    Type: MilestoneType
    /// <summary>
    /// Gets or sets the significance level
    /// </summary>
    SignificanceLevel: int
    /// <summary>
    /// Gets or sets the domains
    /// </summary>
    Domains: string list
    /// <summary>
    /// Gets or sets the skills
    /// </summary>
    Skills: string list
}

/// <summary>
/// Represents an intelligence forecast
/// </summary>
type IntelligenceForecast = {
    /// <summary>
    /// Gets or sets the forecast period in days
    /// </summary>
    ForecastPeriodDays: int
    /// <summary>
    /// Gets or sets the forecast values
    /// </summary>
    ForecastValues: Map<DateTime, float>
    /// <summary>
    /// Gets or sets the forecast confidence intervals
    /// </summary>
    ConfidenceIntervals: Map<DateTime, float * float>
    /// <summary>
    /// Gets or sets the expected milestones
    /// </summary>
    ExpectedMilestones: IntelligenceMilestone list
    /// <summary>
    /// Gets or sets the forecast model type
    /// </summary>
    ForecastModelType: string
    /// <summary>
    /// Gets or sets the forecast accuracy
    /// </summary>
    ForecastAccuracy: float
    /// <summary>
    /// Gets or sets the forecast confidence level
    /// </summary>
    ConfidenceLevel: float
}

/// <summary>
/// Represents an intelligence progression report
/// </summary>
type IntelligenceProgressionReport = {
    /// <summary>
    /// Gets or sets the start time
    /// </summary>
    StartTime: DateTime
    /// <summary>
    /// Gets or sets the end time
    /// </summary>
    EndTime: DateTime
    /// <summary>
    /// Gets or sets the generated at timestamp
    /// </summary>
    GeneratedAt: DateTime
    /// <summary>
    /// Gets or sets the overall intelligence score
    /// </summary>
    OverallIntelligenceScore: float
    /// <summary>
    /// Gets or sets the learning curve analysis
    /// </summary>
    LearningCurveAnalysis: LearningCurveAnalysis
    /// <summary>
    /// Gets or sets the modification analysis
    /// </summary>
    ModificationAnalysis: ModificationAnalysis
    /// <summary>
    /// Gets or sets the visualization data
    /// </summary>
    VisualizationData: VisualizationData
    /// <summary>
    /// Gets or sets the key insights
    /// </summary>
    KeyInsights: string list
    /// <summary>
    /// Gets or sets the recommendations
    /// </summary>
    Recommendations: string list
    /// <summary>
    /// Gets or sets the growth areas
    /// </summary>
    GrowthAreas: string list
    /// <summary>
    /// Gets or sets the strengths
    /// </summary>
    Strengths: string list
    /// <summary>
    /// Gets or sets the intelligence progression milestones
    /// </summary>
    Milestones: IntelligenceMilestone list
    /// <summary>
    /// Gets or sets the intelligence progression forecast
    /// </summary>
    Forecast: IntelligenceForecast
    /// <summary>
    /// Gets the time period in days
    /// </summary>
    TimePeriodDays: int
    /// <summary>
    /// Gets the intelligence growth rate per day
    /// </summary>
    IntelligenceGrowthRatePerDay: float
    /// <summary>
    /// Gets the logarithmic intelligence growth rate per day
    /// </summary>
    LogIntelligenceGrowthRatePerDay: float
    /// <summary>
    /// Gets or sets the domain-specific intelligence scores
    /// </summary>
    DomainSpecificScores: Map<string, float>
    /// <summary>
    /// Gets or sets the skill-specific intelligence scores
    /// </summary>
    SkillSpecificScores: Map<string, float>
}

/// <summary>
/// Represents a modification trend
/// </summary>
type ModificationTrend =
    /// <summary>
    /// Increasing trend
    /// </summary>
    | Increasing
    /// <summary>
    /// Slightly increasing trend
    /// </summary>
    | SlightlyIncreasing
    /// <summary>
    /// Stable trend
    /// </summary>
    | Stable
    /// <summary>
    /// Slightly decreasing trend
    /// </summary>
    | SlightlyDecreasing
    /// <summary>
    /// Decreasing trend
    /// </summary>
    | Decreasing

/// <summary>
/// Represents a code modification
/// </summary>
type CodeModification = {
    /// <summary>
    /// Gets or sets the file path
    /// </summary>
    FilePath: string
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    Timestamp: DateTime
    /// <summary>
    /// Gets or sets the reason
    /// </summary>
    Reason: string
    /// <summary>
    /// Gets or sets the improvement type
    /// </summary>
    ImprovementType: string
    /// <summary>
    /// Gets or sets the original code
    /// </summary>
    OriginalCode: string
    /// <summary>
    /// Gets or sets the modified code
    /// </summary>
    ModifiedCode: string
    /// <summary>
    /// Gets or sets the lines added
    /// </summary>
    LinesAdded: int
    /// <summary>
    /// Gets or sets the lines removed
    /// </summary>
    LinesRemoved: int
    /// <summary>
    /// Gets or sets the lines modified
    /// </summary>
    LinesModified: int
    /// <summary>
    /// Gets or sets the complexity change
    /// </summary>
    ComplexityChange: float
    /// <summary>
    /// Gets or sets the readability change
    /// </summary>
    ReadabilityChange: float
    /// <summary>
    /// Gets or sets the performance impact
    /// </summary>
    PerformanceImpact: float
}

/// <summary>
/// Represents a modification analysis
/// </summary>
type ModificationAnalysis = {
    /// <summary>
    /// Gets or sets the start time
    /// </summary>
    StartTime: DateTime
    /// <summary>
    /// Gets or sets the end time
    /// </summary>
    EndTime: DateTime
    /// <summary>
    /// Gets or sets the total modifications
    /// </summary>
    TotalModifications: int
    /// <summary>
    /// Gets or sets the total lines added
    /// </summary>
    TotalLinesAdded: int
    /// <summary>
    /// Gets or sets the total lines removed
    /// </summary>
    TotalLinesRemoved: int
    /// <summary>
    /// Gets or sets the total lines modified
    /// </summary>
    TotalLinesModified: int
    /// <summary>
    /// Gets or sets the average complexity change
    /// </summary>
    AverageComplexityChange: float
    /// <summary>
    /// Gets or sets the average readability change
    /// </summary>
    AverageReadabilityChange: float
    /// <summary>
    /// Gets or sets the average performance impact
    /// </summary>
    AveragePerformanceImpact: float
    /// <summary>
    /// Gets or sets the modifications by type
    /// </summary>
    ModificationsByType: Map<string, int>
    /// <summary>
    /// Gets or sets the modifications by file
    /// </summary>
    ModificationsByFile: Map<string, int>
    /// <summary>
    /// Gets or sets the modification trend
    /// </summary>
    ModificationTrend: ModificationTrend
}

/// <summary>
/// Represents visualization data
/// </summary>
type VisualizationData = {
    /// <summary>
    /// Gets or sets the chart data
    /// </summary>
    ChartData: Map<string, obj>
    /// <summary>
    /// Gets or sets the graph data
    /// </summary>
    GraphData: Map<string, obj>
    /// <summary>
    /// Gets or sets the timeline data
    /// </summary>
    TimelineData: Map<string, obj>
}
