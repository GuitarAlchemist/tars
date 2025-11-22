namespace TarsEngine.SelfImprovement

open System

/// <summary>
/// Represents an improvement opportunity identified during code analysis with Ollama
/// </summary>
type OllamaImprovementOpportunity = {
    /// <summary>
    /// The file path
    /// </summary>
    FilePath: string

    /// <summary>
    /// The type of improvement (e.g., performance, readability)
    /// </summary>
    Type: string

    /// <summary>
    /// Description of the improvement
    /// </summary>
    Description: string

    /// <summary>
    /// The current code snippet
    /// </summary>
    CurrentCode: string

    /// <summary>
    /// The improved code snippet
    /// </summary>
    ImprovedCode: string

    /// <summary>
    /// Rationale for the improvement
    /// </summary>
    Rationale: string

    /// <summary>
    /// Confidence score for the improvement (0.0 to 1.0)
    /// </summary>
    Confidence: float
}

/// <summary>
/// Represents an improvement opportunity identified during code analysis
/// </summary>
type OllamaAnalysisResult = {
    /// <summary>
    /// The file path
    /// </summary>
    FilePath: string

    /// <summary>
    /// The type of improvement (e.g., performance, readability)
    /// </summary>
    Type: string

    /// <summary>
    /// Description of the improvement
    /// </summary>
    Description: string

    /// <summary>
    /// The current code snippet
    /// </summary>
    CurrentCode: string

    /// <summary>
    /// The improved code snippet
    /// </summary>
    ImprovedCode: string

    /// <summary>
    /// Rationale for the improvement
    /// </summary>
    Rationale: string

    /// <summary>
    /// Confidence score for the improvement (0.0 to 1.0)
    /// </summary>
    Confidence: float
}

/// <summary>
/// Ollama request model
/// </summary>
type OllamaRequest = {
    model: string
    prompt: string
    stream: bool
    options: {|
        temperature: float
        num_predict: int
    |}
}

/// <summary>
/// Ollama response model
/// </summary>
type OllamaResponse = {
    response: string
}
