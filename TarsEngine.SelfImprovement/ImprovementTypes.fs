namespace TarsEngine.SelfImprovement

open System

/// <summary>
/// Represents an applied improvement
/// </summary>
type AppliedImprovement =
    { FilePath: string
      PatternId: string
      PatternName: string
      LineNumber: int option
      OriginalCode: string
      ImprovedCode: string
      AppliedAt: DateTime }
