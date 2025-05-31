namespace TarsEngine.FSharp.SelfImprovement

open System

/// Types for TARS self-improvement system
module ImprovementTypes =
    
    /// Severity level of code issues
    type Severity =
        | Low = 1
        | Medium = 2
        | High = 3
        | Critical = 4
    
    /// Type of improvement pattern
    type PatternType =
        | CodeSmell
        | Performance
        | Maintainability
        | Security
        | Documentation
        | Testing
    
    /// Code improvement pattern
    type ImprovementPattern = {
        Name: string
        Description: string
        PatternType: PatternType
        Severity: Severity
        Example: string option
        Recommendation: string
    }
    
    /// Code analysis result
    type AnalysisResult = {
        FilePath: string
        Issues: ImprovementPattern list
        OverallScore: float
        Recommendations: string list
        AnalyzedAt: DateTime
    }
    
    /// Applied improvement tracking
    type AppliedImprovement = {
        Id: Guid
        FilePath: string
        Pattern: ImprovementPattern
        OriginalCode: string
        ImprovedCode: string
        AppliedAt: DateTime
        Success: bool
        Notes: string option
    }
    
    /// Self-improvement session
    type ImprovementSession = {
        Id: Guid
        StartedAt: DateTime
        CompletedAt: DateTime option
        FilesAnalyzed: int
        ImprovementsApplied: int
        SuccessRate: float
        Summary: string
    }
