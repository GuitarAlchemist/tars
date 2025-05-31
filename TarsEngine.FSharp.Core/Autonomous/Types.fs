namespace TarsEngine.FSharp.Core.Autonomous

open System
open System.Collections.Generic
open System.Threading.Tasks
open TarsEngine.FSharp.Core.LLM
open TarsEngine.FSharp.Core.ChromaDB

/// Improvement suggestion
type ImprovementSuggestion = {
    Id: string
    Title: string
    Description: string
    Priority: int // 1-10, 10 being highest
    Category: string
    EstimatedEffort: string
    ExpectedBenefit: string
    Confidence: float
    GeneratedAt: DateTime
}

/// Improvement execution result
type ImprovementResult = {
    SuggestionId: string
    Success: bool
    ExecutionTime: TimeSpan
    Output: string
    ErrorMessage: string option
    MetricsImprovement: Map<string, float>
}

/// Autonomous improvement cycle
type ImprovementCycle = {
    Id: string
    StartTime: DateTime
    EndTime: DateTime option
    Phase: string // "Analysis", "Planning", "Execution", "Validation", "Complete"
    Suggestions: ImprovementSuggestion list
    Results: ImprovementResult list
    Metrics: Map<string, float>
}

/// Self-modification capability
type SelfModification = {
    Id: string
    TargetComponent: string
    ModificationType: string // "Code", "Config", "Metascript", "Architecture"
    OriginalContent: string
    ModifiedContent: string
    Justification: string
    RiskAssessment: string
    ApprovalRequired: bool
}

/// Autonomous improvement service interface
type IAutonomousImprovementService =
    abstract member StartImprovementCycleAsync: unit -> Task<ImprovementCycle>
    abstract member AnalyzeSystemAsync: unit -> Task<ImprovementSuggestion list>
    abstract member ExecuteImprovementAsync: suggestion: ImprovementSuggestion -> Task<ImprovementResult>
    abstract member ValidateImprovementAsync: result: ImprovementResult -> Task<bool>
    abstract member GenerateMetascriptForTaskAsync: task: string -> Task<string>
    abstract member SelfModifyAsync: modification: SelfModification -> Task<bool>

/// Exploration service interface
type IExplorationService =
    abstract member ExploreUnknownConceptAsync: concept: string -> Task<string>
    abstract member SearchWebForKnowledgeAsync: query: string -> Task<string list>
    abstract member CreateLearningMetascriptAsync: topic: string -> Task<string>
    abstract member UpdateKnowledgeBaseAsync: newKnowledge: string * source: string -> Task<unit>

