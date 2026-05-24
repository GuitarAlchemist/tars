namespace Tars.Core.MetaCognition

open System

// =========================================================================
// Failure Analysis Types
// =========================================================================

/// Root cause categories for failures
[<RequireQualifiedAccess>]
type FailureRootCause =
    | MissingTool of toolName: string
    | WrongPattern of usedPattern: string * suggestedPattern: string
    | KnowledgeGap of domain: string
    | InsufficientContext of missingInfo: string
    | ModelLimitation of detail: string
    | BadPrompt of issue: string
    | ExternalFailure of service: string
    | Unknown of detail: string

/// A single failure record for cluster analysis
type FailureRecord =
    { RunId: string
      Goal: string
      PatternUsed: string
      ErrorMessage: string
      TraceStepCount: int
      FailedAtStep: string option
      Timestamp: DateTime
      Tags: string list
      Score: float }

/// A cluster of similar failures
type FailureCluster =
    { ClusterId: string
      Representative: FailureRecord
      Members: FailureRecord list
      RootCause: FailureRootCause
      Frequency: int
      FirstSeen: DateTime
      LastSeen: DateTime
      AffectedGoalPatterns: string list }

// =========================================================================
// Capability Gap Types
// =========================================================================

/// How to address a detected gap
[<RequireQualifiedAccess>]
type GapRemedy =
    | LearnPattern of patternDescription: string
    | AcquireTool of toolName: string * purpose: string
    | IngestKnowledge of domain: string * sources: string list
    | ImprovePrompt of currentPattern: string * suggestion: string
    | ComposePatterns of patterns: string list

/// A detected capability gap
type CapabilityGap =
    { GapId: string
      Domain: string
      Description: string
      FailureRate: float
      SampleSize: int
      RelatedClusters: string list
      SuggestedRemedy: GapRemedy
      DetectedAt: DateTime
      Confidence: float }

// =========================================================================
// Curriculum Types
// =========================================================================

/// A curriculum task generated to address a gap
type TargetedTask =
    { TaskId: string
      GapId: string
      Description: string
      Difficulty: int
      ExpectedOutcome: string
      ValidationCriteria: string option
      Priority: float }

// =========================================================================
// Adaptive Execution Types
// =========================================================================

/// Step progress snapshot for mid-execution monitoring.
/// Decoupled from Cortex WoTTypes so Core has no upward dependency.
type StepProgress =
    { StepId: string
      Kind: string
      Succeeded: bool
      ErrorMessage: string option
      DurationMs: int64
      Confidence: float option }

/// Adaptation signal during execution
[<RequireQualifiedAccess>]
type AdaptationSignal =
    | ConfidenceDropping of currentConfidence: float * threshold: float
    | StepFailing of stepId: string * error: string
    | ConsecutiveFailures of count: int
    | BudgetExhausting of usedPercent: float
    | PatternMismatch of expected: string * observed: string

/// Adaptation decision
[<RequireQualifiedAccess>]
type AdaptationAction =
    | ContinueCurrent
    | SwitchPattern of newPattern: string * reason: string
    | InsertRecoveryStep of stepDescription: string
    | Abort of reason: string

// =========================================================================
// Reflection Types
// =========================================================================

/// Outcome classification for post-execution reflection
[<RequireQualifiedAccess>]
type ReflectionOutcome =
    | AsExpected
    | BetterThanExpected of detail: string
    | WorseThanExpected of detail: string
    | Unexpected of detail: string

/// Comparison of planned vs actual execution
type IntentOutcomeComparison =
    { PlannedSteps: int
      ExecutedSteps: int
      SkippedSteps: string list
      FailedSteps: string list
      UnexpectedSteps: string list
      OverallAlignment: float }

/// Post-execution reflection report
type ReflectionReport =
    { RunId: string
      Goal: string
      IntendedStrategy: string
      ActualBehavior: string
      Outcome: ReflectionOutcome
      Surprises: string list
      LessonsLearned: string list
      SuggestedImprovements: string list
      Timestamp: DateTime }

// =========================================================================
// Orchestrator Types
// =========================================================================

/// Configuration for the meta-cognitive cycle
type MetaCognitionConfig =
    { FailureClusterThreshold: float
      GapDetectionThreshold: float
      MaxTargetedTasks: int
      AdaptiveMinConfidence: float
      AdaptiveMaxConsecutiveFailures: int
      AdaptiveBudgetWarningPercent: float
      EnableLlmRefinement: bool }

module MetaCognitionConfig =
    let defaults =
        { FailureClusterThreshold = 0.4
          GapDetectionThreshold = 0.5
          MaxTargetedTasks = 5
          AdaptiveMinConfidence = 0.3
          AdaptiveMaxConsecutiveFailures = 3
          AdaptiveBudgetWarningPercent = 0.8
          EnableLlmRefinement = true }

/// Result of a full meta-cognitive cycle
type MetaCognitionResult =
    { FailureClusters: FailureCluster list
      DetectedGaps: CapabilityGap list
      GeneratedTasks: TargetedTask list
      Reflections: ReflectionReport list
      NewBeliefs: int
      Recommendations: string list }
