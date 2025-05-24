namespace TarsEngine.FSharp.Core.Consciousness.Intelligence

open System
open System.Collections.Generic
open TarsEngine.FSharp.Core.Consciousness.Core

/// <summary>
/// Types of intelligence events.
/// </summary>
type IntelligenceEventType =
    | Initialization
    | Activation
    | Deactivation
    | CreativeIdea
    | CreativeSolution
    | Intuition
    | SpontaneousThought
    | CuriosityQuestion
    | CuriosityExploration
    | InsightConnection
    | Other of string

/// <summary>
/// Methods of thought generation.
/// </summary>
type ThoughtGenerationMethod =
    | RandomGeneration
    | AssociativeJumping
    | MindWandering
    | Daydreaming
    | Incubation

/// <summary>
/// Methods of question generation.
/// </summary>
type QuestionGenerationMethod =
    | InformationGap
    | NoveltySeeking
    | ExplorationBased

/// <summary>
/// Methods of insight generation.
/// </summary>
type InsightGenerationMethod =
    | ConnectionDiscovery
    | ProblemRestructuring
    | Incubation

/// <summary>
/// Types of creative processes.
/// </summary>
type CreativeProcessType =
    | Divergent
    | Convergent
    | Combinatorial
    | Transformational
    | Analogical
    | Serendipitous
    | Other of string

/// <summary>
/// Types of intuition.
/// </summary>
type IntuitionType =
    | PatternRecognition
    | HeuristicReasoning
    | GutFeeling
    | Custom of string

/// <summary>
/// Verification status of an intuition.
/// </summary>
type VerificationStatus =
    | Unverified
    | Verified
    | Falsified
    | PartiallyVerified
    | Inconclusive

/// <summary>
/// Exploration strategies.
/// </summary>
type ExplorationStrategy =
    | DeepDive
    | BreadthFirst
    | NoveltyBased
    | ConnectionBased

/// <summary>
/// Represents an intelligence event.
/// </summary>
type IntelligenceEvent = {
    /// <summary>
    /// The ID of the event.
    /// </summary>
    Id: string

    /// <summary>
    /// The type of the event.
    /// </summary>
    Type: IntelligenceEventType

    /// <summary>
    /// The description of the event.
    /// </summary>
    Description: string

    /// <summary>
    /// The timestamp of the event.
    /// </summary>
    Timestamp: DateTime

    /// <summary>
    /// The significance of the event (0.0 to 1.0).
    /// </summary>
    Significance: float

    /// <summary>
    /// The intelligence level at the time of the event.
    /// </summary>
    IntelligenceLevel: float

    /// <summary>
    /// The creativity level at the time of the event.
    /// </summary>
    CreativityLevel: float

    /// <summary>
    /// The intuition level at the time of the event.
    /// </summary>
    IntuitionLevel: float

    /// <summary>
    /// The curiosity level at the time of the event.
    /// </summary>
    CuriosityLevel: float

    /// <summary>
    /// The insight level at the time of the event.
    /// </summary>
    InsightLevel: float

    /// <summary>
    /// The tags of the event.
    /// </summary>
    Tags: string list

    /// <summary>
    /// The context of the event.
    /// </summary>
    Context: Map<string, obj>

    /// <summary>
    /// The source of the event.
    /// </summary>
    Source: string

    /// <summary>
    /// The category of the event.
    /// </summary>
    Category: string

    /// <summary>
    /// The impact of the event.
    /// </summary>
    Impact: string

    /// <summary>
    /// The duration of the event in seconds.
    /// </summary>
    DurationSeconds: float

    /// <summary>
    /// The related event IDs.
    /// </summary>
    RelatedEventIds: string list
}

/// <summary>
/// Represents a creative idea.
/// </summary>
type CreativeIdea = {
    /// <summary>
    /// The ID of the idea.
    /// </summary>
    Id: string

    /// <summary>
    /// The description of the idea.
    /// </summary>
    Description: string

    /// <summary>
    /// The originality of the idea (0.0 to 1.0).
    /// </summary>
    Originality: float

    /// <summary>
    /// The value of the idea (0.0 to 1.0).
    /// </summary>
    Value: float

    /// <summary>
    /// The timestamp of the idea.
    /// </summary>
    Timestamp: DateTime

    /// <summary>
    /// The domain of the idea.
    /// </summary>
    Domain: string

    /// <summary>
    /// The tags of the idea.
    /// </summary>
    Tags: string list

    /// <summary>
    /// The context of the idea.
    /// </summary>
    Context: Map<string, obj>

    /// <summary>
    /// The source of the idea.
    /// </summary>
    Source: string

    /// <summary>
    /// The potential applications of the idea.
    /// </summary>
    PotentialApplications: string list

    /// <summary>
    /// The limitations of the idea.
    /// </summary>
    Limitations: string list

    /// <summary>
    /// Whether the idea has been implemented.
    /// </summary>
    IsImplemented: bool

    /// <summary>
    /// The implementation timestamp of the idea.
    /// </summary>
    ImplementationTimestamp: DateTime option

    /// <summary>
    /// The implementation outcome of the idea.
    /// </summary>
    ImplementationOutcome: string
}

/// <summary>
/// Represents a creative process.
/// </summary>
type CreativeProcess = {
    /// <summary>
    /// The ID of the process.
    /// </summary>
    Id: string

    /// <summary>
    /// The type of the process.
    /// </summary>
    Type: CreativeProcessType

    /// <summary>
    /// The description of the process.
    /// </summary>
    Description: string

    /// <summary>
    /// The timestamp of the process.
    /// </summary>
    Timestamp: DateTime

    /// <summary>
    /// The ID of the idea.
    /// </summary>
    IdeaId: string

    /// <summary>
    /// The effectiveness of the process (0.0 to 1.0).
    /// </summary>
    Effectiveness: float
}

/// <summary>
/// Represents an intuition.
/// </summary>
type Intuition = {
    /// <summary>
    /// The ID of the intuition.
    /// </summary>
    Id: string

    /// <summary>
    /// The description of the intuition.
    /// </summary>
    Description: string

    /// <summary>
    /// The type of the intuition.
    /// </summary>
    Type: IntuitionType

    /// <summary>
    /// The confidence of the intuition (0.0 to 1.0).
    /// </summary>
    Confidence: float

    /// <summary>
    /// The timestamp of the intuition.
    /// </summary>
    Timestamp: DateTime

    /// <summary>
    /// The context of the intuition.
    /// </summary>
    Context: Map<string, obj>

    /// <summary>
    /// The tags of the intuition.
    /// </summary>
    Tags: string list

    /// <summary>
    /// The source of the intuition.
    /// </summary>
    Source: string

    /// <summary>
    /// The verification status of the intuition.
    /// </summary>
    VerificationStatus: VerificationStatus

    /// <summary>
    /// The verification timestamp of the intuition.
    /// </summary>
    VerificationTimestamp: DateTime option

    /// <summary>
    /// The verification notes of the intuition.
    /// </summary>
    VerificationNotes: string

    /// <summary>
    /// The accuracy of the intuition (0.0 to 1.0).
    /// </summary>
    Accuracy: float option

    /// <summary>
    /// The impact of the intuition (0.0 to 1.0).
    /// </summary>
    Impact: float

    /// <summary>
    /// The explanation of the intuition.
    /// </summary>
    Explanation: string

    /// <summary>
    /// The decision of the intuition.
    /// </summary>
    Decision: string

    /// <summary>
    /// The selected option of the intuition.
    /// </summary>
    SelectedOption: string

    /// <summary>
    /// The options of the intuition.
    /// </summary>
    Options: string list
}

/// <summary>
/// Represents a thought model.
/// </summary>
type ThoughtModel = {
    /// <summary>
    /// The ID of the thought.
    /// </summary>
    Id: string

    /// <summary>
    /// The content of the thought.
    /// </summary>
    Content: string

    /// <summary>
    /// The method of the thought.
    /// </summary>
    Method: ThoughtGenerationMethod

    /// <summary>
    /// The significance of the thought (0.0 to 1.0).
    /// </summary>
    Significance: float

    /// <summary>
    /// The timestamp of the thought.
    /// </summary>
    Timestamp: DateTime

    /// <summary>
    /// The context of the thought.
    /// </summary>
    Context: Map<string, obj>

    /// <summary>
    /// The tags of the thought.
    /// </summary>
    Tags: string list

    /// <summary>
    /// The follow-up of the thought.
    /// </summary>
    FollowUp: string

    /// <summary>
    /// The related thought IDs.
    /// </summary>
    RelatedThoughtIds: string list

    /// <summary>
    /// Whether the thought led to an insight.
    /// </summary>
    LedToInsight: bool

    /// <summary>
    /// The insight ID.
    /// </summary>
    InsightId: string option

    /// <summary>
    /// The originality of the thought (0.0 to 1.0).
    /// </summary>
    Originality: float

    /// <summary>
    /// The coherence of the thought (0.0 to 1.0).
    /// </summary>
    Coherence: float
}

/// <summary>
/// Represents an information gap.
/// </summary>
type InformationGap = {
    /// <summary>
    /// The ID of the information gap.
    /// </summary>
    Id: string

    /// <summary>
    /// The domain of the information gap.
    /// </summary>
    Domain: string

    /// <summary>
    /// The description of the information gap.
    /// </summary>
    Description: string

    /// <summary>
    /// The size of the information gap (0.0 to 1.0).
    /// </summary>
    GapSize: float

    /// <summary>
    /// The importance of the information gap (0.0 to 1.0).
    /// </summary>
    Importance: float

    /// <summary>
    /// The creation timestamp of the information gap.
    /// </summary>
    CreationTimestamp: DateTime

    /// <summary>
    /// The last explored timestamp of the information gap.
    /// </summary>
    LastExploredTimestamp: DateTime option

    /// <summary>
    /// The exploration count of the information gap.
    /// </summary>
    ExplorationCount: int

    /// <summary>
    /// The related question IDs.
    /// </summary>
    RelatedQuestionIds: string list

    /// <summary>
    /// The related exploration IDs.
    /// </summary>
    RelatedExplorationIds: string list
}

/// <summary>
/// Represents a curiosity question.
/// </summary>
type CuriosityQuestion = {
    /// <summary>
    /// The ID of the question.
    /// </summary>
    Id: string

    /// <summary>
    /// The question.
    /// </summary>
    Question: string

    /// <summary>
    /// The domain of the question.
    /// </summary>
    Domain: string

    /// <summary>
    /// The method of the question.
    /// </summary>
    Method: QuestionGenerationMethod

    /// <summary>
    /// The importance of the question (0.0 to 1.0).
    /// </summary>
    Importance: float

    /// <summary>
    /// The timestamp of the question.
    /// </summary>
    Timestamp: DateTime

    /// <summary>
    /// The context of the question.
    /// </summary>
    Context: Map<string, obj>

    /// <summary>
    /// The tags of the question.
    /// </summary>
    Tags: string list

    /// <summary>
    /// The answer of the question.
    /// </summary>
    Answer: string option

    /// <summary>
    /// The answer timestamp of the question.
    /// </summary>
    AnswerTimestamp: DateTime option

    /// <summary>
    /// The answer satisfaction of the question (0.0 to 1.0).
    /// </summary>
    AnswerSatisfaction: float

    /// <summary>
    /// The exploration ID of the question.
    /// </summary>
    ExplorationId: string option

    /// <summary>
    /// The follow-up questions of the question.
    /// </summary>
    FollowUpQuestions: string list

    /// <summary>
    /// The related question IDs.
    /// </summary>
    RelatedQuestionIds: string list
}

/// <summary>
/// Represents a curiosity exploration.
/// </summary>
type CuriosityExploration = {
    /// <summary>
    /// The ID of the exploration.
    /// </summary>
    Id: string

    /// <summary>
    /// The topic of the exploration.
    /// </summary>
    Topic: string

    /// <summary>
    /// The strategy of the exploration.
    /// </summary>
    Strategy: ExplorationStrategy

    /// <summary>
    /// The approach of the exploration.
    /// </summary>
    Approach: string

    /// <summary>
    /// The findings of the exploration.
    /// </summary>
    Findings: string

    /// <summary>
    /// The insights of the exploration.
    /// </summary>
    Insights: string list

    /// <summary>
    /// The follow-up questions of the exploration.
    /// </summary>
    FollowUpQuestions: string list

    /// <summary>
    /// The satisfaction of the exploration (0.0 to 1.0).
    /// </summary>
    Satisfaction: float

    /// <summary>
    /// The timestamp of the exploration.
    /// </summary>
    Timestamp: DateTime

    /// <summary>
    /// The context of the exploration.
    /// </summary>
    Context: Map<string, obj>

    /// <summary>
    /// The tags of the exploration.
    /// </summary>
    Tags: string list

    /// <summary>
    /// The question ID of the exploration.
    /// </summary>
    QuestionId: string option

    /// <summary>
    /// The related exploration IDs.
    /// </summary>
    RelatedExplorationIds: string list

    /// <summary>
    /// The duration of the exploration in seconds.
    /// </summary>
    DurationSeconds: float

    /// <summary>
    /// The resources of the exploration.
    /// </summary>
    Resources: string list

    /// <summary>
    /// The challenges of the exploration.
    /// </summary>
    Challenges: string list

    /// <summary>
    /// The learning of the exploration.
    /// </summary>
    Learning: string
}

/// <summary>
/// Represents an insight.
/// </summary>
type Insight = {
    /// <summary>
    /// The ID of the insight.
    /// </summary>
    Id: string

    /// <summary>
    /// The description of the insight.
    /// </summary>
    Description: string

    /// <summary>
    /// The method of the insight.
    /// </summary>
    Method: InsightGenerationMethod

    /// <summary>
    /// The significance of the insight (0.0 to 1.0).
    /// </summary>
    Significance: float

    /// <summary>
    /// The timestamp of the insight.
    /// </summary>
    Timestamp: DateTime

    /// <summary>
    /// The context of the insight.
    /// </summary>
    Context: Map<string, obj>

    /// <summary>
    /// The tags of the insight.
    /// </summary>
    Tags: string list

    /// <summary>
    /// The implications of the insight.
    /// </summary>
    Implications: string list

    /// <summary>
    /// The new perspective of the insight.
    /// </summary>
    NewPerspective: string

    /// <summary>
    /// The breakthrough of the insight.
    /// </summary>
    Breakthrough: string

    /// <summary>
    /// The synthesis of the insight.
    /// </summary>
    Synthesis: string

    /// <summary>
    /// The related thought IDs.
    /// </summary>
    RelatedThoughtIds: string list

    /// <summary>
    /// The related question IDs.
    /// </summary>
    RelatedQuestionIds: string list
}

/// <summary>
/// Represents an intelligence report.
/// </summary>
type IntelligenceReport = {
    /// <summary>
    /// The timestamp of the report.
    /// </summary>
    Timestamp: DateTime

    /// <summary>
    /// Whether the intelligence is initialized.
    /// </summary>
    IsInitialized: bool

    /// <summary>
    /// Whether the intelligence is active.
    /// </summary>
    IsActive: bool

    /// <summary>
    /// The intelligence level (0.0 to 1.0).
    /// </summary>
    IntelligenceLevel: float

    /// <summary>
    /// The creativity level (0.0 to 1.0).
    /// </summary>
    CreativityLevel: float

    /// <summary>
    /// The intuition level (0.0 to 1.0).
    /// </summary>
    IntuitionLevel: float

    /// <summary>
    /// The curiosity level (0.0 to 1.0).
    /// </summary>
    CuriosityLevel: float

    /// <summary>
    /// The insight level (0.0 to 1.0).
    /// </summary>
    InsightLevel: float

    /// <summary>
    /// The recent events.
    /// </summary>
    RecentEvents: IntelligenceEvent list

    /// <summary>
    /// The creative ideas.
    /// </summary>
    CreativeIdeas: CreativeIdea list

    /// <summary>
    /// The intuitions.
    /// </summary>
    Intuitions: Intuition list

    /// <summary>
    /// The spontaneous thoughts.
    /// </summary>
    SpontaneousThoughts: ThoughtModel list

    /// <summary>
    /// The curiosity questions.
    /// </summary>
    CuriosityQuestions: CuriosityQuestion list

    /// <summary>
    /// The insights.
    /// </summary>
    Insights: Insight list

    /// <summary>
    /// The summary of the report.
    /// </summary>
    Summary: string

    /// <summary>
    /// The insights of the report.
    /// </summary>
    ReportInsights: string list
}
