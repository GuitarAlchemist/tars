namespace TarsEngine.FSharp.Core.Consciousness.Intelligence.Services

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Core.Consciousness.Core
open TarsEngine.FSharp.Core.Consciousness.Intelligence

/// <summary>
/// Interface for intelligence spark capabilities.
/// </summary>
type IIntelligenceSpark =
    /// <summary>
    /// Gets whether the intelligence spark is initialized.
    /// </summary>
    abstract member IsInitialized: bool

    /// <summary>
    /// Gets whether the intelligence spark is active.
    /// </summary>
    abstract member IsActive: bool

    /// <summary>
    /// Gets the intelligence level (0.0 to 1.0).
    /// </summary>
    abstract member IntelligenceLevel: float

    /// <summary>
    /// Gets the creativity level (0.0 to 1.0).
    /// </summary>
    abstract member CreativityLevel: float

    /// <summary>
    /// Gets the intuition level (0.0 to 1.0).
    /// </summary>
    abstract member IntuitionLevel: float

    /// <summary>
    /// Gets the curiosity level (0.0 to 1.0).
    /// </summary>
    abstract member CuriosityLevel: float

    /// <summary>
    /// Gets the insight level (0.0 to 1.0).
    /// </summary>
    abstract member InsightLevel: float

    /// <summary>
    /// Gets the intelligence events.
    /// </summary>
    abstract member Events: IntelligenceEvent list

    /// <summary>
    /// Gets the creative thinking component.
    /// </summary>
    abstract member CreativeThinking: ICreativeThinking

    /// <summary>
    /// Gets the intuitive reasoning component.
    /// </summary>
    abstract member IntuitiveReasoning: IIntuitiveReasoning

    /// <summary>
    /// Gets the spontaneous thought component.
    /// </summary>
    abstract member SpontaneousThought: ISpontaneousThought

    /// <summary>
    /// Gets the curiosity drive component.
    /// </summary>
    abstract member CuriosityDrive: ICuriosityDrive

    /// <summary>
    /// Gets the insight generation component.
    /// </summary>
    abstract member InsightGeneration: IInsightGeneration

    /// <summary>
    /// Initializes the intelligence spark.
    /// </summary>
    /// <returns>True if initialization was successful.</returns>
    abstract member InitializeAsync: unit -> Task<bool>

    /// <summary>
    /// Activates the intelligence spark.
    /// </summary>
    /// <returns>True if activation was successful.</returns>
    abstract member ActivateAsync: unit -> Task<bool>

    /// <summary>
    /// Deactivates the intelligence spark.
    /// </summary>
    /// <returns>True if deactivation was successful.</returns>
    abstract member DeactivateAsync: unit -> Task<bool>

    /// <summary>
    /// Updates the intelligence spark.
    /// </summary>
    /// <returns>True if update was successful.</returns>
    abstract member UpdateAsync: unit -> Task<bool>

    /// <summary>
    /// Gets the intelligence report.
    /// </summary>
    /// <returns>The intelligence report.</returns>
    abstract member GetIntelligenceReport: unit -> IntelligenceReport

    /// <summary>
    /// Records an intelligence event.
    /// </summary>
    /// <param name="type">The event type.</param>
    /// <param name="description">The event description.</param>
    /// <param name="significance">The event significance (0.0 to 1.0).</param>
    /// <returns>The recorded event.</returns>
    abstract member RecordEvent: type': IntelligenceEventType * description: string * significance: float -> IntelligenceEvent

    /// <summary>
    /// Gets recent events.
    /// </summary>
    /// <param name="count">The number of events to get.</param>
    /// <returns>The recent events.</returns>
    abstract member GetRecentEvents: count: int -> IntelligenceEvent list

    /// <summary>
    /// Gets events by type.
    /// </summary>
    /// <param name="type">The event type.</param>
    /// <param name="count">The number of events to get.</param>
    /// <returns>The events of the specified type.</returns>
    abstract member GetEventsByType: type': IntelligenceEventType * count: int -> IntelligenceEvent list

    /// <summary>
    /// Gets the most significant events.
    /// </summary>
    /// <param name="count">The number of events to get.</param>
    /// <returns>The most significant events.</returns>
    abstract member GetMostSignificantEvents: count: int -> IntelligenceEvent list

    /// <summary>
    /// Generates a creative solution to a problem.
    /// </summary>
    /// <param name="problem">The problem description.</param>
    /// <param name="constraints">The constraints.</param>
    /// <returns>The creative solution.</returns>
    abstract member GenerateCreativeSolutionAsync: problem: string * constraints: string list option -> Task<CreativeIdea option>

    /// <summary>
    /// Makes an intuitive decision.
    /// </summary>
    /// <param name="options">The options.</param>
    /// <param name="intuitionType">The intuition type.</param>
    /// <returns>The selected option and the intuition.</returns>
    abstract member MakeIntuitiveDecisionAsync: options: string list * intuitionType: IntuitionType option -> Task<string * Intuition>

    /// <summary>
    /// Explores a curiosity topic.
    /// </summary>
    /// <param name="topic">The topic.</param>
    /// <returns>The exploration.</returns>
    abstract member ExploreCuriosityTopicAsync: topic: string -> Task<CuriosityExploration option>

    /// <summary>
    /// Connects ideas for an insight.
    /// </summary>
    /// <param name="ideas">The ideas.</param>
    /// <returns>The insight.</returns>
    abstract member ConnectIdeasForInsightAsync: ideas: string list -> Task<Insight option>
