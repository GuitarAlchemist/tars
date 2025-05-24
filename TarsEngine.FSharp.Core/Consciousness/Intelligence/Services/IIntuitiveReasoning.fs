namespace TarsEngine.FSharp.Core.Consciousness.Intelligence.Services

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Core.Consciousness.Intelligence

/// <summary>
/// Interface for intuitive reasoning capabilities.
/// </summary>
type IIntuitiveReasoning =
    /// <summary>
    /// Gets the intuition level (0.0 to 1.0).
    /// </summary>
    abstract member IntuitionLevel: float

    /// <summary>
    /// Gets the pattern recognition level (0.0 to 1.0).
    /// </summary>
    abstract member PatternRecognitionLevel: float

    /// <summary>
    /// Gets the heuristic reasoning level (0.0 to 1.0).
    /// </summary>
    abstract member HeuristicReasoningLevel: float

    /// <summary>
    /// Gets the gut feeling level (0.0 to 1.0).
    /// </summary>
    abstract member GutFeelingLevel: float

    /// <summary>
    /// Gets the intuitions.
    /// </summary>
    abstract member Intuitions: Intuition list

    /// <summary>
    /// Initializes the intuitive reasoning.
    /// </summary>
    /// <returns>True if initialization was successful.</returns>
    abstract member InitializeAsync: unit -> Task<bool>

    /// <summary>
    /// Activates the intuitive reasoning.
    /// </summary>
    /// <returns>True if activation was successful.</returns>
    abstract member ActivateAsync: unit -> Task<bool>

    /// <summary>
    /// Deactivates the intuitive reasoning.
    /// </summary>
    /// <returns>True if deactivation was successful.</returns>
    abstract member DeactivateAsync: unit -> Task<bool>

    /// <summary>
    /// Updates the intuitive reasoning.
    /// </summary>
    /// <returns>True if update was successful.</returns>
    abstract member UpdateAsync: unit -> Task<bool>

    /// <summary>
    /// Generates an intuition.
    /// </summary>
    /// <returns>The generated intuition.</returns>
    abstract member GenerateIntuitionAsync: unit -> Task<Intuition option>

    /// <summary>
    /// Makes an intuitive decision.
    /// </summary>
    /// <param name="options">The options.</param>
    /// <param name="intuitionType">The intuition type.</param>
    /// <returns>The selected option and the intuition.</returns>
    abstract member MakeIntuitiveDecisionAsync: options: string list * intuitionType: IntuitionType option -> Task<string * Intuition>

    /// <summary>
    /// Evaluates options intuitively.
    /// </summary>
    /// <param name="options">The options.</param>
    /// <param name="intuitionType">The intuition type.</param>
    /// <returns>The option scores.</returns>
    abstract member EvaluateOptionsIntuitivelyAsync: options: string list * intuitionType: IntuitionType option -> Task<Map<string, float>>

    /// <summary>
    /// Gets recent intuitions.
    /// </summary>
    /// <param name="count">The number of intuitions to get.</param>
    /// <returns>The recent intuitions.</returns>
    abstract member GetRecentIntuitions: count: int -> Intuition list

    /// <summary>
    /// Gets intuitions by type.
    /// </summary>
    /// <param name="intuitionType">The intuition type.</param>
    /// <param name="count">The number of intuitions to get.</param>
    /// <returns>The intuitions of the specified type.</returns>
    abstract member GetIntuitionsByType: intuitionType: IntuitionType * count: int -> Intuition list

    /// <summary>
    /// Gets intuitions by tag.
    /// </summary>
    /// <param name="tag">The tag.</param>
    /// <param name="count">The number of intuitions to get.</param>
    /// <returns>The intuitions with the tag.</returns>
    abstract member GetIntuitionsByTag: tag: string * count: int -> Intuition list

    /// <summary>
    /// Gets the most confident intuitions.
    /// </summary>
    /// <param name="count">The number of intuitions to get.</param>
    /// <returns>The most confident intuitions.</returns>
    abstract member GetMostConfidentIntuitions: count: int -> Intuition list

    /// <summary>
    /// Gets the most accurate intuitions.
    /// </summary>
    /// <param name="count">The number of intuitions to get.</param>
    /// <returns>The most accurate intuitions.</returns>
    abstract member GetMostAccurateIntuitions: count: int -> Intuition list

    /// <summary>
    /// Verifies an intuition.
    /// </summary>
    /// <param name="intuitionId">The intuition ID.</param>
    /// <param name="isCorrect">Whether the intuition is correct.</param>
    /// <param name="accuracy">The accuracy.</param>
    /// <param name="notes">The verification notes.</param>
    /// <returns>The updated intuition.</returns>
    abstract member VerifyIntuitionAsync: intuitionId: string * isCorrect: bool * accuracy: float * notes: string -> Task<Intuition option>
