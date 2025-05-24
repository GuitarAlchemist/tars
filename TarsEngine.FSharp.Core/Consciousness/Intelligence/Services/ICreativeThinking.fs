namespace TarsEngine.FSharp.Core.Consciousness.Intelligence.Services

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Core.Consciousness.Intelligence

/// <summary>
/// Interface for creative thinking capabilities.
/// </summary>
type ICreativeThinking =
    /// <summary>
    /// Gets the creativity level (0.0 to 1.0).
    /// </summary>
    abstract member CreativityLevel: float

    /// <summary>
    /// Gets the divergent thinking level (0.0 to 1.0).
    /// </summary>
    abstract member DivergentThinkingLevel: float

    /// <summary>
    /// Gets the convergent thinking level (0.0 to 1.0).
    /// </summary>
    abstract member ConvergentThinkingLevel: float

    /// <summary>
    /// Gets the combinatorial creativity level (0.0 to 1.0).
    /// </summary>
    abstract member CombinatorialCreativityLevel: float

    /// <summary>
    /// Gets the creative ideas.
    /// </summary>
    abstract member CreativeIdeas: CreativeIdea list

    /// <summary>
    /// Gets the creative processes.
    /// </summary>
    abstract member CreativeProcesses: CreativeProcess list

    /// <summary>
    /// Initializes the creative thinking.
    /// </summary>
    /// <returns>True if initialization was successful.</returns>
    abstract member InitializeAsync: unit -> Task<bool>

    /// <summary>
    /// Activates the creative thinking.
    /// </summary>
    /// <returns>True if activation was successful.</returns>
    abstract member ActivateAsync: unit -> Task<bool>

    /// <summary>
    /// Deactivates the creative thinking.
    /// </summary>
    /// <returns>True if deactivation was successful.</returns>
    abstract member DeactivateAsync: unit -> Task<bool>

    /// <summary>
    /// Updates the creative thinking.
    /// </summary>
    /// <returns>True if update was successful.</returns>
    abstract member UpdateAsync: unit -> Task<bool>

    /// <summary>
    /// Generates a creative idea.
    /// </summary>
    /// <returns>The generated creative idea.</returns>
    abstract member GenerateCreativeIdeaAsync: unit -> Task<CreativeIdea option>

    /// <summary>
    /// Generates a creative solution to a problem.
    /// </summary>
    /// <param name="problem">The problem description.</param>
    /// <param name="constraints">The constraints.</param>
    /// <returns>The creative solution.</returns>
    abstract member GenerateCreativeSolutionAsync: problem: string * constraints: string list option -> Task<CreativeIdea option>

    /// <summary>
    /// Gets recent ideas.
    /// </summary>
    /// <param name="count">The number of ideas to get.</param>
    /// <returns>The recent ideas.</returns>
    abstract member GetRecentIdeas: count: int -> CreativeIdea list

    /// <summary>
    /// Gets ideas by domain.
    /// </summary>
    /// <param name="domain">The domain.</param>
    /// <param name="count">The number of ideas to get.</param>
    /// <returns>The ideas in the domain.</returns>
    abstract member GetIdeasByDomain: domain: string * count: int -> CreativeIdea list

    /// <summary>
    /// Gets ideas by tag.
    /// </summary>
    /// <param name="tag">The tag.</param>
    /// <param name="count">The number of ideas to get.</param>
    /// <returns>The ideas with the tag.</returns>
    abstract member GetIdeasByTag: tag: string * count: int -> CreativeIdea list

    /// <summary>
    /// Gets the most original ideas.
    /// </summary>
    /// <param name="count">The number of ideas to get.</param>
    /// <returns>The most original ideas.</returns>
    abstract member GetMostOriginalIdeas: count: int -> CreativeIdea list

    /// <summary>
    /// Gets the most valuable ideas.
    /// </summary>
    /// <param name="count">The number of ideas to get.</param>
    /// <returns>The most valuable ideas.</returns>
    abstract member GetMostValuableIdeas: count: int -> CreativeIdea list

    /// <summary>
    /// Gets the most effective creative processes.
    /// </summary>
    /// <param name="count">The number of processes to get.</param>
    /// <returns>The most effective creative processes.</returns>
    abstract member GetMostEffectiveProcesses: count: int -> CreativeProcess list
