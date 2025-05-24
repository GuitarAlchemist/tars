namespace TarsEngine.FSharp.Core.Consciousness.Services

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Core.Consciousness.Core

/// <summary>
/// Interface for consciousness services.
/// </summary>
type IConsciousnessService =
    /// <summary>
    /// Gets the current mental state.
    /// </summary>
    abstract member GetCurrentMentalState : unit -> Task<MentalState>
    
    /// <summary>
    /// Updates the mental state.
    /// </summary>
    /// <param name="newState">The new mental state.</param>
    abstract member UpdateMentalState : newState:MentalState -> Task<unit>
    
    /// <summary>
    /// Updates the consciousness level.
    /// </summary>
    /// <param name="level">The new consciousness level.</param>
    abstract member UpdateConsciousnessLevel : level:ConsciousnessLevel -> Task<unit>
    
    /// <summary>
    /// Updates the emotional state.
    /// </summary>
    /// <param name="emotionalState">The new emotional state.</param>
    abstract member UpdateEmotionalState : emotionalState:EmotionalState -> Task<unit>
    
    /// <summary>
    /// Adds an emotion.
    /// </summary>
    /// <param name="emotion">The emotion to add.</param>
    abstract member AddEmotion : emotion:Emotion -> Task<unit>
    
    /// <summary>
    /// Sets the current thought process.
    /// </summary>
    /// <param name="thoughtProcess">The thought process to set.</param>
    abstract member SetThoughtProcess : thoughtProcess:ThoughtProcess -> Task<unit>
    
    /// <summary>
    /// Sets the attention focus.
    /// </summary>
    /// <param name="focus">The focus to set.</param>
    abstract member SetAttentionFocus : focus:string -> Task<unit>
    
    /// <summary>
    /// Adds a memory entry.
    /// </summary>
    /// <param name="memory">The memory to add.</param>
    abstract member AddMemory : memory:MemoryEntry -> Task<unit>
    
    /// <summary>
    /// Retrieves memories by tag.
    /// </summary>
    /// <param name="tag">The tag to search for.</param>
    /// <returns>The list of memories with the specified tag.</returns>
    abstract member GetMemoriesByTag : tag:string -> Task<MemoryEntry list>
    
    /// <summary>
    /// Retrieves memories by importance.
    /// </summary>
    /// <param name="minImportance">The minimum importance.</param>
    /// <returns>The list of memories with at least the specified importance.</returns>
    abstract member GetMemoriesByImportance : minImportance:float -> Task<MemoryEntry list>
    
    /// <summary>
    /// Gets the self model.
    /// </summary>
    abstract member GetSelfModel : unit -> Task<SelfModel>
    
    /// <summary>
    /// Updates the self model.
    /// </summary>
    /// <param name="newSelfModel">The new self model.</param>
    abstract member UpdateSelfModel : newSelfModel:SelfModel -> Task<unit>
    
    /// <summary>
    /// Performs self reflection.
    /// </summary>
    /// <param name="topic">The topic to reflect on.</param>
    /// <returns>The self reflection.</returns>
    abstract member PerformSelfReflection : topic:string -> Task<SelfReflection>
    
    /// <summary>
    /// Evaluates a value alignment.
    /// </summary>
    /// <param name="value">The value to evaluate.</param>
    /// <param name="action">The action to evaluate.</param>
    /// <returns>The value alignment.</returns>
    abstract member EvaluateValueAlignment : value:Value * action:string -> Task<ValueAlignment>
    
    /// <summary>
    /// Performs mental optimization.
    /// </summary>
    /// <param name="type">The type of optimization.</param>
    /// <param name="target">The target of the optimization.</param>
    /// <returns>The mental optimization.</returns>
    abstract member PerformMentalOptimization : type':OptimizationType * target:string -> Task<MentalOptimization>
    
    /// <summary>
    /// Generates a consciousness report.
    /// </summary>
    /// <returns>The consciousness report.</returns>
    abstract member GenerateReport : unit -> Task<ConsciousnessReport>
