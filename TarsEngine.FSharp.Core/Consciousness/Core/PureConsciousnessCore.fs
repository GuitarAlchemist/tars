namespace TarsEngine.FSharp.Core.Consciousness.Core

open System
open Microsoft.Extensions.Logging

/// <summary>
/// Pure functional implementation of consciousness functionality.
/// </summary>
type PureConsciousnessCore(logger: ILogger<PureConsciousnessCore>) =
    
    /// <summary>
    /// Creates a new pure mental state.
    /// </summary>
    /// <returns>The new pure mental state.</returns>
    member _.CreateInitialMentalState() =
        {
            ConsciousnessLevel = {
                LevelType = ConsciousnessLevelType.Conscious
                Intensity = 0.7
            }
            EmotionalState = {
                Emotions = []
                DominantEmotion = None
            }
            CurrentThoughtType = None
            AttentionFocus = None
        }
    
    /// <summary>
    /// Updates the consciousness level of a mental state.
    /// </summary>
    /// <param name="state">The mental state to update.</param>
    /// <param name="levelType">The new consciousness level type.</param>
    /// <param name="intensity">The new consciousness level intensity.</param>
    /// <returns>The updated mental state.</returns>
    member _.UpdateConsciousnessLevel(state: PureMentalState, levelType: ConsciousnessLevelType, intensity: float) =
        let newLevel = {
            LevelType = levelType
            Intensity = intensity
        }
        
        { state with ConsciousnessLevel = newLevel }
    
    /// <summary>
    /// Adds an emotion to a mental state.
    /// </summary>
    /// <param name="state">The mental state to update.</param>
    /// <param name="category">The emotion category to add.</param>
    /// <param name="intensity">The emotion intensity.</param>
    /// <returns>The updated mental state.</returns>
    member _.AddEmotion(state: PureMentalState, category: EmotionCategory, intensity: float) =
        let currentEmotions = state.EmotionalState.Emotions
        let newEmotions = (category, intensity) :: currentEmotions
        
        // Determine the dominant emotion
        let dominantEmotion = 
            newEmotions
            |> List.sortByDescending snd
            |> List.tryHead
            |> Option.map fst
        
        let newEmotionalState = {
            Emotions = newEmotions
            DominantEmotion = dominantEmotion
        }
        
        { state with EmotionalState = newEmotionalState }
    
    /// <summary>
    /// Sets the current thought type of a mental state.
    /// </summary>
    /// <param name="state">The mental state to update.</param>
    /// <param name="thoughtType">The new thought type.</param>
    /// <returns>The updated mental state.</returns>
    member _.SetThoughtType(state: PureMentalState, thoughtType: ThoughtType) =
        { state with CurrentThoughtType = Some thoughtType }
    
    /// <summary>
    /// Sets the attention focus of a mental state.
    /// </summary>
    /// <param name="state">The mental state to update.</param>
    /// <param name="focus">The new attention focus.</param>
    /// <returns>The updated mental state.</returns>
    member _.SetAttentionFocus(state: PureMentalState, focus: string) =
        { state with AttentionFocus = Some focus }
    
    /// <summary>
    /// Clears the emotions of a mental state.
    /// </summary>
    /// <param name="state">The mental state to update.</param>
    /// <returns>The updated mental state.</returns>
    member _.ClearEmotions(state: PureMentalState) =
        let newEmotionalState = {
            Emotions = []
            DominantEmotion = None
        }
        
        { state with EmotionalState = newEmotionalState }
    
    /// <summary>
    /// Clears the thought type of a mental state.
    /// </summary>
    /// <param name="state">The mental state to update.</param>
    /// <returns>The updated mental state.</returns>
    member _.ClearThoughtType(state: PureMentalState) =
        { state with CurrentThoughtType = None }
    
    /// <summary>
    /// Clears the attention focus of a mental state.
    /// </summary>
    /// <param name="state">The mental state to update.</param>
    /// <returns>The updated mental state.</returns>
    member _.ClearAttentionFocus(state: PureMentalState) =
        { state with AttentionFocus = None }
    
    /// <summary>
    /// Resets a mental state to its initial state.
    /// </summary>
    /// <returns>The reset mental state.</returns>
    member this.ResetMentalState() =
        this.CreateInitialMentalState()
    
    /// <summary>
    /// Converts a pure mental state to a full mental state.
    /// </summary>
    /// <param name="state">The pure mental state to convert.</param>
    /// <returns>The full mental state.</returns>
    member _.ToPureMentalState(state: MentalState) =
        {
            ConsciousnessLevel = {
                LevelType = state.ConsciousnessLevel.LevelType
                Intensity = state.ConsciousnessLevel.Intensity
            }
            EmotionalState = {
                Emotions = 
                    state.EmotionalState.Emotions
                    |> List.map (fun e -> (e.Category, e.Intensity))
                DominantEmotion = 
                    state.EmotionalState.DominantEmotion
                    |> Option.map (fun e -> e.Category)
            }
            CurrentThoughtType = 
                state.CurrentThoughtProcess
                |> Option.map (fun tp -> tp.Type)
            AttentionFocus = state.AttentionFocus
        }
    
    /// <summary>
    /// Converts a full mental state to a pure mental state.
    /// </summary>
    /// <param name="state">The full mental state to convert.</param>
    /// <returns>The pure mental state.</returns>
    member _.ToFullMentalState(state: PureMentalState) =
        {
            ConsciousnessLevel = {
                LevelType = state.ConsciousnessLevel.LevelType
                Intensity = state.ConsciousnessLevel.Intensity
                Description = 
                    match state.ConsciousnessLevel.LevelType with
                    | ConsciousnessLevelType.Unconscious -> "Unconscious state"
                    | ConsciousnessLevelType.Subconscious -> "Subconscious state"
                    | ConsciousnessLevelType.Conscious -> "Conscious state"
                    | ConsciousnessLevelType.Metaconscious -> "Metaconscious state"
                    | ConsciousnessLevelType.Superconscious -> "Superconscious state"
                    | ConsciousnessLevelType.Custom name -> $"Custom state: {name}"
                Data = Map.empty
            }
            EmotionalState = {
                Emotions = 
                    state.EmotionalState.Emotions
                    |> List.map (fun (category, intensity) -> 
                        {
                            Category = category
                            Intensity = intensity
                            Description = 
                                match category with
                                | EmotionCategory.Joy -> "Feeling of joy"
                                | EmotionCategory.Sadness -> "Feeling of sadness"
                                | EmotionCategory.Anger -> "Feeling of anger"
                                | EmotionCategory.Fear -> "Feeling of fear"
                                | EmotionCategory.Surprise -> "Feeling of surprise"
                                | EmotionCategory.Disgust -> "Feeling of disgust"
                                | EmotionCategory.Trust -> "Feeling of trust"
                                | EmotionCategory.Anticipation -> "Feeling of anticipation"
                                | EmotionCategory.Interest -> "Feeling of interest"
                                | EmotionCategory.Confusion -> "Feeling of confusion"
                                | EmotionCategory.Curiosity -> "Feeling of curiosity"
                                | EmotionCategory.Satisfaction -> "Feeling of satisfaction"
                                | EmotionCategory.Custom name -> $"Feeling of {name}"
                            Trigger = None
                            Duration = None
                            Data = Map.empty
                        }
                    )
                DominantEmotion = 
                    state.EmotionalState.DominantEmotion
                    |> Option.map (fun category -> 
                        let intensity = 
                            state.EmotionalState.Emotions
                            |> List.tryFind (fun (c, _) -> c = category)
                            |> Option.map snd
                            |> Option.defaultValue 0.5
                        
                        {
                            Category = category
                            Intensity = intensity
                            Description = 
                                match category with
                                | EmotionCategory.Joy -> "Feeling of joy"
                                | EmotionCategory.Sadness -> "Feeling of sadness"
                                | EmotionCategory.Anger -> "Feeling of anger"
                                | EmotionCategory.Fear -> "Feeling of fear"
                                | EmotionCategory.Surprise -> "Feeling of surprise"
                                | EmotionCategory.Disgust -> "Feeling of disgust"
                                | EmotionCategory.Trust -> "Feeling of trust"
                                | EmotionCategory.Anticipation -> "Feeling of anticipation"
                                | EmotionCategory.Interest -> "Feeling of interest"
                                | EmotionCategory.Confusion -> "Feeling of confusion"
                                | EmotionCategory.Curiosity -> "Feeling of curiosity"
                                | EmotionCategory.Satisfaction -> "Feeling of satisfaction"
                                | EmotionCategory.Custom name -> $"Feeling of {name}"
                            Trigger = None
                            Duration = None
                            Data = Map.empty
                        }
                    )
                Mood = 
                    match state.EmotionalState.DominantEmotion with
                    | Some EmotionCategory.Joy -> "Happy"
                    | Some EmotionCategory.Sadness -> "Sad"
                    | Some EmotionCategory.Anger -> "Angry"
                    | Some EmotionCategory.Fear -> "Fearful"
                    | Some EmotionCategory.Surprise -> "Surprised"
                    | Some EmotionCategory.Disgust -> "Disgusted"
                    | Some EmotionCategory.Trust -> "Trusting"
                    | Some EmotionCategory.Anticipation -> "Anticipating"
                    | Some EmotionCategory.Interest -> "Interested"
                    | Some EmotionCategory.Confusion -> "Confused"
                    | Some EmotionCategory.Curiosity -> "Curious"
                    | Some EmotionCategory.Satisfaction -> "Satisfied"
                    | Some (EmotionCategory.Custom name) -> name
                    | None -> "Neutral"
                Timestamp = DateTime.Now
                Data = Map.empty
            }
            CurrentThoughtProcess = 
                state.CurrentThoughtType
                |> Option.map (fun thoughtType -> 
                    {
                        Type = thoughtType
                        Content = 
                            match thoughtType with
                            | ThoughtType.Analytical -> "Analytical thought"
                            | ThoughtType.Creative -> "Creative thought"
                            | ThoughtType.Critical -> "Critical thought"
                            | ThoughtType.Intuitive -> "Intuitive thought"
                            | ThoughtType.Reflective -> "Reflective thought"
                            | ThoughtType.Strategic -> "Strategic thought"
                            | ThoughtType.Divergent -> "Divergent thought"
                            | ThoughtType.Convergent -> "Convergent thought"
                            | ThoughtType.Custom name -> $"Custom thought: {name}"
                        Timestamp = DateTime.Now
                        Duration = None
                        AssociatedEmotions = []
                        Data = Map.empty
                    }
                )
            AttentionFocus = state.AttentionFocus
            Timestamp = DateTime.Now
            Data = Map.empty
        }
