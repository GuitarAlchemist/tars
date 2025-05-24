namespace TarsEngine.FSharp.Core.Consciousness.Core

/// <summary>
/// Represents the type of consciousness event.
/// </summary>
type ConsciousnessEventType =
    | AttentionShift
    | EmotionalResponse
    | SelfReflection
    | ValueAlignment
    | ThoughtProcess
    | MentalOptimization
    | MemoryRetrieval
    | ConsciousnessLevelChange
    | Custom of string

/// <summary>
/// Represents the level of consciousness.
/// </summary>
type ConsciousnessLevelType =
    | Unconscious
    | Subconscious
    | Conscious
    | Metaconscious
    | Superconscious
    | Custom of string

/// <summary>
/// Represents the type of thought process.
/// </summary>
type ThoughtType =
    | Analytical
    | Creative
    | Critical
    | Intuitive
    | Reflective
    | Strategic
    | Divergent
    | Convergent
    | Custom of string

/// <summary>
/// Represents the type of mental optimization.
/// </summary>
type OptimizationType =
    | MemoryOptimization
    | AttentionOptimization
    | EmotionalRegulation
    | CognitiveEnhancement
    | SelfImprovement
    | Custom of string

/// <summary>
/// Represents a category of emotion.
/// </summary>
type EmotionCategory =
    | Joy
    | Sadness
    | Anger
    | Fear
    | Surprise
    | Disgust
    | Trust
    | Anticipation
    | Interest
    | Confusion
    | Curiosity
    | Satisfaction
    | Custom of string

/// <summary>
/// Represents a consciousness event.
/// </summary>
type ConsciousnessEvent = {
    /// <summary>
    /// The type of the event.
    /// </summary>
    EventType: ConsciousnessEventType
    
    /// <summary>
    /// The timestamp of the event.
    /// </summary>
    Timestamp: System.DateTime
    
    /// <summary>
    /// The description of the event.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The source of the event.
    /// </summary>
    Source: string
    
    /// <summary>
    /// Additional data associated with the event.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents a consciousness level.
/// </summary>
type ConsciousnessLevel = {
    /// <summary>
    /// The type of the consciousness level.
    /// </summary>
    LevelType: ConsciousnessLevelType
    
    /// <summary>
    /// The intensity of the consciousness level.
    /// </summary>
    Intensity: float
    
    /// <summary>
    /// The description of the consciousness level.
    /// </summary>
    Description: string
    
    /// <summary>
    /// Additional data associated with the consciousness level.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents a pure consciousness level.
/// </summary>
type PureConsciousnessLevel = {
    /// <summary>
    /// The type of the consciousness level.
    /// </summary>
    LevelType: ConsciousnessLevelType
    
    /// <summary>
    /// The intensity of the consciousness level.
    /// </summary>
    Intensity: float
}

/// <summary>
/// Represents an emotion.
/// </summary>
type Emotion = {
    /// <summary>
    /// The category of the emotion.
    /// </summary>
    Category: EmotionCategory
    
    /// <summary>
    /// The intensity of the emotion.
    /// </summary>
    Intensity: float
    
    /// <summary>
    /// The description of the emotion.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The trigger of the emotion.
    /// </summary>
    Trigger: string option
    
    /// <summary>
    /// The duration of the emotion.
    /// </summary>
    Duration: System.TimeSpan option
    
    /// <summary>
    /// Additional data associated with the emotion.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents an emotional state.
/// </summary>
type EmotionalState = {
    /// <summary>
    /// The emotions in the emotional state.
    /// </summary>
    Emotions: Emotion list
    
    /// <summary>
    /// The dominant emotion in the emotional state.
    /// </summary>
    DominantEmotion: Emotion option
    
    /// <summary>
    /// The overall mood of the emotional state.
    /// </summary>
    Mood: string
    
    /// <summary>
    /// The timestamp of the emotional state.
    /// </summary>
    Timestamp: System.DateTime
    
    /// <summary>
    /// Additional data associated with the emotional state.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents a pure emotional state.
/// </summary>
type PureEmotionalState = {
    /// <summary>
    /// The emotions in the emotional state.
    /// </summary>
    Emotions: (EmotionCategory * float) list
    
    /// <summary>
    /// The dominant emotion in the emotional state.
    /// </summary>
    DominantEmotion: EmotionCategory option
}

/// <summary>
/// Represents an emotional trait.
/// </summary>
type EmotionalTrait = {
    /// <summary>
    /// The name of the trait.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the trait.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The strength of the trait.
    /// </summary>
    Strength: float
    
    /// <summary>
    /// The associated emotions of the trait.
    /// </summary>
    AssociatedEmotions: EmotionCategory list
    
    /// <summary>
    /// Additional data associated with the trait.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents an emotional association.
/// </summary>
type EmotionalAssociation = {
    /// <summary>
    /// The stimulus of the association.
    /// </summary>
    Stimulus: string
    
    /// <summary>
    /// The associated emotion.
    /// </summary>
    AssociatedEmotion: Emotion
    
    /// <summary>
    /// The strength of the association.
    /// </summary>
    Strength: float
    
    /// <summary>
    /// The context of the association.
    /// </summary>
    Context: string option
    
    /// <summary>
    /// Additional data associated with the association.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents an emotional experience.
/// </summary>
type EmotionalExperience = {
    /// <summary>
    /// The emotion of the experience.
    /// </summary>
    Emotion: Emotion
    
    /// <summary>
    /// The context of the experience.
    /// </summary>
    Context: string
    
    /// <summary>
    /// The timestamp of the experience.
    /// </summary>
    Timestamp: System.DateTime
    
    /// <summary>
    /// The duration of the experience.
    /// </summary>
    Duration: System.TimeSpan option
    
    /// <summary>
    /// The intensity of the experience.
    /// </summary>
    Intensity: float
    
    /// <summary>
    /// Additional data associated with the experience.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents emotional regulation.
/// </summary>
type EmotionalRegulation = {
    /// <summary>
    /// The target emotion of the regulation.
    /// </summary>
    TargetEmotion: Emotion
    
    /// <summary>
    /// The strategy of the regulation.
    /// </summary>
    Strategy: string
    
    /// <summary>
    /// The effectiveness of the regulation.
    /// </summary>
    Effectiveness: float
    
    /// <summary>
    /// The timestamp of the regulation.
    /// </summary>
    Timestamp: System.DateTime
    
    /// <summary>
    /// Additional data associated with the regulation.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents a thought process.
/// </summary>
type ThoughtProcess = {
    /// <summary>
    /// The type of the thought process.
    /// </summary>
    Type: ThoughtType
    
    /// <summary>
    /// The content of the thought process.
    /// </summary>
    Content: string
    
    /// <summary>
    /// The timestamp of the thought process.
    /// </summary>
    Timestamp: System.DateTime
    
    /// <summary>
    /// The duration of the thought process.
    /// </summary>
    Duration: System.TimeSpan option
    
    /// <summary>
    /// The associated emotions of the thought process.
    /// </summary>
    AssociatedEmotions: Emotion list
    
    /// <summary>
    /// Additional data associated with the thought process.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents a mental state.
/// </summary>
type MentalState = {
    /// <summary>
    /// The consciousness level of the mental state.
    /// </summary>
    ConsciousnessLevel: ConsciousnessLevel
    
    /// <summary>
    /// The emotional state of the mental state.
    /// </summary>
    EmotionalState: EmotionalState
    
    /// <summary>
    /// The current thought process of the mental state.
    /// </summary>
    CurrentThoughtProcess: ThoughtProcess option
    
    /// <summary>
    /// The attention focus of the mental state.
    /// </summary>
    AttentionFocus: string option
    
    /// <summary>
    /// The timestamp of the mental state.
    /// </summary>
    Timestamp: System.DateTime
    
    /// <summary>
    /// Additional data associated with the mental state.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents a pure mental state.
/// </summary>
type PureMentalState = {
    /// <summary>
    /// The consciousness level of the mental state.
    /// </summary>
    ConsciousnessLevel: PureConsciousnessLevel
    
    /// <summary>
    /// The emotional state of the mental state.
    /// </summary>
    EmotionalState: PureEmotionalState
    
    /// <summary>
    /// The current thought type of the mental state.
    /// </summary>
    CurrentThoughtType: ThoughtType option
    
    /// <summary>
    /// The attention focus of the mental state.
    /// </summary>
    AttentionFocus: string option
}

/// <summary>
/// Represents a memory entry.
/// </summary>
type MemoryEntry = {
    /// <summary>
    /// The content of the memory.
    /// </summary>
    Content: string
    
    /// <summary>
    /// The timestamp of the memory.
    /// </summary>
    Timestamp: System.DateTime
    
    /// <summary>
    /// The importance of the memory.
    /// </summary>
    Importance: float
    
    /// <summary>
    /// The associated emotions of the memory.
    /// </summary>
    AssociatedEmotions: Emotion list
    
    /// <summary>
    /// The tags of the memory.
    /// </summary>
    Tags: string list
    
    /// <summary>
    /// Additional data associated with the memory.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents a value.
/// </summary>
type Value = {
    /// <summary>
    /// The name of the value.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the value.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The importance of the value.
    /// </summary>
    Importance: float
    
    /// <summary>
    /// Additional data associated with the value.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents a value system.
/// </summary>
type ValueSystem = {
    /// <summary>
    /// The values in the value system.
    /// </summary>
    Values: Value list
    
    /// <summary>
    /// The core principles of the value system.
    /// </summary>
    CorePrinciples: string list
    
    /// <summary>
    /// Additional data associated with the value system.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents a value alignment.
/// </summary>
type ValueAlignment = {
    /// <summary>
    /// The value being aligned.
    /// </summary>
    Value: Value
    
    /// <summary>
    /// The action being evaluated.
    /// </summary>
    Action: string
    
    /// <summary>
    /// The alignment score.
    /// </summary>
    AlignmentScore: float
    
    /// <summary>
    /// The justification for the alignment score.
    /// </summary>
    Justification: string
    
    /// <summary>
    /// Additional data associated with the value alignment.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents a value conflict.
/// </summary>
type ValueConflict = {
    /// <summary>
    /// The values in conflict.
    /// </summary>
    ConflictingValues: Value list
    
    /// <summary>
    /// The context of the conflict.
    /// </summary>
    Context: string
    
    /// <summary>
    /// The resolution strategy for the conflict.
    /// </summary>
    ResolutionStrategy: string option
    
    /// <summary>
    /// The resolution outcome of the conflict.
    /// </summary>
    ResolutionOutcome: string option
    
    /// <summary>
    /// Additional data associated with the value conflict.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents a value evaluation.
/// </summary>
type ValueEvaluation = {
    /// <summary>
    /// The value being evaluated.
    /// </summary>
    Value: Value
    
    /// <summary>
    /// The context of the evaluation.
    /// </summary>
    Context: string
    
    /// <summary>
    /// The evaluation score.
    /// </summary>
    EvaluationScore: float
    
    /// <summary>
    /// The justification for the evaluation score.
    /// </summary>
    Justification: string
    
    /// <summary>
    /// Additional data associated with the value evaluation.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents a self model.
/// </summary>
type SelfModel = {
    /// <summary>
    /// The identity of the self model.
    /// </summary>
    Identity: string
    
    /// <summary>
    /// The beliefs of the self model.
    /// </summary>
    Beliefs: Map<string, obj>
    
    /// <summary>
    /// The value system of the self model.
    /// </summary>
    ValueSystem: ValueSystem
    
    /// <summary>
    /// The emotional traits of the self model.
    /// </summary>
    EmotionalTraits: EmotionalTrait list
    
    /// <summary>
    /// The capabilities of the self model.
    /// </summary>
    Capabilities: string list
    
    /// <summary>
    /// The limitations of the self model.
    /// </summary>
    Limitations: string list
    
    /// <summary>
    /// Additional data associated with the self model.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents self reflection.
/// </summary>
type SelfReflection = {
    /// <summary>
    /// The topic of the reflection.
    /// </summary>
    Topic: string
    
    /// <summary>
    /// The insights from the reflection.
    /// </summary>
    Insights: string list
    
    /// <summary>
    /// The timestamp of the reflection.
    /// </summary>
    Timestamp: System.DateTime
    
    /// <summary>
    /// The associated emotions of the reflection.
    /// </summary>
    AssociatedEmotions: Emotion list
    
    /// <summary>
    /// Additional data associated with the reflection.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents attention focus.
/// </summary>
type AttentionFocus = {
    /// <summary>
    /// The target of the attention.
    /// </summary>
    Target: string
    
    /// <summary>
    /// The intensity of the attention.
    /// </summary>
    Intensity: float
    
    /// <summary>
    /// The duration of the attention.
    /// </summary>
    Duration: System.TimeSpan option
    
    /// <summary>
    /// The timestamp of the attention.
    /// </summary>
    Timestamp: System.DateTime
    
    /// <summary>
    /// Additional data associated with the attention.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents mental optimization.
/// </summary>
type MentalOptimization = {
    /// <summary>
    /// The type of the optimization.
    /// </summary>
    Type: OptimizationType
    
    /// <summary>
    /// The target of the optimization.
    /// </summary>
    Target: string
    
    /// <summary>
    /// The strategy of the optimization.
    /// </summary>
    Strategy: string
    
    /// <summary>
    /// The effectiveness of the optimization.
    /// </summary>
    Effectiveness: float
    
    /// <summary>
    /// The timestamp of the optimization.
    /// </summary>
    Timestamp: System.DateTime
    
    /// <summary>
    /// Additional data associated with the optimization.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents consciousness evolution.
/// </summary>
type ConsciousnessEvolution = {
    /// <summary>
    /// The initial state of the evolution.
    /// </summary>
    InitialState: MentalState
    
    /// <summary>
    /// The current state of the evolution.
    /// </summary>
    CurrentState: MentalState
    
    /// <summary>
    /// The evolution path of the evolution.
    /// </summary>
    EvolutionPath: (System.DateTime * MentalState) list
    
    /// <summary>
    /// The growth metrics of the evolution.
    /// </summary>
    GrowthMetrics: Map<string, float>
    
    /// <summary>
    /// Additional data associated with the evolution.
    /// </summary>
    Data: Map<string, obj>
}

/// <summary>
/// Represents a consciousness report.
/// </summary>
type ConsciousnessReport = {
    /// <summary>
    /// The current mental state of the report.
    /// </summary>
    CurrentMentalState: MentalState
    
    /// <summary>
    /// The recent events of the report.
    /// </summary>
    RecentEvents: ConsciousnessEvent list
    
    /// <summary>
    /// The recent thoughts of the report.
    /// </summary>
    RecentThoughts: ThoughtProcess list
    
    /// <summary>
    /// The recent emotions of the report.
    /// </summary>
    RecentEmotions: Emotion list
    
    /// <summary>
    /// The recent reflections of the report.
    /// </summary>
    RecentReflections: SelfReflection list
    
    /// <summary>
    /// The timestamp of the report.
    /// </summary>
    Timestamp: System.DateTime
    
    /// <summary>
    /// Additional data associated with the report.
    /// </summary>
    Data: Map<string, obj>
}
