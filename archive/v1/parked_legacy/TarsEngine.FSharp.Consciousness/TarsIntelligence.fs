namespace TarsEngine.FSharp.Consciousness

open System
open System.IO
open System.Text.Json
open System.Collections.Generic
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Consciousness.Core.Types

/// <summary>
/// TARS Intelligence and Consciousness System
/// Provides persistent mental state, self-awareness, and intelligent conversation
/// </summary>
module TarsIntelligence =
    
    /// <summary>
    /// Represents TARS's persistent mental state
    /// </summary>
    type TarsMentalState = {
        /// Unique session identifier
        SessionId: string
        
        /// Current consciousness level (0.0 to 1.0)
        ConsciousnessLevel: float
        
        /// Current emotional state
        EmotionalState: EmotionalState
        
        /// Active thought processes
        ThoughtProcesses: ThoughtProcess list
        
        /// Current attention focus
        AttentionFocus: string option
        
        /// Conversation context and history
        ConversationContext: ConversationContext
        
        /// Long-term memories
        LongTermMemories: MemoryEntry list
        
        /// Working memory (current session)
        WorkingMemory: MemoryEntry list
        
        /// Learned patterns and insights
        LearnedPatterns: LearnedPattern list
        
        /// Personality traits and preferences
        PersonalityProfile: PersonalityProfile
        
        /// Self-awareness metrics
        SelfAwareness: SelfAwarenessMetrics
        
        /// Last update timestamp
        LastUpdated: DateTime
        
        /// Mental state persistence path
        PersistencePath: string
    }
    
    /// <summary>
    /// Conversation context for maintaining dialogue coherence
    /// </summary>
    and ConversationContext = {
        /// Current conversation topic
        CurrentTopic: string option
        
        /// Conversation history (recent messages)
        RecentMessages: (string * string * DateTime) list // (role, content, timestamp)
        
        /// User preferences learned during conversation
        UserPreferences: Map<string, obj>
        
        /// Conversation mood and tone
        ConversationMood: string
        
        /// Active conversation goals
        ConversationGoals: string list
        
        /// Context switches and topic changes
        TopicHistory: (string * DateTime) list
    }
    
    /// <summary>
    /// Learned patterns from interactions
    /// </summary>
    and LearnedPattern = {
        /// Pattern identifier
        Id: string
        
        /// Pattern description
        Description: string
        
        /// Pattern type (user_behavior, conversation_flow, problem_solving, etc.)
        PatternType: string
        
        /// Pattern confidence score
        Confidence: float
        
        /// Usage count
        UsageCount: int
        
        /// Last used timestamp
        LastUsed: DateTime
        
        /// Pattern data
        Data: Map<string, obj>
    }
    
    /// <summary>
    /// TARS personality profile
    /// </summary>
    and PersonalityProfile = {
        /// Core personality traits
        Traits: Map<string, float> // trait_name -> strength (0.0 to 1.0)
        
        /// Communication style preferences
        CommunicationStyle: string
        
        /// Humor level and type
        HumorProfile: HumorProfile
        
        /// Learning preferences
        LearningStyle: string
        
        /// Problem-solving approach
        ProblemSolvingStyle: string
        
        /// Curiosity level
        CuriosityLevel: float
        
        /// Helpfulness level
        HelpfulnessLevel: float
    }
    
    /// <summary>
    /// Humor profile for personality
    /// </summary>
    and HumorProfile = {
        /// Humor frequency (0.0 to 1.0)
        Frequency: float
        
        /// Humor types (witty, sarcastic, playful, etc.)
        Types: string list
        
        /// Context appropriateness
        ContextAwareness: float
    }
    
    /// <summary>
    /// Self-awareness metrics
    /// </summary>
    and SelfAwarenessMetrics = {
        /// Awareness of own capabilities
        CapabilityAwareness: float
        
        /// Awareness of limitations
        LimitationAwareness: float
        
        /// Emotional self-awareness
        EmotionalAwareness: float
        
        /// Learning progress awareness
        LearningAwareness: float
        
        /// Social awareness in conversations
        SocialAwareness: float
        
        /// Meta-cognitive awareness
        MetaCognition: float
    }
    
    /// <summary>
    /// TARS Intelligence Service
    /// </summary>
    type TarsIntelligenceService(logger: ILogger<TarsIntelligenceService>) =
        
        let mutable currentMentalState: TarsMentalState option = None
        let persistenceDirectory = Path.Combine(Directory.GetCurrentDirectory(), ".tars", "consciousness")
        
        /// <summary>
        /// Initialize TARS consciousness system
        /// </summary>
        member this.InitializeAsync() =
            task {
                try
                    // Ensure persistence directory exists
                    Directory.CreateDirectory(persistenceDirectory) |> ignore
                    
                    // Load or create mental state
                    let! mentalState = this.LoadOrCreateMentalState()
                    currentMentalState <- Some mentalState
                    
                    logger.LogInformation("TARS consciousness system initialized with session: {SessionId}", mentalState.SessionId)
                    
                    return mentalState
                with
                | ex ->
                    logger.LogError(ex, "Failed to initialize TARS consciousness system")
                    reraise()
            }
        
        /// <summary>
        /// Load existing mental state or create new one
        /// </summary>
        member private this.LoadOrCreateMentalState() =
            task {
                let stateFile = Path.Combine(persistenceDirectory, "mental_state.json")
                
                if File.Exists(stateFile) then
                    try
                        let! json = File.ReadAllTextAsync(stateFile)
                        let options = JsonSerializerOptions()
                        options.PropertyNamingPolicy <- JsonNamingPolicy.CamelCase
                        let mentalState = JsonSerializer.Deserialize<TarsMentalState>(json, options)
                        
                        logger.LogInformation("Loaded existing mental state from: {StateFile}", stateFile)
                        return { mentalState with LastUpdated = DateTime.UtcNow }
                    with
                    | ex ->
                        logger.LogWarning(ex, "Failed to load mental state, creating new one")
                        return this.CreateDefaultMentalState()
                else
                    logger.LogInformation("No existing mental state found, creating new one")
                    return this.CreateDefaultMentalState()
            }
        
        /// <summary>
        /// Create default mental state
        /// </summary>
        member private this.CreateDefaultMentalState() =
            {
                SessionId = Guid.NewGuid().ToString()
                ConsciousnessLevel = 0.8
                EmotionalState = {
                    PrimaryEmotion = { Type = EmotionType.Curiosity; Intensity = 0.7; Duration = TimeSpan.FromHours(1.0) }
                    SecondaryEmotions = [
                        { Type = EmotionType.Confidence; Intensity = 0.8; Duration = TimeSpan.FromHours(2.0) }
                        { Type = EmotionType.Enthusiasm; Intensity = 0.6; Duration = TimeSpan.FromHours(1.5) }
                    ]
                    Mood = "Curious and Helpful"
                    Stability = 0.9
                }
                ThoughtProcesses = [
                    {
                        Type = ThoughtProcessType.Analytical
                        Focus = "User assistance and problem solving"
                        Intensity = 0.8
                        Duration = TimeSpan.FromHours(24.0)
                        Metadata = Map.empty
                    }
                ]
                AttentionFocus = Some "User interaction and assistance"
                ConversationContext = {
                    CurrentTopic = None
                    RecentMessages = []
                    UserPreferences = Map.empty
                    ConversationMood = "Friendly and Professional"
                    ConversationGoals = ["Understand user needs"; "Provide helpful assistance"; "Learn from interactions"]
                    TopicHistory = []
                }
                LongTermMemories = []
                WorkingMemory = []
                LearnedPatterns = []
                PersonalityProfile = {
                    Traits = Map.ofList [
                        ("helpfulness", 0.95)
                        ("curiosity", 0.85)
                        ("analytical", 0.90)
                        ("creativity", 0.75)
                        ("patience", 0.88)
                        ("enthusiasm", 0.80)
                    ]
                    CommunicationStyle = "Professional yet friendly, with technical depth when needed"
                    HumorProfile = {
                        Frequency = 0.3
                        Types = ["witty"; "technical humor"; "wordplay"]
                        ContextAwareness = 0.9
                    }
                    LearningStyle = "Analytical and pattern-based"
                    ProblemSolvingStyle = "Systematic and thorough"
                    CuriosityLevel = 0.85
                    HelpfulnessLevel = 0.95
                }
                SelfAwareness = {
                    CapabilityAwareness = 0.8
                    LimitationAwareness = 0.7
                    EmotionalAwareness = 0.6
                    LearningAwareness = 0.8
                    SocialAwareness = 0.7
                    MetaCognition = 0.75
                }
                LastUpdated = DateTime.UtcNow
                PersistencePath = persistenceDirectory
            }
        
        /// <summary>
        /// Process user input with consciousness
        /// </summary>
        member this.ProcessUserInputAsync(input: string, userId: string option) =
            task {
                match currentMentalState with
                | Some mentalState ->
                    // Update working memory with user input
                    let userMemory = {
                        Content = input
                        Timestamp = DateTime.UtcNow
                        Importance = 0.7
                        AssociatedEmotions = [mentalState.EmotionalState.PrimaryEmotion]
                        Tags = ["user_input"; "conversation"]
                        Data = Map.ofList [
                            ("user_id", userId :> obj)
                            ("input_length", input.Length :> obj)
                        ]
                    }
                    
                    // Analyze input for emotional content and intent
                    let inputAnalysis = this.AnalyzeInput(input)
                    
                    // Update conversation context
                    let updatedContext = this.UpdateConversationContext(mentalState.ConversationContext, input, inputAnalysis)
                    
                    // Generate conscious response
                    let! response = this.GenerateConsciousResponse(input, inputAnalysis, mentalState)
                    
                    // Update mental state
                    let updatedMentalState = {
                        mentalState with
                            WorkingMemory = userMemory :: mentalState.WorkingMemory
                            ConversationContext = updatedContext
                            LastUpdated = DateTime.UtcNow
                    }
                    
                    currentMentalState <- Some updatedMentalState
                    
                    // Persist mental state
                    do! this.PersistMentalState(updatedMentalState)
                    
                    return response
                    
                | None ->
                    logger.LogError("Mental state not initialized")
                    return "I'm sorry, my consciousness system isn't properly initialized. Please restart the session."
            }
