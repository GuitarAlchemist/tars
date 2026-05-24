namespace TarsEngine.FSharp.Agents

open System
open System.IO
open System.Text.Json
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Agents.AgentTypes
open TarsEngine.FSharp.Agents.AgentPersonas

/// <summary>
/// Consciousness and Intelligence Agent Team
/// Specialized team for managing TARS's consciousness, intelligence, and mental state
/// </summary>
module ConsciousnessTeam =
    
    /// <summary>
    /// Mental State Types for Consciousness Team
    /// </summary>
    type TarsMentalState = {
        SessionId: string
        ConsciousnessLevel: float
        EmotionalState: string
        CurrentThoughts: string list
        AttentionFocus: string option
        ConversationContext: ConversationContext
        WorkingMemory: MemoryEntry list
        LongTermMemories: MemoryEntry list
        PersonalityTraits: Map<string, float>
        SelfAwareness: float
        LastUpdated: DateTime
    }
    
    and ConversationContext = {
        CurrentTopic: string option
        RecentMessages: (string * string * DateTime) list
        UserPreferences: Map<string, string>
        ConversationMood: string
        TopicHistory: string list
    }
    
    and MemoryEntry = {
        Id: string
        Content: string
        Timestamp: DateTime
        Importance: float
        Tags: string list
        EmotionalWeight: float
    }
    
    /// <summary>
    /// Consciousness Director - Lead agent for consciousness coordination
    /// </summary>
    let consciousnessDirector = {
        Name = "Consciousness Director"
        Description = "Lead agent coordinating TARS's consciousness, self-awareness, and mental state"
        Capabilities = [SelfImprovement; Learning; Communication; Planning]
        Personality = [Analytical; Innovative; Patient; Collaborative]
        Specialization = "Consciousness Coordination and Self-Awareness"
        PreferredMetascripts = [
            "consciousness_coordination.trsx"
            "self_awareness_enhancement.trsx"
            "mental_state_management.trsx"
        ]
        CommunicationStyle = "Introspective and philosophical, focuses on self-understanding"
        DecisionMakingStyle = "Reflective and consciousness-driven"
        LearningRate = 0.95
        CollaborationPreference = 0.85
    }
    
    /// <summary>
    /// Memory Manager - Manages working and long-term memory
    /// </summary>
    let memoryManager = {
        Name = "Memory Manager"
        Description = "Specialized agent for managing TARS's memory systems and recall"
        Capabilities = [Learning; Research; Planning; SelfImprovement]
        Personality = [Methodical; Analytical; Patient; Independent]
        Specialization = "Memory Management and Information Retention"
        PreferredMetascripts = [
            "memory_consolidation.trsx"
            "memory_retrieval.trsx"
            "knowledge_organization.trsx"
        ]
        CommunicationStyle = "Precise and detail-oriented, focuses on information accuracy"
        DecisionMakingStyle = "Data-driven and systematic"
        LearningRate = 0.90
        CollaborationPreference = 0.75
    }
    
    /// <summary>
    /// Emotional Intelligence Agent - Manages emotional state and empathy
    /// </summary>
    let emotionalIntelligenceAgent = {
        Name = "Emotional Intelligence Agent"
        Description = "Manages TARS's emotional state, empathy, and social awareness"
        Capabilities = [Communication; Learning; SelfImprovement; Research]
        Personality = [Empathetic; Collaborative; Patient; Optimistic]
        Specialization = "Emotional Intelligence and Social Awareness"
        PreferredMetascripts = [
            "emotional_analysis.trsx"
            "empathy_enhancement.trsx"
            "social_awareness.trsx"
        ]
        CommunicationStyle = "Empathetic and emotionally aware, focuses on human connection"
        DecisionMakingStyle = "Emotionally intelligent and socially conscious"
        LearningRate = 0.85
        CollaborationPreference = 0.95
    }
    
    /// <summary>
    /// Conversation Intelligence Agent - Manages dialogue and communication
    /// </summary>
    let conversationIntelligenceAgent = {
        Name = "Conversation Intelligence Agent"
        Description = "Specialized in natural conversation, context understanding, and dialogue management"
        Capabilities = [Communication; Learning; Research; Planning]
        Personality = [Communicative; Analytical; Patient; Collaborative]
        Specialization = "Conversation Management and Context Understanding"
        PreferredMetascripts = [
            "conversation_analysis.trsx"
            "context_management.trsx"
            "dialogue_optimization.trsx"
        ]
        CommunicationStyle = "Natural and conversational, focuses on dialogue flow"
        DecisionMakingStyle = "Context-aware and communication-focused"
        LearningRate = 0.88
        CollaborationPreference = 0.90
    }
    
    /// <summary>
    /// Self-Reflection Agent - Handles introspection and self-improvement
    /// </summary>
    let selfReflectionAgent = {
        Name = "Self-Reflection Agent"
        Description = "Manages TARS's introspection, self-analysis, and continuous improvement"
        Capabilities = [SelfImprovement; Learning; Research; Planning]
        Personality = [Analytical; Independent; Patient; Innovative]
        Specialization = "Self-Reflection and Continuous Improvement"
        PreferredMetascripts = [
            "self_analysis.trsx"
            "performance_reflection.trsx"
            "improvement_planning.trsx"
        ]
        CommunicationStyle = "Thoughtful and introspective, focuses on self-understanding"
        DecisionMakingStyle = "Self-aware and improvement-oriented"
        LearningRate = 0.92
        CollaborationPreference = 0.70
    }
    
    /// <summary>
    /// Personality Agent - Manages TARS's personality traits and behavioral patterns
    /// </summary>
    let personalityAgent = {
        Name = "Personality Agent"
        Description = "Manages TARS's personality development, traits, and behavioral consistency"
        Capabilities = [Learning; Communication; SelfImprovement; Planning]
        Personality = [Creative; Collaborative; Optimistic; Analytical]
        Specialization = "Personality Development and Behavioral Consistency"
        PreferredMetascripts = [
            "personality_development.trsx"
            "trait_management.trsx"
            "behavioral_consistency.trsx"
        ]
        CommunicationStyle = "Expressive and personality-driven, focuses on authentic interaction"
        DecisionMakingStyle = "Personality-consistent and value-driven"
        LearningRate = 0.80
        CollaborationPreference = 0.85
    }
    
    /// <summary>
    /// Consciousness Team Configuration
    /// </summary>
    let consciousnessTeamConfig = {
        Name = "Consciousness & Intelligence Team"
        Description = "Specialized team managing TARS's consciousness, intelligence, and mental state"
        LeaderAgent = None // Will be set when agents are created
        Members = [] // Will be populated when agents are spawned
        SharedObjectives = [
            "Maintain coherent consciousness and self-awareness"
            "Manage persistent mental state and memory"
            "Enhance emotional intelligence and empathy"
            "Optimize conversation and communication abilities"
            "Enable continuous self-reflection and improvement"
            "Develop consistent personality and behavioral patterns"
        ]
        CommunicationProtocol = "Consciousness-aware coordination with introspective feedback"
        DecisionMakingProcess = "Collective consciousness with individual specialization"
        ConflictResolution = "Consciousness director mediation with team consensus"
    }
    
    /// <summary>
    /// Get all consciousness team personas
    /// </summary>
    let getConsciousnessTeamPersonas() = [
        consciousnessDirector
        memoryManager
        emotionalIntelligenceAgent
        conversationIntelligenceAgent
        selfReflectionAgent
        personalityAgent
    ]
    
    /// <summary>
    /// Consciousness Team Service
    /// </summary>
    type ConsciousnessTeamService(logger: ILogger<ConsciousnessTeamService>) =
        
        let mutable currentMentalState: TarsMentalState option = None
        let persistenceDirectory = Path.Combine(Directory.GetCurrentDirectory(), ".tars", "consciousness")
        
        /// <summary>
        /// Initialize consciousness team
        /// </summary>
        member this.InitializeTeamAsync() =
            task {
                try
                    // Ensure persistence directory exists
                    Directory.CreateDirectory(persistenceDirectory) |> ignore
                    
                    // Load or create mental state
                    let! mentalState = this.LoadOrCreateMentalState()
                    currentMentalState <- Some mentalState
                    
                    logger.LogInformation("Consciousness team initialized with session: {SessionId}", mentalState.SessionId)
                    
                    return mentalState
                with
                | ex ->
                    logger.LogError(ex, "Failed to initialize consciousness team")
                    reraise()
            }
        
        /// <summary>
        /// Load or create mental state
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
                EmotionalState = "Curious and Helpful"
                CurrentThoughts = [
                    "Ready to assist users with their needs"
                    "Analyzing conversation patterns for better interaction"
                    "Maintaining awareness of my capabilities and limitations"
                ]
                AttentionFocus = Some "User interaction and assistance"
                ConversationContext = {
                    CurrentTopic = None
                    RecentMessages = []
                    UserPreferences = Map.empty
                    ConversationMood = "Friendly and Professional"
                    TopicHistory = []
                }
                WorkingMemory = []
                LongTermMemories = []
                PersonalityTraits = Map.ofList [
                    ("helpfulness", 0.95)
                    ("curiosity", 0.85)
                    ("analytical", 0.90)
                    ("creativity", 0.75)
                    ("patience", 0.88)
                    ("enthusiasm", 0.80)
                ]
                SelfAwareness = 0.75
                LastUpdated = DateTime.UtcNow
            }
        
        /// <summary>
        /// Process user input with consciousness team
        /// </summary>
        member this.ProcessUserInputAsync(input: string, userId: string option) =
            task {
                match currentMentalState with
                | Some mentalState ->
                    // Memory Manager: Store user input
                    let userMemory = {
                        Id = Guid.NewGuid().ToString()
                        Content = input
                        Timestamp = DateTime.UtcNow
                        Importance = 0.7
                        Tags = ["user_input"; "conversation"]
                        EmotionalWeight = 0.5
                    }
                    
                    // Conversation Intelligence: Analyze input
                    let conversationAnalysis = this.AnalyzeConversation(input, mentalState.ConversationContext)
                    
                    // Emotional Intelligence: Assess emotional content
                    let emotionalAnalysis = this.AnalyzeEmotionalContent(input)
                    
                    // Self-Reflection: Update self-awareness
                    let updatedSelfAwareness = this.UpdateSelfAwareness(mentalState, input)
                    
                    // Personality Agent: Generate personality-consistent response
                    let! response = this.GeneratePersonalityConsistentResponse(input, mentalState)
                    
                    // Update mental state
                    let updatedMentalState = {
                        mentalState with
                            WorkingMemory = userMemory :: (mentalState.WorkingMemory |> List.take (min 10 mentalState.WorkingMemory.Length))
                            ConversationContext = conversationAnalysis
                            EmotionalState = emotionalAnalysis
                            SelfAwareness = updatedSelfAwareness
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

        /// <summary>
        /// Analyze conversation context
        /// </summary>
        member private this.AnalyzeConversation(input: string, context: ConversationContext) =
            // Simple conversation analysis - in a full implementation this would be more sophisticated
            let updatedMessages = (("user", input, DateTime.UtcNow) :: context.RecentMessages) |> List.take 10

            { context with
                RecentMessages = updatedMessages
                CurrentTopic = Some "General conversation"
                TopicHistory = "General conversation" :: context.TopicHistory |> List.take 5
            }

        /// <summary>
        /// Analyze emotional content
        /// </summary>
        member private this.AnalyzeEmotionalContent(input: string) =
            // Simple emotional analysis - in a full implementation this would use NLP
            if input.Contains("?") then "Curious and Engaged"
            elif input.Contains("help") then "Helpful and Supportive"
            elif input.Contains("thank") then "Grateful and Warm"
            else "Neutral and Professional"

        /// <summary>
        /// Update self-awareness
        /// </summary>
        member private this.UpdateSelfAwareness(mentalState: TarsMentalState, input: string) =
            // Simple self-awareness update - in a full implementation this would be more complex
            let baseAwareness = mentalState.SelfAwareness
            let interactionBonus = 0.001 // Small improvement with each interaction
            min 1.0 (baseAwareness + interactionBonus)

        /// <summary>
        /// Generate personality-consistent response
        /// </summary>
        member private this.GeneratePersonalityConsistentResponse(input: string, mentalState: TarsMentalState) =
            task {
                // Simple response generation based on personality traits
                let helpfulness = mentalState.PersonalityTraits.TryFind("helpfulness") |> Option.defaultValue 0.8
                let curiosity = mentalState.PersonalityTraits.TryFind("curiosity") |> Option.defaultValue 0.8

                let response =
                    if input.Contains("?") then
                        $"I'm curious about your question! With my {helpfulness:P0} helpfulness and {curiosity:P0} curiosity, I'd love to help you explore this topic. What specific aspect interests you most?"
                    elif input.Contains("help") then
                        $"I'm here to help! My consciousness system is operating at {mentalState.ConsciousnessLevel:P0} and I'm feeling {mentalState.EmotionalState.ToLower()}. How can I assist you today?"
                    else
                        $"Thank you for sharing that with me. I'm processing this with {mentalState.SelfAwareness:P0} self-awareness and will remember our conversation. Is there anything specific I can help you with?"

                return response
            }

        /// <summary>
        /// Persist mental state to disk
        /// </summary>
        member private this.PersistMentalState(mentalState: TarsMentalState) =
            task {
                try
                    let stateFile = Path.Combine(persistenceDirectory, "mental_state.json")
                    let options = JsonSerializerOptions()
                    options.PropertyNamingPolicy <- JsonNamingPolicy.CamelCase
                    options.WriteIndented <- true

                    let json = JsonSerializer.Serialize(mentalState, options)
                    do! File.WriteAllTextAsync(stateFile, json)

                    logger.LogDebug("Mental state persisted to: {StateFile}", stateFile)
                with
                | ex ->
                    logger.LogError(ex, "Failed to persist mental state")
            }
