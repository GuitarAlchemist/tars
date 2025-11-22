namespace TarsEngine.FSharp.Cli.Services

open System
open System.Collections.Generic
open System.IO
open System.Text
open Microsoft.Extensions.Logging

/// Role in conversation
type MessageRole =
    | User
    | Assistant
    | System

/// Individual chat message
type ChatMessage = {
    Id: string
    Role: MessageRole
    Content: string
    Timestamp: DateTime
    Model: string option
    ResponseTime: TimeSpan option
    TokensUsed: int option
    Metadata: Map<string, obj>
}

/// Short-term memory item
type MemoryItem = {
    Key: string
    Value: string
    Type: string // "fact", "preference", "context", "variable"
    Confidence: float
    CreatedAt: DateTime
    LastAccessed: DateTime
    AccessCount: int
    Source: string // "user_input", "inference", "web_search", etc.
}

/// Session memory for short-term context
type SessionMemory = {
    Facts: Map<string, MemoryItem>
    UserPreferences: Map<string, MemoryItem>
    ContextVariables: Map<string, MemoryItem>
    ConversationSummary: string option
    LastUpdated: DateTime
}

/// Chat session state
type ChatSession = {
    SessionId: string
    StartTime: DateTime
    LastActivity: DateTime
    Messages: ChatMessage list
    Memory: SessionMemory
    Model: string
    IsActive: bool
    Metadata: Map<string, obj>
}

/// Session statistics
type SessionStats = {
    Duration: TimeSpan
    MessageCount: int
    UserMessages: int
    AssistantMessages: int
    TotalTokens: int option
    AverageResponseTime: TimeSpan option
    MemoryItems: int
    LastActivity: DateTime
}

/// Chat session service
type ChatSessionService(logger: ILogger<ChatSessionService>, ?learningMemoryService: LearningMemoryService) =
    
    // In-memory session storage (could be enhanced with persistence)
    let sessions = Dictionary<string, ChatSession>()
    let sessionLock = obj()
    
    /// Create a new chat session
    member this.CreateSession(model: string) =
        let sessionId = Guid.NewGuid().ToString("N")[..7] // Short session ID
        let session = {
            SessionId = sessionId
            StartTime = DateTime.UtcNow
            LastActivity = DateTime.UtcNow
            Messages = []
            Memory = {
                Facts = Map.empty
                UserPreferences = Map.empty
                ContextVariables = Map.empty
                ConversationSummary = None
                LastUpdated = DateTime.UtcNow
            }
            Model = model
            IsActive = true
            Metadata = Map.empty
        }
        
        lock sessionLock (fun () ->
            sessions.[sessionId] <- session
        )
        
        logger.LogInformation($"🎯 SESSION: Created new chat session {sessionId} with model {model}")
        session
    
    /// Get existing session
    member this.GetSession(sessionId: string) =
        lock sessionLock (fun () ->
            match sessions.TryGetValue(sessionId) with
            | true, session when session.IsActive -> Some session
            | _ -> None
        )
    
    /// Add message to session
    member this.AddMessage(sessionId: string, message: ChatMessage) =
        lock sessionLock (fun () ->
            match sessions.TryGetValue(sessionId) with
            | true, session ->
                let updatedSession = {
                    session with
                        Messages = message :: session.Messages
                        LastActivity = DateTime.UtcNow
                }
                sessions.[sessionId] <- updatedSession
                logger.LogInformation($"💬 SESSION: Added {message.Role} message to session {sessionId}")

                // Automatically extract facts from the message
                this.ExtractFactsFromMessage(sessionId, message)

                Some updatedSession
            | false, _ ->
                logger.LogWarning($"❌ SESSION: Session {sessionId} not found")
                None
        )
    
    /// Update session memory
    member this.UpdateMemory(sessionId: string, memoryUpdate: SessionMemory -> SessionMemory) =
        lock sessionLock (fun () ->
            match sessions.TryGetValue(sessionId) with
            | true, session ->
                let updatedMemory = memoryUpdate session.Memory
                let updatedSession = {
                    session with
                        Memory = { updatedMemory with LastUpdated = DateTime.UtcNow }
                        LastActivity = DateTime.UtcNow
                }
                sessions.[sessionId] <- updatedSession
                logger.LogInformation($"🧠 SESSION: Updated memory for session {sessionId}")
                Some updatedSession
            | false, _ ->
                logger.LogWarning($"❌ SESSION: Session {sessionId} not found for memory update")
                None
        )
    
    /// Add fact to session memory
    member this.AddFact(sessionId: string, key: string, value: string, source: string) =
        let memoryItem = {
            Key = key
            Value = value
            Type = "fact"
            Confidence = 0.8
            CreatedAt = DateTime.UtcNow
            LastAccessed = DateTime.UtcNow
            AccessCount = 0
            Source = source
        }

        logger.LogInformation($"🧠 MEMORY ADD: Adding fact '{key}' to session {sessionId} (Source: {source})")

        this.UpdateMemory(sessionId, fun memory ->
            { memory with Facts = memory.Facts.Add(key, memoryItem) }
        )
    
    /// Add user preference to session memory
    member this.AddPreference(sessionId: string, key: string, value: string) =
        let memoryItem = {
            Key = key
            Value = value
            Type = "preference"
            Confidence = 0.9
            CreatedAt = DateTime.UtcNow
            LastAccessed = DateTime.UtcNow
            AccessCount = 0
            Source = "user_input"
        }

        logger.LogInformation($"🧠 MEMORY ADD: Adding preference '{key}' to session {sessionId}")

        this.UpdateMemory(sessionId, fun memory ->
            { memory with UserPreferences = memory.UserPreferences.Add(key, memoryItem) }
        )

    /// Extract and learn facts from conversation messages
    member this.ExtractFactsFromMessage(sessionId: string, message: ChatMessage) =
        // Simple fact extraction patterns - could be enhanced with NLP
        let content = message.Content.ToLowerInvariant()

        // Extract user preferences
        if message.Role = User then
            // Look for preference patterns like "I prefer...", "I like...", "I use..."
            let preferencePatterns = [
                ("i prefer", "preference")
                ("i like", "preference")
                ("i use", "tool_preference")
                ("my favorite", "preference")
                ("i work with", "work_context")
                ("i'm working on", "current_project")
            ]

            for (pattern, prefType) in preferencePatterns do
                if content.Contains(pattern) then
                    let startIndex = content.IndexOf(pattern) + pattern.Length
                    if startIndex < content.Length - 10 then
                        let remainder = content.Substring(startIndex).Trim()
                        let endIndex = remainder.IndexOfAny([|'.'; '!'; '?'; '\n'|])
                        let value = if endIndex > 0 then remainder.Substring(0, endIndex).Trim() else remainder.Trim()
                        if value.Length > 3 && value.Length < 100 then
                            this.AddPreference(sessionId, prefType, value) |> ignore

        // Extract facts from TARS responses
        elif message.Role = Assistant then
            // Look for factual statements that TARS learned
            if content.Contains("learned") || content.Contains("discovered") || content.Contains("found that") then
                // Extract key facts from TARS responses for future reference
                let sentences = content.Split([|'.'; '!'; '?'|], StringSplitOptions.RemoveEmptyEntries)
                for sentence in sentences |> Array.take (min 3 sentences.Length) do
                    let trimmed = sentence.Trim()
                    if trimmed.Length > 20 && trimmed.Length < 200 then
                        if trimmed.Contains("quantum") || trimmed.Contains("consciousness") ||
                           trimmed.Contains("ai") || trimmed.Contains("tars") then
                            let factKey = $"conversation_fact_{DateTime.UtcNow.Ticks}"
                            this.AddFact(sessionId, factKey, trimmed, "tars_response") |> ignore

    /// Persist session memory to long-term learning storage
    member this.PersistSessionMemoryToLearning(sessionId: string) =
        async {
            match learningMemoryService with
            | Some memoryService ->
                match this.GetSession(sessionId) with
                | Some session ->
                    logger.LogInformation($"💾 LEARNING PERSIST: Persisting session memory to long-term storage for session {sessionId}")

                    // Persist facts
                    for (_, fact) in session.Memory.Facts |> Map.toList do
                        let! storeResult = memoryService.StoreKnowledge(
                            fact.Key,
                            fact.Value,
                            LearningSource.UserInteraction(sessionId),
                            None)
                        match storeResult with
                        | Ok knowledgeId ->
                            logger.LogInformation($"✅ LEARNING PERSIST: Stored fact '{fact.Key}' with ID {knowledgeId}")
                        | Error error ->
                            logger.LogWarning($"⚠️ LEARNING PERSIST: Failed to store fact '{fact.Key}': {error}")

                    // Persist user preferences as knowledge
                    for (_, pref) in session.Memory.UserPreferences |> Map.toList do
                        let prefTopic = $"user_preference_{pref.Key}"
                        let prefContent = $"User preference: {pref.Value}"
                        let! storeResult = memoryService.StoreKnowledge(
                            prefTopic,
                            prefContent,
                            LearningSource.UserInteraction(sessionId),
                            None)
                        match storeResult with
                        | Ok knowledgeId ->
                            logger.LogInformation($"✅ LEARNING PERSIST: Stored preference '{pref.Key}' with ID {knowledgeId}")
                        | Error error ->
                            logger.LogWarning($"⚠️ LEARNING PERSIST: Failed to store preference '{pref.Key}': {error}")

                    return Ok "Session memory persisted to long-term storage"
                | None ->
                    return Error "Session not found"
            | None ->
                logger.LogWarning("⚠️ LEARNING PERSIST: No learning memory service available")
                return Error "No learning memory service available"
        }
    
    /// Get session context for LLM
    member this.GetSessionContext(sessionId: string) =
        match this.GetSession(sessionId) with
        | Some session ->
            let recentMessages = 
                session.Messages 
                |> List.take (min 10 session.Messages.Length) // Last 10 messages
                |> List.rev // Chronological order
            
            let memoryContext = 
                let facts = session.Memory.Facts |> Map.toList |> List.map (fun (_, item) -> $"- {item.Key}: {item.Value}")
                let prefs = session.Memory.UserPreferences |> Map.toList |> List.map (fun (_, item) -> $"- {item.Key}: {item.Value}")
                
                let factStr = if facts.IsEmpty then "" else "Facts learned this session:\n" + String.concat "\n" facts
                let prefStr = if prefs.IsEmpty then "" else "User preferences:\n" + String.concat "\n" prefs
                
                [factStr; prefStr] |> List.filter (fun s -> not (String.IsNullOrEmpty s)) |> String.concat "\n\n"
            
            Some (recentMessages, memoryContext)
        | None -> None
    
    /// Get session statistics
    member this.GetSessionStats(sessionId: string) =
        match this.GetSession(sessionId) with
        | Some session ->
            let userMsgCount = session.Messages |> List.filter (fun m -> m.Role = User) |> List.length
            let assistantMsgCount = session.Messages |> List.filter (fun m -> m.Role = Assistant) |> List.length
            let totalTokens = session.Messages |> List.choose (fun m -> m.TokensUsed) |> List.sum
            let responseTimes = session.Messages |> List.choose (fun m -> m.ResponseTime) |> List.filter (fun t -> t > TimeSpan.Zero)
            let avgResponseTime = if responseTimes.IsEmpty then None else Some (TimeSpan.FromMilliseconds(responseTimes |> List.averageBy (fun t -> t.TotalMilliseconds)))
            let memoryItemCount = session.Memory.Facts.Count + session.Memory.UserPreferences.Count + session.Memory.ContextVariables.Count
            
            Some {
                Duration = DateTime.UtcNow - session.StartTime
                MessageCount = session.Messages.Length
                UserMessages = userMsgCount
                AssistantMessages = assistantMsgCount
                TotalTokens = if totalTokens > 0 then Some totalTokens else None
                AverageResponseTime = avgResponseTime
                MemoryItems = memoryItemCount
                LastActivity = session.LastActivity
            }
        | None -> None
    
    /// Close session
    member this.CloseSession(sessionId: string) =
        lock sessionLock (fun () ->
            match sessions.TryGetValue(sessionId) with
            | true, session ->
                let closedSession = { session with IsActive = false }
                sessions.[sessionId] <- closedSession
                logger.LogInformation($"🔒 SESSION: Closed session {sessionId}")
                true
            | false, _ ->
                logger.LogWarning($"❌ SESSION: Session {sessionId} not found for closing")
                false
        )
    
    /// List active sessions
    member this.GetActiveSessions() =
        lock sessionLock (fun () ->
            sessions.Values 
            |> Seq.filter (fun s -> s.IsActive)
            |> Seq.toList
        )
    
    /// Cleanup old sessions
    member this.CleanupOldSessions(maxAge: TimeSpan) =
        let cutoff = DateTime.UtcNow - maxAge
        lock sessionLock (fun () ->
            let oldSessions =
                sessions.Values
                |> Seq.filter (fun s -> s.LastActivity < cutoff)
                |> Seq.map (fun s -> s.SessionId)
                |> Seq.toList

            for sessionId in oldSessions do
                sessions.Remove(sessionId) |> ignore
                logger.LogInformation($"🧹 SESSION: Cleaned up old session {sessionId}")

            oldSessions.Length
        )

    /// Save session transcript to Markdown file
    member this.SaveTranscript(sessionId: string, ?filePath: string) =
        match this.GetSession(sessionId) with
        | Some session ->
            try
                let fileName =
                    match filePath with
                    | Some path -> path
                    | None ->
                        let timestamp = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss")
                        $"TARS_Session_{sessionId}_{timestamp}.md"

                let markdown = this.GenerateMarkdownTranscript(session)
                File.WriteAllText(fileName, (markdown: string), System.Text.Encoding.UTF8)
                logger.LogInformation($"💾 SESSION: Saved transcript for session {sessionId} to {fileName}")
                Ok fileName
            with
            | ex ->
                logger.LogError(ex, $"❌ SESSION: Failed to save transcript for session {sessionId}")
                Error ex.Message
        | None ->
            logger.LogWarning($"❌ SESSION: Session {sessionId} not found for transcript save")
            Error "Session not found"

    /// Generate Markdown transcript from session
    member private this.GenerateMarkdownTranscript(session: ChatSession) =
        let sb = StringBuilder()

        // Header
        sb.AppendLine($"# TARS Chat Session Transcript") |> ignore
        sb.AppendLine() |> ignore
        sb.AppendLine($"**Session ID:** {session.SessionId}") |> ignore
        sb.AppendLine($"**Model:** {session.Model}") |> ignore
        let startTimeStr = session.StartTime.ToString("yyyy-MM-dd HH:mm:ss")
        sb.AppendLine($"**Start Time:** {startTimeStr} UTC") |> ignore
        sb.AppendLine($"**Duration:** {DateTime.UtcNow - session.StartTime}") |> ignore
        sb.AppendLine($"**Message Count:** {session.Messages.Length}") |> ignore
        sb.AppendLine() |> ignore

        // Session Memory
        if not session.Memory.Facts.IsEmpty || not session.Memory.UserPreferences.IsEmpty then
            sb.AppendLine("## Session Memory") |> ignore
            sb.AppendLine() |> ignore

            if not session.Memory.Facts.IsEmpty then
                sb.AppendLine("### Facts Learned") |> ignore
                for (key, item) in session.Memory.Facts |> Map.toList do
                    sb.AppendLine($"- **{key}:** {item.Value} *(Source: {item.Source})*") |> ignore
                sb.AppendLine() |> ignore

            if not session.Memory.UserPreferences.IsEmpty then
                sb.AppendLine("### User Preferences") |> ignore
                for (key, item) in session.Memory.UserPreferences |> Map.toList do
                    sb.AppendLine($"- **{key}:** {item.Value}") |> ignore
                sb.AppendLine() |> ignore

        // Conversation
        sb.AppendLine("## Conversation") |> ignore
        sb.AppendLine() |> ignore

        let chronologicalMessages = session.Messages |> List.rev
        for (i, message) in chronologicalMessages |> List.indexed do
            let roleIcon =
                match message.Role with
                | User -> "👤"
                | Assistant -> "🤖"
                | System -> "⚙️"

            let roleName =
                match message.Role with
                | User -> "User"
                | Assistant -> "TARS"
                | System -> "System"

            sb.AppendLine($"### {i + 1}. {roleIcon} {roleName}") |> ignore
            sb.AppendLine() |> ignore
            let timestampStr = message.Timestamp.ToString("yyyy-MM-dd HH:mm:ss")
            sb.AppendLine($"**Time:** {timestampStr} UTC") |> ignore

            if message.Model.IsSome then
                sb.AppendLine($"**Model:** {message.Model.Value}") |> ignore

            if message.ResponseTime.IsSome then
                sb.AppendLine($"**Response Time:** {message.ResponseTime.Value.TotalMilliseconds:F0}ms") |> ignore

            if message.TokensUsed.IsSome then
                sb.AppendLine($"**Tokens Used:** {message.TokensUsed.Value}") |> ignore

            sb.AppendLine() |> ignore
            sb.AppendLine(message.Content) |> ignore
            sb.AppendLine() |> ignore
            sb.AppendLine("---") |> ignore
            sb.AppendLine() |> ignore

        // Footer
        sb.AppendLine("## Session Statistics") |> ignore
        sb.AppendLine() |> ignore

        match this.GetSessionStats(session.SessionId) with
        | Some stats ->
            sb.AppendLine($"- **Total Duration:** {stats.Duration}") |> ignore
            sb.AppendLine($"- **Total Messages:** {stats.MessageCount}") |> ignore
            sb.AppendLine($"- **User Messages:** {stats.UserMessages}") |> ignore
            sb.AppendLine($"- **Assistant Messages:** {stats.AssistantMessages}") |> ignore

            if stats.TotalTokens.IsSome then
                sb.AppendLine($"- **Total Tokens:** {stats.TotalTokens.Value}") |> ignore

            if stats.AverageResponseTime.IsSome then
                sb.AppendLine($"- **Average Response Time:** {stats.AverageResponseTime.Value.TotalMilliseconds:F0}ms") |> ignore

            sb.AppendLine($"- **Memory Items:** {stats.MemoryItems}") |> ignore
        | None -> ()

        sb.AppendLine() |> ignore
        sb.AppendLine("---") |> ignore
        let generatedTimeStr = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss")
        sb.AppendLine($"*Generated by TARS Chat Session Service on {generatedTimeStr} UTC*") |> ignore

        sb.ToString()
