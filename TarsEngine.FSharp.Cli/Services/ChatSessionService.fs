namespace TarsEngine.FSharp.Cli.Services

open System
open System.Collections.Concurrent

/// Chat session service for TARS
module ChatSessionService =
    
    type ChatMessage = {
        Id: Guid
        SessionId: Guid
        Role: string // "user" or "assistant"
        Content: string
        Timestamp: DateTime
    }
    
    type ChatSession = {
        Id: Guid
        Name: string
        CreatedAt: DateTime
        LastActivity: DateTime
        Messages: ChatMessage list
    }
    
    let private sessions = ConcurrentDictionary<Guid, ChatSession>()
    
    /// Create a new chat session
    let createSession name =
        let session = {
            Id = Guid.NewGuid()
            Name = name
            CreatedAt = DateTime.UtcNow
            LastActivity = DateTime.UtcNow
            Messages = []
        }
        sessions.TryAdd(session.Id, session) |> ignore
        session
    
    /// Get session by ID
    let getSession sessionId =
        sessions.TryGetValue(sessionId) |> function
        | true, session -> Some session
        | false, _ -> None
    
    /// Get all sessions
    let getAllSessions() =
        sessions.Values |> Seq.toList
    
    /// Add message to session
    let addMessage sessionId role content =
        match sessions.TryGetValue(sessionId) with
        | true, session ->
            let message = {
                Id = Guid.NewGuid()
                SessionId = sessionId
                Role = role
                Content = content
                Timestamp = DateTime.UtcNow
            }
            let updatedSession = {
                session with
                    Messages = message :: session.Messages
                    LastActivity = DateTime.UtcNow
            }
            sessions.TryUpdate(sessionId, updatedSession, session) |> ignore
            Some message
        | false, _ -> None
    
    /// Get messages for session
    let getMessages sessionId =
        match getSession sessionId with
        | Some session -> session.Messages |> List.rev
        | None -> []
    
    /// Delete session
    let deleteSession sessionId =
        sessions.TryRemove(sessionId) |> fst
    
    /// Clear all sessions
    let clearAllSessions() =
        sessions.Clear()
    
    /// Get session statistics
    let getStatistics() = {|
        TotalSessions = sessions.Count
        TotalMessages = sessions.Values |> Seq.sumBy (fun s -> s.Messages.Length)
        ActiveSessions = sessions.Values |> Seq.filter (fun s -> s.LastActivity > DateTime.UtcNow.AddHours(-1.0)) |> Seq.length
    |}
