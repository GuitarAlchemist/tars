namespace TarsEngineFSharp

open System

module ChatService =
    type ChatMessage = {
        Role: string
        Content: string
        Timestamp: DateTime
    }

    type ChatSession = {
        Id: string
        Messages: ChatMessage list
        Created: DateTime
        LastUpdated: DateTime
    }

    let createNewSession() =
        {
            Id = Guid.NewGuid().ToString()
            Messages = []
            Created = DateTime.UtcNow
            LastUpdated = DateTime.UtcNow
        }

    let addMessage (session: ChatSession) (role: string) (content: string) =
        let message = {
            Role = role
            Content = content
            Timestamp = DateTime.UtcNow
        }
        { session with 
            Messages = session.Messages @ [message]
            LastUpdated = DateTime.UtcNow }

    let getLastMessage (session: ChatSession) =
        session.Messages 
        |> List.tryLast

    let getMessageHistory (session: ChatSession) =
        session.Messages
