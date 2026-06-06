namespace TarsEngine.SelfImprovement

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open System.Collections.Generic

/// <summary>
/// Represents the type of message being sent between agents
/// </summary>
type MessageType =
    | Instruction // A directive or instruction to an agent
    | Response    // A response to an instruction
    | Question    // A question that requires a response
    | Information // Informational message that doesn't require a response
    | Error       // An error message

/// <summary>
/// Represents a message sent between agents
/// </summary>
type AgentCommunicationMessage = {
    MessageId: string
    ParentMessageId: string option
    SenderAgentId: string
    ReceiverAgentId: string
    MessageType: MessageType
    Content: string
    Metadata: Dictionary<string, string>
    Timestamp: DateTime
}

/// <summary>
/// Represents a conversation between agents
/// </summary>
type AgentConversation = {
    ConversationId: string
    Messages: AgentCommunicationMessage list
    StartTime: DateTime
    EndTime: DateTime option
    Status: string
}

/// <summary>
/// Functions for working with agent communication
/// </summary>
module AgentCommunication =
    /// <summary>
    /// Creates a new message
    /// </summary>
    let createMessage (senderId: string) (receiverId: string) (messageType: MessageType) (content: string) =
        {
            MessageId = Guid.NewGuid().ToString()
            ParentMessageId = None
            SenderAgentId = senderId
            ReceiverAgentId = receiverId
            MessageType = messageType
            Content = content
            Metadata = new Dictionary<string, string>()
            Timestamp = DateTime.UtcNow
        }
    
    /// <summary>
    /// Creates a response message to a previous message
    /// </summary>
    let createResponseMessage (parentMessage: AgentCommunicationMessage) (content: string) =
        {
            MessageId = Guid.NewGuid().ToString()
            ParentMessageId = Some parentMessage.MessageId
            SenderAgentId = parentMessage.ReceiverAgentId
            ReceiverAgentId = parentMessage.SenderAgentId
            MessageType = Response
            Content = content
            Metadata = new Dictionary<string, string>()
            Timestamp = DateTime.UtcNow
        }
    
    /// <summary>
    /// Creates a new conversation
    /// </summary>
    let createConversation () =
        {
            ConversationId = Guid.NewGuid().ToString()
            Messages = []
            StartTime = DateTime.UtcNow
            EndTime = None
            Status = "Active"
        }
    
    /// <summary>
    /// Adds a message to a conversation
    /// </summary>
    let addMessageToConversation (conversation: AgentConversation) (message: AgentCommunicationMessage) =
        { conversation with Messages = conversation.Messages @ [message] }
    
    /// <summary>
    /// Ends a conversation
    /// </summary>
    let endConversation (conversation: AgentConversation) =
        { conversation with 
            EndTime = Some DateTime.UtcNow
            Status = "Completed" 
        }
    
    /// <summary>
    /// Gets the last message in a conversation
    /// </summary>
    let getLastMessage (conversation: AgentConversation) =
        match conversation.Messages with
        | [] -> None
        | messages -> Some (List.last messages)
    
    /// <summary>
    /// Gets all messages sent by a specific agent
    /// </summary>
    let getMessagesBySender (conversation: AgentConversation) (senderId: string) =
        conversation.Messages
        |> List.filter (fun m -> m.SenderAgentId = senderId)
    
    /// <summary>
    /// Gets all messages sent to a specific agent
    /// </summary>
    let getMessagesByReceiver (conversation: AgentConversation) (receiverId: string) =
        conversation.Messages
        |> List.filter (fun m -> m.ReceiverAgentId = receiverId)
    
    /// <summary>
    /// Gets all messages of a specific type
    /// </summary>
    let getMessagesByType (conversation: AgentConversation) (messageType: MessageType) =
        conversation.Messages
        |> List.filter (fun m -> m.MessageType = messageType)
    
    /// <summary>
    /// Gets a message by its ID
    /// </summary>
    let getMessageById (conversation: AgentConversation) (messageId: string) =
        conversation.Messages
        |> List.tryFind (fun m -> m.MessageId = messageId)
    
    /// <summary>
    /// Gets all responses to a specific message
    /// </summary>
    let getResponsesToMessage (conversation: AgentConversation) (messageId: string) =
        conversation.Messages
        |> List.filter (fun m -> 
            match m.ParentMessageId with
            | Some id -> id = messageId
            | None -> false)
    
    /// <summary>
    /// Serializes a message to JSON
    /// </summary>
    let serializeMessage (message: AgentCommunicationMessage) =
        let options = JsonSerializerOptions()
        options.WriteIndented <- true
        options.Converters.Add(JsonFSharpConverter())
        JsonSerializer.Serialize(message, options)
    
    /// <summary>
    /// Deserializes a message from JSON
    /// </summary>
    let deserializeMessage (json: string) =
        let options = JsonSerializerOptions()
        options.Converters.Add(JsonFSharpConverter())
        JsonSerializer.Deserialize<AgentCommunicationMessage>(json, options)
    
    /// <summary>
    /// Serializes a conversation to JSON
    /// </summary>
    let serializeConversation (conversation: AgentConversation) =
        let options = JsonSerializerOptions()
        options.WriteIndented <- true
        options.Converters.Add(JsonFSharpConverter())
        JsonSerializer.Serialize(conversation, options)
    
    /// <summary>
    /// Deserializes a conversation from JSON
    /// </summary>
    let deserializeConversation (json: string) =
        let options = JsonSerializerOptions()
        options.Converters.Add(JsonFSharpConverter())
        JsonSerializer.Deserialize<AgentConversation>(json, options)
    
    /// <summary>
    /// Saves a conversation to a file
    /// </summary>
    let saveConversationToFile (conversation: AgentConversation) (filePath: string) =
        let json = serializeConversation conversation
        File.WriteAllText(filePath, json)
    
    /// <summary>
    /// Loads a conversation from a file
    /// </summary>
    let loadConversationFromFile (filePath: string) =
        let json = File.ReadAllText(filePath)
        deserializeConversation json
