namespace TarsEngine.SelfImprovement

open System
open System.Text.Json.Serialization
open System.Text.Json

/// <summary>
/// Represents the source type of a knowledge item
/// </summary>
type KnowledgeSourceType =
    | Chat
    | Reflection
    | Documentation
    | Feature
    | Architecture
    | Tutorial

/// <summary>
/// Represents a knowledge item extracted from exploration chats
/// </summary>
type ExplorationKnowledgeItem = {
    Id: string
    Type: string  // Concept, Insight, CodePattern, etc.
    Content: string
    Source: string  // File path
    SourceType: KnowledgeSourceType
    Confidence: float
    Tags: string list
    RelatedItems: string list
    ExtractedAt: DateTime
}

/// <summary>
/// Represents the knowledge base containing all extracted knowledge items
/// </summary>
type KnowledgeBase = {
    Items: ExplorationKnowledgeItem list
    LastUpdated: DateTime
    Version: string
    Statistics: Map<string, string>
}

/// <summary>
/// Alias for ExplorationKnowledgeItem to make it easier to use
/// </summary>
type KnowledgeItem = ExplorationKnowledgeItem
