namespace TarsEngine.FSharp.Cli.Services

open System
open System.Text.Json
open System.Text.Json.Serialization

/// Web search result (shared type)
type WebSearchResult = {
    Title: string
    Url: string
    Snippet: string
    Source: string
}

/// Knowledge quality assessment
[<JsonConverter(typeof<KnowledgeQualityConverter>)>]
type KnowledgeQuality =
    | Unverified
    | Tested
    | Validated
    | Proven
    | Breakthrough

/// JSON converter for KnowledgeQuality discriminated union
and KnowledgeQualityConverter() =
    inherit JsonConverter<KnowledgeQuality>()

    override _.Read(reader: byref<Utf8JsonReader>, typeToConvert: Type, options: JsonSerializerOptions) =
        let value = reader.GetString()
        match value with
        | "Unverified" -> Unverified
        | "Tested" -> Tested
        | "Validated" -> Validated
        | "Proven" -> Proven
        | "Breakthrough" -> Breakthrough
        | _ -> Unverified

    override _.Write(writer: Utf8JsonWriter, value: KnowledgeQuality, options: JsonSerializerOptions) =
        let stringValue =
            match value with
            | Unverified -> "Unverified"
            | Tested -> "Tested"
            | Validated -> "Validated"
            | Proven -> "Proven"
            | Breakthrough -> "Breakthrough"
        writer.WriteStringValue(stringValue)

/// Learning outcome tracking
type LearningOutcome = {
    OriginalProblem: string
    Solution: string
    PerformanceMetrics: Map<string, float>
    ImprovementAchieved: bool
    NoveltyScore: float
    VerificationStatus: KnowledgeQuality
}

/// Learning source type
[<JsonConverter(typeof<LearningSourceConverter>)>]
type LearningSource =
    | WebSearch of query: string
    | UserInteraction of sessionId: string
    | DocumentIngestion of path: string
    | AgentReasoning of agentId: string
    | SelfImprovement of iteration: int
    | CodeAnalysis of repository: string
    | ResearchPaper of arxivId: string
    | PerformanceBenchmark of testName: string

/// JSON converter for LearningSource discriminated union
and LearningSourceConverter() =
    inherit JsonConverter<LearningSource>()

    override _.Read(reader: byref<Utf8JsonReader>, typeToConvert: Type, options: JsonSerializerOptions) =
        let value = reader.GetString()
        // Simple string-based deserialization (could be enhanced for complex cases)
        if value.StartsWith("WebSearch:") then
            WebSearch(value.Substring("WebSearch:".Length).Trim())
        elif value.StartsWith("UserInteraction:") then
            UserInteraction(value.Substring("UserInteraction:".Length).Trim())
        elif value.StartsWith("DocumentIngestion:") then
            DocumentIngestion(value.Substring("DocumentIngestion:".Length).Trim())
        elif value.StartsWith("AgentReasoning:") then
            AgentReasoning(value.Substring("AgentReasoning:".Length).Trim())
        else
            WebSearch(value) // Default fallback

    override _.Write(writer: Utf8JsonWriter, value: LearningSource, options: JsonSerializerOptions) =
        let stringValue =
            match value with
            | WebSearch query -> sprintf "WebSearch: %s" query
            | UserInteraction sessionId -> sprintf "UserInteraction: %s" sessionId
            | DocumentIngestion path -> sprintf "DocumentIngestion: %s" path
            | AgentReasoning agentId -> sprintf "AgentReasoning: %s" agentId
            | SelfImprovement iteration -> sprintf "SelfImprovement: %d" iteration
            | CodeAnalysis repository -> sprintf "CodeAnalysis: %s" repository
            | ResearchPaper arxivId -> sprintf "ResearchPaper: %s" arxivId
            | PerformanceBenchmark testName -> sprintf "PerformanceBenchmark: %s" testName
        writer.WriteStringValue(stringValue)

/// Learned knowledge entry
type LearnedKnowledge = {
    Id: string
    Topic: string
    Content: string
    Source: string
    Confidence: float
    LearnedAt: DateTime
    LastAccessed: DateTime
    AccessCount: int
    Tags: string list
    WebSearchResults: WebSearchResult list option
    Quality: KnowledgeQuality
    LearningOutcome: LearningOutcome option
    RelatedKnowledge: string list
    SupersededBy: string option
    PerformanceImpact: float option
}

/// Mind map node representation
type MindMapNode = {
    Id: string
    Topic: string
    Level: int
    Knowledge: LearnedKnowledge list
    Confidence: float
    ConnectionStrength: float
}

/// Mind map edge representation
type MindMapEdge = {
    From: string
    To: string
    Strength: float
    RelationType: string
}

/// Complete mind map structure
type KnowledgeMindMap = {
    CenterTopic: string
    Nodes: MindMapNode list
    Edges: MindMapEdge list
    MaxDepth: int
    TotalKnowledge: int
}
