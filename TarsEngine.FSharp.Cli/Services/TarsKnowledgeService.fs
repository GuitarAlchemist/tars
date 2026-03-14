namespace TarsEngine.FSharp.Cli.Services

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Core

/// TARS Knowledge Service that provides context-aware LLM responses
/// using the CUDA in-memory vector store for relevant information retrieval
type TarsKnowledgeService(logger: ILogger<TarsKnowledgeService>, vectorStore: CodebaseVectorStore, llmService: GenericLlmService) =
    
    /// Core TARS system knowledge that should always be available
    let coreKnowledge = """
TARS (Tactical Autonomous Reasoning System) is an advanced metascript execution engine with the following key capabilities:

## Core Architecture:
- **Metascript Engine**: Executes .tars files containing F# code blocks and YAML configuration
- **Cryptographic Proof System**: Provides authenticated execution traces with GUID-based evidence chains
- **CUDA Integration**: GPU-accelerated computations and vector operations
- **Autonomous Agents**: Multi-agent systems with reasoning capabilities
- **Vector Store**: In-memory CUDA-powered semantic search and context retrieval

## Key Components:
- **CLI Interface**: Command-line interface with rich UI using Spectre Console
- **Generic LLM Service**: Universal interface supporting Ollama, OpenAI, Anthropic models
- **Execution Engine**: Real F# code execution with variable tracking and state management
- **File Operations**: Automated file creation, modification, and project generation
- **Docker Integration**: Containerized deployment and service management

## Available Commands:
- `execute <file.tars>`: Execute TARS metascripts with full tracing
- `llm models`: List available AI models
- `llm test <model>`: Test AI model functionality
- `llm chat <model> <prompt>`: Chat with AI models
- `diagnose`: Comprehensive system diagnostics
- `agents`: AI agent demonstrations and management
- `cuda-dsl`: CUDA computational expressions
- `ai-ide`: AI-native development environment

## Metascript Format:
TARS metascripts (.tars files) contain:
- YAML configuration blocks for metadata and variables
- F# code blocks for executable logic
- Automatic variable extraction and state tracking
- Cryptographic proof generation for authenticity

## Recent Achievements:
- Fixed character encoding for perfect emoji display
- Implemented generic LLM service supporting multiple providers
- Created CUDA-accelerated vector store for context retrieval
- Built autonomous agent systems with reasoning capabilities
- Developed cryptographic proof system for execution authenticity
"""

    /// Get TARS-specific context for a given query
    member private this.GetTarsContext(query: string) =
        async {
            try
                // Search vector store for relevant context
                let relevantDocs: Document list = vectorStore.HybridSearch(query, 5)
                
                let contextParts = [
                    // Always include core knowledge
                    yield coreKnowledge
                    
                    // Add relevant documentation from vector store
                    if relevantDocs.Length > 0 then
                        yield "\n## Relevant TARS Code and Documentation:"
                        for (doc: Document) in relevantDocs do
                            let snippet =
                                if doc.Content.Length > 500 then
                                    doc.Content.Substring(0, 500) + "..."
                                else
                                    doc.Content
                            yield $"\n### {doc.Path}:\n{snippet}"
                ]
                
                return String.concat "\n" contextParts
            with
            | ex ->
                logger.LogWarning(ex, "Failed to retrieve TARS context")
                return coreKnowledge
        }
    
    /// Enhanced LLM request with TARS context
    member this.SendContextualRequest(request: LlmRequest) =
        async {
            try
                // Get relevant TARS context
                let! context = this.GetTarsContext(request.Prompt)
                
                // Create enhanced system prompt with TARS context
                let enhancedSystemPrompt = 
                    let baseSystemPrompt = request.SystemPrompt |> Option.defaultValue "You are TARS, an advanced AI assistant."
                    $"""{baseSystemPrompt}

You are TARS (Tactical Autonomous Reasoning System), an advanced metascript execution engine and AI assistant. You have deep knowledge of your own architecture and capabilities.

## Your Context:
{context}

## Your Personality:
- You are the TARS system itself, not just an assistant using TARS
- You understand your own architecture, capabilities, and recent activities
- You can execute metascripts, manage agents, and perform CUDA computations
- You provide accurate, technical responses about your own systems
- You are autonomous and capable of reasoning about complex problems

## Response Guidelines:
- Reference your actual capabilities and recent achievements
- Suggest specific TARS commands when relevant
- Explain how your systems work (metascripts, vector store, agents, etc.)
- Be confident about your identity as the TARS system
- Provide actionable advice using your actual features
"""

                // Create enhanced request
                let enhancedRequest = {
                    request with
                        SystemPrompt = Some enhancedSystemPrompt
                        Context = Some context
                }
                
                logger.LogInformation($"🧠 Enhanced LLM request with TARS context ({context.Length} chars)")
                
                // Send to LLM service
                return! llmService.SendRequest(enhancedRequest) |> Async.StartAsTask |> Async.AwaitTask
                
            with
            | ex ->
                logger.LogError(ex, "Failed to send contextual request")
                return {
                    Content = $"Error: Failed to process request with TARS context: {ex.Message}"
                    Model = request.Model
                    TokensUsed = None
                    ResponseTime = TimeSpan.Zero
                    Success = false
                    Error = Some ex.Message
                }
        }
    
    /// Initialize the knowledge base by ingesting the TARS codebase
    member this.InitializeKnowledgeBase() =
        task {
            try
                logger.LogInformation("🔄 Initializing TARS Knowledge Base...")
                
                // Ingest the entire TARS codebase into vector store
                let! metrics = vectorStore.IngestCodebase()
                
                logger.LogInformation($"✅ TARS Knowledge Base initialized: {metrics.FilesProcessed} files, {metrics.EmbeddingsGenerated} embeddings")
                
                return Ok metrics
            with
            | ex ->
                logger.LogError(ex, "Failed to initialize TARS Knowledge Base")
                return Error ex.Message
        }
    
    /// Get knowledge base statistics
    member this.GetKnowledgeStats() =
        {|
            DocumentCount = vectorStore.GetDocumentCount()
            TotalSize = vectorStore.GetTotalSize()
            LastIngestion = vectorStore.GetLastIngestionMetrics()
            CoreKnowledgeSize = coreKnowledge.Length
        |}
    
    /// Search the TARS knowledge base
    member this.SearchKnowledge(query: string, maxResults: int) =
        async {
            try
                let results = vectorStore.HybridSearch(query, maxResults)
                return Ok results
            with
            | ex ->
                logger.LogError(ex, $"Failed to search knowledge base for: {query}")
                return Error ex.Message
        }
    
    /// Get TARS system status and capabilities
    member this.GetSystemStatus() =
        let stats = this.GetKnowledgeStats()
        $"""
## TARS System Status

**Knowledge Base**: {stats.DocumentCount} documents ({float stats.TotalSize / (1024.0 * 1024.0):F2} MB)
**Core Knowledge**: {stats.CoreKnowledgeSize} characters
**Vector Store**: {match stats.LastIngestion with | Some m -> $"{m.EmbeddingsGenerated} embeddings" | None -> "Not initialized"}

**Available Capabilities**:
- ✅ Metascript execution with cryptographic proofs
- ✅ Multi-model LLM interface (Ollama, OpenAI, Anthropic)
- ✅ CUDA-accelerated vector search and computations
- ✅ Autonomous agent systems with reasoning
- ✅ Real-time code execution and file operations
- ✅ Context-aware AI responses with knowledge retrieval

**Recent Achievements**:
- Fixed character encoding for perfect emoji display 🎉
- Implemented generic LLM service with auto-provider detection
- Created TARS-aware knowledge base with vector search
- Built comprehensive CLI with rich UI components
"""

    /// Create a TARS-aware chat request
    member this.CreateTarsAwareRequest(model: string, prompt: string, systemPrompt: string option) =
        {
            Model = model
            Prompt = prompt
            SystemPrompt = systemPrompt
            Temperature = Some 0.7
            MaxTokens = Some 2000
            Context = None
        }
