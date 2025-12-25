namespace Tars.Metascript

open Tars.Core
open Tars.Llm
open Domain
open System.Collections.Generic
open Tars.Connectors.EpisodeIngestion

module Config =
    /// Metadata filter for retrieval
    type MetadataFilter =
        { Field: string
          Operator: string // "eq", "ne", "contains", "gt", "lt", "gte", "lte"
          Value: string }

    /// Query type for routing
    type QueryType =
        | Factual // Direct fact lookup
        | Analytical // Requires reasoning/analysis
        | Conversational // Follow-up or contextual
        | Keyword // Keyword-heavy search
        | Unknown

    /// Metrics for retrieval
    type RetrievalMetrics =
        { mutable TotalQueries: int64
          mutable CacheHits: int64
          mutable CacheMisses: int64
          mutable TotalLatencyMs: int64
          mutable AvgResultCount: float
          mutable FallbackTriggered: int64
          mutable CompressionApplied: int64 }

        static member Create() =
            { TotalQueries = 0L
              CacheHits = 0L
              CacheMisses = 0L
              TotalLatencyMs = 0L
              AvgResultCount = 0.0
              FallbackTriggered = 0L
              CompressionApplied = 0L }

    /// Configuration for RAG (Retrieval Augmented Generation)
    type RagConfig =
        {
            /// Collection name for storing/retrieving embeddings
            CollectionName: string
            /// Number of results to retrieve for context
            TopK: int
            /// Minimum similarity score (0.0-1.0) for including results
            MinScore: float32
            /// Whether to auto-index agent outputs
            AutoIndex: bool
            /// Max characters per stored chunk
            MaxChunkChars: int
            /// Max chunks per document to index
            MaxChunks: int
            /// Max characters to include in assembled context
            MaxContextChars: int
            /// Enable hybrid search (combine keyword + semantic)
            EnableHybridSearch: bool
            /// Weight for semantic score in hybrid search (0.0-1.0)
            SemanticWeight: float32
            /// Rerank results using LLM (slower but more accurate)
            EnableReranking: bool
            /// Use LLM to expand query with related terms
            EnableQueryExpansion: bool
            /// Number of expanded queries to generate
            QueryExpansionCount: int
            /// Enable multi-hop retrieval via knowledge graph
            EnableMultiHop: bool
            /// Max hops in knowledge graph traversal
            MaxHops: int
            /// Metadata filters to apply before scoring
            MetadataFilters: MetadataFilter list
            /// Cache embeddings to avoid recomputation
            EnableEmbeddingCache: bool
            /// Max cache entries
            EmbeddingCacheSize: int
            /// Enable async batching of embeddings
            EnableAsyncBatching: bool
            /// Batch size for async embedding
            BatchSize: int
            /// Use Reciprocal Rank Fusion to combine retrieval methods
            EnableRRF: bool
            /// RRF constant k (typically 60)
            RRFConstant: int
            // ===== NEW BATCH 2 OPTIONS =====
            /// Use LLM to extract only relevant portions from retrieved docs
            EnableContextualCompression: bool
            /// Max chars to keep per doc after compression
            CompressionMaxChars: int
            /// Enable parent document retrieval (store small, retrieve large)
            EnableParentDocRetrieval: bool
            /// Collection name for parent documents
            ParentCollectionName: string
            /// Enable sentence window expansion
            EnableSentenceWindow: bool
            /// Number of sentences to expand on each side
            SentenceWindowSize: int
            /// Apply time decay to fresher documents
            EnableTimeDecay: bool
            /// Half-life for time decay in days (after this, score halves)
            TimeDecayHalfLifeDays: float
            /// Enable semantic chunking (vs fixed-size)
            EnableSemanticChunking: bool
            /// Min chars per semantic chunk
            SemanticChunkMinChars: int
            /// Max chars per semantic chunk
            SemanticChunkMaxChars: int
            /// Enable cross-encoder reranking (lighter than full LLM)
            EnableCrossEncoder: bool
            /// Cross-encoder model hint
            CrossEncoderModel: string
            /// Enable automatic query routing
            EnableQueryRouting: bool
            /// Enable answer attribution tracking
            EnableAnswerAttribution: bool
            /// Enable retrieval metrics collection
            EnableMetrics: bool
            /// Shared metrics instance
            Metrics: RetrievalMetrics option
            /// Enable fallback chain when results are insufficient
            EnableFallbackChain: bool
            /// Minimum results before triggering fallback
            FallbackMinResults: int
        }

        static member Default =
            { CollectionName = "tars_context"
              TopK = 5
              MinScore = 0.3f
              AutoIndex = true
              MaxChunkChars = 800
              MaxChunks = 8
              MaxContextChars = 4000
              EnableHybridSearch = true
              SemanticWeight = 0.7f
              EnableReranking = false
              EnableQueryExpansion = false
              QueryExpansionCount = 3
              EnableMultiHop = false
              MaxHops = 2
              MetadataFilters = []
              EnableEmbeddingCache = true
              EmbeddingCacheSize = 1000
              EnableAsyncBatching = false
              BatchSize = 10
              EnableRRF = false
              RRFConstant = 60
              EnableContextualCompression = false
              CompressionMaxChars = 2000
              EnableParentDocRetrieval = false
              ParentCollectionName = "tars_context_parents"
              EnableSentenceWindow = false
              SentenceWindowSize = 1
              EnableTimeDecay = false
              TimeDecayHalfLifeDays = 365.0
              EnableSemanticChunking = false
              SemanticChunkMinChars = 100
              SemanticChunkMaxChars = 2000
              EnableCrossEncoder = false
              CrossEncoderModel = "fast"
              EnableQueryRouting = false
              EnableAnswerAttribution = false
              EnableMetrics = false
              Metrics = None
              EnableFallbackChain = false
              FallbackMinResults = 3 }

    type MetascriptContext =
        { Llm: ILlmService
          Tools: IToolRegistry
          Budget: BudgetGovernor option
          VectorStore: IVectorStore option
          KnowledgeGraph: TemporalKnowledgeGraph.TemporalGraph option
          SemanticMemory: ISemanticMemory option
          EpisodeService: IEpisodeIngestionService option
          RagConfig: RagConfig
          MacroRegistry: IMacroRegistry option
          MetascriptRegistry: IMetascriptRegistry option }
