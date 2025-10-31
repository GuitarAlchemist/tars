namespace TarsEngine.FSharp.Core.Context

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Context.Types

/// Main context assembly pipeline that orchestrates all context engineering components
type ContextAssemblyPipeline(
    intentRouter: IIntentRouter,
    retriever: IContextRetriever,
    budget: IContextBudget,
    compressor: IContextCompressor option,
    guard: IContextGuard,
    memory: IContextMemory,
    validator: IOutputValidator,
    logger: ILogger<ContextAssemblyPipeline>) =
    
    /// Current context policy
    let mutable currentPolicy = {
        StepTokenBudget = 12000
        CompressionEnabled = true
        CompressionStrategy = "extractive-v1"
        CompressionTargets = ["low_salience_notes"; "long_logs"; "verbose_traces"]
        MaxCompressionRatio = 0.55
        RetrievalProfiles = []
        FewShotMaxExamples = 6
        FewShotPolicy = "salience-topk"
    }
    
    /// Context assembly statistics
    let mutable assemblyStats = Map.empty<string, float>
    
    /// Update assembly statistics
    let updateStats (key: string) (value: float) =
        assemblyStats <- assemblyStats.Add(key, value)
    
    /// Log context assembly metrics
    let logContextMetrics (stepName: string) (intent: Intent) (spans: ContextSpan list) (processingTime: TimeSpan) =
        let totalTokens = spans |> List.sumBy (fun s -> s.Tokens)
        let averageSalience = 
            if spans.IsEmpty then 0.0
            else spans |> List.map (fun s -> s.Salience) |> List.average
        
        let salienceDistribution = 
            spans 
            |> List.map (fun s -> s.Salience)
            |> List.sort
        
        let compressionSavings = 
            spans
            |> List.choose (fun s -> 
                match s.Metadata.TryFind("compressed") with
                | Some "true" -> Some 1
                | _ -> None)
            |> List.length
        
        logger.LogInformation("Context assembly completed for {StepName} ({Intent}): {SpanCount} spans, {TotalTokens} tokens, avg salience {AvgSalience:F2}, {ProcessingTime}ms",
            stepName, intent, spans.Length, totalTokens, averageSalience, processingTime.TotalMilliseconds)
        
        // Update statistics
        updateStats "last_span_count" (float spans.Length)
        updateStats "last_token_count" (float totalTokens)
        updateStats "last_avg_salience" averageSalience
        updateStats "last_processing_time_ms" processingTime.TotalMilliseconds
        updateStats "compression_savings" (float compressionSavings)
    
    /// Create context span from text
    let createContextSpan (id: string) (text: string) (source: string) (salience: float) (intent: Intent option) =
        {
            Id = id
            Text = text
            Tokens = Math.Max(1, text.Length / 4) // Simple token estimation
            Salience = salience
            Source = source
            Timestamp = DateTime.UtcNow
            Intent = intent |> Option.map (fun i -> i.ToString())
            Metadata = Map.empty
        }
    
    /// Store context spans in memory for future retrieval
    let storeContextSpans (spans: ContextSpan list) (intent: Intent) =
        task {
            try
                // Tag spans with current intent for better future retrieval
                let taggedSpans = 
                    spans
                    |> List.map (fun span -> 
                        { span with 
                            Intent = Some (intent.ToString())
                            Metadata = span.Metadata.Add("assembly_timestamp", DateTime.UtcNow.ToString("O")) })
                
                // Store in ephemeral memory
                do! memory.StoreEphemeralAsync(taggedSpans)
                
                // Promote high-salience spans to working set
                let highSalienceSpans = spans |> List.filter (fun s -> s.Salience >= 0.6)
                if not highSalienceSpans.IsEmpty then
                    do! memory.PromoteToWorkingSetAsync(highSalienceSpans)
                
                logger.LogDebug("Stored {TotalSpans} spans in memory ({HighSalience} promoted to working set)",
                    spans.Length, highSalienceSpans.Length)
                    
            with
            | ex ->
                logger.LogError(ex, "Failed to store context spans in memory")
        }
    
    interface IContextAssembly with
        
        member _.AssembleContextAsync(stepName, input, policy) =
            task {
                let startTime = DateTime.UtcNow
                logger.LogInformation("Starting context assembly for step: {StepName}", stepName)
                
                try
                    // Step 1: Classify intent
                    let intent = intentRouter.ClassifyIntent(stepName, input)
                    let confidence = intentRouter.GetConfidence(stepName, input, intent)
                    
                    logger.LogDebug("Classified intent: {Intent} (confidence: {Confidence:F2})", intent, confidence)
                    
                    // Step 2: Sanitize input
                    let sanitizedInput = guard.SanitizeContext(input)
                    
                    if sanitizedInput <> input then
                        logger.LogWarning("Input was sanitized for security")
                    
                    // Step 3: Retrieve relevant context
                    let! retrievedSpans = retriever.RetrieveAsync(intent, sanitizedInput)
                    
                    logger.LogDebug("Retrieved {SpanCount} context spans", retrievedSpans.Length)
                    
                    // Step 4: Score and prioritize spans
                    let scoredSpans = budget.ScoreSpans(intent, retrievedSpans)
                    
                    // Step 5: Apply compression if enabled
                    let! processedSpans = 
                        if policy.CompressionEnabled then
                            match compressor with
                            | Some comp ->
                                let! compressionResult = comp.CompressSpans(scoredSpans)
                                logger.LogInformation("Compression completed: ratio {Ratio:F2}, quality {Quality:F2}",
                                    compressionResult.CompressionRatio, compressionResult.QualityEstimate)
                                return compressionResult.CompressedSpans
                            | None ->
                                logger.LogWarning("Compression enabled but no compressor available")
                                return scoredSpans
                        else
                            return scoredSpans
                    
                    // Step 6: Enforce token budget
                    let budgetedSpans = budget.EnforceTokenBudget(policy.StepTokenBudget, processedSpans)
                    
                    // Step 7: Final security check
                    let finalSpans = 
                        budgetedSpans
                        |> List.map (fun span -> 
                            let sanitizedText = guard.SanitizeContext(span.Text)
                            if sanitizedText <> span.Text then
                                { span with 
                                    Text = sanitizedText
                                    Tokens = Math.Max(1, sanitizedText.Length / 4)
                                    Metadata = span.Metadata.Add("sanitized", "true") }
                            else
                                span)
                    
                    // Step 8: Store spans for future retrieval
                    do! storeContextSpans finalSpans intent
                    
                    // Step 9: Log metrics and update statistics
                    let processingTime = DateTime.UtcNow - startTime
                    logContextMetrics stepName intent finalSpans processingTime
                    
                    logger.LogInformation("Context assembly completed: {FinalSpanCount} spans, {TotalTokens} tokens",
                        finalSpans.Length, finalSpans |> List.sumBy (fun s -> s.Tokens))
                    
                    return finalSpans
                    
                with
                | ex ->
                    logger.LogError(ex, "Context assembly failed for step: {StepName}", stepName)
                    
                    // Return minimal context on failure
                    let fallbackSpan = createContextSpan 
                        "fallback" 
                        $"Context assembly failed: {ex.Message}" 
                        "error" 
                        0.1 
                        (Some intent)
                    
                    return [fallbackSpan]
            }
        
        member _.GetContextStats() =
            assemblyStats
        
        member _.UpdatePolicy(policy) =
            currentPolicy <- policy
            logger.LogInformation("Context policy updated: budget={Budget}, compression={Compression}",
                policy.StepTokenBudget, policy.CompressionEnabled)

/// Factory for creating context assembly pipeline
type ContextAssemblyFactory() =
    
    /// Create a complete context assembly pipeline with default configuration
    static member CreatePipeline(logger: ILogger<ContextAssemblyPipeline>) =
        
        // Token estimator function
        let tokenEstimator (text: string) = Math.Max(1, text.Length / 4)
        
        // Create memory manager
        let memoryConfig = {
            EphemeralPath = "memory/ephemeral.jsonl"
            WorkingSetPath = "memory/working_set.jsonl"
            LongTermPath = "memory/long_term.jsonl"
            EphemeralMaxSpans = 100
            WorkingSetMaxSpans = 500
            SalienceDecayRate = 0.95
            PromotionThreshold = 0.6
            ConsolidationFrequency = "daily"
        }
        let memoryLogger = logger :> ILogger<TieredMemoryManager>
        let memory = TieredMemoryManager(memoryConfig, memoryLogger) :> IContextMemory
        
        // Create intent router
        let routerLogger = logger :> ILogger<IntentClassificationRouter>
        let intentRouter = IntentClassificationRouter(routerLogger) :> IIntentRouter
        
        // Create context budget
        let budgetConfig = {
            MaxTokens = 12000
            SalienceWeight = 0.6
            RecencyWeight = 0.2
            IntentWeight = 0.15
            SourceWeight = 0.05
            TokenEstimator = tokenEstimator
        }
        let budgetLogger = logger :> ILogger<SalienceContextBudget>
        let budget = SalienceContextBudget(budgetConfig, budgetLogger) :> IContextBudget
        
        // Create retriever
        let retrievalConfig = {
            DefaultMaxSpans = 20
            DefaultSalienceThreshold = 0.3
            RetrievalProfiles = Map.empty // Will be populated from policy
            EnableCaching = true
            CacheExpiryMinutes = 30
        }
        let retrieverLogger = logger :> ILogger<IntentAwareContextRetriever>
        let retriever = IntentAwareContextRetriever(retrievalConfig, memory, retrieverLogger) :> IContextRetriever
        
        // Create compressor
        let compressionConfig = {
            MaxCompressionRatio = 0.55
            QualityThreshold = 0.8
            PreservePatterns = []
            CompressionTargets = ["verbose", "debug", "trace"]
            TokenEstimator = tokenEstimator
        }
        let compressorLogger = logger :> ILogger<ExtractiveContextCompressor>
        let compressor = ExtractiveContextCompressor(compressionConfig, compressorLogger) :> IContextCompressor
        
        // Create security guard
        let securityConfig = {
            AllowedTools = Set.ofList ["fs.read"; "git.diff"; "run.tests"; "cuda.benchmark"; "metascript.execute"]
            EnableSanitization = true
            EnableInjectionDetection = true
            ContentFilters = ["credentials"; "private_keys"; "system_prompts"]
            MaxContextLength = 50000
            SuspiciousPatterns = []
        }
        let guardLogger = logger :> ILogger<ContextSecurityGuard>
        let guard = ContextSecurityGuard(securityConfig, guardLogger) :> IContextGuard
        
        // Create output validator
        let validationConfig = {
            SchemaDirectory = "schemas"
            EnableStrictValidation = true
            EnableAutoRepair = true
            MaxRepairAttempts = 3
        }
        let validatorLogger = logger :> ILogger<JsonSchemaOutputValidator>
        let validator = JsonSchemaOutputValidator(validationConfig, validatorLogger) :> IOutputValidator
        
        // Create the main pipeline
        ContextAssemblyPipeline(
            intentRouter,
            retriever,
            budget,
            Some compressor,
            guard,
            memory,
            validator,
            logger
        ) :> IContextAssembly
