namespace TarsEngine.FSharp.Core.Grammar

open System
open System.Collections.Generic
open TarsEngine.FSharp.Core.Grammar.EmergentTierEvolution
open Tars.Engine.Grammar.FractalGrammar
open Tars.Engine.Grammar
open TarsEngine.FSharp.Core.Tracing.AgenticTraceCapture

/// Unified Grammar Evolution System
/// Integrates Emergent Tier Evolution with Fractal Grammar Generation
/// Provides comprehensive autonomous language evolution capabilities
module UnifiedGrammarEvolution =

    // ============================================================================
    // UNIFIED EVOLUTION TYPES
    // ============================================================================

    type UnifiedEvolutionContext = {
        ProblemContext: ProblemContext
        FractalProperties: FractalProperties
        EvolutionHistory: EvolutionStep list
        CurrentCapabilities: string list
        PerformanceMetrics: Map<string, float>
        ResourceConstraints: Map<string, float>
    }

    and EvolutionStep = {
        StepId: string
        Timestamp: DateTime
        TierLevel: int
        EvolutionType: EvolutionType
        GeneratedContent: string list
        PerformanceImpact: float
        ResourceUsage: Map<string, float>
    }

    and EvolutionType =
        | TierEvolution of GrammarEvolutionResult
        | FractalExpansion of FractalGenerationResult
        | HybridEvolution of GrammarEvolutionResult * FractalGenerationResult

    type UnifiedEvolutionResult = {
        Success: bool
        NewTierLevel: int
        GeneratedGrammar: string
        FractalStructure: FractalNode option
        EvolutionSteps: EvolutionStep list
        PerformanceImprovement: float
        ResourceEfficiency: float
        NextEvolutionSuggestions: string list
        ComprehensiveTrace: string
    }

    // ============================================================================
    // UNIFIED EVOLUTION ENGINE
    // ============================================================================

    type UnifiedGrammarEvolutionEngine() =
        let fractalEngine = FractalGrammarEngine()
        let mutable evolutionHistory = []

        /// Create unified evolution context from problem domain
        member this.CreateEvolutionContext(domain: string, constraints: Map<string, float * float>, capabilities: string list) : UnifiedEvolutionContext =
            let problemContext = {
                Domain = domain
                Constraints = constraints
                Tensions = []
                RequiredCapabilities = capabilities
                CurrentTier = 5  // Start at Tier 5 for advanced evolution
                SuccessMetrics = Map.empty
            }

            let fractalProperties = {
                Dimension = 2.0
                ScalingFactor = 0.618
                IterationDepth = 7
                SelfSimilarityRatio = 0.5
                RecursionLimit = 12
                CompositionRules = ["scale"; "compose"; "recursive"; "adaptive"; "emergent"]
            }

            {
                ProblemContext = problemContext
                FractalProperties = fractalProperties
                EvolutionHistory = []
                CurrentCapabilities = capabilities
                PerformanceMetrics = Map.empty
                ResourceConstraints = Map.empty
            }

        /// Analyze evolution potential and recommend strategy
        member this.AnalyzeEvolutionPotential(context: UnifiedEvolutionContext) : string * EvolutionType =
            let tierEvolutionPotential = 
                let directions = analyzeProblemContext context.ProblemContext
                directions.Length

            let fractalExpansionPotential =
                let complexity = fractalEngine.AnalyzeFractalComplexity({
                    Name = context.ProblemContext.Domain
                    Version = "1.0"
                    BaseGrammar = Grammar.createInline "base" ""
                    FractalRules = []
                    GlobalProperties = context.FractalProperties
                    CompositionGraph = Map.empty
                    GenerationHistory = []
                    Metadata = GrammarMetadata.createDefault context.ProblemContext.Domain
                })
                complexity.["total_rules"] :?> int

            let strategy, evolutionType = 
                match tierEvolutionPotential, fractalExpansionPotential with
                | t, f when t > 3 && f > 2 -> 
                    "Hybrid Evolution - High potential for both tier advancement and fractal expansion", 
                    HybridEvolution (evolveGrammarTier context.ProblemContext, fractalEngine.GenerateFractalGrammar({
                        Name = context.ProblemContext.Domain
                        Version = "1.0"
                        BaseGrammar = Grammar.createInline "base" ""
                        FractalRules = []
                        GlobalProperties = context.FractalProperties
                        CompositionGraph = Map.empty
                        GenerationHistory = []
                        Metadata = GrammarMetadata.createDefault context.ProblemContext.Domain
                    }))
                | t, _ when t > 2 -> 
                    "Tier Evolution - Focus on advancing computational expressions and meta-constructs", 
                    TierEvolution (evolveGrammarTier context.ProblemContext)
                | _, f when f > 1 -> 
                    "Fractal Expansion - Focus on self-similar pattern generation", 
                    FractalExpansion (fractalEngine.GenerateFractalGrammar({
                        Name = context.ProblemContext.Domain
                        Version = "1.0"
                        BaseGrammar = Grammar.createInline "base" ""
                        FractalRules = []
                        GlobalProperties = context.FractalProperties
                        CompositionGraph = Map.empty
                        GenerationHistory = []
                        Metadata = GrammarMetadata.createDefault context.ProblemContext.Domain
                    }))
                | _ -> 
                    "Conservative Evolution - Incremental tier advancement", 
                    TierEvolution (evolveGrammarTier context.ProblemContext)

            strategy, evolutionType

        /// Execute unified grammar evolution
        member this.ExecuteUnifiedEvolution(context: UnifiedEvolutionContext) : UnifiedEvolutionResult =
            let startTime = DateTime.UtcNow
            let stepId = Guid.NewGuid().ToString("N").[..7]

            // Log evolution start
            GlobalTraceCapture.LogAgentEvent(
                "unified_grammar_evolution_agent",
                "UnifiedEvolution",
                sprintf "Starting unified evolution for domain: %s" context.ProblemContext.Domain,
                Map.ofList [("domain", context.ProblemContext.Domain :> obj); ("tier", context.ProblemContext.CurrentTier :> obj)],
                Map.empty,
                0.0,
                context.ProblemContext.CurrentTier + 1,
                []
            )

            try
                let strategy, evolutionType = this.AnalyzeEvolutionPotential(context)
                
                let evolutionStep = {
                    StepId = stepId
                    Timestamp = startTime
                    TierLevel = context.ProblemContext.CurrentTier + 1
                    EvolutionType = evolutionType
                    GeneratedContent = []
                    PerformanceImpact = 0.0
                    ResourceUsage = Map.empty
                }

                let result = 
                    match evolutionType with
                    | TierEvolution tierResult ->
                        {
                            Success = true
                            NewTierLevel = tierResult.NewTier
                            GeneratedGrammar = String.concat "\n" (tierResult.GeneratedExpressions @ tierResult.PhysicsModifications @ tierResult.MetaConstructs)
                            FractalStructure = None
                            EvolutionSteps = [evolutionStep]
                            PerformanceImprovement = tierResult.ExpectedImprovement
                            ResourceEfficiency = 0.8
                            NextEvolutionSuggestions = ["Consider fractal expansion"; "Explore cross-domain integration"]
                            ComprehensiveTrace = sprintf "Tier Evolution: %s\nStrategy: %s\nGenerated %d new capabilities" tierResult.EvolutionReasoning strategy (tierResult.GeneratedExpressions.Length + tierResult.PhysicsModifications.Length + tierResult.MetaConstructs.Length)
                        }
                    
                    | FractalExpansion fractalResult ->
                        {
                            Success = fractalResult.Success
                            NewTierLevel = context.ProblemContext.CurrentTier + 1
                            GeneratedGrammar = fractalResult.GeneratedGrammar
                            FractalStructure = Some fractalResult.FractalTree
                            EvolutionSteps = [evolutionStep]
                            PerformanceImprovement = 0.7
                            ResourceEfficiency = 0.9
                            NextEvolutionSuggestions = ["Integrate with tier evolution"; "Expand fractal depth"]
                            ComprehensiveTrace = sprintf "Fractal Expansion: %s\nStrategy: %s\nIterations: %d" fractalResult.GeneratedGrammar strategy fractalResult.IterationsPerformed
                        }
                    
                    | HybridEvolution (tierResult, fractalResult) ->
                        let combinedGrammar = sprintf "%s\n\n// Fractal Extensions\n%s" 
                                                (String.concat "\n" (tierResult.GeneratedExpressions @ tierResult.PhysicsModifications @ tierResult.MetaConstructs))
                                                fractalResult.GeneratedGrammar
                        {
                            Success = tierResult.ExpectedImprovement > 0.5 && fractalResult.Success
                            NewTierLevel = tierResult.NewTier
                            GeneratedGrammar = combinedGrammar
                            FractalStructure = Some fractalResult.FractalTree
                            EvolutionSteps = [evolutionStep]
                            PerformanceImprovement = (tierResult.ExpectedImprovement + 0.7) / 2.0
                            ResourceEfficiency = 0.85
                            NextEvolutionSuggestions = ["Optimize hybrid integration"; "Explore emergent properties"; "Scale to higher tiers"]
                            ComprehensiveTrace = sprintf "Hybrid Evolution:\nTier: %s\nFractal: %s\nStrategy: %s" tierResult.EvolutionReasoning fractalResult.GeneratedGrammar strategy
                        }

                // Log successful evolution
                GlobalTraceCapture.LogAgentEvent(
                    "unified_grammar_evolution_agent",
                    "EvolutionComplete",
                    sprintf "Evolution completed successfully. New tier: %d" result.NewTierLevel,
                    Map.ofList [("success", result.Success :> obj); ("new_tier", result.NewTierLevel :> obj)],
                    Map.empty,
                    result.PerformanceImprovement,
                    result.NewTierLevel,
                    []
                )

                result

            with
            | ex ->
                // Log evolution failure
                GlobalTraceCapture.LogAgentEvent(
                    "unified_grammar_evolution_agent",
                    "EvolutionError",
                    sprintf "Evolution failed: %s" ex.Message,
                    Map.ofList [("error", ex.Message :> obj)],
                    Map.empty,
                    0.0,
                    context.ProblemContext.CurrentTier,
                    []
                )

                {
                    Success = false
                    NewTierLevel = context.ProblemContext.CurrentTier
                    GeneratedGrammar = ""
                    FractalStructure = None
                    EvolutionSteps = []
                    PerformanceImprovement = 0.0
                    ResourceEfficiency = 0.0
                    NextEvolutionSuggestions = ["Debug evolution failure"; "Simplify evolution strategy"]
                    ComprehensiveTrace = sprintf "Evolution failed: %s" ex.Message
                }

        /// Generate comprehensive evolution roadmap
        member this.GenerateEvolutionRoadmap(domains: string list) : Map<string, UnifiedEvolutionResult> =
            let roadmap = Dictionary<string, UnifiedEvolutionResult>()
            
            for domain in domains do
                let context = this.CreateEvolutionContext(domain, Map.empty, [])
                let result = this.ExecuteUnifiedEvolution(context)
                roadmap.[domain] <- result
            
            roadmap |> Seq.map (|KeyValue|) |> Map.ofSeq

    /// Unified Grammar Evolution Service
    type UnifiedGrammarEvolutionService() =
        let engine = UnifiedGrammarEvolutionEngine()
        
        /// Execute evolution for multiple domains simultaneously
        member this.EvolveMultipleDomains(domains: string list) : Map<string, UnifiedEvolutionResult> =
            engine.GenerateEvolutionRoadmap(domains)
        
        /// Get evolution recommendations for a specific domain
        member this.GetEvolutionRecommendations(domain: string, currentCapabilities: string list) : string list =
            let context = engine.CreateEvolutionContext(domain, Map.empty, currentCapabilities)
            let strategy, _ = engine.AnalyzeEvolutionPotential(context)
            [strategy; "Consider cross-domain integration"; "Explore emergent capabilities"; "Optimize resource efficiency"]
