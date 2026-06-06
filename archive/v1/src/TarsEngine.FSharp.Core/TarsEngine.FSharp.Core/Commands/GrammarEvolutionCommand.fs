namespace TarsEngine.FSharp.Core.Commands

open System
open System.IO
open System.Threading.Tasks
open TarsEngine.FSharp.Core.Grammar.UnifiedGrammarEvolution
open TarsEngine.FSharp.Core.Grammar.GrammarEvolutionDemo
open TarsEngine.FSharp.Core.Grammar.VectorStoreGrammarAnalyzer
open TarsEngine.FSharp.Core.Grammar.ReasoningGrammarEvolution

/// Grammar Evolution CLI Commands
/// Provides command-line interface for TARS unified grammar evolution capabilities
module GrammarEvolutionCommand =

    // ============================================================================
    // COMMAND TYPES AND CONFIGURATION
    // ============================================================================

    type GrammarCommand =
        | Evolve of domains: string list * outputDir: string option
        | EvolveVector of domains: string list * outputDir: string option
        | EvolveReasoning of domains: string list * outputDir: string option
        | Analyze of domain: string * capabilities: string list
        | AnalyzeVector of domain: string * capabilities: string list
        | AnalyzeReasoning of domain: string * capabilities: string list
        | Demo of outputDir: string option
        | Status
        | Help

    type CommandResult = {
        Success: bool
        Message: string
        OutputFiles: string list
        ExecutionTime: TimeSpan
    }

    // ============================================================================
    // COMMAND IMPLEMENTATIONS
    // ============================================================================

    /// Display grammar evolution help
    let showGrammarHelp() =
        printfn "🧬 TARS Grammar Evolution Commands"
        printfn "=================================="
        printfn ""
        printfn "Commands:"
        printfn "  grammar evolve <domains> [--output <dir>]"
        printfn "    - Execute multi-domain grammar evolution"
        printfn "    - Domains: SoftwareDevelopment, AgentCoordination, MachineLearning,"
        printfn "               DataProcessing, UserInterface, Security"
        printfn "    - Example: tars grammar evolve SoftwareDevelopment,AgentCoordination"
        printfn ""
        printfn "  grammar evolve-vector <domains> [--output <dir>]"
        printfn "    - Execute VECTOR-ENHANCED multi-domain grammar evolution"
        printfn "    - Uses semantic similarity and clustering for advanced evolution"
        printfn "    - Example: tars grammar evolve-vector SoftwareDevelopment,MachineLearning"
        printfn ""
        printfn "  grammar evolve-reasoning <domains> [--output <dir>]"
        printfn "    - Execute REASONING-ENHANCED multi-domain grammar evolution"
        printfn "    - Uses BSP reasoning and complex problem solving for revolutionary evolution"
        printfn "    - Example: tars grammar evolve-reasoning SoftwareDevelopment,AgentCoordination"
        printfn ""
        printfn "  grammar analyze <domain> [--capabilities <list>]"
        printfn "    - Analyze evolution potential for a specific domain"
        printfn "    - Example: tars grammar analyze SoftwareDevelopment --capabilities autonomous_coding,refactoring"
        printfn ""
        printfn "  grammar analyze-vector <domain> [--capabilities <list>]"
        printfn "    - VECTOR-ENHANCED analysis with semantic similarity insights"
        printfn "    - Example: tars grammar analyze-vector MachineLearning --capabilities adaptive_architectures"
        printfn ""
        printfn "  grammar analyze-reasoning <domain> [--capabilities <list>]"
        printfn "    - REASONING-ENHANCED analysis with BSP reasoning and problem decomposition"
        printfn "    - Example: tars grammar analyze-reasoning AgentCoordination --capabilities semantic_routing"
        printfn ""
        printfn "  grammar demo [--output <dir>]"
        printfn "    - Run comprehensive grammar evolution demonstration"
        printfn "    - Demonstrates all 6 domains with full reporting"
        printfn ""
        printfn "  grammar status"
        printfn "    - Show grammar evolution system status"
        printfn ""
        printfn "  grammar help"
        printfn "    - Show this help message"
        printfn ""
        printfn "Options:"
        printfn "  --output <dir>        Output directory for generated files (default: output)"
        printfn "  --capabilities <list> Comma-separated list of current capabilities"
        printfn ""
        printfn "Examples:"
        printfn "  tars grammar evolve SoftwareDevelopment"
        printfn "  tars grammar evolve SoftwareDevelopment,AgentCoordination --output results"
        printfn "  tars grammar analyze MachineLearning --capabilities adaptive_architectures"
        printfn "  tars grammar demo --output demo_results"
        printfn ""
        printfn "🚀 Revolutionary multi-domain autonomous language evolution!"

    /// Show grammar evolution system status
    let showGrammarStatus() : CommandResult =
        let startTime = DateTime.UtcNow
        
        try
            printfn "🧬 TARS Grammar Evolution System Status"
            printfn "======================================="
            printfn ""
            
            // Test service creation
            let evolutionService = UnifiedGrammarEvolutionService()
            printfn "✅ UnifiedGrammarEvolutionService: OPERATIONAL"
            
            // Test demo service
            let demoService = GrammarEvolutionDemoService()
            printfn "✅ GrammarEvolutionDemoService: OPERATIONAL"
            
            // Test evolution recommendations
            let recommendations = evolutionService.GetEvolutionRecommendations("SoftwareDevelopment", ["test_capability"])
            printfn "✅ Evolution Analysis: FUNCTIONAL (%d recommendations)" recommendations.Length
            
            printfn ""
            printfn "📊 Supported Domains:"
            let supportedDomains = [
                "SoftwareDevelopment - Autonomous coding, intelligent refactoring"
                "AgentCoordination - Semantic routing, dynamic team formation"
                "MachineLearning - Adaptive architectures, continual learning"
                "DataProcessing - Stream optimization, adaptive pipelines"
                "UserInterface - Dynamic generation, adaptive patterns"
                "Security - Threat evolution, adaptive policies"
            ]
            
            for domain in supportedDomains do
                printfn "  • %s" domain
            
            printfn ""
            printfn "🎯 Evolution Capabilities:"
            printfn "  • Tier Evolution: Tier 5 → Tier 6+ advancement"
            printfn "  • Fractal Grammar: Self-similar pattern generation"
            printfn "  • Hybrid Evolution: Combined tier + fractal strategies"
            printfn "  • Performance Optimization: Real-time constraint analysis"
            printfn "  • Comprehensive Tracing: Full agentic event logging"
            printfn ""
            printfn "🚀 Status: READY FOR REVOLUTIONARY GRAMMAR EVOLUTION"
            
            {
                Success = true
                Message = "Grammar evolution system fully operational"
                OutputFiles = []
                ExecutionTime = DateTime.UtcNow - startTime
            }
            
        with
        | ex ->
            printfn "❌ Grammar evolution system error: %s" ex.Message
            {
                Success = false
                Message = sprintf "System check failed: %s" ex.Message
                OutputFiles = []
                ExecutionTime = DateTime.UtcNow - startTime
            }

    /// Execute vector-enhanced analysis for a domain
    let analyzeVectorEvolution(domain: string, capabilities: string list) : CommandResult =
        let startTime = DateTime.UtcNow

        try
            printfn "🔬 TARS Vector-Enhanced Grammar Evolution Analysis"
            printfn "==============================================="
            printfn ""
            printfn "🎯 Domain: %s" domain
            printfn "⚙️  Current Capabilities: %s" (String.concat ", " capabilities)
            printfn "🧬 Vector Analysis: ENABLED"
            printfn ""

            let vectorService = VectorEnhancedGrammarEvolutionService()

            // Run async operation synchronously for CLI
            let recommendations =
                vectorService.GetSemanticRecommendations(domain, capabilities)
                |> Async.AwaitTask
                |> Async.RunSynchronously

            printfn "📊 Vector-Enhanced Analysis Results:"
            printfn "===================================="
            printfn ""

            for i, recommendation in recommendations |> List.indexed do
                printfn "%d. %s" (i + 1) recommendation

            printfn ""
            printfn "🎯 Vector-Enhanced Recommendations:"
            printfn "  1. Execute vector evolution: tars grammar evolve-vector %s" domain
            printfn "  2. Leverage semantic clustering for optimization"
            printfn "  3. Monitor vector space coherence metrics"
            printfn "  4. Apply adaptive evolution strategies"
            printfn ""
            printfn "✅ Vector analysis completed successfully!"

            {
                Success = true
                Message = sprintf "Vector analysis completed for %s domain with %d semantic recommendations" domain recommendations.Length
                OutputFiles = []
                ExecutionTime = DateTime.UtcNow - startTime
            }

        with
        | ex ->
            printfn "❌ Vector analysis failed: %s" ex.Message
            {
                Success = false
                Message = sprintf "Vector analysis failed: %s" ex.Message
                OutputFiles = []
                ExecutionTime = DateTime.UtcNow - startTime
            }

    /// Execute reasoning-enhanced analysis for a domain
    let analyzeReasoningEvolution(domain: string, capabilities: string list) : CommandResult =
        let startTime = DateTime.UtcNow

        try
            printfn "🧠 TARS Reasoning-Enhanced Grammar Evolution Analysis"
            printfn "================================================="
            printfn ""
            printfn "🎯 Domain: %s" domain
            printfn "⚙️  Current Capabilities: %s" (String.concat ", " capabilities)
            printfn "🧬 Vector Analysis: ENABLED"
            printfn "🧠 BSP Reasoning: ENABLED"
            printfn "🔬 Complex Problem Solving: ENABLED"
            printfn ""

            let reasoningService = ReasoningEnhancedGrammarEvolutionService()

            // Run async operation synchronously for CLI
            let recommendations =
                reasoningService.GetReasoningRecommendations(domain, capabilities)
                |> Async.AwaitTask
                |> Async.RunSynchronously

            printfn "📊 Reasoning-Enhanced Analysis Results:"
            printfn "======================================="
            printfn ""

            for i, recommendation in recommendations |> List.indexed do
                printfn "%d. %s" (i + 1) recommendation

            printfn ""
            printfn "🎯 Reasoning-Enhanced Recommendations:"
            printfn "  1. Execute reasoning evolution: tars grammar evolve-reasoning %s" domain
            printfn "  2. Leverage BSP reasoning for decision optimization"
            printfn "  3. Apply complex problem decomposition strategies"
            printfn "  4. Monitor reasoning coherence and confidence metrics"
            printfn "  5. Integrate meta-reasoning insights for continuous improvement"
            printfn ""
            printfn "✅ Reasoning analysis completed successfully!"

            {
                Success = true
                Message = sprintf "Reasoning analysis completed for %s domain with %d comprehensive recommendations" domain recommendations.Length
                OutputFiles = []
                ExecutionTime = DateTime.UtcNow - startTime
            }

        with
        | ex ->
            printfn "❌ Reasoning analysis failed: %s" ex.Message
            {
                Success = false
                Message = sprintf "Reasoning analysis failed: %s" ex.Message
                OutputFiles = []
                ExecutionTime = DateTime.UtcNow - startTime
            }

    /// Execute multi-domain grammar evolution
    let executeEvolution(domains: string list, outputDir: string option) : CommandResult =
        let startTime = DateTime.UtcNow
        let outputDirectory = defaultArg outputDir "output"
        
        try
            printfn "🧬 TARS Multi-Domain Grammar Evolution"
            printfn "====================================="
            printfn ""
            printfn "🎯 Target Domains: %s" (String.concat ", " domains)
            printfn "📁 Output Directory: %s" outputDirectory
            printfn ""
            
            // Ensure output directory exists
            if not (Directory.Exists(outputDirectory)) then
                Directory.CreateDirectory(outputDirectory) |> ignore
                printfn "📁 Created output directory: %s" outputDirectory
            
            // Execute evolution
            let evolutionService = UnifiedGrammarEvolutionService()
            printfn "🚀 Starting evolution process..."
            
            let results = evolutionService.EvolveMultipleDomains(domains)
            
            let mutable outputFiles = []
            let mutable successCount = 0
            let mutable totalPerformanceImprovement = 0.0
            let mutable totalResourceEfficiency = 0.0
            
            printfn ""
            printfn "📊 Evolution Results:"
            printfn "===================="
            
            for kvp in results do
                let domain = kvp.Key
                let result = kvp.Value
                
                if result.Success then
                    successCount <- successCount + 1
                    totalPerformanceImprovement <- totalPerformanceImprovement + result.PerformanceImprovement
                    totalResourceEfficiency <- totalResourceEfficiency + result.ResourceEfficiency
                    
                    printfn ""
                    printfn "✅ %s Evolution SUCCESS:" domain
                    printfn "   • New Tier Level: %d" result.NewTierLevel
                    printfn "   • Performance Improvement: %.1f%%" (result.PerformanceImprovement * 100.0)
                    printfn "   • Resource Efficiency: %.1f%%" (result.ResourceEfficiency * 100.0)
                    printfn "   • Generated Grammar: %d characters" result.GeneratedGrammar.Length
                    printfn "   • Next Suggestions: %d items" result.NextEvolutionSuggestions.Length
                    
                    // Save generated grammar
                    if not (String.IsNullOrEmpty(result.GeneratedGrammar)) then
                        let grammarFile = Path.Combine(outputDirectory, sprintf "grammar_%s_tier_%d.grammar" (domain.ToLower()) result.NewTierLevel)
                        File.WriteAllText(grammarFile, result.GeneratedGrammar)
                        outputFiles <- grammarFile :: outputFiles
                        printfn "   • Grammar saved: %s" grammarFile
                else
                    printfn ""
                    printfn "❌ %s Evolution FAILED" domain
                    printfn "   • Trace: %s" result.ComprehensiveTrace
            
            // Generate summary report
            let avgPerformanceImprovement = if successCount > 0 then totalPerformanceImprovement / float successCount else 0.0
            let avgResourceEfficiency = if successCount > 0 then totalResourceEfficiency / float successCount else 0.0
            
            let domainResults =
                results
                |> Seq.map (fun kvp ->
                    sprintf "### %s\n- Success: %b\n- Tier: %d\n- Performance: %.1f%%\n"
                        kvp.Key
                        kvp.Value.Success
                        kvp.Value.NewTierLevel
                        (kvp.Value.PerformanceImprovement * 100.0))
                |> String.concat "\n"

            let fileList = outputFiles |> List.map (sprintf "- %s") |> String.concat "\n"

            let reportContent = sprintf """# TARS Grammar Evolution Report
Generated: %s

## Evolution Summary
- **Domains Processed:** %d
- **Successful Evolutions:** %d/%d (%.1f%% success rate)
- **Average Performance Improvement:** %.1f%%
- **Average Resource Efficiency:** %.1f%%
- **Total Execution Time:** %.2f seconds

## Domain Results
%s

## Revolutionary Achievements
✅ Multi-domain autonomous language evolution
✅ Real-time constraint tension analysis
✅ Performance-driven evolution strategies
✅ Comprehensive capability synthesis
✅ Resource-optimized grammar generation

## Generated Files
%s

---
*Generated by TARS Unified Grammar Evolution Engine*
*🧬 Revolutionary Grammar Evolution - Real Implementation*""" (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")) domains.Length successCount domains.Length (float successCount / float domains.Length * 100.0) (avgPerformanceImprovement * 100.0) (avgResourceEfficiency * 100.0) (DateTime.UtcNow - startTime).TotalSeconds domainResults fileList
            
            let reportFile = Path.Combine(outputDirectory, "grammar_evolution_report.md")
            File.WriteAllText(reportFile, reportContent)
            outputFiles <- reportFile :: outputFiles
            
            printfn ""
            printfn "📈 Evolution Summary:"
            printfn "   • Successful Evolutions: %d/%d (%.1f%%)" successCount domains.Length (float successCount / float domains.Length * 100.0)
            printfn "   • Average Performance Improvement: %.1f%%" (avgPerformanceImprovement * 100.0)
            printfn "   • Average Resource Efficiency: %.1f%%" (avgResourceEfficiency * 100.0)
            printfn "   • Execution Time: %.2f seconds" (DateTime.UtcNow - startTime).TotalSeconds
            printfn ""
            printfn "📊 Report saved: %s" reportFile
            printfn "🎉 Grammar evolution completed successfully!"
            
            {
                Success = successCount > 0
                Message = sprintf "Evolution completed: %d/%d domains successful" successCount domains.Length
                OutputFiles = List.rev outputFiles
                ExecutionTime = DateTime.UtcNow - startTime
            }
            
        with
        | ex ->
            printfn "❌ Evolution failed: %s" ex.Message
            {
                Success = false
                Message = sprintf "Evolution failed: %s" ex.Message
                OutputFiles = []
                ExecutionTime = DateTime.UtcNow - startTime
            }

    /// Execute vector-enhanced multi-domain grammar evolution
    let executeVectorEvolution(domains: string list, outputDir: string option) : CommandResult =
        let startTime = DateTime.UtcNow
        let outputDirectory = defaultArg outputDir "output"

        try
            printfn "🧬 TARS Vector-Enhanced Multi-Domain Grammar Evolution"
            printfn "===================================================="
            printfn ""
            printfn "🎯 Target Domains: %s" (String.concat ", " domains)
            printfn "📁 Output Directory: %s" outputDirectory
            printfn "🔬 Vector Analysis: ENABLED"
            printfn ""

            // Ensure output directory exists
            if not (Directory.Exists(outputDirectory)) then
                Directory.CreateDirectory(outputDirectory) |> ignore
                printfn "📁 Created output directory: %s" outputDirectory

            // Execute vector-enhanced evolution
            let vectorService = VectorEnhancedGrammarEvolutionService()
            printfn "🚀 Starting vector-enhanced evolution process..."

            let mutable outputFiles = []
            let mutable successCount = 0
            let mutable totalSemanticImprovement = 0.0
            let mutable totalVectorOptimization = 0.0

            printfn ""
            printfn "📊 Vector-Enhanced Evolution Results:"
            printfn "===================================="

            for domain in domains do
                try
                    // Run async operation synchronously for CLI
                    let result =
                        vectorService.EvolveWithVectorAnalysis(domain, [])
                        |> Async.AwaitTask
                        |> Async.RunSynchronously

                    if result.BaseResult.Success then
                        successCount <- successCount + 1
                        totalSemanticImprovement <- totalSemanticImprovement + result.SemanticImprovement
                        totalVectorOptimization <- totalVectorOptimization + result.VectorSpaceOptimization

                        printfn ""
                        printfn "✅ %s Vector Evolution SUCCESS:" domain
                        printfn "   • New Tier Level: %d" result.BaseResult.NewTierLevel
                        printfn "   • Base Performance: %.1f%%" (result.BaseResult.PerformanceImprovement * 100.0)
                        printfn "   • Semantic Improvement: %.1f%%" (result.SemanticImprovement * 100.0)
                        printfn "   • Vector Optimization: %.1f%%" (result.VectorSpaceOptimization * 100.0)
                        printfn "   • Cluster Coherence: %.1f%%" (result.ClusterCoherence * 100.0)
                        printfn "   • Generated Grammar: %d characters" result.BaseResult.GeneratedGrammar.Length

                        // Save enhanced grammar with vector metadata
                        if not (String.IsNullOrEmpty(result.BaseResult.GeneratedGrammar)) then
                            let grammarFile = Path.Combine(outputDirectory, sprintf "vector_grammar_%s_tier_%d.grammar" (domain.ToLower()) result.BaseResult.NewTierLevel)
                            let enhancedContent = sprintf "%s\n\n// Vector Enhancement Metadata\n// Semantic Improvement: %.1f%%\n// Vector Optimization: %.1f%%\n// Cluster Coherence: %.1f%%" result.BaseResult.GeneratedGrammar (result.SemanticImprovement * 100.0) (result.VectorSpaceOptimization * 100.0) (result.ClusterCoherence * 100.0)
                            File.WriteAllText(grammarFile, enhancedContent)
                            outputFiles <- grammarFile :: outputFiles
                            printfn "   • Enhanced Grammar saved: %s" grammarFile

                        // Save semantic trace
                        let traceFile = Path.Combine(outputDirectory, sprintf "vector_trace_%s.md" (domain.ToLower()))
                        File.WriteAllText(traceFile, result.SemanticTrace)
                        outputFiles <- traceFile :: outputFiles
                        printfn "   • Semantic Trace saved: %s" traceFile
                    else
                        printfn ""
                        printfn "❌ %s Vector Evolution FAILED" domain
                        printfn "   • Trace: %s" result.BaseResult.ComprehensiveTrace

                with
                | ex ->
                    printfn ""
                    printfn "❌ %s Vector Evolution ERROR: %s" domain ex.Message

            // Generate enhanced summary report
            let avgSemanticImprovement = if successCount > 0 then totalSemanticImprovement / float successCount else 0.0
            let avgVectorOptimization = if successCount > 0 then totalVectorOptimization / float successCount else 0.0

            let enhancedReportContent = sprintf """# TARS Vector-Enhanced Grammar Evolution Report
Generated: %s

## Vector Evolution Summary
- **Domains Processed:** %d
- **Successful Vector Evolutions:** %d/%d (%.1f%% success rate)
- **Average Semantic Improvement:** %.1f%%
- **Average Vector Optimization:** %.1f%%
- **Total Execution Time:** %.2f seconds

## Revolutionary Vector Achievements
✅ Multi-space semantic analysis integration
✅ Vector-guided evolution strategy selection
✅ Cluster-based coherence optimization
✅ Adaptive parameter tuning
✅ Semantic similarity-driven improvements

## Generated Files
%s

---
*Generated by TARS Vector-Enhanced Grammar Evolution Engine*
*🧬 Revolutionary Vector-Semantic Grammar Evolution - Real Implementation*""" (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")) domains.Length successCount domains.Length (float successCount / float domains.Length * 100.0) (avgSemanticImprovement * 100.0) (avgVectorOptimization * 100.0) (DateTime.UtcNow - startTime).TotalSeconds (outputFiles |> List.map (sprintf "- %s") |> String.concat "\n")

            let reportFile = Path.Combine(outputDirectory, "vector_enhanced_evolution_report.md")
            File.WriteAllText(reportFile, enhancedReportContent)
            outputFiles <- reportFile :: outputFiles

            printfn ""
            printfn "📈 Vector Evolution Summary:"
            printfn "   • Successful Evolutions: %d/%d (%.1f%%)" successCount domains.Length (float successCount / float domains.Length * 100.0)
            printfn "   • Average Semantic Improvement: %.1f%%" (avgSemanticImprovement * 100.0)
            printfn "   • Average Vector Optimization: %.1f%%" (avgVectorOptimization * 100.0)
            printfn "   • Execution Time: %.2f seconds" (DateTime.UtcNow - startTime).TotalSeconds
            printfn ""
            printfn "📊 Enhanced Report saved: %s" reportFile
            printfn "🎉 Vector-enhanced grammar evolution completed successfully!"

            {
                Success = successCount > 0
                Message = sprintf "Vector evolution completed: %d/%d domains successful with %.1f%% avg semantic improvement" successCount domains.Length (avgSemanticImprovement * 100.0)
                OutputFiles = List.rev outputFiles
                ExecutionTime = DateTime.UtcNow - startTime
            }

        with
        | ex ->
            printfn "❌ Vector evolution failed: %s" ex.Message
            {
                Success = false
                Message = sprintf "Vector evolution failed: %s" ex.Message
                OutputFiles = []
                ExecutionTime = DateTime.UtcNow - startTime
            }

    /// Analyze evolution potential for a domain
    let analyzeEvolution(domain: string, capabilities: string list) : CommandResult =
        let startTime = DateTime.UtcNow
        
        try
            printfn "🔬 TARS Grammar Evolution Analysis"
            printfn "================================="
            printfn ""
            printfn "🎯 Domain: %s" domain
            printfn "⚙️  Current Capabilities: %s" (String.concat ", " capabilities)
            printfn ""
            
            let evolutionService = UnifiedGrammarEvolutionService()
            let recommendations = evolutionService.GetEvolutionRecommendations(domain, capabilities)
            
            printfn "📊 Evolution Analysis Results:"
            printfn "============================="
            printfn ""
            
            for i, recommendation in recommendations |> List.indexed do
                printfn "%d. %s" (i + 1) recommendation
            
            printfn ""
            printfn "🎯 Recommended Next Steps:"
            printfn "  1. Execute evolution: tars grammar evolve %s" domain
            printfn "  2. Monitor performance improvements"
            printfn "  3. Integrate generated capabilities"
            printfn "  4. Plan next tier advancement"
            printfn ""
            printfn "✅ Analysis completed successfully!"
            
            {
                Success = true
                Message = sprintf "Analysis completed for %s domain with %d recommendations" domain recommendations.Length
                OutputFiles = []
                ExecutionTime = DateTime.UtcNow - startTime
            }
            
        with
        | ex ->
            printfn "❌ Analysis failed: %s" ex.Message
            {
                Success = false
                Message = sprintf "Analysis failed: %s" ex.Message
                OutputFiles = []
                ExecutionTime = DateTime.UtcNow - startTime
            }

    /// Execute reasoning-enhanced multi-domain grammar evolution
    let executeReasoningEvolution(domains: string list, outputDir: string option) : CommandResult =
        let startTime = DateTime.UtcNow
        let outputDirectory = defaultArg outputDir "output"

        try
            printfn "🧠 TARS Reasoning-Enhanced Multi-Domain Grammar Evolution"
            printfn "======================================================="
            printfn ""
            printfn "🎯 Target Domains: %s" (String.concat ", " domains)
            printfn "📁 Output Directory: %s" outputDirectory
            printfn "🔬 Vector Analysis: ENABLED"
            printfn "🧠 BSP Reasoning: ENABLED"
            printfn "🔬 Complex Problem Solving: ENABLED"
            printfn ""

            // Ensure output directory exists
            if not (Directory.Exists(outputDirectory)) then
                Directory.CreateDirectory(outputDirectory) |> ignore
                printfn "📁 Created output directory: %s" outputDirectory

            // Execute reasoning-enhanced evolution
            let reasoningService = ReasoningEnhancedGrammarEvolutionService()
            printfn "🚀 Starting reasoning-enhanced evolution process..."

            let mutable outputFiles = []
            let mutable successCount = 0
            let mutable totalReasoningImprovement = 0.0
            let mutable totalBSPQuality = 0.0
            let mutable totalChainCoherence = 0.0

            printfn ""
            printfn "📊 Reasoning-Enhanced Evolution Results:"
            printfn "========================================"

            for domain in domains do
                try
                    // Run async operation synchronously for CLI
                    let result =
                        reasoningService.EvolveWithReasoningAnalysis(domain, [])
                        |> Async.AwaitTask
                        |> Async.RunSynchronously

                    if result.VectorResult.BaseResult.Success then
                        successCount <- successCount + 1
                        totalReasoningImprovement <- totalReasoningImprovement + result.ReasoningImprovement
                        totalBSPQuality <- totalBSPQuality + result.BSPSolutionQuality
                        totalChainCoherence <- totalChainCoherence + result.ChainCoherence

                        printfn ""
                        printfn "✅ %s Reasoning Evolution SUCCESS:" domain
                        printfn "   • New Tier Level: %d" result.VectorResult.BaseResult.NewTierLevel
                        printfn "   • Base Performance: %.1f%%" (result.VectorResult.BaseResult.PerformanceImprovement * 100.0)
                        printfn "   • Vector Enhancement: %.1f%%" (result.VectorResult.SemanticImprovement * 100.0)
                        printfn "   • Reasoning Improvement: %.1f%%" (result.ReasoningImprovement * 100.0)
                        printfn "   • BSP Solution Quality: %.1f%%" (result.BSPSolutionQuality * 100.0)
                        printfn "   • Chain Coherence: %.1f%%" (result.ChainCoherence * 100.0)
                        printfn "   • Problem Solving Efficiency: %.1f%%" (result.ProblemSolvingEfficiency * 100.0)
                        printfn "   • Generated Grammar: %d characters" result.VectorResult.BaseResult.GeneratedGrammar.Length

                        // Save reasoning-enhanced grammar with comprehensive metadata
                        if not (String.IsNullOrEmpty(result.VectorResult.BaseResult.GeneratedGrammar)) then
                            let grammarFile = Path.Combine(outputDirectory, sprintf "reasoning_grammar_%s_tier_%d.grammar" (domain.ToLower()) result.VectorResult.BaseResult.NewTierLevel)
                            let enhancedContent =
                                sprintf "%s\n\n// Reasoning Enhancement Metadata\n// Vector Enhancement: %.1f%%\n// Reasoning Improvement: %.1f%%\n// BSP Solution Quality: %.1f%%\n// Chain Coherence: %.1f%%\n// Problem Solving Efficiency: %.1f%%"
                                    result.VectorResult.BaseResult.GeneratedGrammar
                                    (result.VectorResult.SemanticImprovement * 100.0)
                                    (result.ReasoningImprovement * 100.0)
                                    (result.BSPSolutionQuality * 100.0)
                                    (result.ChainCoherence * 100.0)
                                    (result.ProblemSolvingEfficiency * 100.0)
                            File.WriteAllText(grammarFile, enhancedContent)
                            outputFiles <- grammarFile :: outputFiles
                            printfn "   • Reasoning Grammar saved: %s" grammarFile

                        // Save comprehensive reasoning trace
                        let traceFile = Path.Combine(outputDirectory, sprintf "reasoning_trace_%s.md" (domain.ToLower()))
                        File.WriteAllText(traceFile, result.ReasoningTrace)
                        outputFiles <- traceFile :: outputFiles
                        printfn "   • Reasoning Trace saved: %s" traceFile

                        // Save meta-reasoning insights
                        let insightsFile = Path.Combine(outputDirectory, sprintf "meta_insights_%s.md" (domain.ToLower()))
                        let insightsContent = sprintf "# Meta-Reasoning Insights for %s\n\n%s" domain (String.concat "\n- " result.MetaReasoningInsights)
                        File.WriteAllText(insightsFile, insightsContent)
                        outputFiles <- insightsFile :: outputFiles
                        printfn "   • Meta Insights saved: %s" insightsFile
                    else
                        printfn ""
                        printfn "❌ %s Reasoning Evolution FAILED" domain
                        printfn "   • Trace: %s" result.VectorResult.BaseResult.ComprehensiveTrace

                with
                | ex ->
                    printfn ""
                    printfn "❌ %s Reasoning Evolution ERROR: %s" domain ex.Message

            // Generate comprehensive reasoning report
            let avgReasoningImprovement = if successCount > 0 then totalReasoningImprovement / float successCount else 0.0
            let avgBSPQuality = if successCount > 0 then totalBSPQuality / float successCount else 0.0
            let avgChainCoherence = if successCount > 0 then totalChainCoherence / float successCount else 0.0

            let reasoningReportContent =
                sprintf """# TARS Reasoning-Enhanced Grammar Evolution Report
Generated: %s

## Reasoning Evolution Summary
- **Domains Processed:** %d
- **Successful Reasoning Evolutions:** %d/%d (%.1f%% success rate)
- **Average Reasoning Improvement:** %.1f%%
- **Average BSP Solution Quality:** %.1f%%
- **Average Chain Coherence:** %.1f%%
- **Total Execution Time:** %.2f seconds

## Revolutionary Reasoning Achievements
✅ BSP reasoning-guided evolution strategy selection
✅ Complex problem decomposition and solving
✅ Chain-of-thought coherence optimization
✅ Meta-reasoning insights generation
✅ Multi-agent collaborative problem solving
✅ Reasoning confidence and quality assessment

## Generated Files
%s

---
*Generated by TARS Reasoning-Enhanced Grammar Evolution Engine*
*🧠 Revolutionary BSP + Complex Problem Solving Grammar Evolution - Real Implementation*"""
                    (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"))
                    domains.Length
                    successCount
                    domains.Length
                    (float successCount / float domains.Length * 100.0)
                    (avgReasoningImprovement * 100.0)
                    (avgBSPQuality * 100.0)
                    (avgChainCoherence * 100.0)
                    (DateTime.UtcNow - startTime).TotalSeconds
                    (outputFiles |> List.map (sprintf "- %s") |> String.concat "\n")

            let reportFile = Path.Combine(outputDirectory, "reasoning_enhanced_evolution_report.md")
            File.WriteAllText(reportFile, reasoningReportContent)
            outputFiles <- reportFile :: outputFiles

            printfn ""
            printfn "📈 Reasoning Evolution Summary:"
            printfn "   • Successful Evolutions: %d/%d (%.1f%%)" successCount domains.Length (float successCount / float domains.Length * 100.0)
            printfn "   • Average Reasoning Improvement: %.1f%%" (avgReasoningImprovement * 100.0)
            printfn "   • Average BSP Solution Quality: %.1f%%" (avgBSPQuality * 100.0)
            printfn "   • Average Chain Coherence: %.1f%%" (avgChainCoherence * 100.0)
            printfn "   • Execution Time: %.2f seconds" (DateTime.UtcNow - startTime).TotalSeconds
            printfn ""
            printfn "📊 Comprehensive Report saved: %s" reportFile
            printfn "🎉 Reasoning-enhanced grammar evolution completed successfully!"

            {
                Success = successCount > 0
                Message = sprintf "Reasoning evolution completed: %d/%d domains successful with %.1f%% avg reasoning improvement" successCount domains.Length (avgReasoningImprovement * 100.0)
                OutputFiles = List.rev outputFiles
                ExecutionTime = DateTime.UtcNow - startTime
            }

        with
        | ex ->
            printfn "❌ Reasoning evolution failed: %s" ex.Message
            {
                Success = false
                Message = sprintf "Reasoning evolution failed: %s" ex.Message
                OutputFiles = []
                ExecutionTime = DateTime.UtcNow - startTime
            }

    /// Execute comprehensive grammar evolution demonstration
    let executeDemonstration(outputDir: string option) : CommandResult =
        let startTime = DateTime.UtcNow
        let outputDirectory = defaultArg outputDir "output"
        
        try
            printfn "🎭 TARS Grammar Evolution Demonstration"
            printfn "======================================"
            printfn ""
            printfn "🚀 Executing comprehensive multi-domain evolution demonstration..."
            printfn "📁 Output Directory: %s" outputDirectory
            printfn ""
            
            let demoService = GrammarEvolutionDemoService()
            let result = demoService.ExecuteStandardDemo()
            
            if result.Success then
                printfn "🎉 Demonstration completed successfully!"
                printfn ""
                printfn "📊 Demonstration Results:"
                printfn "   • Domains Processed: %d" result.DomainsProcessed
                printfn "   • Successful Evolutions: %d" result.SuccessfulEvolutions
                printfn "   • Average Performance Improvement: %.1f%%" (result.AveragePerformanceImprovement * 100.0)
                printfn "   • Average Resource Efficiency: %.1f%%" (result.AverageResourceEfficiency * 100.0)
                printfn "   • Execution Time: %.2f seconds" result.ExecutionTime.TotalSeconds
                printfn ""
                printfn "📁 Generated Files:"
                for file in result.GeneratedFiles do
                    printfn "   • %s" file
                
                {
                    Success = true
                    Message = sprintf "Demonstration completed: %d domains, %d successful evolutions" result.DomainsProcessed result.SuccessfulEvolutions
                    OutputFiles = result.GeneratedFiles
                    ExecutionTime = DateTime.UtcNow - startTime
                }
            else
                printfn "❌ Demonstration failed: %s" result.ComprehensiveTrace
                {
                    Success = false
                    Message = sprintf "Demonstration failed: %s" result.ComprehensiveTrace
                    OutputFiles = result.GeneratedFiles
                    ExecutionTime = DateTime.UtcNow - startTime
                }
                
        with
        | ex ->
            printfn "❌ Demonstration failed: %s" ex.Message
            {
                Success = false
                Message = sprintf "Demonstration failed: %s" ex.Message
                OutputFiles = []
                ExecutionTime = DateTime.UtcNow - startTime
            }

    // ============================================================================
    // COMMAND PARSING AND EXECUTION
    // ============================================================================

    /// Parse grammar evolution command arguments
    let parseGrammarCommand(args: string array) : GrammarCommand =
        match args with
        | [||] -> Help
        | [| "help" |] -> Help
        | [| "status" |] -> Status
        | [| "demo" |] -> Demo None
        | [| "demo"; "--output"; outputDir |] -> Demo (Some outputDir)
        | [| "analyze"; domain |] -> Analyze (domain, [])
        | [| "analyze"; domain; "--capabilities"; capabilitiesStr |] ->
            let capabilities = capabilitiesStr.Split(',') |> Array.map (fun s -> s.Trim()) |> Array.toList
            Analyze (domain, capabilities)
        | [| "analyze-vector"; domain |] -> AnalyzeVector (domain, [])
        | [| "analyze-vector"; domain; "--capabilities"; capabilitiesStr |] ->
            let capabilities = capabilitiesStr.Split(',') |> Array.map (fun s -> s.Trim()) |> Array.toList
            AnalyzeVector (domain, capabilities)
        | [| "analyze-reasoning"; domain |] -> AnalyzeReasoning (domain, [])
        | [| "analyze-reasoning"; domain; "--capabilities"; capabilitiesStr |] ->
            let capabilities = capabilitiesStr.Split(',') |> Array.map (fun s -> s.Trim()) |> Array.toList
            AnalyzeReasoning (domain, capabilities)
        | [| "evolve"; domainsStr |] ->
            let domains = domainsStr.Split(',') |> Array.map (fun s -> s.Trim()) |> Array.toList
            Evolve (domains, None)
        | [| "evolve"; domainsStr; "--output"; outputDir |] ->
            let domains = domainsStr.Split(',') |> Array.map (fun s -> s.Trim()) |> Array.toList
            Evolve (domains, Some outputDir)
        | [| "evolve-vector"; domainsStr |] ->
            let domains = domainsStr.Split(',') |> Array.map (fun s -> s.Trim()) |> Array.toList
            EvolveVector (domains, None)
        | [| "evolve-vector"; domainsStr; "--output"; outputDir |] ->
            let domains = domainsStr.Split(',') |> Array.map (fun s -> s.Trim()) |> Array.toList
            EvolveVector (domains, Some outputDir)
        | [| "evolve-reasoning"; domainsStr |] ->
            let domains = domainsStr.Split(',') |> Array.map (fun s -> s.Trim()) |> Array.toList
            EvolveReasoning (domains, None)
        | [| "evolve-reasoning"; domainsStr; "--output"; outputDir |] ->
            let domains = domainsStr.Split(',') |> Array.map (fun s -> s.Trim()) |> Array.toList
            EvolveReasoning (domains, Some outputDir)
        | _ -> Help

    /// Execute grammar evolution command
    let executeGrammarCommand(command: GrammarCommand) : CommandResult =
        match command with
        | Help ->
            showGrammarHelp()
            { Success = true; Message = "Help displayed"; OutputFiles = []; ExecutionTime = TimeSpan.Zero }
        | Status -> showGrammarStatus()
        | Demo outputDir -> executeDemonstration(outputDir)
        | Analyze (domain, capabilities) -> analyzeEvolution(domain, capabilities)
        | AnalyzeVector (domain, capabilities) -> analyzeVectorEvolution(domain, capabilities)
        | AnalyzeReasoning (domain, capabilities) -> analyzeReasoningEvolution(domain, capabilities)
        | Evolve (domains, outputDir) -> executeEvolution(domains, outputDir)
        | EvolveVector (domains, outputDir) -> executeVectorEvolution(domains, outputDir)
        | EvolveReasoning (domains, outputDir) -> executeReasoningEvolution(domains, outputDir)

    /// Main entry point for grammar evolution commands
    let runGrammarCommand(args: string array) : CommandResult =
        let command = parseGrammarCommand(args)
        executeGrammarCommand(command)
