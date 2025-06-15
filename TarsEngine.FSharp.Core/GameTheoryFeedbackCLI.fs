namespace TarsEngine.FSharp.Core

open System
open System.IO
open System.Text.Json
open TarsEngine.FSharp.Core.FeedbackTracker
open TarsEngine.FSharp.Core.ModernGameTheory

/// CLI Commands for Game Theory Enhanced Feedback Analysis
module GameTheoryFeedbackCLI =

    /// Parse enhanced feedback file
    let parseEnhancedFeedback (path: string) : FeedbackGraphEntry =
        let content = File.ReadAllText(path)
        // Extract JSON from .trsx format
        let jsonStart = content.IndexOf("{")
        let jsonEnd = content.LastIndexOf("}")
        let jsonContent = content.Substring(jsonStart, jsonEnd - jsonStart + 1)
        JsonSerializer.Deserialize<FeedbackGraphEntry>(jsonContent)

    /// Print detailed analysis of a single feedback entry
    let printDetailedAnalysis (entry: FeedbackGraphEntry) =
        printfn "🎯 ENHANCED GAME THEORY FEEDBACK ANALYSIS"
        printfn "=========================================="
        printfn ""
        printfn "📎 Agent ID: %s" entry.agent_id
        printfn "📌 Task: %s" entry.task_id
        printfn "🕒 Timestamp: %s" (entry.timestamp.ToString("yyyy-MM-dd HH:mm:ss UTC"))
        printfn "🎲 Game Theory Model: %A" entry.game_theory_model
        printfn "🤝 Coordination Score: %.3f" entry.coordination_score
        printfn "⚙️ Update Policy: %s" entry.regret_update_policy
        printfn ""

        // Confidence Analysis
        printfn "🧠 CONFIDENCE ANALYSIS"
        printfn "======================"
        printfn "Before: %.3f" entry.confidence_shift.before
        printfn "After:  %.3f" entry.confidence_shift.after
        printfn "Delta:  %+.3f" entry.confidence_shift.delta
        printfn "Model Influence: %s" entry.confidence_shift.model_influence
        printfn ""

        // Decision Table
        printfn "📊 DECISION ANALYSIS"
        printfn "===================="
        printfn "┌─────────────────┬──────────┬──────────┬────────┬─────────┬──────────────────────┐"
        printfn "│ Action          │ Estimated│ Actual   │ Regret │ Cog.Lvl │ Context              │"
        printfn "├─────────────────┼──────────┼──────────┼────────┼─────────┼──────────────────────┤"

        for d in entry.decisions do
            let cogLevel = d.cognitive_level |> Option.map string |> Option.defaultValue "N/A"
            printfn "│ %-15s │ %8.3f │ %8.3f │ %6.3f │ %7s │ %-20s │" d.action d.estimated_reward d.actual_reward d.regret cogLevel d.context

        printfn "└─────────────────┴──────────┴──────────┴────────┴─────────┴──────────────────────┘"
        printfn ""

        // Game Theory Metrics
        let avgRegret = entry.decisions |> List.averageBy (fun d -> Math.Abs(d.regret))
        printfn "🎲 GAME THEORY METRICS"
        printfn "======================"
        printfn "Average Regret: %.3f" avgRegret
        printfn "Coordination Score: %.3f" entry.coordination_score

        match entry.convergence_metrics with
        | Some metrics ->
            let convergenceStatus = if metrics.IsConverged then "✅ YES" else "❌ NO"
            printfn "Convergence: %s" convergenceStatus
            printfn "Convergence Rate: %.3f" metrics.ConvergenceRate
            printfn "Stability Score: %.3f" metrics.StabilityScore
            printfn "Equilibrium Type: %s" metrics.EquilibriumType
        | None ->
            printfn "Convergence: Not analyzed"

        printfn ""

        // Recommendations
        printfn "💡 RECOMMENDATIONS"
        printfn "=================="

        match entry.game_theory_model with
        | QuantalResponseEquilibrium temp when avgRegret > 0.2 ->
            printfn "⚠️ High regret with QRE - consider adjusting temperature (current: %.2f)" temp
        | NoRegretLearning decay when entry.coordination_score < 0.5 ->
            printfn "⚠️ Low coordination with No-Regret Learning - consider Correlated Equilibrium"
        | CognitiveHierarchy level when avgRegret < 0.1 ->
            printfn "✅ Excellent performance with Cognitive Hierarchy - consider advancing to level %d" (level + 1)
        | _ when entry.coordination_score > 0.8 ->
            printfn "✅ Excellent coordination - current model is working well"
        | _ when avgRegret > 0.3 ->
            printfn "⚠️ High regret detected - recommend switching to No-Regret Learning"
        | _ ->
            printfn "ℹ️ Performance within normal parameters"

    /// CLI command handlers
    let runAnalyzeCommand (path: string) =
        if not (File.Exists(path)) then
            printfn "❌ File not found: %s" path
        else
            try
                let entry = parseEnhancedFeedback path
                printDetailedAnalysis entry
            with ex ->
                printfn "❌ Failed to parse enhanced feedback file: %s" ex.Message

    /// Main CLI router for enhanced feedback commands
    let routeCommand (args: string[]) =
        match args with
        | [| "game-theory-feedback"; "analyze"; path |] ->
            runAnalyzeCommand path
        | [| "game-theory-feedback"; "help" |] ->
            printfn "Enhanced Game Theory Feedback Commands:"
            printfn "  analyze <path>     - Detailed analysis of single feedback file"
            printfn "  help               - Show this help message"
        | _ ->
            printfn "Unknown command. Use 'game-theory-feedback help' for available commands."