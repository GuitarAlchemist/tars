namespace Tars.Cortex

open System
open Tars.Core

/// <summary>
/// An agent responsible for analyzing the performance and cognition of a completed run.
/// It provides post-action reflection and optimization suggestions.
/// </summary>
module DiagnosticsAgent =
    
    type AnalysisResult =
        { Summary: string
          Warnings: string list
          Recommendations: string list
          Score: float } // 0.0 to 1.0 health score

    let analyze (run: RunDiagnostics) =
        let warnings = ResizeArray<string>()
        let recommendations = ResizeArray<string>()
        
        // 1. Throughput Analysis
        let tokens = int run.Cost.Tokens
        let durationSec = run.Duration.TotalSeconds
        let tps = if durationSec > 0.1 then float tokens / durationSec else 0.0
        
        if tps < 10.0 && tokens > 100 then
            warnings.Add(sprintf "Low throughput detected: %.1f tok/s" tps)
            recommendations.Add("Consider using a faster model (e.g., Flash Attention enabled) or checking network latency.")
        
        if tps > 50.0 then
            recommendations.Add("High throughput achieved. Consider increasing model complexity if reasoning quality is low.")

        // 2. Cost Analysis
        if tokens > 100000 then
            warnings.Add(sprintf "High token consumption: %d tokens" tokens)
            recommendations.Add("Consider summarizing context or using RAG to reduce input size.")

        // 3. Outcome Analysis
        let outcomeScore =
            match run.Outcome.ToLowerInvariant() with
            | "success" -> 1.0
            | "partialsuccess" | "partial" -> 0.6
            | _ -> 0.0

        if outcomeScore < 1.0 then
             warnings.Add($"Run outcome was {run.Outcome}")
             recommendations.Add("Review the reasoning trace for logical gaps or tool failures.")

        // 4. Time Analysis
        if durationSec > 60.0 && run.Outcome = "Success" then
             recommendations.Add("Run was successful but slow (> 60s). Check for redundant steps.")

        // Calculate final score
        // Base score from outcome, penalized by warnings
        let penalty = float warnings.Count * 0.1
        let score = Math.Max(0.0, outcomeScore - penalty)

        { Summary = sprintf "Run completed in %.2fs. Cost: %d tokens. Throughput: %.1f tok/s. Outcome: %s." durationSec tokens tps run.Outcome
          Warnings = warnings |> Seq.toList
          Recommendations = recommendations |> Seq.toList
          Score = score }
