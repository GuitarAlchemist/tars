namespace TarsEngine.FSharp.Core.Context

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Context.Types

/// Configuration for context budgeting
type ContextBudgetConfig = {
    MaxTokens: int
    SalienceWeight: float
    RecencyWeight: float
    IntentWeight: float
    SourceWeight: float
    TokenEstimator: string -> int
}

/// Salience-based context budget manager
type SalienceContextBudget(config: ContextBudgetConfig, logger: ILogger<SalienceContextBudget>) =
    
    /// Calculate intent relevance score
    let calculateIntentRelevance (spanIntent: string option) (targetIntent: Intent) =
        match spanIntent, targetIntent with
        | Some intent, target ->
            let intentStr = target.ToString().ToLower()
            if intent.ToLower().Contains(intentStr) then 1.0
            elif intentStr.Contains(intent.ToLower()) then 0.8
            else 0.3
        | None, _ -> 0.5 // Neutral if no intent specified
    
    /// Calculate recency score
    let calculateRecencyScore (timestamp: DateTime) =
        let ageHours = (DateTime.UtcNow - timestamp).TotalHours
        Math.Exp(-ageHours / 24.0) // Exponential decay over days
    
    /// Calculate source reliability score
    let calculateSourceScore (source: string) =
        match source.ToLower() with
        | s when s.Contains("test") -> 0.9
        | s when s.Contains("doc") -> 0.8
        | s when s.Contains("code") -> 0.95
        | s when s.Contains("trace") -> 0.7
        | s when s.Contains("log") -> 0.6
        | s when s.Contains("consolidated") -> 0.85
        | _ -> 0.75
    
    /// Calculate comprehensive salience score
    let calculateComprehensiveSalience (span: ContextSpan) (intent: Intent) =
        let baseSalience = span.Salience
        let intentRelevance = calculateIntentRelevance span.Intent intent
        let recencyScore = calculateRecencyScore span.Timestamp
        let sourceScore = calculateSourceScore span.Source
        
        let weightedScore = 
            config.SalienceWeight * baseSalience +
            config.IntentWeight * intentRelevance +
            config.RecencyWeight * recencyScore +
            config.SourceWeight * sourceScore
        
        // Normalize to 0-1 range
        Math.Min(1.0, Math.Max(0.0, weightedScore))
    
    /// Score and rank spans by relevance
    let scoreSpans (intent: Intent) (spans: ContextSpan list) =
        spans
        |> List.map (fun span ->
            let score = calculateComprehensiveSalience span intent
            (span, score))
        |> List.sortByDescending snd
        |> List.map fst
    
    /// Enforce token budget using greedy selection
    let enforceTokenBudget (maxTokens: int) (spans: ContextSpan list) =
        let mutable totalTokens = 0
        let mutable selectedSpans = []
        
        for span in spans do
            if totalTokens + span.Tokens <= maxTokens then
                totalTokens <- totalTokens + span.Tokens
                selectedSpans <- span :: selectedSpans
        
        List.rev selectedSpans
    
    interface IContextBudget with
        
        member _.ScoreSpans(intent, spans) =
            logger.LogDebug("Scoring {SpanCount} spans for intent {Intent}", spans.Length, intent)
            
            let scoredSpans = scoreSpans intent spans
            
            logger.LogDebug("Scored spans - top 3 salience scores: {TopScores}", 
                scoredSpans 
                |> List.take (Math.Min(3, scoredSpans.Length))
                |> List.map (fun s -> s.Salience))
            
            scoredSpans
        
        member _.EnforceTokenBudget(maxTokens, spans) =
            logger.LogDebug("Enforcing token budget of {MaxTokens} on {SpanCount} spans", maxTokens, spans.Length)
            
            let selectedSpans = enforceTokenBudget maxTokens spans
            let totalTokens = selectedSpans |> List.sumBy (fun s -> s.Tokens)
            
            logger.LogInformation("Selected {SelectedCount}/{TotalCount} spans using {UsedTokens}/{MaxTokens} tokens", 
                selectedSpans.Length, spans.Length, totalTokens, maxTokens)
            
            selectedSpans
        
        member _.CalculateSalience(span, intent) =
            calculateComprehensiveSalience span intent
