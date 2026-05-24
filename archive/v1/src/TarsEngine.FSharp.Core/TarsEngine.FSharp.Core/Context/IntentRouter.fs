namespace TarsEngine.FSharp.Core.Context

open System
open System.Text.RegularExpressions
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Context.Types

/// Intent classification router
type IntentClassificationRouter(logger: ILogger<IntentClassificationRouter>) =
    
    /// Keywords for each intent type
    let intentKeywords = Map.ofList [
        (Plan, [
            "plan"; "strategy"; "roadmap"; "approach"; "design"; "architecture"
            "objective"; "goal"; "milestone"; "phase"; "step"; "workflow"
            "organize"; "structure"; "outline"; "framework"
        ])
        (CodeGen, [
            "generate"; "create"; "implement"; "build"; "develop"; "code"
            "function"; "method"; "class"; "module"; "service"; "component"
            "scaffold"; "template"; "boilerplate"; "write"; "add"
        ])
        (Eval, [
            "test"; "evaluate"; "assess"; "validate"; "verify"; "check"
            "measure"; "benchmark"; "analyze"; "review"; "audit"
            "score"; "rate"; "compare"; "examine"
        ])
        (Refactor, [
            "refactor"; "improve"; "optimize"; "clean"; "restructure"
            "simplify"; "enhance"; "modernize"; "update"; "fix"
            "reorganize"; "streamline"; "polish"
        ])
        (Reasoning, [
            "reason"; "think"; "analyze"; "deduce"; "infer"; "conclude"
            "logic"; "reasoning"; "inference"; "deduction"; "analysis"
            "understand"; "explain"; "interpret"; "solve"
        ])
        (MetascriptExecution, [
            "metascript"; "execute"; "run"; "process"; "interpret"
            "tars"; "trsx"; "flux"; "autonomous"; "agent"
            "workflow"; "automation"; "script"
        ])
        (AutonomousImprovement, [
            "autonomous"; "self"; "improve"; "enhance"; "evolve"
            "learn"; "adapt"; "optimize"; "upgrade"; "advance"
            "superintelligence"; "evolution"; "modification"
        ])
    ]
    
    /// Calculate keyword match score for intent
    let calculateKeywordScore (text: string) (intent: Intent) =
        match intentKeywords.TryFind intent with
        | Some keywords ->
            let lowerText = text.ToLower()
            let matches = 
                keywords 
                |> List.filter (fun keyword -> lowerText.Contains(keyword))
                |> List.length
            float matches / float keywords.Length
        | None -> 0.0
    
    /// Detect TARS-specific patterns
    let detectTarsPatterns (text: string) =
        let patterns = [
            (@"\b(CUDA|GPU)\b.*\b(acceleration|performance)\b", AutonomousImprovement)
            (@"\b184M\+.*searches/second\b", AutonomousImprovement)
            (@"\b\.tars\b|\b\.trsx\b|\bFLUX\b", MetascriptExecution)
            (@"\bAgent\s+OS\b", AutonomousImprovement)
            (@"\bautonomous.*agent\b", AutonomousImprovement)
            (@"\bself.*modification\b", AutonomousImprovement)
            (@"\brepository.*management\b", AutonomousImprovement)
            (@"\bsuperintelligence\b", AutonomousImprovement)
            (@"\bF#.*functional\b", CodeGen)
            (@"\bC#.*infrastructure\b", CodeGen)
            (@"\btest.*coverage\b", Eval)
            (@"\bFS0988\b", Eval)
        ]
        
        patterns
        |> List.choose (fun (pattern, intent) ->
            if Regex.IsMatch(text, pattern, RegexOptions.IgnoreCase) then
                Some (intent, 0.8) // High confidence for pattern matches
            else
                None)
    
    /// Calculate context-based score
    let calculateContextScore (stepName: string) (input: string) (intent: Intent) =
        let combinedText = $"{stepName} {input}"
        
        // Base keyword score
        let keywordScore = calculateKeywordScore combinedText intent
        
        // Pattern-based boost
        let patternMatches = detectTarsPatterns combinedText
        let patternBoost = 
            patternMatches
            |> List.filter (fun (matchedIntent, _) -> matchedIntent = intent)
            |> List.map snd
            |> List.tryHead
            |> Option.defaultValue 0.0
        
        // Step name specific scoring
        let stepNameScore = 
            match stepName.ToLower(), intent with
            | s, Plan when s.Contains("plan") || s.Contains("design") -> 0.5
            | s, CodeGen when s.Contains("generate") || s.Contains("create") -> 0.5
            | s, Eval when s.Contains("test") || s.Contains("validate") -> 0.5
            | s, Refactor when s.Contains("refactor") || s.Contains("improve") -> 0.5
            | s, Reasoning when s.Contains("reason") || s.Contains("analyze") -> 0.5
            | s, MetascriptExecution when s.Contains("metascript") || s.Contains("execute") -> 0.5
            | s, AutonomousImprovement when s.Contains("autonomous") || s.Contains("evolve") -> 0.5
            | _ -> 0.0
        
        // Combine scores with weights
        let finalScore = 
            0.4 * keywordScore + 
            0.4 * patternBoost + 
            0.2 * stepNameScore
        
        Math.Min(1.0, finalScore)
    
    /// Classify intent with confidence
    let classifyWithConfidence (stepName: string) (input: string) =
        let allIntents = [Plan; CodeGen; Eval; Refactor; Reasoning; MetascriptExecution; AutonomousImprovement]
        
        let scores = 
            allIntents
            |> List.map (fun intent -> 
                let score = calculateContextScore stepName input intent
                (intent, score))
            |> List.sortByDescending snd
        
        match scores with
        | (topIntent, topScore) :: _ when topScore > 0.3 ->
            (topIntent, topScore)
        | _ ->
            (Unknown stepName, 0.1)
    
    interface IIntentRouter with
        
        member _.ClassifyIntent(stepName, input) =
            let (intent, confidence) = classifyWithConfidence stepName input
            
            logger.LogDebug("Classified intent for step '{StepName}': {Intent} (confidence: {Confidence:F2})",
                stepName, intent, confidence)
            
            intent
        
        member _.GetConfidence(stepName, input, intent) =
            let score = calculateContextScore stepName input intent
            
            logger.LogDebug("Intent confidence for {Intent}: {Confidence:F2}", intent, score)
            
            score
