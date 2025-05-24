namespace TarsEngine.FSharp.Core.Consciousness.Intelligence.Services.IntuitiveReasoning

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Intelligence

/// <summary>
/// Implementation of intuitive decision making methods.
/// </summary>
module IntuitiveDecisionMaking =
    /// <summary>
    /// Evaluates options using pattern recognition.
    /// </summary>
    /// <param name="options">The options.</param>
    /// <param name="patternRecognitionLevel">The pattern recognition level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The option scores.</returns>
    let evaluateOptionsWithPatternRecognition (options: string list) (patternRecognitionLevel: float) (random: Random) =
        options
        |> List.map (fun option ->
            // Base score is random but influenced by pattern recognition level
            let baseScore = 0.3 + (0.4 * random.NextDouble())
            
            // Pattern recognition bonus
            let patternBonus = 
                // Longer options might contain more recognizable patterns
                let lengthFactor = Math.Min(1.0, option.Length / 20.0) * 0.1
                
                // Options with certain keywords might trigger pattern recognition
                let keywordBonus =
                    let keywords = ["pattern"; "similar"; "like"; "recognize"; "familiar"; "seen"; "before"; "known"]
                    let optionLower = option.ToLowerInvariant()
                    keywords
                    |> List.sumBy (fun keyword -> if optionLower.Contains(keyword) then 0.05 else 0.0)
                
                // Combine factors
                (lengthFactor + keywordBonus) * patternRecognitionLevel
            
            // Calculate final score
            let finalScore = Math.Min(1.0, baseScore + patternBonus)
            
            (option, finalScore))
        |> Map.ofList
    
    /// <summary>
    /// Evaluates options using heuristic reasoning.
    /// </summary>
    /// <param name="options">The options.</param>
    /// <param name="heuristicReasoningLevel">The heuristic reasoning level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The option scores.</returns>
    let evaluateOptionsWithHeuristicReasoning (options: string list) (heuristicReasoningLevel: float) (random: Random) =
        options
        |> List.map (fun option ->
            // Base score is random but influenced by heuristic reasoning level
            let baseScore = 0.3 + (0.3 * random.NextDouble())
            
            // Apply various heuristics
            
            // Simplicity heuristic - shorter options are preferred
            let simplicityBonus = (10.0 - Math.Min(10.0, float option.Length / 5.0)) * 0.01
            
            // Familiarity heuristic - options with familiar terms are preferred
            let familiarityBonus =
                let optionLower = option.ToLowerInvariant()
                if optionLower.Contains("familiar") || 
                   optionLower.Contains("known") || 
                   optionLower.Contains("proven") then
                    0.1
                else
                    0.0
            
            // Availability heuristic - options that are easy to think of examples for
            let availabilityBonus =
                let optionLower = option.ToLowerInvariant()
                if optionLower.Contains("example") || 
                   optionLower.Contains("instance") || 
                   optionLower.Contains("case") then
                    0.1
                else
                    0.0
            
            // Apply heuristic reasoning level to bonuses
            let heuristicBonus = (simplicityBonus + familiarityBonus + availabilityBonus) * heuristicReasoningLevel
            
            // Calculate final score
            let finalScore = Math.Min(1.0, baseScore + heuristicBonus)
            
            (option, finalScore))
        |> Map.ofList
    
    /// <summary>
    /// Evaluates options using gut feeling.
    /// </summary>
    /// <param name="options">The options.</param>
    /// <param name="gutFeelingLevel">The gut feeling level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The option scores.</returns>
    let evaluateOptionsWithGutFeeling (options: string list) (gutFeelingLevel: float) (random: Random) =
        options
        |> List.map (fun option ->
            // Base score is mostly random for gut feeling
            let baseScore = 0.3 + (0.4 * random.NextDouble())
            
            // Gut feeling is more random but influenced by gut feeling level
            let gutBonus = (random.NextDouble() - 0.5) * gutFeelingLevel
            
            // Calculate final score
            let finalScore = Math.Min(1.0, Math.Max(0.0, baseScore + gutBonus))
            
            (option, finalScore))
        |> Map.ofList
    
    /// <summary>
    /// Evaluates options intuitively.
    /// </summary>
    /// <param name="options">The options.</param>
    /// <param name="intuitionType">The intuition type.</param>
    /// <param name="patternRecognitionLevel">The pattern recognition level.</param>
    /// <param name="heuristicReasoningLevel">The heuristic reasoning level.</param>
    /// <param name="gutFeelingLevel">The gut feeling level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The option scores.</returns>
    let evaluateOptionsIntuitively (options: string list) (intuitionType: IntuitionType option) 
                                  (patternRecognitionLevel: float) (heuristicReasoningLevel: float) 
                                  (gutFeelingLevel: float) (random: Random) =
        // Determine which intuition type to use
        let actualType =
            match intuitionType with
            | Some t -> t
            | None ->
                // Choose based on levels
                IntuitionGeneration.chooseIntuitionType patternRecognitionLevel heuristicReasoningLevel gutFeelingLevel random
        
        // Evaluate based on intuition type
        match actualType with
        | IntuitionType.PatternRecognition ->
            evaluateOptionsWithPatternRecognition options patternRecognitionLevel random
        | IntuitionType.HeuristicReasoning ->
            evaluateOptionsWithHeuristicReasoning options heuristicReasoningLevel random
        | IntuitionType.GutFeeling ->
            evaluateOptionsWithGutFeeling options gutFeelingLevel random
        | _ ->
            // For custom or unknown types, use a weighted combination
            let patternScores = evaluateOptionsWithPatternRecognition options patternRecognitionLevel random
            let heuristicScores = evaluateOptionsWithHeuristicReasoning options heuristicReasoningLevel random
            let gutScores = evaluateOptionsWithGutFeeling options gutFeelingLevel random
            
            // Combine scores with weights based on levels
            let totalLevel = patternRecognitionLevel + heuristicReasoningLevel + gutFeelingLevel
            let patternWeight = patternRecognitionLevel / totalLevel
            let heuristicWeight = heuristicReasoningLevel / totalLevel
            let gutWeight = gutFeelingLevel / totalLevel
            
            options
            |> List.map (fun option ->
                let patternScore = Map.find option patternScores
                let heuristicScore = Map.find option heuristicScores
                let gutScore = Map.find option gutScores
                
                let combinedScore = 
                    (patternScore * patternWeight) + 
                    (heuristicScore * heuristicWeight) + 
                    (gutScore * gutWeight)
                
                (option, combinedScore))
            |> Map.ofList
    
    /// <summary>
    /// Makes an intuitive decision.
    /// </summary>
    /// <param name="options">The options.</param>
    /// <param name="intuitionType">The intuition type.</param>
    /// <param name="patternRecognitionLevel">The pattern recognition level.</param>
    /// <param name="heuristicReasoningLevel">The heuristic reasoning level.</param>
    /// <param name="gutFeelingLevel">The gut feeling level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The selected option and the intuition.</returns>
    let makeIntuitiveDecision (options: string list) (intuitionType: IntuitionType option) 
                             (patternRecognitionLevel: float) (heuristicReasoningLevel: float) 
                             (gutFeelingLevel: float) (random: Random) =
        // Evaluate options
        let scores = evaluateOptionsIntuitively options intuitionType patternRecognitionLevel heuristicReasoningLevel gutFeelingLevel random
        
        // Determine which intuition type was used
        let actualType =
            match intuitionType with
            | Some t -> t
            | None ->
                // Choose based on levels
                IntuitionGeneration.chooseIntuitionType patternRecognitionLevel heuristicReasoningLevel gutFeelingLevel random
        
        // Find the option with the highest score
        let selectedOption =
            scores
            |> Map.toList
            |> List.maxBy snd
            |> fst
        
        // Get the score of the selected option
        let selectedScore = Map.find selectedOption scores
        
        // Generate an explanation based on the intuition type
        let explanation =
            match actualType with
            | IntuitionType.PatternRecognition ->
                sprintf "This option matches patterns I've observed in successful outcomes. It has a familiar structure that suggests it will work well."
            | IntuitionType.HeuristicReasoning ->
                sprintf "This option aligns with proven heuristics for this type of decision. It has qualities that are typically associated with good outcomes."
            | IntuitionType.GutFeeling ->
                sprintf "I have a strong intuitive sense that this is the right choice. While I can't fully articulate why, it feels like the best direction."
            | IntuitionType.Custom name ->
                sprintf "Using %s intuition, this option stands out as the most promising choice." name
            | _ ->
                sprintf "This option intuitively feels like the best choice based on a combination of factors."
        
        // Create an intuition for the decision
        let intuition = {
            Id = Guid.NewGuid().ToString()
            Description = sprintf "Intuitively, '%s' seems like the best option." selectedOption
            Type = actualType
            Confidence = selectedScore
            Timestamp = DateTime.UtcNow
            Context = Map.ofList [
                "Options", box options
                "Scores", box (scores |> Map.toList)
            ]
            Tags = ["decision"; "intuitive"; actualType.ToString().ToLowerInvariant()]
            Source = "Intuitive Decision Making"
            VerificationStatus = VerificationStatus.Unverified
            VerificationTimestamp = None
            VerificationNotes = ""
            Accuracy = None
            Impact = 0.7 // Decisions typically have high impact
            Explanation = explanation
            Decision = "Option Selection"
            SelectedOption = selectedOption
            Options = options
        }
        
        (selectedOption, intuition)
