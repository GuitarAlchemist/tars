namespace TarsEngine.FSharp.Core.Consciousness.Intelligence.Services.IntuitiveReasoning

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Intelligence

/// <summary>
/// Implementation of intuition generation methods.
/// </summary>
module IntuitionGeneration =
    /// <summary>
    /// Chooses an intuition type based on current levels.
    /// </summary>
    /// <param name="patternRecognitionLevel">The pattern recognition level.</param>
    /// <param name="heuristicReasoningLevel">The heuristic reasoning level.</param>
    /// <param name="gutFeelingLevel">The gut feeling level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The chosen intuition type.</returns>
    let chooseIntuitionType (patternRecognitionLevel: float) (heuristicReasoningLevel: float) 
                           (gutFeelingLevel: float) (random: Random) =
        // Calculate probabilities based on current levels
        let patternProb = patternRecognitionLevel * 0.4
        let heuristicProb = heuristicReasoningLevel * 0.3
        let gutProb = gutFeelingLevel * 0.3
        
        // Normalize probabilities
        let total = patternProb + heuristicProb + gutProb
        let patternProb = patternProb / total
        let heuristicProb = heuristicProb / total
        
        // Choose type based on probabilities
        let rand = random.NextDouble()
        
        if rand < patternProb then
            IntuitionType.PatternRecognition
        else if rand < patternProb + heuristicProb then
            IntuitionType.HeuristicReasoning
        else
            IntuitionType.GutFeeling
    
    /// <summary>
    /// Generates an intuition using pattern recognition.
    /// </summary>
    /// <param name="patternRecognitionLevel">The pattern recognition level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The generated intuition.</returns>
    let generatePatternRecognitionIntuition (patternRecognitionLevel: float) (random: Random) =
        // Domains for pattern recognition
        let domains = [
            "Data Analysis"; "Market Trends"; "Human Behavior"; "Natural Phenomena"; 
            "System Behavior"; "Learning Patterns"; "Social Dynamics"
        ]
        
        // Choose a random domain
        let domain = domains.[random.Next(domains.Length)]
        
        // Generate a random intuition based on domain
        let (description, tags) =
            match domain with
            | "Data Analysis" ->
                let patterns = ["correlation"; "anomaly"; "trend"; "cluster"; "outlier"]
                let pattern = patterns.[random.Next(patterns.Length)]
                let insights = ["indicates a hidden factor"; "suggests a causal relationship"; 
                               "reveals an underlying structure"; "points to a significant shift"]
                let insight = insights.[random.Next(insights.Length)]
                (sprintf "The %s in the data %s that wasn't immediately obvious" pattern insight,
                 [domain; pattern; "analysis"; "pattern recognition"])
            | "Market Trends" ->
                let indicators = ["price movement"; "volume change"; "sentiment shift"; 
                                 "adoption rate"; "competitive response"]
                let indicator = indicators.[random.Next(indicators.Length)]
                let predictions = ["market correction"; "emerging opportunity"; "shift in consumer behavior"; 
                                  "competitive disruption"; "regulatory impact"]
                let prediction = predictions.[random.Next(predictions.Length)]
                (sprintf "The recent %s suggests an upcoming %s" indicator prediction,
                 [domain; indicator; prediction; "market"; "pattern recognition"])
            | "Human Behavior" ->
                let behaviors = ["communication pattern"; "decision making"; "emotional response"; 
                                "group dynamic"; "learning approach"]
                let behavior = behaviors.[random.Next(behaviors.Length)]
                let insights = ["reveals underlying motivation"; "indicates a shift in perspective"; 
                               "suggests a hidden concern"; "points to an unmet need"]
                let insight = insights.[random.Next(insights.Length)]
                (sprintf "Their %s %s that should be addressed" behavior insight,
                 [domain; behavior; insight; "psychology"; "pattern recognition"])
            | _ ->
                let patterns = ["recurring pattern"; "emerging trend"; "subtle shift"; 
                               "underlying structure"; "hidden connection"]
                let pattern = patterns.[random.Next(patterns.Length)]
                let insights = ["indicates a significant change"; "suggests an important relationship", 
                               "reveals a key insight", "points to a valuable opportunity"]
                let insight = insights.[random.Next(insights.Length)]
                (sprintf "The %s in %s %s worth investigating further" pattern domain insight,
                 [domain; "pattern"; "recognition"; "intuition"])
        
        // Calculate confidence based on pattern recognition level
        let confidence = 0.6 + (0.4 * patternRecognitionLevel * random.NextDouble())
        
        // Create the intuition
        {
            Id = Guid.NewGuid().ToString()
            Description = description
            Type = IntuitionType.PatternRecognition
            Confidence = confidence
            Timestamp = DateTime.UtcNow
            Context = Map.empty
            Tags = tags
            Source = "Pattern Recognition"
            VerificationStatus = VerificationStatus.Unverified
            VerificationTimestamp = None
            VerificationNotes = ""
            Accuracy = None
            Impact = 0.5 + (0.3 * random.NextDouble())
            Explanation = "Based on recognition of patterns in observed data and experiences"
            Decision = ""
            SelectedOption = ""
            Options = []
        }
    
    /// <summary>
    /// Generates an intuition using heuristic reasoning.
    /// </summary>
    /// <param name="heuristicReasoningLevel">The heuristic reasoning level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The generated intuition.</returns>
    let generateHeuristicReasoningIntuition (heuristicReasoningLevel: float) (random: Random) =
        // Heuristics
        let heuristics = [
            "Availability", "judging likelihood based on how easily examples come to mind"
            "Representativeness", "judging probability by similarity to typical cases"
            "Anchoring", "relying too heavily on the first piece of information"
            "Recognition", "choosing what is recognized over what is not"
            "Affect", "making decisions based on emotional response"
            "Simplicity", "preferring simpler explanations over complex ones"
            "Familiarity", "preferring known options over unknown ones"
            "Scarcity", "valuing things that are rare or limited"
            "Social Proof", "following what others are doing"
            "Authority", "deferring to expertise or authority"
        ]
        
        // Choose a random heuristic
        let (heuristic, explanation) = heuristics.[random.Next(heuristics.Length)]
        
        // Domains for application
        let domains = [
            "Decision Making"; "Problem Solving"; "Risk Assessment"; 
            "Investment Strategy"; "Product Selection"; "Career Choice"
        ]
        
        // Choose a random domain
        let domain = domains.[random.Next(domains.Length)]
        
        // Generate a random intuition based on heuristic and domain
        let description = sprintf "Using the %s heuristic, the best approach for %s is to focus on what has worked well in similar situations" heuristic domain
        
        // Tags
        let tags = [domain; heuristic; "heuristic"; "reasoning"; "intuition"]
        
        // Calculate confidence based on heuristic reasoning level
        let confidence = 0.5 + (0.4 * heuristicReasoningLevel * random.NextDouble())
        
        // Create the intuition
        {
            Id = Guid.NewGuid().ToString()
            Description = description
            Type = IntuitionType.HeuristicReasoning
            Confidence = confidence
            Timestamp = DateTime.UtcNow
            Context = Map.ofList [
                "Heuristic", box heuristic
                "Explanation", box explanation
                "Domain", box domain
            ]
            Tags = tags
            Source = "Heuristic Reasoning"
            VerificationStatus = VerificationStatus.Unverified
            VerificationTimestamp = None
            VerificationNotes = ""
            Accuracy = None
            Impact = 0.4 + (0.3 * random.NextDouble())
            Explanation = sprintf "Based on the %s heuristic: %s" heuristic explanation
            Decision = ""
            SelectedOption = ""
            Options = []
        }
    
    /// <summary>
    /// Generates an intuition using gut feeling.
    /// </summary>
    /// <param name="gutFeelingLevel">The gut feeling level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The generated intuition.</returns>
    let generateGutFeelingIntuition (gutFeelingLevel: float) (random: Random) =
        // Gut feeling phrases
        let phrases = [
            "Something doesn't feel right about this"
            "This feels like the right direction"
            "I have a strong sense that we should proceed cautiously"
            "My instinct says this is a valuable opportunity"
            "I have a nagging feeling we're missing something important"
            "This approach intuitively feels more promising"
            "I sense there's a better solution we haven't considered"
            "Something about this situation feels familiar in a concerning way"
            "I have a good feeling about this direction"
            "My gut says we should reconsider our assumptions"
        ]
        
        // Choose a random phrase
        let description = phrases.[random.Next(phrases.Length)]
        
        // Feeling types
        let feelings = [
            "Unease"; "Confidence"; "Caution"; "Excitement"; "Doubt"; 
            "Trust"; "Suspicion"; "Optimism"; "Concern"; "Certainty"
        ]
        
        // Choose a random feeling
        let feeling = feelings.[random.Next(feelings.Length)]
        
        // Tags
        let tags = ["gut feeling"; feeling.ToLowerInvariant(); "intuition"; "instinct"]
        
        // Calculate confidence based on gut feeling level
        let confidence = 0.3 + (0.5 * gutFeelingLevel * random.NextDouble())
        
        // Create the intuition
        {
            Id = Guid.NewGuid().ToString()
            Description = description
            Type = IntuitionType.GutFeeling
            Confidence = confidence
            Timestamp = DateTime.UtcNow
            Context = Map.ofList [
                "Feeling", box feeling
            ]
            Tags = tags
            Source = "Gut Feeling"
            VerificationStatus = VerificationStatus.Unverified
            VerificationTimestamp = None
            VerificationNotes = ""
            Accuracy = None
            Impact = 0.3 + (0.4 * random.NextDouble())
            Explanation = sprintf "Based on a %s feeling that's difficult to articulate but feels significant" feeling.ToLowerInvariant()
            Decision = ""
            SelectedOption = ""
            Options = []
        }
    
    /// <summary>
    /// Generates an intuition by a specific type.
    /// </summary>
    /// <param name="intuitionType">The intuition type.</param>
    /// <param name="patternRecognitionLevel">The pattern recognition level.</param>
    /// <param name="heuristicReasoningLevel">The heuristic reasoning level.</param>
    /// <param name="gutFeelingLevel">The gut feeling level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The generated intuition.</returns>
    let generateIntuitionByType (intuitionType: IntuitionType) (patternRecognitionLevel: float) 
                               (heuristicReasoningLevel: float) (gutFeelingLevel: float) (random: Random) =
        match intuitionType with
        | IntuitionType.PatternRecognition ->
            generatePatternRecognitionIntuition patternRecognitionLevel random
        | IntuitionType.HeuristicReasoning ->
            generateHeuristicReasoningIntuition heuristicReasoningLevel random
        | IntuitionType.GutFeeling ->
            generateGutFeelingIntuition gutFeelingLevel random
        | IntuitionType.Custom name ->
            // For custom types, default to a mix of pattern recognition and gut feeling
            let baseIntuition = 
                if random.NextDouble() < 0.5 then
                    generatePatternRecognitionIntuition patternRecognitionLevel random
                else
                    generateGutFeelingIntuition gutFeelingLevel random
            
            { baseIntuition with 
                Type = IntuitionType.Custom name
                Description = sprintf "[%s] %s" name baseIntuition.Description
                Tags = name :: baseIntuition.Tags
                Source = sprintf "Custom Intuition: %s" name }
        | _ ->
            // Default to gut feeling for unknown types
            generateGutFeelingIntuition gutFeelingLevel random
