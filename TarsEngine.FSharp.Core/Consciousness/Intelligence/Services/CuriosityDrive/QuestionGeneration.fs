namespace TarsEngine.FSharp.Core.Consciousness.Intelligence.Services.CuriosityDrive

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Intelligence

/// <summary>
/// Implementation of question generation methods.
/// </summary>
module QuestionGeneration =
    /// <summary>
    /// Chooses a question generation method based on current levels.
    /// </summary>
    /// <param name="noveltySeekingLevel">The novelty seeking level.</param>
    /// <param name="questionGenerationLevel">The question generation level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The chosen question generation method.</returns>
    let chooseQuestionMethod (noveltySeekingLevel: float) (questionGenerationLevel: float) (random: Random) =
        // Calculate probabilities based on current levels
        let informationGapProb = questionGenerationLevel * 0.5
        let noveltySeekingProb = noveltySeekingLevel * 0.3
        let explorationProb = 0.2 // Fixed probability for exploration-based
        
        // Normalize probabilities
        let total = informationGapProb + noveltySeekingProb + explorationProb
        let informationGapProb = informationGapProb / total
        let noveltySeekingProb = noveltySeekingProb / total
        
        // Choose method based on probabilities
        let rand = random.NextDouble()
        
        if rand < informationGapProb then
            QuestionGenerationMethod.InformationGap
        else if rand < informationGapProb + noveltySeekingProb then
            QuestionGenerationMethod.NoveltySeeking
        else
            QuestionGenerationMethod.ExplorationBased
    
    /// <summary>
    /// Generates a question using the information gap method.
    /// </summary>
    /// <param name="questionGenerationLevel">The question generation level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The generated question.</returns>
    let generateInformationGapQuestion (questionGenerationLevel: float) (random: Random) =
        // Domains for information gap questions
        let domains = [
            "Knowledge Representation"; "Learning Systems"; "Decision Making"; 
            "Pattern Recognition"; "Consciousness"; "Emergent Behavior";
            "Cognitive Architecture"; "Information Processing"; "Memory Systems";
            "Adaptive Systems"; "Computational Models"; "Neural Networks"
        ]
        
        // Choose a random domain
        let domain = domains.[random.Next(domains.Length)]
        
        // Information gap question templates
        let templates = [
            "What is the relationship between %s and %s?"
            "How does %s influence %s?"
            "What are the key factors that determine %s?"
            "What would happen if %s were fundamentally different?"
            "How can we measure %s more effectively?"
            "What are the limitations of current approaches to %s?"
            "How does %s emerge from simpler components?"
            "What is missing from our understanding of %s?"
            "How can we bridge the gap between %s and %s?"
            "What would a unified theory of %s look like?"
        ]
        
        // Choose a random template
        let templateIndex = random.Next(templates.Length)
        let template = templates.[templateIndex]
        
        // Domain-specific concepts
        let concepts =
            match domain with
            | "Knowledge Representation" -> 
                ["symbolic representation"; "semantic networks"; "ontologies"; 
                 "knowledge graphs"; "logical formalisms"; "frame systems"]
            | "Learning Systems" -> 
                ["supervised learning"; "unsupervised learning"; "reinforcement learning"; 
                 "transfer learning"; "meta-learning"; "continual learning"]
            | "Decision Making" -> 
                ["utility functions"; "risk assessment"; "preference ordering"; 
                 "multi-criteria decisions"; "bounded rationality"; "intuitive decisions"]
            | "Consciousness" -> 
                ["self-awareness"; "qualia"; "attention"; "intentionality"; 
                 "phenomenal experience"; "integrated information"]
            | _ -> 
                ["complexity"; "emergence"; "adaptation"; "optimization"; 
                 "representation"; "computation"; "information"; "structure"]
        
        // Choose random concepts for the template
        let concept1 = concepts.[random.Next(concepts.Length)]
        let concept2 = 
            let mutable c2 = concepts.[random.Next(concepts.Length)]
            while c2 = concept1 do
                c2 <- concepts.[random.Next(concepts.Length)]
            c2
        
        // Generate the question
        let question =
            if template.Contains("%s") && template.IndexOf("%s") <> template.LastIndexOf("%s") then
                // Template with two placeholders
                String.Format(template, concept1, concept2)
            else
                // Template with one placeholder
                String.Format(template, concept1)
        
        // Calculate importance based on question generation level
        let importance = 0.5 + (0.4 * questionGenerationLevel * random.NextDouble())
        
        // Generate tags
        let domainTags = domain.ToLowerInvariant().Split([|' '|]) |> Array.toList
        let conceptTags = 
            [concept1; concept2]
            |> List.collect (fun c -> c.ToLowerInvariant().Split([|' '|]) |> Array.toList)
            |> List.distinct
        
        let baseTags = ["information gap"; "question"; "curiosity"]
        let tags = domainTags @ conceptTags @ baseTags |> List.distinct |> List.truncate 8
        
        // Create the question
        {
            Id = Guid.NewGuid().ToString()
            Question = question
            Domain = domain
            Method = QuestionGenerationMethod.InformationGap
            Importance = importance
            Timestamp = DateTime.UtcNow
            Context = Map.empty
            Tags = tags
            Answer = None
            AnswerTimestamp = None
            AnswerSatisfaction = 0.0
            ExplorationId = None
            FollowUpQuestions = []
            RelatedQuestionIds = []
        }
    
    /// <summary>
    /// Generates a question using the novelty seeking method.
    /// </summary>
    /// <param name="noveltySeekingLevel">The novelty seeking level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The generated question.</returns>
    let generateNoveltySeekingQuestion (noveltySeekingLevel: float) (random: Random) =
        // Domains for novelty seeking questions
        let domains = [
            "Emerging Technologies"; "Theoretical Approaches"; "Novel Architectures"; 
            "Unconventional Computing"; "Speculative Systems"; "Future Directions";
            "Cross-Disciplinary Concepts"; "Paradigm Shifts"; "Experimental Methods"
        ]
        
        // Choose a random domain
        let domain = domains.[random.Next(domains.Length)]
        
        // Novelty seeking question templates
        let templates = [
            "What would happen if we combined %s with %s?"
            "How might %s be reimagined from first principles?"
            "What if %s operated on completely different assumptions?"
            "Could %s be approached from a radically different perspective?"
            "What novel properties might emerge if %s were %s?"
            "How could %s transform our understanding of %s?"
            "What unexplored directions exist for %s?"
            "What would a revolutionary approach to %s look like?"
            "How might %s evolve in unexpected ways?"
            "What if the fundamental assumptions about %s are incorrect?"
        ]
        
        // Choose a random template
        let templateIndex = random.Next(templates.Length)
        let template = templates.[templateIndex]
        
        // Domain-specific concepts and modifiers
        let concepts =
            match domain with
            | "Emerging Technologies" -> 
                ["quantum computing"; "neuromorphic hardware"; "biological computing"; 
                 "molecular information processing"; "self-organizing systems"]
            | "Theoretical Approaches" -> 
                ["non-classical logics"; "topological computation"; "quantum cognition"; 
                 "hyperdimensional computing"; "morphological computation"]
            | "Novel Architectures" -> 
                ["liquid neural networks"; "reservoir computing"; "cortical computing"; 
                 "attention-based architectures"; "sparse distributed representations"]
            | "Cross-Disciplinary Concepts" -> 
                ["cognitive neuroscience"; "quantum physics"; "evolutionary biology"; 
                 "complex systems theory"; "information theory"; "linguistics"]
            | _ -> 
                ["self-organization"; "emergence"; "non-linearity"; "adaptation"; 
                 "evolution"; "complexity"; "intelligence"; "consciousness"]
        
        let modifiers = [
            "fundamentally reimagined"; "radically transformed"; "completely inverted";
            "operating at different scales"; "based on different principles";
            "evolved in a different direction"; "optimized for different goals"
        ]
        
        // Choose random concepts and modifiers for the template
        let concept1 = concepts.[random.Next(concepts.Length)]
        let concept2 = 
            let mutable c2 = concepts.[random.Next(concepts.Length)]
            while c2 = concept1 do
                c2 <- concepts.[random.Next(concepts.Length)]
            c2
        
        let modifier = modifiers.[random.Next(modifiers.Length)]
        
        // Generate the question
        let question =
            if template.Contains("%s") && template.IndexOf("%s") <> template.LastIndexOf("%s") then
                if template.Contains("were %s") then
                    // Template with "were %s" pattern
                    String.Format(template, concept1, modifier)
                else
                    // Template with two different concepts
                    String.Format(template, concept1, concept2)
            else
                // Template with one placeholder
                String.Format(template, concept1)
        
        // Calculate importance based on novelty seeking level
        let importance = 0.6 + (0.3 * noveltySeekingLevel * random.NextDouble())
        
        // Generate tags
        let domainTags = domain.ToLowerInvariant().Split([|' '|]) |> Array.toList
        let conceptTags = 
            [concept1; concept2]
            |> List.collect (fun c -> c.ToLowerInvariant().Split([|' '|]) |> Array.toList)
            |> List.distinct
        
        let baseTags = ["novelty seeking"; "question"; "curiosity"; "innovation"]
        let tags = domainTags @ conceptTags @ baseTags |> List.distinct |> List.truncate 8
        
        // Create the question
        {
            Id = Guid.NewGuid().ToString()
            Question = question
            Domain = domain
            Method = QuestionGenerationMethod.NoveltySeeking
            Importance = importance
            Timestamp = DateTime.UtcNow
            Context = Map.ofList [
                "Concept1", box concept1
                "Concept2", box concept2
            ]
            Tags = tags
            Answer = None
            AnswerTimestamp = None
            AnswerSatisfaction = 0.0
            ExplorationId = None
            FollowUpQuestions = []
            RelatedQuestionIds = []
        }
    
    /// <summary>
    /// Generates a question using the exploration-based method.
    /// </summary>
    /// <param name="explorations">The previous explorations.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The generated question.</returns>
    let generateExplorationBasedQuestion (explorations: CuriosityExploration list) (random: Random) =
        // If no previous explorations, generate a novelty seeking question
        if List.isEmpty explorations then
            let question = generateNoveltySeekingQuestion 0.5 random
            { question with Method = QuestionGenerationMethod.ExplorationBased }
        else
            // Pick a random previous exploration to build upon
            let sourceExploration = explorations.[random.Next(explorations.Length)]
            
            // Exploration-based question templates
            let templates = [
                "Building on our exploration of %s, what would happen if %s?"
                "Following our investigation into %s, how might %s be different?"
                "Based on what we learned about %s, what are the implications for %s?"
                "Our exploration of %s revealed patterns - how do they apply to %s?"
                "Given our findings about %s, what new questions arise about %s?"
                "How can we extend our understanding of %s to explore %s?"
                "What deeper aspects of %s remain unexplored?"
                "What contradictions or tensions exist in our current understanding of %s?"
                "How might our exploration of %s connect to broader questions about %s?"
                "What would a more comprehensive exploration of %s include?"
            ]
            
            // Choose a random template
            let templateIndex = random.Next(templates.Length)
            let template = templates.[templateIndex]
            
            // Extract key terms from the source exploration
            let topic = sourceExploration.Topic
            
            // Generate related concepts based on the exploration topic
            let relatedConcepts =
                if not (List.isEmpty sourceExploration.Insights) then
                    // Use insights if available
                    sourceExploration.Insights
                elif not (String.IsNullOrEmpty sourceExploration.Learning) then
                    // Use learning if available
                    [sourceExploration.Learning]
                else
                    // Default related concepts
                    ["deeper implications"; "underlying mechanisms"; "broader applications"; 
                     "edge cases"; "alternative perspectives"; "theoretical foundations"]
            
            // Choose a random related concept
            let relatedConcept = 
                if List.isEmpty relatedConcepts then
                    "related aspects"
                else
                    relatedConcepts.[random.Next(relatedConcepts.Length)]
            
            // Generate the question
            let question =
                if template.Contains("%s") && template.IndexOf("%s") <> template.LastIndexOf("%s") then
                    // Template with two placeholders
                    String.Format(template, topic, relatedConcept)
                else
                    // Template with one placeholder
                    String.Format(template, topic)
            
            // Calculate importance based on source exploration satisfaction
            let importance = 0.4 + (0.4 * sourceExploration.Satisfaction)
            
            // Generate tags
            let topicTags = topic.ToLowerInvariant().Split([|' '|]) |> Array.toList
            let conceptTags = 
                relatedConcept.ToLowerInvariant().Split([|' '|]) |> Array.toList
            
            let baseTags = ["exploration-based"; "question"; "curiosity"; "follow-up"]
            let tags = topicTags @ conceptTags @ baseTags |> List.distinct |> List.truncate 8
            
            // Create the question
            {
                Id = Guid.NewGuid().ToString()
                Question = question
                Domain = sourceExploration.Topic
                Method = QuestionGenerationMethod.ExplorationBased
                Importance = importance
                Timestamp = DateTime.UtcNow
                Context = Map.ofList [
                    "SourceExplorationId", box sourceExploration.Id
                    "SourceTopic", box topic
                    "RelatedConcept", box relatedConcept
                ]
                Tags = tags
                Answer = None
                AnswerTimestamp = None
                AnswerSatisfaction = 0.0
                ExplorationId = None
                FollowUpQuestions = []
                RelatedQuestionIds = []
            }
    
    /// <summary>
    /// Generates a question by a specific method.
    /// </summary>
    /// <param name="method">The question generation method.</param>
    /// <param name="explorations">The previous explorations.</param>
    /// <param name="noveltySeekingLevel">The novelty seeking level.</param>
    /// <param name="questionGenerationLevel">The question generation level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The generated question.</returns>
    let generateQuestionByMethod (method: QuestionGenerationMethod) (explorations: CuriosityExploration list)
                                (noveltySeekingLevel: float) (questionGenerationLevel: float) (random: Random) =
        match method with
        | QuestionGenerationMethod.InformationGap ->
            generateInformationGapQuestion questionGenerationLevel random
        | QuestionGenerationMethod.NoveltySeeking ->
            generateNoveltySeekingQuestion noveltySeekingLevel random
        | QuestionGenerationMethod.ExplorationBased ->
            generateExplorationBasedQuestion explorations random
        | _ ->
            // Default to information gap for unknown methods
            generateInformationGapQuestion questionGenerationLevel random
