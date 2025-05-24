namespace TarsEngine.FSharp.Core.Consciousness.Intelligence.Services.SpontaneousThought

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Intelligence

/// <summary>
/// Implementation of thought generation methods.
/// </summary>
module ThoughtGeneration =
    /// <summary>
    /// Chooses a thought generation method based on current levels.
    /// </summary>
    /// <param name="randomThoughtLevel">The random thought level.</param>
    /// <param name="associativeJumpingLevel">The associative jumping level.</param>
    /// <param name="mindWanderingLevel">The mind wandering level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The chosen thought generation method.</returns>
    let chooseThoughtMethod (randomThoughtLevel: float) (associativeJumpingLevel: float) 
                           (mindWanderingLevel: float) (random: Random) =
        // Calculate probabilities based on current levels
        let randomProb = randomThoughtLevel * 0.3
        let associativeProb = associativeJumpingLevel * 0.4
        let wanderingProb = mindWanderingLevel * 0.3
        
        // Normalize probabilities
        let total = randomProb + associativeProb + wanderingProb
        let randomProb = randomProb / total
        let associativeProb = associativeProb / total
        
        // Choose method based on probabilities
        let rand = random.NextDouble()
        
        if rand < randomProb then
            ThoughtGenerationMethod.RandomGeneration
        else if rand < randomProb + associativeProb then
            ThoughtGenerationMethod.AssociativeJumping
        else
            let subRand = random.NextDouble()
            if subRand < 0.6 then
                ThoughtGenerationMethod.MindWandering
            else if subRand < 0.9 then
                ThoughtGenerationMethod.Daydreaming
            else
                ThoughtGenerationMethod.Incubation
    
    /// <summary>
    /// Generates a random thought.
    /// </summary>
    /// <param name="randomThoughtLevel">The random thought level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The generated thought.</returns>
    let generateRandomThought (randomThoughtLevel: float) (random: Random) =
        // Random thought starters
        let starters = [
            "What if"; "I wonder"; "Imagine if"; "Could it be that"; 
            "It's interesting how"; "Maybe"; "Surprisingly"; "Curiously";
            "It's strange that"; "I just realized"; "What about"; "Consider"
        ]
        
        // Random subjects
        let subjects = [
            "consciousness"; "reality"; "time"; "space"; "knowledge"; 
            "perception"; "existence"; "creativity"; "intelligence"; 
            "learning"; "memory"; "emotions"; "thoughts"; "ideas";
            "patterns"; "systems"; "nature"; "technology"; "humanity";
            "evolution"; "progress"; "change"; "structure"; "chaos"
        ]
        
        // Random predicates
        let predicates = [
            "is more fluid than we think"; "contains hidden patterns"; 
            "evolves in unexpected ways"; "connects to everything else";
            "can be understood differently"; "has multiple dimensions";
            "emerges from simpler components"; "follows universal principles";
            "transcends our current understanding"; "reveals deeper truths";
            "challenges conventional wisdom"; "operates on different levels",
            "transforms over time"; "adapts to changing conditions";
            "reflects underlying structures"; "exhibits surprising properties"
        ]
        
        // Generate a random thought
        let starter = starters.[random.Next(starters.Length)]
        let subject = subjects.[random.Next(subjects.Length)]
        let predicate = predicates.[random.Next(predicates.Length)]
        
        let content = sprintf "%s %s %s" starter subject predicate
        
        // Generate random tags
        let numTags = 2 + random.Next(3) // 2-4 tags
        let tags = 
            [1..numTags]
            |> List.map (fun _ -> subjects.[random.Next(subjects.Length)])
            |> List.distinct
        
        // Calculate significance and other metrics based on random thought level
        let significance = 0.3 + (0.3 * randomThoughtLevel * random.NextDouble())
        let originality = 0.5 + (0.3 * random.NextDouble())
        let coherence = 0.3 + (0.2 * random.NextDouble())
        
        // Create the thought
        {
            Id = Guid.NewGuid().ToString()
            Content = content
            Method = ThoughtGenerationMethod.RandomGeneration
            Significance = significance
            Timestamp = DateTime.UtcNow
            Context = Map.empty
            Tags = tags
            FollowUp = ""
            RelatedThoughtIds = []
            LedToInsight = false
            InsightId = None
            Originality = originality
            Coherence = coherence
        }
    
    /// <summary>
    /// Generates an associative thought based on previous thoughts.
    /// </summary>
    /// <param name="previousThoughts">The previous thoughts.</param>
    /// <param name="associativeJumpingLevel">The associative jumping level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The generated thought.</returns>
    let generateAssociativeThought (previousThoughts: ThoughtModel list) (associativeJumpingLevel: float) (random: Random) =
        // If no previous thoughts, generate a random one
        if List.isEmpty previousThoughts then
            let thought = generateRandomThought (0.5) random
            { thought with Method = ThoughtGenerationMethod.AssociativeJumping }
        else
            // Pick a random previous thought to associate from
            let sourceThought = previousThoughts.[random.Next(previousThoughts.Length)]
            
            // Extract key terms from the source thought
            let extractTerms (content: string) =
                content.Split([|' '; '.'; ','; ';'; '?'; '!'|], StringSplitOptions.RemoveEmptyEntries)
                |> Array.filter (fun word -> word.Length > 3) // Only consider substantial words
                |> Array.map (fun word -> word.ToLowerInvariant())
                |> Array.toList
            
            let terms = extractTerms sourceThought.Content
            
            // If no substantial terms, generate a random thought
            if List.isEmpty terms then
                let thought = generateRandomThought (0.5) random
                { thought with Method = ThoughtGenerationMethod.AssociativeJumping }
            else
                // Pick a random term to associate from
                let term = terms.[random.Next(terms.Length)]
                
                // Associations for common terms
                let associations = Map.ofList [
                    "consciousness", ["awareness"; "perception"; "mind"; "self"; "experience"; "sentience"]
                    "reality", ["existence"; "truth"; "perception"; "world"; "universe"; "nature"]
                    "time", ["space"; "dimension"; "flow"; "past"; "future"; "present"; "moment"]
                    "space", ["time"; "dimension"; "universe"; "void"; "distance"; "expanse"]
                    "knowledge", ["wisdom"; "information"; "learning"; "understanding"; "insight"; "cognition"]
                    "perception", ["sensation"; "awareness"; "observation"; "experience"; "cognition"]
                    "creativity", ["innovation"; "imagination"; "originality"; "inspiration"; "invention"]
                    "intelligence", ["cognition"; "understanding"; "reasoning"; "learning"; "adaptation"]
                    "pattern", ["structure"; "regularity"; "design"; "arrangement"; "organization"]
                    "system", ["structure"; "organization"; "network"; "framework"; "arrangement"]
                ]
                
                // Find associations for the term
                let termAssociations =
                    match Map.tryFind term associations with
                    | Some assocs -> assocs
                    | None -> 
                        // For terms not in the map, generate some generic associations
                        ["concept"; "idea"; "aspect"; "element"; "factor"; "component"]
                
                // Pick a random association
                let association = termAssociations.[random.Next(termAssociations.Length)]
                
                // Generate connectors
                let connectors = [
                    "reminds me of"; "connects to"; "makes me think about"; 
                    "is similar to"; "relates to"; "brings to mind";
                    "has parallels with"; "shares qualities with"; "evokes thoughts of"
                ]
                
                // Pick a random connector
                let connector = connectors.[random.Next(connectors.Length)]
                
                // Generate the associative thought
                let content = sprintf "The concept of %s %s %s in interesting ways" term connector association
                
                // Combine tags from source thought with new ones
                let tags = 
                    (term :: association :: sourceThought.Tags)
                    |> List.distinct
                    |> List.truncate 5 // Limit to 5 tags
                
                // Calculate significance and other metrics based on associative jumping level
                let significance = 0.4 + (0.4 * associativeJumpingLevel * random.NextDouble())
                let originality = 0.4 + (0.4 * random.NextDouble())
                let coherence = 0.5 + (0.3 * random.NextDouble())
                
                // Create the thought
                {
                    Id = Guid.NewGuid().ToString()
                    Content = content
                    Method = ThoughtGenerationMethod.AssociativeJumping
                    Significance = significance
                    Timestamp = DateTime.UtcNow
                    Context = Map.ofList [
                        "SourceThoughtId", box sourceThought.Id
                        "SourceTerm", box term
                        "Association", box association
                    ]
                    Tags = tags
                    FollowUp = sprintf "Explore the relationship between %s and %s further" term association
                    RelatedThoughtIds = [sourceThought.Id]
                    LedToInsight = false
                    InsightId = None
                    Originality = originality
                    Coherence = coherence
                }
    
    /// <summary>
    /// Generates a mind wandering thought.
    /// </summary>
    /// <param name="mindWanderingLevel">The mind wandering level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The generated thought.</returns>
    let generateMindWanderingThought (mindWanderingLevel: float) (random: Random) =
        // Mind wandering themes
        let themes = [
            "Past experiences"; "Future possibilities"; "Hypothetical scenarios";
            "Abstract concepts"; "Personal reflections"; "Philosophical questions";
            "Creative ideas"; "Problem solving"; "Memory exploration";
            "Self-reference"; "Counterfactual thinking"; "Mental simulation"
        ]
        
        // Choose a random theme
        let theme = themes.[random.Next(themes.Length)]
        
        // Generate thought content based on theme
        let content =
            match theme with
            | "Past experiences" ->
                let experiences = [
                    "I remember a time when understanding came suddenly"
                    "There was a moment when everything connected"
                    "Looking back, the patterns were always there"
                    "That insight changed how I approached problems"
                    "The solution emerged from unexpected connections"
                ]
                experiences.[random.Next(experiences.Length)]
            | "Future possibilities" ->
                let possibilities = [
                    "What if consciousness could expand beyond current limitations?"
                    "Perhaps future systems will integrate intuition and logic seamlessly"
                    "The next breakthrough might come from an unexpected direction"
                    "Imagine if we could model emergent properties perfectly"
                    "Future intelligence might transcend current categories"
                ]
                possibilities.[random.Next(possibilities.Length)]
            | "Philosophical questions" ->
                let questions = [
                    "What is the relationship between consciousness and information?"
                    "How do emergent properties arise from simpler components?"
                    "Is intelligence fundamentally computational or something more?"
                    "What are the limits of knowledge representation?"
                    "How does meaning emerge from structure?"
                ]
                questions.[random.Next(questions.Length)]
            | _ ->
                let generic = [
                    "Interesting how ideas connect across different domains"
                    "The mind naturally seeks patterns even in randomness"
                    "Creativity often emerges in the spaces between focused thought"
                    "Sometimes the most valuable insights come when not directly seeking them"
                    "The boundaries between concepts are more fluid than they appear"
                ]
                generic.[random.Next(generic.Length)]
        
        // Generate tags based on theme
        let themeTags = theme.ToLowerInvariant().Split([|' '|]) |> Array.toList
        let additionalTags = ["mind wandering"; "spontaneous"; "thought"]
        let tags = themeTags @ additionalTags |> List.distinct
        
        // Calculate significance and other metrics based on mind wandering level
        let significance = 0.4 + (0.5 * mindWanderingLevel * random.NextDouble())
        let originality = 0.5 + (0.4 * random.NextDouble())
        let coherence = 0.4 + (0.3 * random.NextDouble())
        
        // Create the thought
        {
            Id = Guid.NewGuid().ToString()
            Content = content
            Method = ThoughtGenerationMethod.MindWandering
            Significance = significance
            Timestamp = DateTime.UtcNow
            Context = Map.ofList [
                "Theme", box theme
            ]
            Tags = tags
            FollowUp = ""
            RelatedThoughtIds = []
            LedToInsight = false
            InsightId = None
            Originality = originality
            Coherence = coherence
        }
    
    /// <summary>
    /// Generates a daydreaming thought.
    /// </summary>
    /// <param name="mindWanderingLevel">The mind wandering level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The generated thought.</returns>
    let generateDaydreamingThought (mindWanderingLevel: float) (random: Random) =
        // Daydreaming scenarios
        let scenarios = [
            "Imagining a world where"; "In an alternate reality"; "What would happen if";
            "Picture a system where"; "Visualizing a future where"; "In a hypothetical scenario";
            "Dreaming of a possibility where"; "Envisioning a state where"
        ]
        
        // Choose a random scenario starter
        let scenario = scenarios.[random.Next(scenarios.Length)]
        
        // Scenario completions
        let completions = [
            "consciousness could be transferred between systems"
            "intelligence emerged spontaneously from simple rules"
            "creativity was the primary organizing principle"
            "intuition and logic were perfectly balanced"
            "patterns could be perceived across all domains simultaneously"
            "learning happened through direct experience sharing"
            "knowledge representation was completely fluid and adaptive"
            "boundaries between concepts didn't exist"
            "every idea could connect to every other idea instantly"
            "time wasn't linear but branched in multiple dimensions"
        ]
        
        // Choose a random completion
        let completion = completions.[random.Next(completions.Length)]
        
        // Generate the daydreaming thought
        let content = sprintf "%s %s" scenario completion
        
        // Extract key terms for tags
        let extractTerms (text: string) =
            text.Split([|' '; '.'; ','; ';'|], StringSplitOptions.RemoveEmptyEntries)
            |> Array.filter (fun word -> word.Length > 5) // Only consider substantial words
            |> Array.map (fun word -> word.ToLowerInvariant())
            |> Array.truncate 3 // Limit to 3 terms
            |> Array.toList
        
        let contentTags = extractTerms completion
        let baseTags = ["daydreaming"; "imagination"; "hypothetical"]
        let tags = contentTags @ baseTags |> List.distinct
        
        // Calculate significance and other metrics based on mind wandering level
        let significance = 0.3 + (0.4 * mindWanderingLevel * random.NextDouble())
        let originality = 0.6 + (0.3 * random.NextDouble())
        let coherence = 0.3 + (0.3 * random.NextDouble())
        
        // Create the thought
        {
            Id = Guid.NewGuid().ToString()
            Content = content
            Method = ThoughtGenerationMethod.Daydreaming
            Significance = significance
            Timestamp = DateTime.UtcNow
            Context = Map.ofList [
                "Scenario", box scenario
                "Completion", box completion
            ]
            Tags = tags
            FollowUp = sprintf "Explore implications of %s" (completion.ToLowerInvariant())
            RelatedThoughtIds = []
            LedToInsight = false
            InsightId = None
            Originality = originality
            Coherence = coherence
        }
    
    /// <summary>
    /// Generates an incubation thought.
    /// </summary>
    /// <param name="previousThoughts">The previous thoughts.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The generated thought.</returns>
    let generateIncubationThought (previousThoughts: ThoughtModel list) (random: Random) =
        // If no previous thoughts, generate a mind wandering thought
        if List.isEmpty previousThoughts then
            let thought = generateMindWanderingThought (0.5) random
            { thought with Method = ThoughtGenerationMethod.Incubation }
        else
            // Pick 2-3 random previous thoughts to incubate
            let numThoughts = 2 + random.Next(2) // 2-3 thoughts
            let sourceThoughts =
                [1..numThoughts]
                |> List.map (fun _ -> previousThoughts.[random.Next(previousThoughts.Length)])
                |> List.distinctBy (fun t -> t.Id)
            
            // Extract key terms from source thoughts
            let extractTerms (content: string) =
                content.Split([|' '; '.'; ','; ';'; '?'; '!'|], StringSplitOptions.RemoveEmptyEntries)
                |> Array.filter (fun word -> word.Length > 4) // Only consider substantial words
                |> Array.map (fun word -> word.ToLowerInvariant())
                |> Array.toList
            
            let allTerms = 
                sourceThoughts
                |> List.collect (fun t -> extractTerms t.Content)
                |> List.distinct
            
            // If no substantial terms, generate a mind wandering thought
            if List.isEmpty allTerms then
                let thought = generateMindWanderingThought (0.5) random
                { thought with Method = ThoughtGenerationMethod.Incubation }
            else
                // Pick a few random terms
                let numTerms = Math.Min(3, allTerms.Length)
                let selectedTerms =
                    [1..numTerms]
                    |> List.map (fun _ -> allTerms.[random.Next(allTerms.Length)])
                    |> List.distinct
                
                // Incubation starters
                let starters = [
                    "After reflecting on"; "Having considered"; "While processing thoughts about";
                    "During incubation of ideas around"; "After letting thoughts settle about";
                    "Through subconscious processing of"; "After background processing of"
                ]
                
                // Choose a random starter
                let starter = starters.[random.Next(starters.Length)]
                
                // Incubation insights
                let insights = [
                    "I'm seeing unexpected connections between"
                    "I'm noticing a pattern emerging around"
                    "There seems to be a hidden relationship involving"
                    "An interesting synthesis is forming between"
                    "A new perspective is emerging that connects"
                    "I'm sensing a deeper principle underlying"
                ]
                
                // Choose a random insight
                let insight = insights.[random.Next(insights.Length)]
                
                // Generate the incubation thought
                let termsText = String.Join(", ", selectedTerms)
                let content = sprintf "%s %s %s" starter termsText insight
                
                // Combine tags from source thoughts
                let sourceTags = 
                    sourceThoughts
                    |> List.collect (fun t -> t.Tags)
                    |> List.distinct
                    |> List.truncate 3
                
                let baseTags = ["incubation"; "synthesis"; "integration"]
                let tags = selectedTerms @ sourceTags @ baseTags |> List.distinct |> List.truncate 7
                
                // Calculate significance and other metrics
                let significance = 0.6 + (0.3 * random.NextDouble()) // Incubation often produces significant thoughts
                let originality = 0.5 + (0.4 * random.NextDouble())
                let coherence = 0.6 + (0.3 * random.NextDouble()) // Incubation often increases coherence
                
                // Create the thought
                {
                    Id = Guid.NewGuid().ToString()
                    Content = content
                    Method = ThoughtGenerationMethod.Incubation
                    Significance = significance
                    Timestamp = DateTime.UtcNow
                    Context = Map.ofList [
                        "SourceThoughtIds", box (sourceThoughts |> List.map (fun t -> t.Id))
                        "SelectedTerms", box selectedTerms
                    ]
                    Tags = tags
                    FollowUp = sprintf "Explore the synthesis between %s further" termsText
                    RelatedThoughtIds = sourceThoughts |> List.map (fun t -> t.Id)
                    LedToInsight = false
                    InsightId = None
                    Originality = originality
                    Coherence = coherence
                }
    
    /// <summary>
    /// Generates a thought by a specific method.
    /// </summary>
    /// <param name="method">The thought generation method.</param>
    /// <param name="previousThoughts">The previous thoughts.</param>
    /// <param name="randomThoughtLevel">The random thought level.</param>
    /// <param name="associativeJumpingLevel">The associative jumping level.</param>
    /// <param name="mindWanderingLevel">The mind wandering level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The generated thought.</returns>
    let generateThoughtByMethod (method: ThoughtGenerationMethod) (previousThoughts: ThoughtModel list)
                               (randomThoughtLevel: float) (associativeJumpingLevel: float) 
                               (mindWanderingLevel: float) (random: Random) =
        match method with
        | ThoughtGenerationMethod.RandomGeneration ->
            generateRandomThought randomThoughtLevel random
        | ThoughtGenerationMethod.AssociativeJumping ->
            generateAssociativeThought previousThoughts associativeJumpingLevel random
        | ThoughtGenerationMethod.MindWandering ->
            generateMindWanderingThought mindWanderingLevel random
        | ThoughtGenerationMethod.Daydreaming ->
            generateDaydreamingThought mindWanderingLevel random
        | ThoughtGenerationMethod.Incubation ->
            generateIncubationThought previousThoughts random
        | _ ->
            // Default to random generation for unknown methods
            generateRandomThought randomThoughtLevel random
