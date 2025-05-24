namespace TarsEngine.FSharp.Core.Consciousness.Intelligence.Services.InsightGeneration

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Intelligence

/// <summary>
/// Implementation of connection discovery methods.
/// </summary>
module ConnectionDiscovery =
    /// <summary>
    /// Extracts concepts from ideas.
    /// </summary>
    /// <param name="ideas">The ideas.</param>
    /// <returns>The extracted concepts.</returns>
    let extractConcepts (ideas: string list) =
        // Common stop words to filter out
        let stopWords = Set.ofList [
            "a"; "an"; "the"; "and"; "or"; "but"; "if"; "then"; "else"; "when";
            "at"; "from"; "to"; "in"; "on"; "by"; "for"; "with"; "about"; "against";
            "between"; "into"; "through"; "during"; "before"; "after"; "above"; "below";
            "of"; "is"; "are"; "was"; "were"; "be"; "been"; "being"; "have"; "has";
            "had"; "having"; "do"; "does"; "did"; "doing"; "can"; "could"; "should";
            "would"; "may"; "might"; "must"; "shall"; "will"
        ]
        
        // Extract words from ideas
        let words = 
            ideas
            |> List.collect (fun idea -> 
                idea.ToLowerInvariant().Split([|' '; ','; '.'; ';'; ':'; '?'; '!'; '('; ')'; '['; ']'; '{'; '}'; '"'; '\''; '-'; '_'|], 
                                            StringSplitOptions.RemoveEmptyEntries)
                |> Array.toList)
            |> List.filter (fun word -> 
                word.Length > 3 && not (Set.contains word stopWords))
        
        // Count word frequencies
        let wordCounts = 
            words
            |> List.groupBy id
            |> List.map (fun (word, occurrences) -> (word, List.length occurrences))
            |> List.sortByDescending snd
        
        // Take the top N most frequent words as concepts
        let topN = 10
        wordCounts
        |> List.truncate topN
        |> List.map fst
    
    /// <summary>
    /// Finds connections between concepts.
    /// </summary>
    /// <param name="concepts">The concepts.</param>
    /// <param name="ideas">The ideas.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The concept connections.</returns>
    let findConceptConnections (concepts: string list) (ideas: string list) (random: Random) =
        // For each pair of concepts, check if they co-occur in ideas
        let conceptPairs = 
            [for i in 0 .. concepts.Length - 2 do
                for j in i + 1 .. concepts.Length - 1 do
                    yield (concepts.[i], concepts.[j])]
        
        // Check co-occurrence in ideas
        let coOccurrences = 
            conceptPairs
            |> List.map (fun (concept1, concept2) ->
                let coOccurrenceCount = 
                    ideas
                    |> List.filter (fun idea -> 
                        idea.ToLowerInvariant().Contains(concept1) && 
                        idea.ToLowerInvariant().Contains(concept2))
                    |> List.length
                
                ((concept1, concept2), coOccurrenceCount))
            |> List.filter (fun (_, count) -> count > 0)
            |> List.sortByDescending snd
        
        // Return the concept pairs with co-occurrences
        coOccurrences
        |> List.map fst
    
    /// <summary>
    /// Generates an insight description based on connected concepts.
    /// </summary>
    /// <param name="concept1">The first concept.</param>
    /// <param name="concept2">The second concept.</param>
    /// <param name="connectionDiscoveryLevel">The connection discovery level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The insight description.</returns>
    let generateConnectionInsightDescription (concept1: string) (concept2: string) 
                                            (connectionDiscoveryLevel: float) (random: Random) =
        // Templates for connection insights
        let templates = [
            "The connection between %s and %s reveals a deeper pattern of %s"
            "When %s and %s are considered together, a new understanding of %s emerges"
            "The relationship between %s and %s suggests an underlying principle of %s"
            "Connecting %s with %s provides insight into how %s operates"
            "The intersection of %s and %s illuminates the nature of %s"
            "By linking %s and %s, we can see a fundamental aspect of %s"
            "%s and %s are connected through their shared relationship to %s"
            "The bridge between %s and %s reveals important insights about %s"
        ]
        
        // Possible higher-level concepts that might emerge from connections
        let emergentConcepts = [
            "self-organization"; "emergent complexity"; "adaptive systems";
            "information processing"; "pattern formation"; "structural dynamics";
            "functional integration"; "hierarchical organization"; "network effects";
            "feedback mechanisms"; "evolutionary processes"; "cognitive frameworks";
            "distributed intelligence"; "collective behavior"; "systemic resilience"
        ]
        
        // Choose a random template and emergent concept
        let template = templates.[random.Next(templates.Length)]
        let emergentConcept = emergentConcepts.[random.Next(emergentConcepts.Length)]
        
        // Generate the insight description
        String.Format(template, concept1, concept2, emergentConcept)
    
    /// <summary>
    /// Generates implications based on connected concepts.
    /// </summary>
    /// <param name="concept1">The first concept.</param>
    /// <param name="concept2">The second concept.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The implications.</returns>
    let generateImplications (concept1: string) (concept2: string) (random: Random) =
        // Templates for implications
        let templates = [
            "This connection suggests that %s could be approached through the lens of %s"
            "We might apply principles of %s to better understand %s"
            "This insight could lead to new approaches for integrating %s and %s"
            "The relationship implies that changes in %s might influence %s in unexpected ways"
            "This connection points to potential innovations at the intersection of %s and %s"
            "Understanding this relationship could help resolve challenges in both %s and %s"
            "This insight suggests reconsidering how %s and %s interact in complex systems"
        ]
        
        // Generate 2-4 implications
        let numImplications = 2 + random.Next(3) // 2-4 implications
        
        [1..numImplications]
        |> List.map (fun _ ->
            let template = templates.[random.Next(templates.Length)]
            // Randomly decide which concept goes first
            if random.NextDouble() < 0.5 then
                String.Format(template, concept1, concept2)
            else
                String.Format(template, concept2, concept1))
    
    /// <summary>
    /// Generates a new perspective based on connected concepts.
    /// </summary>
    /// <param name="concept1">The first concept.</param>
    /// <param name="concept2">The second concept.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The new perspective.</returns>
    let generateNewPerspective (concept1: string) (concept2: string) (random: Random) =
        // Templates for new perspectives
        let templates = [
            "Rather than viewing %s and %s as separate domains, we can see them as complementary aspects of a unified system"
            "This connection invites us to reconsider %s through the lens of %s, revealing new dimensions"
            "Instead of treating %s and %s in isolation, we can explore their dynamic interplay"
            "This insight shifts our perspective from seeing %s and %s as distinct to recognizing their interdependence"
            "The connection between %s and %s challenges conventional boundaries between these domains"
        ]
        
        // Choose a random template
        let template = templates.[random.Next(templates.Length)]
        
        // Generate the new perspective
        if random.NextDouble() < 0.5 then
            String.Format(template, concept1, concept2)
        else
            String.Format(template, concept2, concept1)
    
    /// <summary>
    /// Generates a synthesis based on connected concepts.
    /// </summary>
    /// <param name="concept1">The first concept.</param>
    /// <param name="concept2">The second concept.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The synthesis.</returns>
    let generateSynthesis (concept1: string) (concept2: string) (random: Random) =
        // Templates for synthesis
        let templates = [
            "The integration of %s and %s creates a framework that encompasses both while transcending their individual limitations"
            "By synthesizing principles from %s and %s, we can develop a more comprehensive understanding that honors both perspectives"
            "This connection allows us to create a unified approach that leverages the strengths of both %s and %s"
            "The synthesis of %s and %s offers a more nuanced and complete picture than either provides alone"
            "Bringing together %s and %s allows us to address challenges that neither domain could fully resolve independently"
        ]
        
        // Choose a random template
        let template = templates.[random.Next(templates.Length)]
        
        // Generate the synthesis
        if random.NextDouble() < 0.5 then
            String.Format(template, concept1, concept2)
        else
            String.Format(template, concept2, concept1)
    
    /// <summary>
    /// Generates a breakthrough based on connected concepts.
    /// </summary>
    /// <param name="concept1">The first concept.</param>
    /// <param name="concept2">The second concept.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The breakthrough.</returns>
    let generateBreakthrough (concept1: string) (concept2: string) (random: Random) =
        // Templates for breakthroughs
        let templates = [
            "This connection breaks through conventional boundaries between %s and %s, opening new possibilities"
            "Recognizing the relationship between %s and %s overcomes a significant conceptual barrier"
            "This insight represents a breakthrough in understanding how %s and %s interact and influence each other"
            "The connection between %s and %s challenges fundamental assumptions about both domains"
            "This relationship reveals an unexpected bridge between %s and %s that was previously overlooked"
        ]
        
        // Choose a random template
        let template = templates.[random.Next(templates.Length)]
        
        // Generate the breakthrough
        if random.NextDouble() < 0.5 then
            String.Format(template, concept1, concept2)
        else
            String.Format(template, concept2, concept1)
    
    /// <summary>
    /// Generates an insight from connected ideas.
    /// </summary>
    /// <param name="ideas">The ideas.</param>
    /// <param name="connectionDiscoveryLevel">The connection discovery level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The generated insight.</returns>
    let generateConnectionInsight (ideas: string list) (connectionDiscoveryLevel: float) (random: Random) =
        // Extract concepts from ideas
        let concepts = extractConcepts ideas
        
        // If not enough concepts, return None
        if List.length concepts < 2 then
            None
        else
            // Find connections between concepts
            let connections = findConceptConnections concepts ideas random
            
            // If no connections found, return None
            if List.isEmpty connections then
                None
            else
                // Choose a random connection
                let (concept1, concept2) = connections.[random.Next(connections.Length)]
                
                // Generate insight description
                let description = generateConnectionInsightDescription concept1 concept2 connectionDiscoveryLevel random
                
                // Generate implications
                let implications = generateImplications concept1 concept2 random
                
                // Generate new perspective
                let newPerspective = generateNewPerspective concept1 concept2 random
                
                // Generate synthesis
                let synthesis = generateSynthesis concept1 concept2 random
                
                // Generate breakthrough
                let breakthrough = generateBreakthrough concept1 concept2 random
                
                // Calculate significance based on connection discovery level
                let significance = 0.5 + (0.4 * connectionDiscoveryLevel * random.NextDouble())
                
                // Generate tags
                let baseTags = ["connection"; "insight"; "relationship"; concept1; concept2]
                
                // Create the insight
                let insight = {
                    Id = Guid.NewGuid().ToString()
                    Description = description
                    Method = InsightGenerationMethod.ConnectionDiscovery
                    Significance = significance
                    Timestamp = DateTime.UtcNow
                    Context = Map.ofList [
                        "Concept1", box concept1
                        "Concept2", box concept2
                        "Ideas", box ideas
                    ]
                    Tags = baseTags
                    Implications = implications
                    NewPerspective = newPerspective
                    Breakthrough = breakthrough
                    Synthesis = synthesis
                    RelatedThoughtIds = []
                    RelatedQuestionIds = []
                }
                
                Some insight
