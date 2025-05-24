namespace TarsEngine.FSharp.Core.Consciousness.Intelligence.Services.InsightGeneration

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Intelligence

/// <summary>
/// Implementation of problem restructuring methods.
/// </summary>
module ProblemRestructuring =
    /// <summary>
    /// Extracts key elements from a problem.
    /// </summary>
    /// <param name="problem">The problem.</param>
    /// <returns>The key elements.</returns>
    let extractKeyElements (problem: string) =
        // Split the problem into sentences
        let sentences = 
            problem.Split([|'.'; '?'; '!'|], StringSplitOptions.RemoveEmptyEntries)
            |> Array.map (fun s -> s.Trim())
            |> Array.filter (fun s -> not (String.IsNullOrWhiteSpace(s)))
        
        // Common stop words to filter out
        let stopWords = Set.ofList [
            "a"; "an"; "the"; "and"; "or"; "but"; "if"; "then"; "else"; "when";
            "at"; "from"; "to"; "in"; "on"; "by"; "for"; "with"; "about"; "against";
            "between"; "into"; "through"; "during"; "before"; "after"; "above"; "below";
            "of"; "is"; "are"; "was"; "were"; "be"; "been"; "being"; "have"; "has";
            "had"; "having"; "do"; "does"; "did"; "doing"; "can"; "could"; "should";
            "would"; "may"; "might"; "must"; "shall"; "will"
        ]
        
        // Extract key words from each sentence
        let keyWords = 
            sentences
            |> Array.collect (fun sentence -> 
                sentence.Split([|' '; ','; ';'; ':'; '('; ')'; '['; ']'; '{'; '}'; '"'; '\''; '-'; '_'|], 
                              StringSplitOptions.RemoveEmptyEntries)
                |> Array.map (fun word -> word.ToLowerInvariant())
                |> Array.filter (fun word -> 
                    word.Length > 3 && not (Set.contains word stopWords)))
            |> Array.toList
        
        // Count word frequencies
        let wordCounts = 
            keyWords
            |> List.groupBy id
            |> List.map (fun (word, occurrences) -> (word, List.length occurrences))
            |> List.sortByDescending snd
        
        // Take the top N most frequent words as key elements
        let topN = 5
        wordCounts
        |> List.truncate topN
        |> List.map fst
    
    /// <summary>
    /// Identifies assumptions in a problem.
    /// </summary>
    /// <param name="problem">The problem.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The assumptions.</returns>
    let identifyAssumptions (problem: string) (random: Random) =
        // Common assumption patterns
        let assumptionPatterns = [
            "must"; "should"; "always"; "never"; "only"; "all"; "none"; "every"; 
            "cannot"; "impossible"; "necessary"; "required"; "need to"; "has to";
            "obvious"; "clearly"; "certainly"; "undoubtedly"; "definitely"
        ]
        
        // Check if any assumption patterns are present in the problem
        let problemLower = problem.ToLowerInvariant()
        let foundPatterns = 
            assumptionPatterns
            |> List.filter (fun pattern -> problemLower.Contains(pattern))
        
        // If assumption patterns found, generate specific assumptions
        if not (List.isEmpty foundPatterns) then
            // Generate assumptions based on found patterns
            foundPatterns
            |> List.map (fun pattern ->
                // Find the context around the pattern
                let patternIndex = problemLower.IndexOf(pattern)
                let startIndex = Math.Max(0, patternIndex - 20)
                let endIndex = Math.Min(problemLower.Length, patternIndex + pattern.Length + 30)
                let context = problemLower.Substring(startIndex, endIndex - startIndex)
                
                // Generate an assumption based on the pattern and context
                match pattern with
                | "must" | "should" | "need to" | "has to" | "required" | "necessary" ->
                    sprintf "The assumption that there is only one correct approach or solution"
                | "always" | "never" | "all" | "none" | "every" ->
                    sprintf "The assumption that the situation is absolute or binary without exceptions"
                | "only" ->
                    sprintf "The assumption that options are limited to what's explicitly stated"
                | "cannot" | "impossible" ->
                    sprintf "The assumption that certain approaches are impossible or not viable"
                | "obvious" | "clearly" | "certainly" | "undoubtedly" | "definitely" ->
                    sprintf "The assumption that certain aspects are self-evident and don't need examination"
                | _ ->
                    sprintf "An implicit assumption related to '%s'" pattern)
        else
            // If no specific patterns found, generate generic assumptions
            let genericAssumptions = [
                "The problem must be solved within the current framework"
                "The stated constraints are fixed and cannot be changed"
                "All relevant information is included in the problem statement"
                "The problem has a single correct solution"
                "The problem should be approached using conventional methods"
                "The problem is well-defined and complete as stated"
                "The goals and success criteria are clear and appropriate"
                "The problem exists in isolation from other related issues"
            ]
            
            // Choose 2-3 random generic assumptions
            let numAssumptions = 2 + random.Next(2) // 2-3 assumptions
            [1..numAssumptions]
            |> List.map (fun _ -> genericAssumptions.[random.Next(genericAssumptions.Length)])
            |> List.distinct
    
    /// <summary>
    /// Generates alternative framings for a problem.
    /// </summary>
    /// <param name="problem">The problem.</param>
    /// <param name="keyElements">The key elements.</param>
    /// <param name="assumptions">The assumptions.</param>
    /// <param name="problemRestructuringLevel">The problem restructuring level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The alternative framings.</returns>
    let generateAlternativeFramings (problem: string) (keyElements: string list) (assumptions: string list)
                                   (problemRestructuringLevel: float) (random: Random) =
        // Framing templates
        let framingTemplates = [
            "Instead of seeing this as a problem of %s, we could view it as an opportunity for %s"
            "Rather than focusing on %s, we might reframe this as a question of %s"
            "What if we approached this not as %s, but as %s?"
            "The issue could be reframed from %s to %s"
            "Shifting perspective from %s to %s might reveal new insights"
            "Consider transforming the challenge from %s into %s"
        ]
        
        // Positive framing concepts
        let positiveFramings = [
            "innovation"; "growth"; "learning"; "adaptation"; "evolution";
            "integration"; "transformation"; "emergence"; "collaboration";
            "exploration"; "discovery"; "creativity"; "connection"; "synthesis"
        ]
        
        // Generate 2-4 alternative framings
        let numFramings = 2 + random.Next(3) // 2-4 framings
        
        [1..numFramings]
        |> List.map (fun _ ->
            // Choose a random template
            let template = framingTemplates.[random.Next(framingTemplates.Length)]
            
            // Choose a random key element for the first part
            let keyElement = 
                if List.isEmpty keyElements then
                    "the current approach"
                else
                    keyElements.[random.Next(keyElements.Length)]
            
            // Choose a random positive framing for the second part
            let positiveFraming = positiveFramings.[random.Next(positiveFramings.Length)]
            
            // Generate the alternative framing
            String.Format(template, keyElement, positiveFraming))
    
    /// <summary>
    /// Generates new perspectives for a problem.
    /// </summary>
    /// <param name="problem">The problem.</param>
    /// <param name="keyElements">The key elements.</param>
    /// <param name="problemRestructuringLevel">The problem restructuring level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The new perspectives.</returns>
    let generateNewPerspectives (problem: string) (keyElements: string list) 
                               (problemRestructuringLevel: float) (random: Random) =
        // Perspective templates
        let perspectiveTemplates = [
            "From a %s perspective, this problem reveals opportunities for %s"
            "Looking through a %s lens, we might see this as %s"
            "A %s approach would emphasize %s as the central consideration"
            "Viewing this from a %s standpoint shifts focus to %s"
            "Through a %s framework, the key aspect becomes %s"
        ]
        
        // Perspective types
        let perspectiveTypes = [
            "systems thinking"; "evolutionary"; "ecological"; "network";
            "complexity"; "adaptive"; "emergent"; "holistic"; "integrative";
            "transformative"; "developmental"; "process-oriented"; "relational"
        ]
        
        // Focus areas
        let focusAreas = [
            "interconnections and relationships"; "patterns and dynamics";
            "feedback loops and cycles"; "emergence and self-organization";
            "adaptation and learning"; "resilience and robustness";
            "diversity and variation"; "context and environment";
            "boundaries and interfaces"; "scales and hierarchies"
        ]
        
        // Generate 2-3 new perspectives
        let numPerspectives = 2 + random.Next(2) // 2-3 perspectives
        
        [1..numPerspectives]
        |> List.map (fun _ ->
            // Choose a random template
            let template = perspectiveTemplates.[random.Next(perspectiveTemplates.Length)]
            
            // Choose a random perspective type
            let perspectiveType = perspectiveTypes.[random.Next(perspectiveTypes.Length)]
            
            // Choose a random focus area
            let focusArea = focusAreas.[random.Next(focusAreas.Length)]
            
            // Generate the new perspective
            String.Format(template, perspectiveType, focusArea))
    
    /// <summary>
    /// Generates breakthrough insights for a problem.
    /// </summary>
    /// <param name="problem">The problem.</param>
    /// <param name="keyElements">The key elements.</param>
    /// <param name="assumptions">The assumptions.</param>
    /// <param name="problemRestructuringLevel">The problem restructuring level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The breakthrough insights.</returns>
    let generateBreakthroughInsights (problem: string) (keyElements: string list) (assumptions: string list)
                                    (problemRestructuringLevel: float) (random: Random) =
        // Breakthrough templates
        let breakthroughTemplates = [
            "By challenging the assumption that %s, we can explore %s"
            "What if %s is not the real issue, but rather %s?"
            "The breakthrough comes from recognizing that %s is actually %s"
            "Inverting our thinking from %s to %s opens new possibilities"
            "The key insight is that %s can be transformed through %s"
        ]
        
        // Generate 1-2 breakthrough insights
        let numBreakthroughs = 1 + random.Next(2) // 1-2 breakthroughs
        
        [1..numBreakthroughs]
        |> List.map (fun _ ->
            // Choose a random template
            let template = breakthroughTemplates.[random.Next(breakthroughTemplates.Length)]
            
            // Choose a random assumption or key element for the first part
            let firstPart = 
                if not (List.isEmpty assumptions) && random.NextDouble() < 0.7 then
                    // Use an assumption (70% chance if available)
                    let assumption = assumptions.[random.Next(assumptions.Length)]
                    // Extract the core of the assumption
                    if assumption.StartsWith("The assumption that ") then
                        assumption.Substring("The assumption that ".Length)
                    else if assumption.StartsWith("An implicit assumption ") then
                        assumption.Substring("An implicit assumption ".Length)
                    else
                        assumption
                elif not (List.isEmpty keyElements) then
                    // Use a key element
                    sprintf "%s is the central issue" keyElements.[random.Next(keyElements.Length)]
                else
                    // Fallback
                    "the problem must be solved as currently framed"
            
            // Generate a second part that contrasts with the first
            let secondPart =
                if random.NextDouble() < 0.5 then
                    // Transformative approach
                    let transformations = [
                        "a different level of organization"; "a dynamic process rather than a static state";
                        "an emergent property of the system"; "a symptom rather than the root cause";
                        "an opportunity for fundamental redesign"; "a leverage point for systemic change";
                        "a different scale or timeframe"; "a different pattern of relationships"
                    ]
                    transformations.[random.Next(transformations.Length)]
                else
                    // Alternative focus
                    let alternatives = [
                        "focusing on the relationships between elements";
                        "examining the boundaries and interfaces";
                        "considering the system's adaptive capacity";
                        "exploring the underlying patterns and principles";
                        "investigating the feedback loops and dynamics";
                        "understanding the context and environment";
                        "looking at the process of change itself"
                    ]
                    alternatives.[random.Next(alternatives.Length)]
            
            // Generate the breakthrough insight
            String.Format(template, firstPart, secondPart))
    
    /// <summary>
    /// Generates an insight from a restructured problem.
    /// </summary>
    /// <param name="problem">The problem.</param>
    /// <param name="problemRestructuringLevel">The problem restructuring level.</param>
    /// <param name="random">The random number generator.</param>
    /// <returns>The generated insight.</returns>
    let generateProblemRestructuringInsight (problem: string) (problemRestructuringLevel: float) (random: Random) =
        // Extract key elements from the problem
        let keyElements = extractKeyElements problem
        
        // Identify assumptions in the problem
        let assumptions = identifyAssumptions problem random
        
        // Generate alternative framings
        let alternativeFramings = generateAlternativeFramings problem keyElements assumptions problemRestructuringLevel random
        
        // Generate new perspectives
        let newPerspectives = generateNewPerspectives problem keyElements problemRestructuringLevel random
        
        // Generate breakthrough insights
        let breakthroughInsights = generateBreakthroughInsights problem keyElements assumptions problemRestructuringLevel random
        
        // Choose a primary insight from the breakthrough insights
        let primaryInsight = 
            if List.isEmpty breakthroughInsights then
                "Restructuring this problem reveals new approaches and possibilities"
            else
                breakthroughInsights.[0]
        
        // Generate implications
        let implications = 
            alternativeFramings @ 
            (if List.length newPerspectives > 0 then [newPerspectives.[0]] else []) @
            (if List.length breakthroughInsights > 1 then [breakthroughInsights.[1]] else [])
        
        // Generate new perspective
        let newPerspective = 
            if List.length newPerspectives > 0 then
                newPerspectives.[0]
            else if List.length alternativeFramings > 0 then
                alternativeFramings.[0]
            else
                "This problem can be viewed from multiple perspectives, each revealing different aspects and possibilities"
        
        // Generate synthesis
        let synthesis = 
            let synthesisTemplates = [
                "By restructuring this problem from %s to %s, we can develop more effective approaches"
                "Reframing from %s to %s allows us to see both the problem and potential solutions differently"
                "Shifting perspective from %s to %s integrates multiple viewpoints into a more comprehensive understanding"
                "Transforming our view from %s to %s creates a synthesis that honors complexity while enabling action"
            ]
            
            let template = synthesisTemplates.[random.Next(synthesisTemplates.Length)]
            
            let originalFrame = 
                if List.isEmpty keyElements then
                    "the conventional approach"
                else
                    sprintf "a focus on %s" keyElements.[0]
            
            let newFrame = 
                if List.isEmpty alternativeFramings then
                    "a more integrative perspective"
                else
                    // Extract the second part of an alternative framing
                    let framing = alternativeFramings.[0]
                    let parts = framing.Split([|" to "; " into "; " as "|], StringSplitOptions.None)
                    if parts.Length > 1 then
                        parts.[1]
                    else
                        "a more integrative perspective"
            
            String.Format(template, originalFrame, newFrame)
        
        // Generate breakthrough
        let breakthrough = 
            if List.isEmpty breakthroughInsights then
                "The breakthrough comes from questioning fundamental assumptions and reframing the problem entirely"
            else
                breakthroughInsights.[0]
        
        // Calculate significance based on problem restructuring level
        let significance = 0.5 + (0.4 * problemRestructuringLevel * random.NextDouble())
        
        // Generate tags
        let baseTags = ["problem restructuring"; "reframing"; "insight"]
        let elementTags = keyElements |> List.truncate 3
        let tags = baseTags @ elementTags
        
        // Create the insight
        let insight = {
            Id = Guid.NewGuid().ToString()
            Description = primaryInsight
            Method = InsightGenerationMethod.ProblemRestructuring
            Significance = significance
            Timestamp = DateTime.UtcNow
            Context = Map.ofList [
                "Problem", box problem
                "KeyElements", box keyElements
                "Assumptions", box assumptions
                "AlternativeFramings", box alternativeFramings
            ]
            Tags = tags
            Implications = implications
            NewPerspective = newPerspective
            Breakthrough = breakthrough
            Synthesis = synthesis
            RelatedThoughtIds = []
            RelatedQuestionIds = []
        }
        
        Some insight
