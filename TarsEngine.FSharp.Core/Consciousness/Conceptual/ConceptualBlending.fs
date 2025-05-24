namespace TarsEngine.FSharp.Core.Consciousness.Conceptual

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Conceptual

/// <summary>
/// Implements conceptual blending capabilities for creative idea generation.
/// </summary>
type ConceptualBlending(logger: ILogger<ConceptualBlending>) =
    // Random number generator for simulating conceptual blending
    let random = System.Random()
    
    // Conceptual blending level (0.0 to 1.0)
    let mutable conceptualBlendingLevel = 0.5 // Starting with moderate conceptual blending
    
    // Concept models: concept -> model
    let conceptModels = System.Collections.Generic.Dictionary<string, ConceptModel>()
    
    // Blend spaces
    let blendSpaces = System.Collections.Generic.List<BlendSpace>()
    
    /// <summary>
    /// Gets the conceptual blending level (0.0 to 1.0).
    /// </summary>
    member _.ConceptualBlendingLevel = conceptualBlendingLevel
    
    /// <summary>
    /// Initializes the concept models with seed concepts.
    /// </summary>
    member private _.InitializeConceptModels() =
        // Add seed concepts with attributes
        this.AddConceptModel("algorithm", Map.ofList [
            "procedural", 0.9
            "step-by-step", 0.8
            "deterministic", 0.7
            "computational", 0.9
            "problem-solving", 0.8
        ])
        
        this.AddConceptModel("pattern", Map.ofList [
            "repetitive", 0.8
            "structured", 0.9
            "recognizable", 0.7
            "predictable", 0.6
            "template", 0.8
        ])
        
        this.AddConceptModel("abstraction", Map.ofList [
            "conceptual", 0.9
            "simplified", 0.8
            "generalized", 0.9
            "high-level", 0.7
            "essential", 0.6
        ])
        
        this.AddConceptModel("modularity", Map.ofList [
            "component-based", 0.9
            "reusable", 0.8
            "encapsulated", 0.7
            "independent", 0.6
            "composable", 0.8
        ])
        
        this.AddConceptModel("recursion", Map.ofList [
            "self-referential", 0.9
            "nested", 0.8
            "repetitive", 0.7
            "hierarchical", 0.8
            "elegant", 0.6
        ])
        
        logger.LogInformation("Initialized conceptual blending with {ConceptCount} concepts", conceptModels.Count)
    
    /// <summary>
    /// Initializes a new instance of the <see cref="ConceptualBlending"/> class.
    /// </summary>
    do
        logger.LogInformation("Initializing ConceptualBlending")
        this.InitializeConceptModels()
    
    /// <summary>
    /// Updates the conceptual blending level.
    /// </summary>
    /// <returns>True if the update was successful, false otherwise.</returns>
    member _.Update() =
        try
            // Gradually increase conceptual blending level over time (very slowly)
            if conceptualBlendingLevel < 0.95 then
                conceptualBlendingLevel <- conceptualBlendingLevel + 0.0001 * random.NextDouble()
                conceptualBlendingLevel <- Math.Min(conceptualBlendingLevel, 1.0)
            
            true
        with
        | ex ->
            logger.LogError(ex, "Error updating conceptual blending")
            false
    
    /// <summary>
    /// Adds a concept model.
    /// </summary>
    /// <param name="concept">The concept.</param>
    /// <param name="attributes">The attributes.</param>
    /// <returns>The created concept model.</returns>
    member _.AddConceptModel(concept: string, attributes: Map<string, float>) =
        if conceptModels.ContainsKey(concept) then
            conceptModels.[concept]
        else
            // Create concept model
            let model = {
                Name = concept
                Attributes = if Map.isEmpty attributes then Map.empty else attributes
            }
            
            // Add to concept models
            conceptModels.[concept] <- model
            
            logger.LogInformation("Added concept model: {Concept}", concept)
            
            model
    
    /// <summary>
    /// Gets random concepts from the models.
    /// </summary>
    /// <param name="count">The number of concepts to get.</param>
    /// <returns>The random concepts.</returns>
    member _.GetRandomConcepts(count: int) =
        let concepts = ResizeArray<string>()
        
        // Ensure we don't try to get more concepts than exist
        let adjustedCount = Math.Min(count, conceptModels.Count)
        
        // Get random concepts
        let shuffled = 
            conceptModels.Keys 
            |> Seq.sortBy (fun _ -> random.Next())
            |> Seq.take adjustedCount
            |> Seq.toList
        
        concepts.AddRange(shuffled)
        
        concepts |> Seq.toList
    
    /// <summary>
    /// Creates a blend space between concepts.
    /// </summary>
    /// <param name="inputConcepts">The input concepts.</param>
    /// <returns>The created blend space.</returns>
    member _.CreateBlendSpace(inputConcepts: string list) =
        try
            logger.LogInformation("Creating blend space for concepts: {Concepts}", String.Join(", ", inputConcepts))
            
            // Ensure all concepts exist in the model
            for concept in inputConcepts do
                if not (conceptModels.ContainsKey(concept)) then
                    this.AddConceptModel(concept, Map.empty) |> ignore
            
            // Create blend space
            let blendSpace = {
                Id = Guid.NewGuid().ToString()
                InputConcepts = inputConcepts
                ConceptMappings = []
                CreatedAt = DateTime.UtcNow
            }
            
            // Create concept mappings
            let mutable conceptMappings = []
            
            for i = 0 to inputConcepts.Length - 1 do
                for j = i + 1 to inputConcepts.Length - 1 do
                    let mapping = this.CreateConceptMapping(inputConcepts.[i], inputConcepts.[j])
                    conceptMappings <- mapping :: conceptMappings
            
            // Create final blend space with mappings
            let finalBlendSpace = { blendSpace with ConceptMappings = conceptMappings }
            
            // Add to blend spaces
            blendSpaces.Add(finalBlendSpace)
            
            logger.LogInformation("Created blend space: {BlendSpaceId} with {MappingCount} mappings", 
                finalBlendSpace.Id, finalBlendSpace.ConceptMappings.Length)
            
            finalBlendSpace
        with
        | ex ->
            logger.LogError(ex, "Error creating blend space")
            
            // Return empty blend space
            {
                Id = Guid.NewGuid().ToString()
                InputConcepts = inputConcepts
                ConceptMappings = []
                CreatedAt = DateTime.UtcNow
            }
    
    /// <summary>
    /// Creates a concept mapping between two concepts.
    /// </summary>
    /// <param name="concept1">The first concept.</param>
    /// <param name="concept2">The second concept.</param>
    /// <returns>The created concept mapping.</returns>
    member private _.CreateConceptMapping(concept1: string, concept2: string) =
        let mapping = {
            SourceConcept = concept1
            TargetConcept = concept2
            AttributeMappings = []
        }
        
        // Get concept models
        let model1 = conceptModels.[concept1]
        let model2 = conceptModels.[concept2]
        
        // Find common attributes
        let attributeMappings =
            model1.Attributes
            |> Map.toSeq
            |> Seq.choose (fun (attr1, value1) ->
                match Map.tryFind attr1 model2.Attributes with
                | Some value2 ->
                    // Add attribute mapping
                    Some {
                        SourceAttribute = attr1
                        TargetAttribute = attr1
                        Strength = (value1 + value2) / 2.0
                    }
                | None -> None
            )
            |> Seq.toList
        
        { mapping with AttributeMappings = attributeMappings }
    
    /// <summary>
    /// Generates a conceptual blend idea.
    /// </summary>
    /// <returns>The generated creative idea.</returns>
    member _.GenerateConceptualBlendIdea() =
        // Get random seed concepts
        let seedConcepts = this.GetRandomConcepts(3)
        
        // Create blend space
        let blendSpace = this.CreateBlendSpace(seedConcepts)
        
        // Create blend descriptions
        let blendDescriptions = [
            sprintf "A hybrid of %s and %s with aspects of %s" seedConcepts.[0] seedConcepts.[1] seedConcepts.[2]
            sprintf "A new approach that merges %s with %s, influenced by %s" seedConcepts.[0] seedConcepts.[1] seedConcepts.[2]
            sprintf "A %s-%s fusion system with %s characteristics" seedConcepts.[0] seedConcepts.[1] seedConcepts.[2]
            sprintf "A %s-inspired blend of %s and %s" seedConcepts.[2] seedConcepts.[0] seedConcepts.[1]
        ]
        
        // Choose a random description
        let description = blendDescriptions.[random.Next(blendDescriptions.Length)]
        
        // Calculate average association strength
        let mutable totalAssociation = 0.0
        let mutable pairs = 0
        
        for mapping in blendSpace.ConceptMappings do
            let avgStrength =
                if not (List.isEmpty mapping.AttributeMappings) then
                    mapping.AttributeMappings
                    |> List.averageBy (fun am -> am.Strength)
                else
                    0.5
            
            totalAssociation <- totalAssociation + avgStrength
            pairs <- pairs + 1
        
        let avgAssociation = if pairs > 0 then totalAssociation / float pairs else 0.5
        
        // Calculate originality (lower average association = higher originality)
        let originality = 0.6 + (0.4 * (1.0 - avgAssociation)) * conceptualBlendingLevel
        
        // Calculate value (somewhat random but influenced by conceptual blending level)
        let value = 0.4 + (0.6 * random.NextDouble() * conceptualBlendingLevel)
        
        {
            Id = Guid.NewGuid().ToString()
            Description = description
            Originality = originality
            Value = value
            Timestamp = DateTime.UtcNow
            ProcessType = CreativeProcessType.ConceptualBlending
            Concepts = seedConcepts
            Tags = []
            Context = Map.ofList [
                "BlendSpaceId", box blendSpace.Id
            ]
            Problem = ""
            Constraints = []
            ImplementationSteps = []
            PotentialImpact = ""
            Limitations = []
            EvaluationScore = 0.0
            IsImplemented = false
            ImplementationTimestamp = None
            ImplementationOutcome = ""
        }
    
    /// <summary>
    /// Generates a blended solution for a problem.
    /// </summary>
    /// <param name="problem">The problem description.</param>
    /// <param name="constraints">The constraints.</param>
    /// <returns>The blended solution.</returns>
    member _.GenerateBlendedSolution(problem: string, ?constraints: string list) =
        try
            logger.LogInformation("Generating blended solution for problem: {Problem}", problem)
            
            // Extract concepts from problem
            let problemConcepts = this.ExtractConcepts(problem)
            
            // Get additional concepts
            let additionalConcepts = this.GetRandomConcepts(2)
            
            // Create all concepts list
            let allConcepts = List.append problemConcepts additionalConcepts
            
            // Create blend space
            let blendSpace = this.CreateBlendSpace(allConcepts)
            
            // Generate solution description
            let description = 
                sprintf "Solution approach: Create a hybrid solution that blends %s with %s, incorporating elements of %s to address the core challenge."
                    problemConcepts.[0]
                    additionalConcepts.[0]
                    additionalConcepts.[1]
            
            // Apply constraints if provided
            let finalDescription =
                match constraints with
                | Some constraintList when not (List.isEmpty constraintList) ->
                    sprintf "%s While ensuring %s." description (String.Join(" and ", constraintList))
                | _ -> description
            
            // Calculate originality
            let originality = 0.6 + (0.4 * conceptualBlendingLevel)
            
            // Calculate value
            let value = 0.7 + (0.3 * conceptualBlendingLevel)
            
            // Create implementation steps
            let implementationSteps = [
                sprintf "1. Analyze the core elements of %s" problemConcepts.[0]
                sprintf "2. Identify the key principles of %s that can be applied" additionalConcepts.[0]
                sprintf "3. Determine how %s can enhance the solution" additionalConcepts.[1]
                "4. Create a prototype that combines these elements"
                "5. Test and refine the blended solution"
            ]
            
            {
                Id = Guid.NewGuid().ToString()
                Description = finalDescription
                Originality = originality
                Value = value
                Timestamp = DateTime.UtcNow
                ProcessType = CreativeProcessType.ConceptualBlending
                Concepts = allConcepts
                Tags = []
                Context = Map.ofList [
                    "BlendSpaceId", box blendSpace.Id
                ]
                Problem = problem
                Constraints = defaultArg constraints []
                ImplementationSteps = implementationSteps
                PotentialImpact = ""
                Limitations = []
                EvaluationScore = 0.0
                IsImplemented = false
                ImplementationTimestamp = None
                ImplementationOutcome = ""
            }
        with
        | ex ->
            logger.LogError(ex, "Error generating blended solution")
            
            // Return basic idea
            {
                Id = Guid.NewGuid().ToString()
                Description = "A blended approach to solving the problem by combining multiple perspectives"
                Originality = 0.5
                Value = 0.5
                Timestamp = DateTime.UtcNow
                ProcessType = CreativeProcessType.ConceptualBlending
                Concepts = []
                Tags = []
                Context = Map.empty
                Problem = problem
                Constraints = []
                ImplementationSteps = []
                PotentialImpact = ""
                Limitations = []
                EvaluationScore = 0.0
                IsImplemented = false
                ImplementationTimestamp = None
                ImplementationOutcome = ""
            }
    
    /// <summary>
    /// Extracts concepts from text.
    /// </summary>
    /// <param name="text">The text.</param>
    /// <returns>The extracted concepts.</returns>
    member private _.ExtractConcepts(text: string) =
        let concepts = ResizeArray<string>()
        
        // Simple concept extraction by splitting and filtering
        let words = 
            text.Split([|' '; ','; '.'; ':'; ';'; '('; ')'; '['; ']'; '{'; '}'; '\n'; '\r'; '\t'|], 
                StringSplitOptions.RemoveEmptyEntries)
        
        for word in words do
            // Only consider words of reasonable length
            if word.Length >= 4 && word.Length <= 20 then
                // Convert to lowercase
                let concept = word.ToLowerInvariant()
                
                // Add if not already in list
                if not (concepts.Contains(concept)) then
                    concepts.Add(concept)
        
        // If no concepts found, add some default ones
        if concepts.Count = 0 then
            concepts.Add("problem")
            concepts.Add("solution")
        
        concepts |> Seq.toList
    
    /// <summary>
    /// Evaluates the emergent structure of a blend.
    /// </summary>
    /// <param name="blendSpaceId">The blend space ID.</param>
    /// <returns>The evaluation score (0.0 to 1.0).</returns>
    member _.EvaluateEmergentStructure(blendSpaceId: string) =
        try
            // Find blend space
            let blendSpace = 
                blendSpaces 
                |> Seq.tryFind (fun bs -> bs.Id = blendSpaceId)
            
            match blendSpace with
            | None ->
                logger.LogWarning("Blend space not found: {BlendSpaceId}", blendSpaceId)
                0.5
            | Some bs ->
                // Calculate coherence
                let coherence =
                    if not (List.isEmpty bs.ConceptMappings) then
                        let avgMappings = 
                            bs.ConceptMappings 
                            |> List.averageBy (fun m -> float m.AttributeMappings.Length)
                        Math.Min(1.0, avgMappings / 5.0)
                    else
                        0.3
                
                // Calculate integration
                let integration =
                    if bs.InputConcepts.Length > 0 then
                        let maxPossibleMappings = bs.InputConcepts.Length * (bs.InputConcepts.Length - 1) / 2
                        Math.Min(1.0, float bs.ConceptMappings.Length / float maxPossibleMappings)
                    else
                        0.3
                
                // Calculate emergent structure score
                let emergentScore = (coherence * 0.6) + (integration * 0.4)
                
                emergentScore
        with
        | ex ->
            logger.LogError(ex, "Error evaluating emergent structure")
            0.5
