namespace TarsEngine.FSharp.WindowsService.Semantic

open System
open System.Collections.Concurrent
open System.Text.RegularExpressions
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Intent classification result
/// </summary>
type IntentClassification = {
    Intent: string
    Confidence: float
    AlternativeIntents: (string * float) list
    Entities: Map<string, string>
    Keywords: string list
}

/// <summary>
/// Entity extraction result
/// </summary>
type EntityExtractionResult = {
    EntityType: string
    Value: string
    StartPosition: int
    EndPosition: int
    Confidence: float
    Context: string
}

/// <summary>
/// Semantic similarity result
/// </summary>
type SimilarityResult = {
    Score: float
    MatchingTerms: string list
    ConceptualSimilarity: float
    StructuralSimilarity: float
    SemanticDistance: float
}

/// <summary>
/// Task complexity analysis
/// </summary>
type ComplexityAnalysis = {
    OverallComplexity: TaskComplexity
    ComplexityScore: float
    ComplexityFactors: Map<string, float>
    EstimatedDuration: TimeSpan
    RequiredSkillLevel: CapabilityLevel
    Reasoning: string list
}

/// <summary>
/// Domain classification
/// </summary>
type DomainClassification = {
    PrimaryDomain: string
    Confidence: float
    SecondaryDomains: (string * float) list
    TechnicalTerms: string list
    DomainSpecificConcepts: string list
}

/// <summary>
/// Semantic embeddings (simplified representation)
/// </summary>
type SemanticEmbedding = {
    Dimensions: int
    Vector: float array
    Magnitude: float
    NormalizedVector: float array
}

/// <summary>
/// NLP processing pipeline result
/// </summary>
type NLPProcessingResult = {
    OriginalText: string
    CleanedText: string
    Tokens: string list
    Keywords: string list
    Entities: EntityExtractionResult list
    Intent: IntentClassification
    Domain: DomainClassification
    Complexity: ComplexityAnalysis
    Sentiment: float
    Embeddings: SemanticEmbedding option
    ProcessingTime: TimeSpan
}

/// <summary>
/// Semantic analyzer for natural language processing and task analysis
/// </summary>
type SemanticAnalyzer(logger: ILogger<SemanticAnalyzer>) =
    
    let stopWords = Set.ofList [
        "a"; "an"; "and"; "are"; "as"; "at"; "be"; "by"; "for"; "from"; "has"; "he"; "in"; "is"; "it"; 
        "its"; "of"; "on"; "that"; "the"; "to"; "was"; "will"; "with"; "would"; "could"; "should";
        "this"; "these"; "those"; "they"; "them"; "their"; "there"; "where"; "when"; "what"; "how"
    ]
    
    let technicalTerms = Map.ofList [
        ("api", "WebDevelopment")
        ("rest", "WebDevelopment")
        ("graphql", "WebDevelopment")
        ("database", "DataManagement")
        ("sql", "DataManagement")
        ("mongodb", "DataManagement")
        ("redis", "DataManagement")
        ("docker", "Infrastructure")
        ("kubernetes", "Infrastructure")
        ("microservices", "Architecture")
        ("machine learning", "AI")
        ("neural network", "AI")
        ("algorithm", "Programming")
        ("framework", "Programming")
        ("library", "Programming")
        ("testing", "QualityAssurance")
        ("deployment", "DevOps")
        ("ci/cd", "DevOps")
        ("monitoring", "Operations")
        ("analytics", "DataAnalysis")
    ]
    
    let intentPatterns = [
        ("create", [|"create"; "build"; "make"; "generate"; "develop"; "implement"|])
        ("analyze", [|"analyze"; "examine"; "review"; "inspect"; "evaluate"; "assess"|])
        ("fix", [|"fix"; "repair"; "resolve"; "debug"; "troubleshoot"; "solve"|])
        ("optimize", [|"optimize"; "improve"; "enhance"; "speed up"; "performance"|])
        ("test", [|"test"; "validate"; "verify"; "check"; "ensure"|])
        ("deploy", [|"deploy"; "release"; "publish"; "launch"; "rollout"|])
        ("monitor", [|"monitor"; "track"; "observe"; "watch"; "measure"|])
        ("document", [|"document"; "write"; "explain"; "describe"; "record"|])
    ]
    
    let complexityIndicators = Map.ofList [
        ("simple", 1.0)
        ("basic", 1.0)
        ("easy", 1.0)
        ("straightforward", 1.5)
        ("moderate", 2.0)
        ("intermediate", 2.5)
        ("complex", 3.0)
        ("advanced", 3.5)
        ("difficult", 4.0)
        ("expert", 4.5)
        ("sophisticated", 4.5)
        ("enterprise", 4.0)
        ("scalable", 3.0)
        ("distributed", 3.5)
        ("real-time", 3.0)
        ("high-performance", 3.5)
        ("mission-critical", 4.0)
    ]
    
    /// Analyze text using NLP pipeline
    member this.AnalyzeTextAsync(text: string) = task {
        try
            let startTime = DateTime.UtcNow
            logger.LogDebug($"Analyzing text: {text.Substring(0, min 50 text.Length)}...")
            
            // Clean and tokenize text
            let cleanedText = this.CleanText(text)
            let tokens = this.TokenizeText(cleanedText)
            
            // Extract keywords
            let keywords = this.ExtractKeywords(tokens)
            
            // Extract entities
            let entities = this.ExtractEntities(cleanedText)
            
            // Classify intent
            let intent = this.ClassifyIntent(cleanedText, keywords)
            
            // Classify domain
            let domain = this.ClassifyDomain(cleanedText, keywords, entities)
            
            // Analyze complexity
            let complexity = this.AnalyzeComplexity(cleanedText, keywords, entities)
            
            // Analyze sentiment
            let sentiment = this.AnalyzeSentiment(cleanedText)
            
            // Generate embeddings (simplified)
            let embeddings = this.GenerateEmbeddings(cleanedText, keywords)
            
            let processingTime = DateTime.UtcNow - startTime
            
            let result = {
                OriginalText = text
                CleanedText = cleanedText
                Tokens = tokens
                Keywords = keywords
                Entities = entities
                Intent = intent
                Domain = domain
                Complexity = complexity
                Sentiment = sentiment
                Embeddings = Some embeddings
                ProcessingTime = processingTime
            }
            
            logger.LogDebug($"Text analysis completed in {processingTime.TotalMilliseconds:F0}ms")
            return Ok result
            
        with
        | ex ->
            logger.LogError(ex, $"Error analyzing text: {text.Substring(0, min 50 text.Length)}")
            return Error ex.Message
    }
    
    /// Calculate semantic similarity between two texts
    member this.CalculateSimilarityAsync(text1: string, text2: string) = task {
        try
            logger.LogDebug("Calculating semantic similarity between texts")
            
            let! analysis1 = this.AnalyzeTextAsync(text1)
            let! analysis2 = this.AnalyzeTextAsync(text2)
            
            match analysis1, analysis2 with
            | Ok result1, Ok result2 ->
                // Keyword similarity
                let keywords1 = result1.Keywords |> Set.ofList
                let keywords2 = result2.Keywords |> Set.ofList
                let keywordIntersection = Set.intersect keywords1 keywords2
                let keywordUnion = Set.union keywords1 keywords2
                let keywordSimilarity = 
                    if keywordUnion.Count = 0 then 0.0
                    else float keywordIntersection.Count / float keywordUnion.Count
                
                // Intent similarity
                let intentSimilarity = 
                    if result1.Intent.Intent = result2.Intent.Intent then 1.0
                    else 0.0
                
                // Domain similarity
                let domainSimilarity = 
                    if result1.Domain.PrimaryDomain = result2.Domain.PrimaryDomain then 1.0
                    else 
                        // Check secondary domains
                        let domains1 = result1.Domain.PrimaryDomain :: (result1.Domain.SecondaryDomains |> List.map fst)
                        let domains2 = result2.Domain.PrimaryDomain :: (result2.Domain.SecondaryDomains |> List.map fst)
                        let domainOverlap = Set.intersect (Set.ofList domains1) (Set.ofList domains2)
                        if domainOverlap.Count > 0 then 0.5 else 0.0
                
                // Complexity similarity
                let complexitySimilarity = 
                    let diff = abs (result1.Complexity.ComplexityScore - result2.Complexity.ComplexityScore)
                    max 0.0 (1.0 - diff / 5.0) // Normalize to 0-1 scale
                
                // Embeddings similarity (cosine similarity)
                let embeddingSimilarity = 
                    match result1.Embeddings, result2.Embeddings with
                    | Some emb1, Some emb2 -> this.CalculateCosineSimilarity(emb1.NormalizedVector, emb2.NormalizedVector)
                    | _ -> 0.0
                
                // Weighted overall similarity
                let overallSimilarity = 
                    keywordSimilarity * 0.3 +
                    intentSimilarity * 0.2 +
                    domainSimilarity * 0.2 +
                    complexitySimilarity * 0.1 +
                    embeddingSimilarity * 0.2
                
                let similarityResult = {
                    Score = overallSimilarity
                    MatchingTerms = keywordIntersection |> List.ofSeq
                    ConceptualSimilarity = (intentSimilarity + domainSimilarity) / 2.0
                    StructuralSimilarity = complexitySimilarity
                    SemanticDistance = 1.0 - embeddingSimilarity
                }
                
                logger.LogDebug($"Similarity calculated: {overallSimilarity:F3}")
                return Ok similarityResult
            
            | Error error1, _ -> return Error error1
            | _, Error error2 -> return Error error2
            
        with
        | ex ->
            logger.LogError(ex, "Error calculating semantic similarity")
            return Error ex.Message
    }
    
    /// Extract capability requirements from task description
    member this.ExtractCapabilityRequirementsAsync(taskDescription: string) = task {
        try
            logger.LogDebug("Extracting capability requirements from task description")
            
            let! analysisResult = this.AnalyzeTextAsync(taskDescription)
            match analysisResult with
            | Ok analysis ->
                let requirements = ResizeArray<CapabilityRequirement>()
                
                // Extract capabilities based on domain and technical terms
                for entity in analysis.Entities do
                    match technicalTerms.TryGetValue(entity.Value.ToLower()) with
                    | true, domain ->
                        let requirement = {
                            Name = entity.Value
                            Level = this.DetermineRequiredLevel(analysis.Complexity.OverallComplexity)
                            Required = true
                            Weight = entity.Confidence
                            Description = $"Required for {domain} tasks"
                        }
                        requirements.Add(requirement)
                    | false, _ -> ()
                
                // Add domain-specific capabilities
                let domainCapability = {
                    Name = analysis.Domain.PrimaryDomain
                    Level = this.DetermineRequiredLevel(analysis.Complexity.OverallComplexity)
                    Required = true
                    Weight = analysis.Domain.Confidence
                    Description = $"Primary domain capability"
                }
                requirements.Add(domainCapability)
                
                // Add intent-based capabilities
                let intentCapability = {
                    Name = analysis.Intent.Intent
                    Level = this.DetermineRequiredLevel(analysis.Complexity.OverallComplexity)
                    Required = true
                    Weight = analysis.Intent.Confidence
                    Description = $"Required for {analysis.Intent.Intent} operations"
                }
                requirements.Add(intentCapability)
                
                logger.LogDebug($"Extracted {requirements.Count} capability requirements")
                return Ok (requirements |> List.ofSeq)
            
            | Error error ->
                return Error error
                
        with
        | ex ->
            logger.LogError(ex, "Error extracting capability requirements")
            return Error ex.Message
    }
    
    /// Clean text by removing special characters and normalizing
    member private this.CleanText(text: string) =
        let cleaned = Regex.Replace(text, @"[^\w\s]", " ")
        let normalized = Regex.Replace(cleaned, @"\s+", " ")
        normalized.Trim().ToLower()
    
    /// Tokenize text into words
    member private this.TokenizeText(text: string) =
        text.Split([|' '; '\t'; '\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
        |> Array.filter (fun word -> word.Length > 2 && not (stopWords.Contains(word)))
        |> List.ofArray
    
    /// Extract keywords from tokens
    member private this.ExtractKeywords(tokens: string list) =
        tokens
        |> List.groupBy id
        |> List.map (fun (word, occurrences) -> (word, occurrences.Length))
        |> List.sortByDescending snd
        |> List.take (min 10 tokens.Length)
        |> List.map fst
    
    /// Extract entities from text
    member private this.ExtractEntities(text: string) =
        let entities = ResizeArray<EntityExtractionResult>()
        
        // Extract technical terms
        for kvp in technicalTerms do
            let pattern = $@"\b{Regex.Escape(kvp.Key)}\b"
            let matches = Regex.Matches(text, pattern, RegexOptions.IgnoreCase)
            for match' in matches do
                let entity = {
                    EntityType = "TechnicalTerm"
                    Value = kvp.Key
                    StartPosition = match'.Index
                    EndPosition = match'.Index + match'.Length
                    Confidence = 0.9
                    Context = kvp.Value
                }
                entities.Add(entity)
        
        entities |> List.ofSeq
    
    /// Classify intent from text
    member private this.ClassifyIntent(text: string, keywords: string list) =
        let intentScores = ResizeArray<string * float>()
        
        for (intent, patterns) in intentPatterns do
            let score = 
                patterns
                |> Array.sumBy (fun pattern ->
                    if text.Contains(pattern) then 1.0
                    elif keywords |> List.contains pattern then 0.8
                    else 0.0)
            
            if score > 0.0 then
                intentScores.Add((intent, score))
        
        let sortedIntents = intentScores |> Seq.sortByDescending snd |> List.ofSeq
        
        match sortedIntents with
        | (primaryIntent, score) :: alternatives ->
            {
                Intent = primaryIntent
                Confidence = min 1.0 (score / 2.0)
                AlternativeIntents = alternatives |> List.take (min 3 alternatives.Length)
                Entities = Map.empty
                Keywords = keywords
            }
        | [] ->
            {
                Intent = "unknown"
                Confidence = 0.0
                AlternativeIntents = []
                Entities = Map.empty
                Keywords = keywords
            }
    
    /// Classify domain from text
    member private this.ClassifyDomain(text: string, keywords: string list, entities: EntityExtractionResult list) =
        let domainScores = ConcurrentDictionary<string, float>()
        
        // Score based on technical terms
        for entity in entities do
            if entity.EntityType = "TechnicalTerm" then
                domainScores.AddOrUpdate(entity.Context, entity.Confidence, fun _ current -> current + entity.Confidence) |> ignore
        
        // Score based on keywords
        for keyword in keywords do
            match technicalTerms.TryGetValue(keyword) with
            | true, domain ->
                domainScores.AddOrUpdate(domain, 0.5, fun _ current -> current + 0.5) |> ignore
            | false, _ -> ()
        
        let sortedDomains = 
            domainScores
            |> Seq.map (fun kvp -> (kvp.Key, kvp.Value))
            |> Seq.sortByDescending snd
            |> List.ofSeq
        
        match sortedDomains with
        | (primaryDomain, score) :: alternatives ->
            {
                PrimaryDomain = primaryDomain
                Confidence = min 1.0 (score / 3.0)
                SecondaryDomains = alternatives |> List.take (min 2 alternatives.Length)
                TechnicalTerms = entities |> List.map (fun e -> e.Value)
                DomainSpecificConcepts = keywords |> List.filter (fun k -> technicalTerms.ContainsKey(k))
            }
        | [] ->
            {
                PrimaryDomain = "General"
                Confidence = 0.5
                SecondaryDomains = []
                TechnicalTerms = []
                DomainSpecificConcepts = []
            }
    
    /// Analyze task complexity
    member private this.AnalyzeComplexity(text: string, keywords: string list, entities: EntityExtractionResult list) =
        let complexityFactors = ConcurrentDictionary<string, float>()
        
        // Analyze complexity indicators in text
        for kvp in complexityIndicators do
            if text.Contains(kvp.Key) then
                complexityFactors.["TextComplexity"] <- kvp.Value
        
        // Analyze based on number of technical terms
        let technicalTermCount = entities |> List.filter (fun e -> e.EntityType = "TechnicalTerm") |> List.length
        complexityFactors.["TechnicalComplexity"] <- float technicalTermCount * 0.5
        
        // Analyze based on text length and structure
        let textLength = text.Length
        let lengthComplexity = 
            if textLength > 500 then 3.0
            elif textLength > 200 then 2.0
            elif textLength > 100 then 1.5
            else 1.0
        complexityFactors.["LengthComplexity"] <- lengthComplexity
        
        // Calculate overall complexity score
        let totalScore = complexityFactors.Values |> Seq.sum
        let averageScore = if complexityFactors.Count > 0 then totalScore / float complexityFactors.Count else 1.0
        
        let overallComplexity = 
            if averageScore >= 4.0 then TaskComplexity.Expert
            elif averageScore >= 3.0 then TaskComplexity.Complex
            elif averageScore >= 2.0 then TaskComplexity.Moderate
            else TaskComplexity.Simple
        
        let estimatedDuration = 
            match overallComplexity with
            | TaskComplexity.Simple -> TimeSpan.FromMinutes(15.0)
            | TaskComplexity.Moderate -> TimeSpan.FromMinutes(60.0)
            | TaskComplexity.Complex -> TimeSpan.FromHours(4.0)
            | TaskComplexity.Expert -> TimeSpan.FromHours(16.0)
            | TaskComplexity.Collaborative -> TimeSpan.FromHours(8.0)
        
        {
            OverallComplexity = overallComplexity
            ComplexityScore = averageScore
            ComplexityFactors = complexityFactors |> Map.ofSeq
            EstimatedDuration = estimatedDuration
            RequiredSkillLevel = this.DetermineRequiredLevel(overallComplexity)
            Reasoning = [
                $"Text complexity indicators: {complexityFactors.GetOrAdd("TextComplexity", 0.0)}"
                $"Technical term count: {technicalTermCount}"
                $"Text length complexity: {lengthComplexity}"
            ]
        }
    
    /// Analyze sentiment (simplified)
    member private this.AnalyzeSentiment(text: string) =
        let positiveWords = ["good"; "great"; "excellent"; "amazing"; "perfect"; "love"; "like"; "best"]
        let negativeWords = ["bad"; "terrible"; "awful"; "hate"; "worst"; "problem"; "issue"; "error"]
        
        let positiveCount = positiveWords |> List.sumBy (fun word -> if text.Contains(word) then 1 else 0)
        let negativeCount = negativeWords |> List.sumBy (fun word -> if text.Contains(word) then 1 else 0)
        
        let sentiment = float (positiveCount - negativeCount) / float (max 1 (positiveCount + negativeCount))
        sentiment
    
    /// Generate semantic embeddings (simplified)
    member private this.GenerateEmbeddings(text: string, keywords: string list) =
        // Simplified embedding generation (in production, would use proper ML models)
        let dimensions = 100
        let vector = Array.create dimensions 0.0
        
        // Generate pseudo-embeddings based on keywords
        for i, keyword in keywords |> List.indexed do
            if i < dimensions then
                vector.[i] <- float (keyword.Length) / 10.0
        
        let magnitude = sqrt (vector |> Array.sumBy (fun x -> x * x))
        let normalizedVector = 
            if magnitude > 0.0 then vector |> Array.map (fun x -> x / magnitude)
            else vector
        
        {
            Dimensions = dimensions
            Vector = vector
            Magnitude = magnitude
            NormalizedVector = normalizedVector
        }
    
    /// Calculate cosine similarity between two vectors
    member private this.CalculateCosineSimilarity(vector1: float array, vector2: float array) =
        if vector1.Length <> vector2.Length then 0.0
        else
            let dotProduct = Array.zip vector1 vector2 |> Array.sumBy (fun (a, b) -> a * b)
            let magnitude1 = sqrt (vector1 |> Array.sumBy (fun x -> x * x))
            let magnitude2 = sqrt (vector2 |> Array.sumBy (fun x -> x * x))
            
            if magnitude1 = 0.0 || magnitude2 = 0.0 then 0.0
            else dotProduct / (magnitude1 * magnitude2)
    
    /// Determine required capability level based on complexity
    member private this.DetermineRequiredLevel(complexity: TaskComplexity) =
        match complexity with
        | TaskComplexity.Simple -> CapabilityLevel.Beginner
        | TaskComplexity.Moderate -> CapabilityLevel.Intermediate
        | TaskComplexity.Complex -> CapabilityLevel.Advanced
        | TaskComplexity.Expert -> CapabilityLevel.Expert
        | TaskComplexity.Collaborative -> CapabilityLevel.Advanced

    /// Extract semantic metadata from text
    member this.ExtractSemanticMetadata(text: string) = task {
        let! analysisResult = this.AnalyzeTextAsync(text)
        match analysisResult with
        | Ok analysis ->
            return {
                Keywords = analysis.Keywords
                Entities = analysis.Entities |> List.map (fun e -> (e.EntityType, e.Value)) |> Map.ofList
                Intent = analysis.Intent.Intent
                Domain = analysis.Domain.PrimaryDomain
                Language = "en"
                Confidence = analysis.Intent.Confidence
                Embeddings = analysis.Embeddings |> Option.map (fun e -> e.Vector)
                Sentiment = analysis.Sentiment
                Topics = analysis.Domain.DomainSpecificConcepts
            }
        | Error _ ->
            return {
                Keywords = []
                Entities = Map.empty
                Intent = "unknown"
                Domain = "General"
                Language = "en"
                Confidence = 0.0
                Embeddings = None
                Sentiment = 0.0
                Topics = []
            }
    }
