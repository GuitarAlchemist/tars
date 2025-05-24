namespace TarsEngine.SelfImprovement

open System
open System.Net.Http
open System.Net.Http.Json
open System.Text.Json
open System.Text.RegularExpressions
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Module for extracting knowledge from exploration files using Ollama
/// </summary>
module OllamaKnowledgeExtractor =

    /// <summary>
    /// Knowledge extraction result model
    /// </summary>
    type KnowledgeExtractionResult = {
        concepts: {| name: string; definition: string |}[]
        insights: {| description: string; importance: string |}[]
        technical_details: {| topic: string; details: string |}[]
        design_decisions: {| decision: string; rationale: string |}[]
        relationships: {| from: string; ``to``: string; relationship: string |}[]
    }

    /// <summary>
    /// Extracts knowledge from text content using Ollama
    /// </summary>
    let extractKnowledgeWithOllama (logger: ILogger) (content: string) (ollamaEndpoint: string) (model: string) =
        task {
            try
                logger.LogInformation("Extracting knowledge using Ollama with model {Model}", model)

                // Create a prompt for knowledge extraction
                let prompt =
                    $"You are an expert at extracting structured knowledge from documentation.\n\n" +
                    $"I'll provide you with content from a file. Please extract key knowledge, concepts, and insights.\n\n" +
                    $"Focus on:\n" +
                    $"1. Key concepts and their definitions\n" +
                    $"2. Important insights and conclusions\n" +
                    $"3. Technical details and specifications\n" +
                    $"4. Design decisions and rationales\n" +
                    $"5. Relationships between concepts\n\n" +
                    $"Here's the content:\n\n{content}\n\n" +
                    $"Please provide your extracted knowledge in JSON format with the following structure:\n" +
                    $"{{\n" +
                    $"  \"concepts\": [\n" +
                    $"    {{ \"name\": \"Concept name\", \"definition\": \"Concept definition\" }}\n" +
                    $"  ],\n" +
                    $"  \"insights\": [\n" +
                    $"    {{ \"description\": \"Description of the insight\", \"importance\": \"Why this insight is important\" }}\n" +
                    $"  ],\n" +
                    $"  \"technical_details\": [\n" +
                    $"    {{ \"topic\": \"Topic name\", \"details\": \"Technical details\" }}\n" +
                    $"  ],\n" +
                    $"  \"design_decisions\": [\n" +
                    $"    {{ \"decision\": \"The decision made\", \"rationale\": \"Why this decision was made\" }}\n" +
                    $"  ],\n" +
                    $"  \"relationships\": [\n" +
                    $"    {{ \"from\": \"Concept A\", \"to\": \"Concept B\", \"relationship\": \"How A relates to B\" }}\n" +
                    $"  ]\n" +
                    $"}}\n\n" +
                    $"Only return the JSON, no other text."

                // Create HTTP client
                use client = new HttpClient()

                // Create request
                let request = {
                    model = model
                    prompt = prompt
                    stream = false
                    options = {|
                        temperature = 0.3
                        num_predict = 4000
                    |}
                }

                // Send request to Ollama
                let! response = client.PostAsJsonAsync($"{ollamaEndpoint}/api/generate", request)

                // Check if request was successful
                if not response.IsSuccessStatusCode then
                    logger.LogError("Failed to extract knowledge using Ollama: {StatusCode} - {ReasonPhrase}", response.StatusCode, response.ReasonPhrase)
                    return []
                else
                    // Parse response
                    let! result = response.Content.ReadFromJsonAsync<OllamaResponse>()

                    // Extract JSON from response
                    let jsonPattern = "\{[\s\S]*\}"
                    let jsonMatch = Regex.Match(result.response, jsonPattern)

                    if jsonMatch.Success then
                        let jsonResponse = jsonMatch.Value

                        // Parse JSON
                        let options = JsonSerializerOptions()
                        options.PropertyNameCaseInsensitive <- true

                        let extractionResult = JsonSerializer.Deserialize<KnowledgeExtractionResult>(jsonResponse, options)

                        // Convert to knowledge items
                        let mutable items = []

                        // Add concepts
                        for concept in extractionResult.concepts do
                            let item = {
                                Id = Guid.NewGuid().ToString()
                                Type = "Concept"
                                Content = $"{concept.name}: {concept.definition}"
                                Source = ""  // Will be set by the caller
                                SourceType = KnowledgeSourceType.Documentation  // Will be set by the caller
                                Confidence = 0.9
                                Tags = ["concept"; concept.name]
                                RelatedItems = []
                                ExtractedAt = DateTime.UtcNow
                            }
                            items <- items @ [item]

                        // Add insights
                        for insight in extractionResult.insights do
                            let item = {
                                Id = Guid.NewGuid().ToString()
                                Type = "Insight"
                                Content = if insight.importance <> "" then $"{insight.description} (Importance: {insight.importance})" else insight.description
                                Source = ""  // Will be set by the caller
                                SourceType = KnowledgeSourceType.Documentation  // Will be set by the caller
                                Confidence = 0.8
                                Tags = ["insight"]
                                RelatedItems = []
                                ExtractedAt = DateTime.UtcNow
                            }
                            items <- items @ [item]

                        // Add technical details
                        for detail in extractionResult.technical_details do
                            let item = {
                                Id = Guid.NewGuid().ToString()
                                Type = "TechnicalDetail"
                                Content = $"{detail.topic}: {detail.details}"
                                Source = ""  // Will be set by the caller
                                SourceType = KnowledgeSourceType.Documentation  // Will be set by the caller
                                Confidence = 0.85
                                Tags = ["technical"; detail.topic]
                                RelatedItems = []
                                ExtractedAt = DateTime.UtcNow
                            }
                            items <- items @ [item]

                        // Add design decisions
                        for decision in extractionResult.design_decisions do
                            let item = {
                                Id = Guid.NewGuid().ToString()
                                Type = "DesignDecision"
                                Content = $"{decision.decision} (Rationale: {decision.rationale})"
                                Source = ""  // Will be set by the caller
                                SourceType = KnowledgeSourceType.Documentation  // Will be set by the caller
                                Confidence = 0.9
                                Tags = ["design"; "decision"]
                                RelatedItems = []
                                ExtractedAt = DateTime.UtcNow
                            }
                            items <- items @ [item]

                        // Add relationships
                        for relationship in extractionResult.relationships do
                            let item = {
                                Id = Guid.NewGuid().ToString()
                                Type = "Relationship"
                                Content = $"{relationship.from} {relationship.relationship} {relationship.``to``}"
                                Source = ""  // Will be set by the caller
                                SourceType = KnowledgeSourceType.Documentation  // Will be set by the caller
                                Confidence = 0.75
                                Tags = ["relationship"; relationship.from; relationship.``to``]
                                RelatedItems = []
                                ExtractedAt = DateTime.UtcNow
                            }
                            items <- items @ [item]

                        return items
                    else
                        logger.LogWarning("Failed to extract knowledge using Ollama: Invalid response format")
                        return []
            with ex ->
                logger.LogError(ex, "Error extracting knowledge using Ollama")
                return []
        }
