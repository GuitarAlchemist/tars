namespace Tars.Knowledge

open System
open Tars.Core
open Tars.Llm

/// Extract structured beliefs from Wikipedia content using LLM
module WikipediaExtractor =

    /// Extract beliefs from a Wikipedia article section
    let extractBeliefs (llmService: ILlmService) (title: string) (content: string) =
        async {
            let prompt = $"""You are a knowledge extraction agent. Extract factual beliefs from this Wikipedia content as subject-predicate-object triples.

Article: {title}

Content:
{content}

Instructions:
1. Extract ONLY factual, verifiable statements
2. Format each as: (Subject, Predicate, Object)
3. Use simple predicates: "is", "has", "invented", "created", "located_in", "part_of", "causes", etc.
4. Be specific and atomic - one fact per triple
5. Limit to the 10 most important facts

Output format (one per line):
(Subject, Predicate, Object)

Example:
(Paris, is, capital of France)
(Eiffel Tower, located_in, Paris)
(Python, is, programming language)
"""

            // Create LLM request with proper types
            let request = {
                LlmRequest.Default with
                    Messages = [ { Role = Role.User; Content = prompt } ]
                    MaxTokens = Some 1000
                    Temperature = Some 0.3 // Low temperature for factual extraction
                    ModelHint = Some "fast"
                    Stream = false
            }

            let! response = llmService.CompleteAsync(request) |> Async.AwaitTask
            let responseText : string = response.Text

            // Parse response into ProposedAssertions
            let lines : string array = 
                responseText.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
                |> Array.filter (fun (line: string) -> line.Trim().StartsWith("(") && line.Contains(","))

            let proposals = ResizeArray<ProposedAssertion>()

            for line in lines do
                try
                    // Parse: (Subject, Predicate, Object)
                    let cleaned = (line : string).Trim().TrimStart('(').TrimEnd(')')
                    let parts : string array = cleaned.Split([|','|], 3)
                    
                    if parts.Length >= 3 then
                        let subject : string = parts.[0].Trim()
                        let predicate : string = parts.[1].Trim()
                        let obj : string = parts.[2].Trim()
                        
                        if not (String.IsNullOrWhiteSpace subject) && 
                           not (String.IsNullOrWhiteSpace predicate) &&
                           not (String.IsNullOrWhiteSpace obj) then
                            
                            proposals.Add({
                                Id = Guid.NewGuid()
                                Subject = subject
                                Predicate = predicate
                                Object = obj
                                SourceSection = content.Substring(0, Math.Min(200, content.Length))
                                Confidence = 0.8 // LLM extraction confidence
                                ExtractorAgent = AgentId.System
                                ExtractedAt = DateTime.UtcNow
                            })
                with ex ->
                    Logging.warn $"Failed to parse line: {line} - {ex.Message}"

            return proposals |> Seq.toList
        }

    /// Extract beliefs from full Wikipedia article (chunked processing)
    let extractFromArticle (llmService: ILlmService) (title: string) (fullContent: string) =
        async {
            // Split into sections (Wikipedia typically has sections)
            let sections : string array = 
                fullContent.Split([|"=="|], StringSplitOptions.RemoveEmptyEntries)
                |> Array.map (fun s -> s.Trim())
                |> Array.filter (fun s -> s.Length > 100) // Skip tiny sections
                |> Array.truncate 5 // Process max 5 sections to avoid overwhelming

            let! allProposals = 
                sections
                |> Array.map (fun section -> extractBeliefs llmService title section)
                |> Async.Parallel

            // Flatten the array of lists
            return allProposals |> Array.toList |> List.concat
        }
