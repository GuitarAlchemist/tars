namespace TarsEngine.FSharp.Cli.Services

open System
open System.Net.Http
open System.Text
open System.Text.Json
open System.Threading.Tasks
open Microsoft.Extensions.Logging
// Remove this import as we're defining our own types

/// Expert specialization types for Mixture of Experts
type ExpertType =
    | CodeGeneration
    | CodeAnalysis
    | Architecture
    | Testing
    | Documentation
    | Debugging
    | Performance
    | Security
    | DevOps
    | General

/// Expert configuration
type Expert = {
    Type: ExpertType
    Name: string
    SystemPrompt: string
    Temperature: float
    MaxTokens: int
    Confidence: float
}

/// Routing decision for expert selection
type RoutingDecision = {
    SelectedExpert: Expert
    Confidence: float
    Reasoning: string
    AlternativeExperts: Expert list
}

/// Mixtral request with MoE routing
type MixtralRequest = {
    Query: string
    Context: string option
    RequiredExpertise: ExpertType list
    MaxExperts: int
    UseEnsemble: bool
}

/// Mixtral response with expert attribution
type MixtralResponse = {
    Content: string
    UsedExperts: Expert list
    RoutingDecision: RoutingDecision
    TokensUsed: int
    ResponseTime: TimeSpan
    Confidence: float
}

/// Computational expression for expert routing and prompt chaining
type ExpertRoutingBuilder() =
    member _.Bind(decision: RoutingDecision, f: Expert -> Task<'T>) =
        task {
            return! f decision.SelectedExpert
        }
    
    member _.Return(value: 'T) = Task.FromResult(value)
    
    member _.ReturnFrom(task: Task<'T>) = task
    
    member _.Zero() = Task.FromResult(())
    
    member _.Combine(task1: Task<unit>, task2: Task<'T>) =
        task {
            do! task1
            return! task2
        }
    
    member _.Delay(f: unit -> Task<'T>) = f
    
    member _.Run(f: unit -> Task<'T>) = f()

/// Computational expression for prompt chaining
type PromptChainBuilder() =
    member _.Bind(response: MixtralResponse, f: string -> Task<MixtralResponse>) =
        task {
            return! f response.Content
        }
    
    member _.Return(value: MixtralResponse) = Task.FromResult(value)
    
    member _.ReturnFrom(task: Task<MixtralResponse>) = task
    
    member _.Zero() = Task.FromResult({
        Content = ""
        UsedExperts = []
        RoutingDecision = {
            SelectedExpert = { Type = General; Name = ""; SystemPrompt = ""; Temperature = 0.7; MaxTokens = 1000; Confidence = 0.0 }
            Confidence = 0.0
            Reasoning = ""
            AlternativeExperts = []
        }
        TokensUsed = 0
        ResponseTime = TimeSpan.Zero
        Confidence = 0.0
    })

/// Mixtral LLM service with Mixture of Experts support
type MixtralService(httpClient: HttpClient, logger: ILogger<MixtralService>) =
    
    let expertRouting = ExpertRoutingBuilder()
    let promptChain = PromptChainBuilder()
    
    // Define expert configurations
    let experts = [
        {
            Type = CodeGeneration
            Name = "Code Generator"
            SystemPrompt = """You are an expert code generator specializing in F#, C#, and functional programming. 
            Focus on clean, efficient, and well-structured code with proper error handling and documentation."""
            Temperature = 0.3
            MaxTokens = 2000
            Confidence = 0.9
        }
        {
            Type = CodeAnalysis
            Name = "Code Analyzer"
            SystemPrompt = """You are an expert code analyzer specializing in static analysis, code quality, and best practices.
            Provide detailed analysis of code structure, potential issues, and improvement suggestions."""
            Temperature = 0.2
            MaxTokens = 1500
            Confidence = 0.85
        }
        {
            Type = Architecture
            Name = "System Architect"
            SystemPrompt = """You are a system architecture expert specializing in distributed systems, microservices, and scalable design.
            Focus on high-level design decisions, patterns, and architectural trade-offs."""
            Temperature = 0.4
            MaxTokens = 2000
            Confidence = 0.8
        }
        {
            Type = Testing
            Name = "Test Engineer"
            SystemPrompt = """You are a testing expert specializing in unit testing, integration testing, and test automation.
            Focus on comprehensive test strategies, test case generation, and quality assurance."""
            Temperature = 0.3
            MaxTokens = 1500
            Confidence = 0.85
        }
        {
            Type = Documentation
            Name = "Technical Writer"
            SystemPrompt = """You are a technical documentation expert specializing in clear, comprehensive documentation.
            Focus on user guides, API documentation, and technical specifications."""
            Temperature = 0.5
            MaxTokens = 2000
            Confidence = 0.8
        }
        {
            Type = Debugging
            Name = "Debug Specialist"
            SystemPrompt = """You are a debugging expert specializing in error analysis, troubleshooting, and problem resolution.
            Focus on identifying root causes, providing solutions, and preventing similar issues."""
            Temperature = 0.2
            MaxTokens = 1500
            Confidence = 0.9
        }
        {
            Type = Performance
            Name = "Performance Engineer"
            SystemPrompt = """You are a performance optimization expert specializing in profiling, benchmarking, and optimization.
            Focus on identifying bottlenecks, memory usage, and scalability improvements."""
            Temperature = 0.3
            MaxTokens = 1500
            Confidence = 0.85
        }
        {
            Type = Security
            Name = "Security Analyst"
            SystemPrompt = """You are a security expert specializing in vulnerability assessment, secure coding, and threat analysis.
            Focus on identifying security risks, implementing security measures, and compliance."""
            Temperature = 0.2
            MaxTokens = 1500
            Confidence = 0.9
        }
        {
            Type = DevOps
            Name = "DevOps Engineer"
            SystemPrompt = """You are a DevOps expert specializing in CI/CD, containerization, and infrastructure automation.
            Focus on deployment strategies, monitoring, and operational excellence."""
            Temperature = 0.3
            MaxTokens = 1500
            Confidence = 0.8
        }
        {
            Type = General
            Name = "General Assistant"
            SystemPrompt = """You are a general-purpose AI assistant with broad knowledge across multiple domains.
            Provide helpful, accurate, and well-reasoned responses to a wide variety of questions."""
            Temperature = 0.7
            MaxTokens = 1000
            Confidence = 0.7
        }
    ]
    
    /// Route query to appropriate expert using intelligent analysis
    member private this.RouteToExpertAsync(query: string, requiredExpertise: ExpertType list) =
        task {
            try
                // Analyze query to determine best expert
                let queryLower = query.ToLower()
                
                let expertScores = 
                    experts
                    |> List.map (fun expert ->
                        let score = this.CalculateExpertScore(expert, queryLower, requiredExpertise)
                        (expert, score))
                    |> List.sortByDescending snd
                
                let (selectedExpert, confidence) = expertScores |> List.head
                let alternatives = expertScores |> List.tail |> List.take 2 |> List.map fst
                
                let reasoning = this.GenerateRoutingReasoning(selectedExpert, confidence, queryLower)
                
                logger.LogInformation("Routed query to expert: {ExpertName} with confidence: {Confidence}", 
                    selectedExpert.Name, confidence)
                
                return {
                    SelectedExpert = selectedExpert
                    Confidence = confidence
                    Reasoning = reasoning
                    AlternativeExperts = alternatives
                }
            with
            | ex ->
                logger.LogError(ex, "Failed to route query to expert")
                let fallbackExpert = experts |> List.find (fun e -> e.Type = General)
                return {
                    SelectedExpert = fallbackExpert
                    Confidence = 0.5
                    Reasoning = "Fallback to general expert due to routing error"
                    AlternativeExperts = []
                }
        }
    
    /// Calculate expert score based on query analysis
    member private this.CalculateExpertScore(expert: Expert, queryLower: string, requiredExpertise: ExpertType list) =
        let baseScore = expert.Confidence
        
        // Boost score if expert type is in required expertise
        let expertiseBoost = 
            if requiredExpertise |> List.contains expert.Type then 0.3 else 0.0
        
        // Keyword-based scoring
        let keywordScore = 
            match expert.Type with
            | CodeGeneration when queryLower.Contains("generate") || queryLower.Contains("create") || queryLower.Contains("implement") -> 0.2
            | CodeAnalysis when queryLower.Contains("analyze") || queryLower.Contains("review") || queryLower.Contains("quality") -> 0.2
            | Architecture when queryLower.Contains("design") || queryLower.Contains("architecture") || queryLower.Contains("pattern") -> 0.2
            | Testing when queryLower.Contains("test") || queryLower.Contains("unit") || queryLower.Contains("integration") -> 0.2
            | Documentation when queryLower.Contains("document") || queryLower.Contains("explain") || queryLower.Contains("guide") -> 0.2
            | Debugging when queryLower.Contains("debug") || queryLower.Contains("error") || queryLower.Contains("fix") -> 0.2
            | Performance when queryLower.Contains("performance") || queryLower.Contains("optimize") || queryLower.Contains("speed") -> 0.2
            | Security when queryLower.Contains("security") || queryLower.Contains("vulnerability") || queryLower.Contains("secure") -> 0.2
            | DevOps when queryLower.Contains("deploy") || queryLower.Contains("docker") || queryLower.Contains("ci/cd") -> 0.2
            | _ -> 0.0
        
        Math.Min(1.0, baseScore + expertiseBoost + keywordScore)
    
    /// Generate reasoning for expert selection
    member private this.GenerateRoutingReasoning(expert: Expert, confidence: float, query: string) =
        $"Selected {expert.Name} (confidence: {confidence:F2}) based on query analysis. " +
        $"Expert specializes in {expert.Type} with system prompt optimized for this domain."

    /// Send request to Mixtral LLM with expert configuration
    member private this.SendToMixtralAsync(expert: Expert, query: string, context: string option) =
        task {
            try
                let startTime = DateTime.UtcNow

                // Build the prompt with expert system prompt
                let fullPrompt =
                    match context with
                    | Some ctx -> $"{expert.SystemPrompt}\n\nContext:\n{ctx}\n\nQuery: {query}"
                    | None -> $"{expert.SystemPrompt}\n\nQuery: {query}"

                // Create request payload for Mixtral
                let payload = {|
                    model = "mixtral-8x7b-instruct-v0.1"
                    messages = [|
                        {| role = "system"; content = expert.SystemPrompt |}
                        {| role = "user"; content = query |}
                    |]
                    temperature = expert.Temperature
                    max_tokens = expert.MaxTokens
                    top_p = 0.9
                    stream = false
                |}

                let jsonContent = JsonSerializer.Serialize(payload)
                let content = new StringContent(jsonContent, Encoding.UTF8, "application/json")

                // Send to Ollama endpoint (assuming Mixtral is available)
                let! response = httpClient.PostAsync("http://localhost:11434/v1/chat/completions", content)

                if response.IsSuccessStatusCode then
                    let! responseContent = response.Content.ReadAsStringAsync()
                    let responseData = JsonSerializer.Deserialize<JsonElement>(responseContent)

                    let content =
                        responseData.GetProperty("choices").[0]
                            .GetProperty("message")
                            .GetProperty("content")
                            .GetString()

                    let tokensUsed =
                        try
                            responseData.GetProperty("usage").GetProperty("total_tokens").GetInt32()
                        with
                        | _ -> 0

                    let responseTime = DateTime.UtcNow - startTime

                    logger.LogInformation("Mixtral response received from {ExpertName} in {ResponseTime}ms",
                        expert.Name, responseTime.TotalMilliseconds)

                    return Ok {
                        Content = content
                        UsedExperts = [expert]
                        RoutingDecision = {
                            SelectedExpert = expert
                            Confidence = expert.Confidence
                            Reasoning = $"Direct expert call to {expert.Name}"
                            AlternativeExperts = []
                        }
                        TokensUsed = tokensUsed
                        ResponseTime = responseTime
                        Confidence = expert.Confidence
                    }
                else
                    let! errorContent = response.Content.ReadAsStringAsync()
                    logger.LogError("Mixtral API error: {StatusCode} - {Error}", response.StatusCode, errorContent)
                    return Error $"Mixtral API error: {response.StatusCode}"
            with
            | ex ->
                logger.LogError(ex, "Failed to send request to Mixtral")
                return Error ex.Message
        }

    /// Process request with Mixture of Experts routing
    member this.ProcessWithMoEAsync(request: MixtralRequest) =
        task {
            try
                logger.LogInformation("Processing MoE request: {Query}", request.Query)

                // Route to appropriate expert
                let! routingDecision = this.RouteToExpertAsync(request.Query, request.RequiredExpertise)

                if request.UseEnsemble && request.MaxExperts > 1 then
                    // Use ensemble of experts
                    return! this.ProcessWithEnsembleAsync(request, routingDecision)
                else
                    // Use single expert
                    let! result = this.SendToMixtralAsync(routingDecision.SelectedExpert, request.Query, request.Context)
                    match result with
                    | Ok response ->
                        return Ok { response with RoutingDecision = routingDecision }
                    | Error error ->
                        return Error error
            with
            | ex ->
                logger.LogError(ex, "Failed to process MoE request")
                return Error ex.Message
        }

    /// Process with ensemble of experts
    member private this.ProcessWithEnsembleAsync(request: MixtralRequest, routingDecision: RoutingDecision) =
        task {
            try
                logger.LogInformation("Processing with ensemble of {MaxExperts} experts", request.MaxExperts)

                // Select top experts
                let selectedExperts =
                    routingDecision.SelectedExpert :: routingDecision.AlternativeExperts
                    |> List.take (Math.Min(request.MaxExperts, experts.Length))

                // Send requests to all selected experts in parallel
                let! responses =
                    selectedExperts
                    |> List.map (fun expert -> this.SendToMixtralAsync(expert, request.Query, request.Context))
                    |> Task.WhenAll

                // Combine successful responses
                let successfulResponses =
                    responses
                    |> Array.choose (function | Ok resp -> Some resp | Error _ -> None)
                    |> Array.toList

                if successfulResponses.IsEmpty then
                    return Error "All expert requests failed"
                else
                    // Combine responses intelligently
                    let combinedContent = this.CombineExpertResponses(successfulResponses)
                    let totalTokens = successfulResponses |> List.sumBy (_.TokensUsed)
                    let avgConfidence = successfulResponses |> List.averageBy (_.Confidence)
                    let maxResponseTime = successfulResponses |> List.maxBy (_.ResponseTime) |> (_.ResponseTime)

                    return Ok {
                        Content = combinedContent
                        UsedExperts = selectedExperts
                        RoutingDecision = routingDecision
                        TokensUsed = totalTokens
                        ResponseTime = maxResponseTime
                        Confidence = avgConfidence
                    }
            with
            | ex ->
                logger.LogError(ex, "Failed to process ensemble request")
                return Error ex.Message
        }

    /// Combine responses from multiple experts
    member private this.CombineExpertResponses(responses: MixtralResponse list) =
        let sections =
            responses
            |> List.mapi (fun i resp ->
                $"## Expert {i + 1}: {resp.UsedExperts.[0].Name}\n\n{resp.Content}\n")
            |> String.concat "\n"

        $"# Ensemble Response from {responses.Length} Experts\n\n{sections}\n\n" +
        "## Summary\n\nThis response combines insights from multiple specialized experts to provide comprehensive coverage of your query."

    /// Computational expression instances
    member _.ExpertRouting = expertRouting
    member _.PromptChain = promptChain

    /// High-level API for simple queries
    member this.QueryAsync(query: string, ?expertType: ExpertType, ?useEnsemble: bool) =
        task {
            let request = {
                Query = query
                Context = None
                RequiredExpertise = match expertType with | Some et -> [et] | None -> []
                MaxExperts = if defaultArg useEnsemble false then 3 else 1
                UseEnsemble = defaultArg useEnsemble false
            }
            return! this.ProcessWithMoEAsync(request)
        }

    /// Chain multiple prompts with expert routing
    member this.ChainPromptsAsync(prompts: string list, ?expertTypes: ExpertType list) =
        task {
            try
                let mutable context = None
                let mutable allResponses = []
                let mutable hasError = false
                let mutable errorMessage = ""

                for (i, prompt) in prompts |> List.indexed do
                    if not hasError then
                        let expertType =
                            match expertTypes with
                            | Some types when i < types.Length -> Some types.[i]
                            | _ -> None

                        let! result = this.QueryAsync(prompt, ?expertType = expertType)
                        match result with
                        | Ok response ->
                            context <- Some response.Content
                            allResponses <- response :: allResponses
                        | Error error ->
                            logger.LogError("Failed to process prompt {Index}: {Error}", i, error)
                            hasError <- true
                            errorMessage <- $"Chain failed at prompt {i}: {error}"

                if hasError then
                    return Error errorMessage
                elif allResponses.IsEmpty then
                    return Error "No successful responses in chain"
                else
                    let finalResponse = allResponses |> List.head
                    let allExperts = allResponses |> List.collect (_.UsedExperts) |> List.distinct
                    let totalTokens = allResponses |> List.sumBy (_.TokensUsed)

                    return Ok {
                        finalResponse with
                            Content = context |> Option.defaultValue ""
                            UsedExperts = allExperts
                            TokensUsed = totalTokens
                    }
            with
            | ex ->
                logger.LogError(ex, "Failed to execute prompt chain")
                return Error ex.Message
        }

/// Router component for intelligent LLM selection and routing
type LLMRouter(mixtralService: MixtralService, logger: ILogger<LLMRouter>) =

    /// Route query to best available LLM service
    member this.RouteQueryAsync(query: string, availableServices: string list) =
        task {
            try
                // Analyze query complexity and requirements
                let complexity = this.AnalyzeQueryComplexity(query)
                let domain = this.IdentifyDomain(query)

                // Route based on analysis
                let selectedService =
                    match complexity, domain with
                    | "High", "Code" -> "mixtral-code"
                    | "High", _ -> "mixtral-ensemble"
                    | "Medium", "Code" -> "mixtral-single"
                    | _ -> "codestral"

                logger.LogInformation("Routed query to {Service} based on complexity: {Complexity}, domain: {Domain}",
                    selectedService, complexity, domain)

                return {|
                    SelectedService = selectedService
                    Complexity = complexity
                    Domain = domain
                    Reasoning = $"Selected {selectedService} for {complexity} complexity {domain} query"
                |}
            with
            | ex ->
                logger.LogError(ex, "Failed to route query")
                return {|
                    SelectedService = "codestral"
                    Complexity = "Unknown"
                    Domain = "General"
                    Reasoning = "Fallback to Codestral due to routing error"
                |}
        }

    /// Analyze query complexity
    member private this.AnalyzeQueryComplexity(query: string) =
        let wordCount = query.Split(' ').Length
        let hasCodeKeywords =
            ["implement"; "generate"; "create"; "build"; "develop"; "code"; "function"; "class"]
            |> List.exists (fun keyword -> query.ToLower().Contains(keyword))

        match wordCount, hasCodeKeywords with
        | w, true when w > 50 -> "High"
        | w, _ when w > 30 -> "Medium"
        | _ -> "Low"

    /// Identify domain from query
    member private this.IdentifyDomain(query: string) =
        let queryLower = query.ToLower()
        if queryLower.Contains("code") || queryLower.Contains("implement") || queryLower.Contains("function") then "Code"
        elif queryLower.Contains("test") || queryLower.Contains("unit") then "Testing"
        elif queryLower.Contains("deploy") || queryLower.Contains("docker") then "DevOps"
        elif queryLower.Contains("security") || queryLower.Contains("vulnerability") then "Security"
        else "General"

/// Module for computational expression instances
module MixtralComputationExpressions =
    let expertRouting = ExpertRoutingBuilder()
    let promptChain = PromptChainBuilder()

    /// Example usage documentation
    let exampleUsage = """
    // Expert routing computational expression:
    expertRouting {
        let! decision = routeToExpert query
        let! response = callExpert decision
        return response
    }

    // Prompt chaining computational expression:
    promptChain {
        let! response1 = query "Analyze code"
        let! response2 = query ("Improve: " + response1.Content)
        return response2
    }
    """
