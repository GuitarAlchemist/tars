namespace TarsEngine.FSharp.Reasoning

open System
open System.Threading.Tasks
open System.Text.Json
open Microsoft.Extensions.Logging

/// Qwen3 reasoning modes
type ReasoningMode =
    | Thinking      // Deep step-by-step reasoning
    | NonThinking   // Fast responses
    | Hybrid        // Dynamic mode switching

/// Qwen3 model specifications
type Qwen3Model =
    | Qwen3_235B_A22B   // Flagship reasoning model
    | Qwen3_30B_A3B     // Efficient MoE reasoning
    | Qwen3_32B         // Dense reasoning model
    | Qwen3_14B         // Balanced reasoning
    | Qwen3_8B          // Fast reasoning

/// Reasoning request configuration
type ReasoningRequest = {
    Problem: string
    Context: string option
    Mode: ReasoningMode
    ThinkingBudget: int option  // Computational budget for thinking
    RequiredCapabilities: string list
    Priority: int
}

/// Reasoning response with thinking process
type ReasoningResponse = {
    Problem: string
    ThinkingContent: string option  // Step-by-step reasoning process
    FinalAnswer: string
    Confidence: float
    ReasoningSteps: int
    ProcessingTime: TimeSpan
    Model: Qwen3Model
    Mode: ReasoningMode
}

/// Qwen3 reasoning engine interface
type IQwenReasoningEngine =
    abstract member ReasonAsync: ReasoningRequest -> Task<ReasoningResponse>
    abstract member SetThinkingMode: ReasoningMode -> unit
    abstract member GetAvailableModels: unit -> Qwen3Model list
    abstract member IsModelAvailable: Qwen3Model -> bool

/// Qwen3 reasoning engine implementation
type QwenReasoningEngine(logger: ILogger<QwenReasoningEngine>) =
    
    let mutable currentMode = ReasoningMode.Hybrid
    let mutable availableModels = []
    
    /// Convert Qwen3Model to string identifier
    let modelToString = function
        | Qwen3_235B_A22B -> "qwen3:235b-a22b"
        | Qwen3_30B_A3B -> "qwen3:30b-a3b"
        | Qwen3_32B -> "qwen3:32b"
        | Qwen3_14B -> "qwen3:14b"
        | Qwen3_8B -> "qwen3:8b"
    
    /// Select optimal model based on request complexity
    let selectOptimalModel (request: ReasoningRequest) =
        let complexity = 
            request.Problem.Length + 
            (request.Context |> Option.map (fun c -> c.Length) |> Option.defaultValue 0) +
            (request.RequiredCapabilities.Length * 100)
        
        match complexity with
        | c when c > 5000 -> Qwen3_235B_A22B  // Complex problems
        | c when c > 2000 -> Qwen3_30B_A3B    // Moderate complexity
        | c when c > 1000 -> Qwen3_32B        // Standard problems
        | c when c > 500 -> Qwen3_14B         // Simple problems
        | _ -> Qwen3_8B                       // Quick responses
    
    /// Format prompt for Qwen3 with thinking mode control
    let formatPrompt (request: ReasoningRequest) (model: Qwen3Model) =
        let thinkingControl = 
            match request.Mode with
            | Thinking -> "/think"
            | NonThinking -> "/no_think"
            | Hybrid -> ""  // Let model decide
        
        let contextPart = 
            request.Context 
            |> Option.map (fun c -> $"Context: {c}\n\n")
            |> Option.defaultValue ""
        
        let capabilitiesPart =
            if request.RequiredCapabilities.IsEmpty then ""
            else $"Required capabilities: {String.Join(", ", request.RequiredCapabilities)}\n\n"
        
        $"{contextPart}{capabilitiesPart}Problem: {request.Problem} {thinkingControl}"
    
    /// Parse Qwen3 response to extract thinking and final answer
    let parseResponse (rawResponse: string) =
        // Look for thinking tags in Qwen3 format
        let thinkingPattern = @"<think>(.*?)</think>"
        let regex = System.Text.RegularExpressions.Regex(thinkingPattern, System.Text.RegularExpressions.RegexOptions.Singleline)
        let matches = regex.Matches(rawResponse)
        
        if matches.Count > 0 then
            let thinkingContent = matches.[0].Groups.[1].Value.Trim()
            let finalAnswer = regex.Replace(rawResponse, "").Trim()
            (Some thinkingContent, finalAnswer)
        else
            (None, rawResponse.Trim())
    
    /// Call Qwen3 model via Ollama API
    let callQwen3Model (model: Qwen3Model) (prompt: string) = async {
        try
            let modelName = modelToString model
            logger.LogInformation($"Calling Qwen3 model: {modelName}")
            
            // Simulate Ollama API call (replace with actual HTTP client)
            let requestBody = JsonSerializer.Serialize({|
                model = modelName
                prompt = prompt
                stream = false
                options = {|
                    temperature = 0.3
                    top_p = 0.9
                    max_tokens = 32768
                |}
            |})
            
            // TODO: Replace with actual HTTP client call to Ollama
            // REAL IMPLEMENTATION NEEDED
            let simulatedResponse = $"<think>Let me analyze this problem step by step...</think>Based on my analysis, the answer is: [Simulated response for {modelName}]"
            
            return simulatedResponse
        with
        | ex ->
            logger.LogError(ex, $"Error calling Qwen3 model: {model}")
            return $"Error: {ex.Message}"
    }
    
    /// Initialize available models by checking Ollama
    member private this.InitializeAvailableModels() = async {
        try
            // TODO: Check which Qwen3 models are available in Ollama
            // For now, assume all models are available
            availableModels <- [
                Qwen3_8B
                Qwen3_14B
                Qwen3_32B
                Qwen3_30B_A3B
                Qwen3_235B_A22B
            ]
            logger.LogInformation($"Initialized {availableModels.Length} Qwen3 models")
        with
        | ex ->
            logger.LogError(ex, "Failed to initialize Qwen3 models")
            availableModels <- []
    }
    
    interface IQwenReasoningEngine with
        
        member this.ReasonAsync(request: ReasoningRequest) = task {
            let startTime = DateTime.UtcNow
            
            try
                // Select optimal model for the request
                let selectedModel = selectOptimalModel request
                
                // Check if model is available
                if not (availableModels |> List.contains selectedModel) then
                    logger.LogWarning($"Requested model {selectedModel} not available, falling back to Qwen3_8B")
                    let fallbackModel = Qwen3_8B
                    
                    if not (availableModels |> List.contains fallbackModel) then
                        failwith "No Qwen3 models available"
                
                // Format prompt for Qwen3
                let prompt = formatPrompt request selectedModel
                logger.LogDebug($"Formatted prompt: {prompt}")
                
                // Call Qwen3 model
                let! rawResponse = callQwen3Model selectedModel prompt
                
                // Parse response
                let (thinkingContent, finalAnswer) = parseResponse rawResponse
                
                let processingTime = DateTime.UtcNow - startTime
                
                // Calculate confidence based on response quality
                let confidence = 
                    match thinkingContent with
                    | Some thinking when thinking.Length > 100 -> 0.9
                    | Some _ -> 0.7
                    | None -> 0.6
                
                let reasoningSteps = 
                    thinkingContent
                    |> Option.map (fun t -> t.Split([|'\n'; '.'; ';'|], StringSplitOptions.RemoveEmptyEntries).Length)
                    |> Option.defaultValue 1
                
                return {
                    Problem = request.Problem
                    ThinkingContent = thinkingContent
                    FinalAnswer = finalAnswer
                    Confidence = confidence
                    ReasoningSteps = reasoningSteps
                    ProcessingTime = processingTime
                    Model = selectedModel
                    Mode = request.Mode
                }
                
            with
            | ex ->
                logger.LogError(ex, $"Error processing reasoning request: {request.Problem}")
                let processingTime = DateTime.UtcNow - startTime
                
                return {
                    Problem = request.Problem
                    ThinkingContent = Some $"Error occurred: {ex.Message}"
                    FinalAnswer = $"Unable to process request: {ex.Message}"
                    Confidence = 0.0
                    ReasoningSteps = 0
                    ProcessingTime = processingTime
                    Model = Qwen3_8B
                    Mode = request.Mode
                }
        }
        
        member this.SetThinkingMode(mode: ReasoningMode) =
            currentMode <- mode
            logger.LogInformation($"Reasoning mode set to: {mode}")
        
        member this.GetAvailableModels() =
            availableModels
        
        member this.IsModelAvailable(model: Qwen3Model) =
            availableModels |> List.contains model
    
    /// Initialize the reasoning engine
    member this.InitializeAsync() = async {
        logger.LogInformation("Initializing Qwen3 Reasoning Engine...")
        do! this.InitializeAvailableModels()
        logger.LogInformation("Qwen3 Reasoning Engine initialized successfully")
    }

/// Reasoning agent specializations
type ReasoningSpecialization =
    | MathematicalReasoning
    | LogicalReasoning
    | CausalReasoning
    | StrategicReasoning
    | MetaReasoning
    | CollaborativeReasoning

/// Specialized reasoning agent
type ReasoningAgent = {
    Id: string
    Name: string
    Specialization: ReasoningSpecialization
    Model: Qwen3Model
    Engine: IQwenReasoningEngine
    Capabilities: string list
}

/// Factory for creating Qwen3 reasoning engines
module QwenReasoningEngineFactory =

    let create (logger: ILogger<QwenReasoningEngine>) =
        let engine = new QwenReasoningEngine(logger)
        engine.InitializeAsync() |> Async.RunSynchronously
        engine :> IQwenReasoningEngine

    let createSpecializedAgent (specialization: ReasoningSpecialization) (logger: ILogger<QwenReasoningEngine>) =
        let engine = create logger
        let (name, model, capabilities) =
            match specialization with
            | MathematicalReasoning ->
                ("Mathematical Reasoner", Qwen3_30B_A3B, ["proofs"; "calculations"; "optimization"; "statistics"])
            | LogicalReasoning ->
                ("Logical Reasoner", Qwen3_32B, ["deduction"; "induction"; "formal_logic"; "consistency_checking"])
            | CausalReasoning ->
                ("Causal Reasoner", Qwen3_14B, ["cause_effect"; "root_cause_analysis"; "system_dynamics"])
            | StrategicReasoning ->
                ("Strategic Reasoner", Qwen3_8B, ["decision_trees"; "game_theory"; "optimization"; "resource_allocation"])
            | MetaReasoning ->
                ("Meta Reasoner", Qwen3_14B, ["reasoning_strategy"; "cognitive_bias_detection"; "quality_evaluation"])
            | CollaborativeReasoning ->
                ("Collaborative Reasoner", Qwen3_235B_A22B, ["consensus_building"; "distributed_solving"; "knowledge_synthesis"])

        {
            Id = Guid.NewGuid().ToString()
            Name = name
            Specialization = specialization
            Model = model
            Engine = engine
            Capabilities = capabilities
        }

