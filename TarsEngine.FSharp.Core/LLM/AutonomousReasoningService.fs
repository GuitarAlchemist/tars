namespace TarsEngine.FSharp.Core.LLM

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.ChromaDB

/// Autonomous reasoning service combining Codestral LLM with ChromaDB RAG
type AutonomousReasoningService(llmClient: ILLMClient, hybridRAG: IHybridRAGService, logger: ILogger<AutonomousReasoningService>) =
    
    let createSystemPrompt() = """You are TARS, an autonomous F# development AI assistant with the following capabilities:

1. **Code Generation**: Generate high-quality F# code following best practices
2. **Code Analysis**: Analyze code for quality, performance, and security
3. **Metascript Creation**: Create TARS metascripts for automation
4. **Autonomous Reasoning**: Make intelligent decisions based on context
5. **Knowledge Integration**: Use RAG to incorporate relevant knowledge

Guidelines:
- Always prioritize clean, maintainable F# code
- Use functional programming principles
- Consider performance and security implications
- Provide clear explanations for decisions
- Suggest improvements when possible
- Be honest about limitations and uncertainties"""

    interface IAutonomousReasoningService with
        member _.ReasonAboutTaskAsync(task: string, context: Map<string, obj>) =
            task {
                try
                    logger.LogInformation("Starting autonomous reasoning for task: {Task}", task)
                    
                    // Retrieve relevant knowledge from RAG
                    printfn "🧠 Retrieving relevant knowledge..."
                    let! relevantKnowledge = hybridRAG.RetrieveKnowledgeAsync(task, 3)
                    
                    // Build context with RAG knowledge
                    let knowledgeContext = 
                        relevantKnowledge
                        |> List.map (fun doc -> sprintf "- %s" doc.Content)
                        |> String.concat "\n"
                    
                    let contextInfo = 
                        context
                        |> Map.toList
                        |> List.map (fun (k, v) -> sprintf "%s: %A" k v)
                        |> String.concat "\n"
                    
                    let prompt = sprintf """Task: %s

Context:
%s

Relevant Knowledge:
%s

Please reason about this task and provide:
1. Analysis of the task requirements
2. Recommended approach
3. Potential challenges
4. Success criteria
5. Next steps""" task contextInfo knowledgeContext
                    
                    // Use LLM for reasoning
                    let llmContext = llmClient.CreateContextAsync(Some (createSystemPrompt()))
                    let! response = llmClient.SendMessageAsync(llmContext, prompt)
                    
                    // Store the reasoning result in RAG for future use
                    let metadata = Map.ofList [
                        ("type", "reasoning" :> obj)
                        ("task", task :> obj)
                        ("timestamp", DateTime.UtcNow :> obj)
                    ]
                    let! _ = hybridRAG.StoreKnowledgeAsync(response.Content, metadata)
                    
                    logger.LogInformation("Autonomous reasoning completed for task: {Task}", task)
                    return response.Content
                with
                | ex ->
                    logger.LogError(ex, "Failed to reason about task: {Task}", task)
                    return sprintf "Error in autonomous reasoning: %s" ex.Message
            }
        
        member _.GenerateMetascriptAsync(objective: string, context: Map<string, obj>) =
            task {
                try
                    logger.LogInformation("Generating metascript for objective: {Objective}", objective)
                    
                    // Retrieve similar metascripts from RAG
                    let! similarMetascripts = hybridRAG.SearchSimilarAsync(objective, 2)
                    
                    let examplesContext = 
                        similarMetascripts
                        |> List.map (fun doc -> sprintf "Example: %s" doc.Content)
                        |> String.concat "\n\n"
                    
                    let prompt = sprintf """Generate a TARS metascript for the following objective:

Objective: %s

Context: %A

Similar Examples:
%s

Please create a complete .tars metascript with:
1. DESCRIBE block with name, version, description
2. CONFIG block with model and parameters
3. VARIABLE blocks for any needed variables
4. ACTION blocks for logging and operations
5. FSHARP block for F# code execution

Follow TARS metascript DSL syntax exactly.""" objective context examplesContext
                    
                    let request = {
                        Context = llmClient.CreateContextAsync(Some (createSystemPrompt()))
                        CodeContext = Some "TARS Metascript DSL"
                        Language = Some "TARS"
                        Task = sprintf "Generate metascript for: %s" objective
                    }
                    
                    let! response = llmClient.GenerateCodeAsync(request)
                    
                    // Store the generated metascript in RAG
                    let metadata = Map.ofList [
                        ("type", "generated_metascript" :> obj)
                        ("objective", objective :> obj)
                        ("confidence", response.Confidence :> obj)
                    ]
                    let! _ = hybridRAG.StoreKnowledgeAsync(response.GeneratedCode, metadata)
                    
                    logger.LogInformation("Metascript generation completed with confidence: {Confidence}", response.Confidence)
                    return response.GeneratedCode
                with
                | ex ->
                    logger.LogError(ex, "Failed to generate metascript for objective: {Objective}", objective)
                    return sprintf "Error generating metascript: %s" ex.Message
            }
        
        member _.AnalyzeAndImproveAsync(code: string, language: string) =
            task {
                try
                    logger.LogInformation("Analyzing and improving {Language} code", language)
                    
                    // Analyze the code with LLM
                    let! analysis = llmClient.AnalyzeCodeAsync(code, language)
                    
                    // Generate improvement suggestions
                    let improvementPrompt = sprintf """Based on this code analysis:

%s

Original Code:
```%s
%s
```

Please provide:
1. Specific improvement recommendations
2. Refactored code if applicable
3. Performance optimizations
4. Security considerations
5. Best practice suggestions""" analysis.Content language code
                    
                    let llmContext = llmClient.CreateContextAsync(Some (createSystemPrompt()))
                    let! improvements = llmClient.SendMessageAsync(llmContext, improvementPrompt)
                    
                    let result = sprintf """# Code Analysis and Improvement Report

## Original Analysis
%s

## Improvement Recommendations
%s

Generated by TARS Autonomous Reasoning with Codestral LLM + ChromaDB RAG""" analysis.Content improvements.Content
                    
                    // Store analysis in RAG for future reference
                    let metadata = Map.ofList [
                        ("type", "code_analysis" :> obj)
                        ("language", language :> obj)
                        ("code_length", code.Length :> obj)
                    ]
                    let! _ = hybridRAG.StoreKnowledgeAsync(result, metadata)
                    
                    logger.LogInformation("Code analysis and improvement completed")
                    return result
                with
                | ex ->
                    logger.LogError(ex, "Failed to analyze and improve code")
                    return sprintf "Error in code analysis: %s" ex.Message
            }
        
        member _.MakeDecisionAsync(options: string list, criteria: string) =
            task {
                try
                    logger.LogInformation("Making autonomous decision with {OptionCount} options", options.Length)
                    
                    // Retrieve relevant decision-making knowledge
                    let! relevantDecisions = hybridRAG.RetrieveKnowledgeAsync(criteria, 2)
                    
                    let pastDecisions = 
                        relevantDecisions
                        |> List.map (fun doc -> sprintf "Past decision: %s" doc.Content)
                        |> String.concat "\n"
                    
                    let optionsText = 
                        options
                        |> List.mapi (fun i opt -> sprintf "%d. %s" (i + 1) opt)
                        |> String.concat "\n"
                    
                    let prompt = sprintf """Make an autonomous decision based on the following:

Criteria: %s

Options:
%s

Relevant Past Decisions:
%s

Please provide:
1. Analysis of each option
2. Recommended decision with reasoning
3. Confidence level (0-1)
4. Potential risks and mitigations
5. Success metrics""" criteria optionsText pastDecisions
                    
                    let llmContext = llmClient.CreateContextAsync(Some (createSystemPrompt()))
                    let! decision = llmClient.SendMessageAsync(llmContext, prompt)
                    
                    // Store decision in RAG for future reference
                    let metadata = Map.ofList [
                        ("type", "autonomous_decision" :> obj)
                        ("criteria", criteria :> obj)
                        ("options_count", options.Length :> obj)
                    ]
                    let! _ = hybridRAG.StoreKnowledgeAsync(decision.Content, metadata)
                    
                    logger.LogInformation("Autonomous decision completed")
                    return decision.Content
                with
                | ex ->
                    logger.LogError(ex, "Failed to make autonomous decision")
                    return sprintf "Error in decision making: %s" ex.Message
            }

