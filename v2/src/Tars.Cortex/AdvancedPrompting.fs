/// <summary>
/// Phase 8: Advanced Prompting Techniques
/// =========================================
/// Implements state-of-the-art prompting patterns for improved LLM reasoning:
/// - Self-Ask: Decompose questions into sub-questions
/// - Least-to-Most: Solve simple sub-problems first
/// - Generate Knowledge: Generate relevant facts before answering
/// - Directional Stimulus: Provide hints to guide reasoning
/// - PAL (Program-Aided Language Models): Generate code to solve problems
/// </summary>
namespace Tars.Cortex

open System
open System.Text.RegularExpressions
open Tars.Core
open Tars.Llm
open Tars.Llm.LlmService

/// Advanced prompting strategies for enhanced reasoning
module AdvancedPrompting =

    // =========================================================================
    // Core Types
    // =========================================================================

    /// A sub-question generated during Self-Ask
    type SubQuestion = {
        Question: string
        Answer: string option
        Confidence: float
    }

    /// Result of a prompting technique
    type PromptResult = {
        FinalAnswer: string
        IntermediateSteps: string list
        TotalTokens: int
        Strategy: string
    }

    /// Configuration for prompting strategies
    type PromptConfig = {
        MaxSubQuestions: int
        MaxIterations: int
        Temperature: float
        Verbose: bool
    }

    let defaultConfig = {
        MaxSubQuestions = 5
        MaxIterations = 5
        Temperature = 0.3
        Verbose = false
    }

    // =========================================================================
    // Self-Ask Prompting
    // =========================================================================
    /// Decomposes complex questions into simpler sub-questions and answers them
    
    let private selfAskSystemPrompt = """You are an expert at breaking down complex questions.
When given a question, determine if you need to ask follow-up questions to answer it.

Format your response as:
- If you need more information: "Follow-up: [your follow-up question]"
- If you can answer directly: "Final Answer: [your answer]"

Always think step by step. Only ask one follow-up question at a time."""

    let private parseFollowUp (response: string) =
        let followUpMatch = Regex.Match(response, @"Follow-up:\s*(.+?)(?:\n|$)", RegexOptions.IgnoreCase)
        let answerMatch = Regex.Match(response, @"Final Answer:\s*(.+)", RegexOptions.IgnoreCase ||| RegexOptions.Singleline)
        
        if answerMatch.Success then
            None, Some (answerMatch.Groups.[1].Value.Trim())
        elif followUpMatch.Success then
            Some (followUpMatch.Groups.[1].Value.Trim()), None
        else
            None, Some response // Treat as final answer if no format detected

    /// Execute Self-Ask prompting pattern
    let selfAsk (llm: ILlmService) (config: PromptConfig) (question: string) =
        task {
            let mutable subQuestions: SubQuestion list = []
            let mutable finalAnswer = None
            let mutable iteration = 0
            let mutable totalTokens = 0
            let mutable context = $"Original Question: {question}\n\n"

            while finalAnswer.IsNone && iteration < config.MaxIterations do
                iteration <- iteration + 1
                
                let prompt = context + "Based on the above, what follow-up question do you need to ask, or can you provide the final answer?"
                
                let! response =
                    Prompt.ask prompt
                    |> Prompt.withSystem selfAskSystemPrompt
                    |> Prompt.withTemp config.Temperature
                    |> Prompt.withHint "reasoning"
                    |> Prompt.complete llm
                totalTokens <- totalTokens + (response.Usage |> Option.map (fun u -> u.TotalTokens) |> Option.defaultValue 0)
                
                let followUp, answer = parseFollowUp response.Text
                
                match followUp with
                | Some q ->
                    // Answer the follow-up question
                    let! followUpResponse =
                        Prompt.ask q
                        |> Prompt.withTemp config.Temperature
                        |> Prompt.complete llm
                    totalTokens <- totalTokens + (followUpResponse.Usage |> Option.map (fun u -> u.TotalTokens) |> Option.defaultValue 0)
                    
                    subQuestions <- subQuestions @ [{
                        Question = q
                        Answer = Some followUpResponse.Text
                        Confidence = 0.8
                    }]
                    context <- context + $"Q: {q}\nA: {followUpResponse.Text}\n\n"
                | None -> ()
                
                match answer with
                | Some a -> finalAnswer <- Some a
                | None -> ()

            return {
                FinalAnswer = finalAnswer |> Option.defaultValue "Unable to determine answer"
                IntermediateSteps = 
                    subQuestions 
                    |> List.map (fun sq -> 
                        let answer = sq.Answer |> Option.defaultValue ""
                        sprintf "Q: %s\nA: %s" sq.Question answer)
                TotalTokens = totalTokens
                Strategy = "Self-Ask"
            }
        }

    // =========================================================================
    // Least-to-Most Prompting
    // =========================================================================
    /// Solves problems by decomposing into simpler sub-problems and solving in order
    
    let private decomposePrompt = """Break down this problem into a list of simpler sub-problems.
Output ONLY a numbered list of sub-problems, from simplest to most complex.
Each sub-problem should be solvable independently or with knowledge from previous sub-problems.

Format:
1. [simplest sub-problem]
2. [next sub-problem, building on 1]
3. [and so on...]"""

    let private parseDecomposition (response: string) =
        let lines = response.Split([|'\n'|], StringSplitOptions.RemoveEmptyEntries)
        lines 
        |> Array.choose (fun line ->
            let m = Regex.Match(line, @"^\d+\.\s*(.+)$")
            if m.Success then Some (m.Groups.[1].Value.Trim())
            else None)
        |> Array.toList

    /// Execute Least-to-Most prompting pattern
    let leastToMost (llm: ILlmService) (config: PromptConfig) (problem: string) =
        task {
            let mutable totalTokens = 0
            let mutable steps: string list = []
            
            // Step 1: Decompose the problem
            let! decomposeResponse =
                Prompt.ask problem
                |> Prompt.withSystem decomposePrompt
                |> Prompt.withTemp config.Temperature
                |> Prompt.complete llm
            totalTokens <- totalTokens + (decomposeResponse.Usage |> Option.map (fun u -> u.TotalTokens) |> Option.defaultValue 0)
            
            let subProblems = parseDecomposition decomposeResponse.Text
            steps <- steps @ [$"Decomposed into {subProblems.Length} sub-problems"]
            
            // Step 2: Solve each sub-problem in order
            let mutable context = $"Main Problem: {problem}\n\nSolutions so far:\n"
            let mutable solutions: string list = []
            
            for i, subProblem in subProblems |> List.indexed do
                let solvePrompt = context + $"\nNow solve this sub-problem: {subProblem}"
                
                let! solveResponse =
                    Prompt.ask solvePrompt
                    |> Prompt.withTemp config.Temperature
                    |> Prompt.complete llm
                totalTokens <- totalTokens + (solveResponse.Usage |> Option.map (fun u -> u.TotalTokens) |> Option.defaultValue 0)
                
                solutions <- solutions @ [solveResponse.Text]
                context <- context + $"\n{i+1}. {subProblem}: {solveResponse.Text}"
                steps <- steps @ [$"Solved: {subProblem}"]
            
            // Step 3: Synthesize final answer
            let synthesizePrompt = context + "\n\nBased on solving all sub-problems above, provide the final complete answer to the original problem."
            
            let! synthesizeResponse =
                Prompt.ask synthesizePrompt
                |> Prompt.withTemp config.Temperature
                |> Prompt.complete llm
            totalTokens <- totalTokens + (synthesizeResponse.Usage |> Option.map (fun u -> u.TotalTokens) |> Option.defaultValue 0)
            
            return {
                FinalAnswer = synthesizeResponse.Text
                IntermediateSteps = steps @ solutions
                TotalTokens = totalTokens
                Strategy = "Least-to-Most"
            }
        }

    // =========================================================================
    // Generate Knowledge Prompting
    // =========================================================================
    /// Generates relevant background knowledge before answering
    
    let private knowledgeGenPrompt = """Generate 3-5 relevant facts or pieces of background knowledge that would help answer this question.
Output ONLY the facts, one per line, prefixed with "FACT:".

Example:
FACT: The capital of France is Paris.
FACT: Paris has a population of about 2.1 million in the city proper."""

    let private parseKnowledge (response: string) =
        response.Split([|'\n'|], StringSplitOptions.RemoveEmptyEntries)
        |> Array.choose (fun line ->
            let m = Regex.Match(line, @"^FACT:\s*(.+)$", RegexOptions.IgnoreCase)
            if m.Success then Some (m.Groups.[1].Value.Trim())
            else None)
        |> Array.toList

    /// Execute Generate Knowledge prompting pattern
    let generateKnowledge (llm: ILlmService) (config: PromptConfig) (question: string) =
        task {
            let mutable totalTokens = 0
            
            // Step 1: Generate relevant knowledge
            let! knowledgeResponse =
                Prompt.ask question
                |> Prompt.withSystem knowledgeGenPrompt
                |> Prompt.withTemp 0.5 // Slightly higher for creativity
                |> Prompt.complete llm
            totalTokens <- totalTokens + (knowledgeResponse.Usage |> Option.map (fun u -> u.TotalTokens) |> Option.defaultValue 0)
            
            let facts = parseKnowledge knowledgeResponse.Text
            
            // Step 2: Answer using the generated knowledge
            let knowledgeContext = facts |> String.concat "\n- "
            let answerPrompt = $"""Using the following background knowledge:
- {knowledgeContext}

Please answer this question: {question}"""
            
            let! answerResponse =
                Prompt.ask answerPrompt
                |> Prompt.withTemp config.Temperature
                |> Prompt.complete llm
            totalTokens <- totalTokens + (answerResponse.Usage |> Option.map (fun u -> u.TotalTokens) |> Option.defaultValue 0)
            
            return {
                FinalAnswer = answerResponse.Text
                IntermediateSteps = ["Generated Knowledge:"] @ (facts |> List.map (fun f -> $"  - {f}"))
                TotalTokens = totalTokens
                Strategy = "Generate-Knowledge"
            }
        }

    // =========================================================================
    // Directional Stimulus Prompting
    // =========================================================================
    /// Provides hints or directions to guide the LLM's reasoning
    
    /// Execute Directional Stimulus prompting with custom hints
    let directionalStimulus (llm: ILlmService) (config: PromptConfig) (question: string) (hints: string list) =
        task {
            let hintsText = hints |> List.mapi (fun i h -> $"{i+1}. {h}") |> String.concat "\n"
            
            let prompt = $"""Question: {question}

Consider these hints while formulating your answer:
{hintsText}

Using the hints above to guide your reasoning, provide a thorough answer."""
            
            let! response =
                Prompt.ask prompt
                |> Prompt.withTemp config.Temperature
                |> Prompt.withHint "reasoning"
                |> Prompt.complete llm
            let tokens = response.Usage |> Option.map (fun u -> u.TotalTokens) |> Option.defaultValue 0
            
            return {
                FinalAnswer = response.Text
                IntermediateSteps = ["Applied Hints:"] @ hints
                TotalTokens = tokens
                Strategy = "Directional-Stimulus"
            }
        }

    // =========================================================================
    // PAL (Program-Aided Language Models)
    // =========================================================================
    /// Generates code to solve problems programmatically
    
    let private palPrompt = """You are a programmer solving problems by writing code.
For the given problem, write a complete, executable F# script that solves it.
The script should print the final answer.

Output ONLY the F# code wrapped in ```fsharp``` tags.
Do not include explanations before or after the code."""

    let private extractCode (response: string) =
        let m = Regex.Match(response, @"```fsharp\s*([\s\S]*?)```")
        if m.Success then Some (m.Groups.[1].Value.Trim())
        else None

    /// Execute PAL prompting pattern (generates code, does not execute)
    let programAided (llm: ILlmService) (config: PromptConfig) (problem: string) =
        task {
            let! response =
                Prompt.ask problem
                |> Prompt.withSystem palPrompt
                |> Prompt.withTemp 0.2 // Lower for code generation
                |> Prompt.withHint "coding"
                |> Prompt.complete llm
            let tokens = response.Usage |> Option.map (fun u -> u.TotalTokens) |> Option.defaultValue 0
            
            let code = extractCode response.Text
            
            return {
                FinalAnswer = code |> Option.defaultValue response.Text
                IntermediateSteps = ["Generated F# code to solve the problem"]
                TotalTokens = tokens
                Strategy = "PAL"
            }
        }

    // =========================================================================
    // Meta-Prompting (Prompt about Prompting)
    // =========================================================================
    /// Asks the LLM to design the optimal prompt for a task
    
    let metaPrompt (llm: ILlmService) (config: PromptConfig) (taskDescription: string) =
        task {
            let! metaResponse =
                Prompt.ask (sprintf "Design a detailed, effective prompt for this task:\n\n%s\n\nOutput the prompt in a <prompt></prompt> block." taskDescription)
                |> Prompt.withSystem "You are an expert prompt engineer. Design the optimal prompt for the given task."
                |> Prompt.withTemp 0.4
                |> Prompt.complete llm
            let tokens = metaResponse.Usage |> Option.map (fun u -> u.TotalTokens) |> Option.defaultValue 0
            
            // Extract the designed prompt
            let promptMatch = Regex.Match(metaResponse.Text, @"<prompt>([\s\S]*?)</prompt>")
            let designedPrompt = 
                if promptMatch.Success then promptMatch.Groups.[1].Value.Trim()
                else metaResponse.Text
            
            // Execute the designed prompt
            let! executeResponse =
                Prompt.ask designedPrompt
                |> Prompt.withTemp config.Temperature
                |> Prompt.complete llm
            let totalTokens = tokens + (executeResponse.Usage |> Option.map (fun u -> u.TotalTokens) |> Option.defaultValue 0)
            
            return {
                FinalAnswer = executeResponse.Text
                IntermediateSteps = [sprintf "Designed Prompt:\n%s" designedPrompt]
                TotalTokens = totalTokens
                Strategy = "Meta-Prompt"
            }
        }

    // =========================================================================
    // Utility: Auto-Select Strategy
    // =========================================================================
    
    /// Analyzes a question and selects the best prompting strategy
    let autoSelectStrategy (question: string) =
        let lower = question.ToLowerInvariant()
        
        if lower.Contains("calculate") || lower.Contains("compute") || lower.Contains("math") then
            "PAL" // Use code for calculations
        elif lower.Contains("why") || lower.Contains("explain") || lower.Contains("how does") then
            "Self-Ask" // Decompose explanatory questions
        elif lower.Contains("list") || lower.Contains("steps") || lower.Contains("procedure") then
            "Least-to-Most" // Break down procedural questions
        elif lower.Contains("what is") || lower.Contains("define") || lower.Contains("who is") then
            "Generate-Knowledge" // Fact-based questions
        else
            "Self-Ask" // Default to Self-Ask for complex questions
