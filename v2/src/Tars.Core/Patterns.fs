namespace Tars.Core

open System
open Tars.Core.AgentWorkflow

module Patterns =

    /// Chain of Thought (CoT) Pattern
    /// Executes a linear sequence of reasoning steps.
    let chainOfThought (steps: (string -> AgentWorkflow<string>) list) (initialInput: string) : AgentWorkflow<string> =
        let rec loop remainingSteps currentInput =
            match remainingSteps with
            | [] -> succeed currentInput
            | step :: rest ->
                agent {
                    let! result = step currentInput
                    return! loop rest result
                }

        loop steps initialInput

    /// ReAct (Reason + Act) Pattern
    /// Interleaves reasoning and action execution in a loop until a termination condition is met.
    let reAct (maxSteps: int) (goal: string) : AgentWorkflow<string> =
        let rec loop stepCount context =
            agent {
                if stepCount >= maxSteps then
                    return! fail (PartialFailure.Warning "ReAct loop reached max steps")
                else
                    // 1. Reason (Thought)
                    // In a real implementation, this would call the LLM to generate a thought
                    let! thought = succeed $"Thought {stepCount}: Analyzing {goal}..."

                    // 2. Act (Action)
                    // This would call the LLM to select a tool and arguments
                    let! action = succeed "Action: Search(query='...')"

                    // 3. Observe (Observation)
                    // This would execute the tool
                    let! observation = succeed "Observation: Results found..."

                    // Check for completion
                    if observation.Contains("Final Answer") then
                        return observation
                    else
                        return! loop (stepCount + 1) (context + "\n" + observation)
            }

        loop 0 goal

    /// Plan & Execute Pattern
    /// Generates a high-level plan and then executes it step-by-step.
    let planAndExecute
        (planner: AgentWorkflow<string list>)
        (executor: string -> AgentWorkflow<string>)
        : AgentWorkflow<string list> =
        agent {
            // 1. Generate Plan
            let! plan = planner

            // 2. Execute Plan
            // We use aggregateResults to run steps, potentially in parallel if the executor supports it
            // For sequential execution, we'd use a fold/loop
            let! results = plan |> List.map executor |> aggregateResults

            return results
        }
