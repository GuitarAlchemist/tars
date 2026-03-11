namespace Tars.Core.WorkflowOfThought

open System
open System.Threading.Tasks
open Tars.Core

// =============================================================================
// PHASE 8.4.1: WoT CONTROLLER COMPONENTS
// =============================================================================
// These components steer the reasoning process (Graph-of-Thoughts).
// They effectively act as the "Operating System" for the thought graph.

module WotController =

    /// Decisions made by the Router about what to do with a node
    type RouterDecision =
        | Expand of nodeId: string * branchingFactor: int
        | Refine of nodeId: string
        | Merge of nodeIds: string list
        | Backtrack of fromNodeId: string
        | Finalize of nodeId: string
        | Escalate of nodeId: string * reason: string
        | Wait // For async operations

    // -------------------------------------------------------------------------
    // 1. PLANNER
    // -------------------------------------------------------------------------
    module Planner =
        /// Decomposition of a high-level goal into sub-goals
        type Plan =
            { Goal: string
              SubGoals: string list
              Strategy: string } // e.g. "BFS", "DFS", "A*"

        /// Heuristic to estimate plan complexity
        let estimateComplexity (goal: string) : int =
            // Stub: In reality, use LLM or keyword heuristics
            if goal.Length > 200 then 5 else 1

        /// Generates an initial plan (seeds the graph)
        let generatePlan (goal: string) (reasoner: IReasoner) (ctx: ExecContext) : Async<Result<Plan, string>> =
            async {
                // Prompt LLM to break down the goal
                let prompt =
                    $"Break down this goal into 3-5 sub-steps: {goal}. Return as JSON list."

                match! reasoner.Reason("planner", ctx, Some goal, Some prompt, None) with
                | Result.Ok res ->
                    // Stub parsing logic - normally use structured output or parser
                    let subGoals =
                        res.Content.Split('\n')
                        |> Array.filter (fun s -> s.Trim().StartsWith("-"))
                        |> Array.map (fun s -> s.Trim().TrimStart('-').Trim())
                        |> Array.toList

                    return
                        Result.Ok
                            { Goal = goal
                              SubGoals = subGoals
                              Strategy = "TreeOfThoughts" }
                | Result.Error e -> return Result.Error e
            }

    // -------------------------------------------------------------------------
    // 2. GENERATOR
    // -------------------------------------------------------------------------
    module Generator =
        /// Expands a thought node into k candidates
        let expand
            (nodeId: string)
            (k: int)
            (context: string)
            (reasoner: IReasoner)
            (ctx: ExecContext)
            : Async<Result<string list, string>> =
            async {
                // In a real implementation, we'd parallelize k calls or ask for k options
                let instruction =
                    $"Generate {k} alternative distinct approaches/thoughts for: {context}"

                match! reasoner.Reason(nodeId, ctx, None, Some instruction, None) with
                | Result.Ok res ->
                    // Naive split, real impl would use N reasoning calls
                    let thoughts =
                        res.Content.Split("---") // Assuming separator
                        |> Array.map (fun s -> s.Trim())
                        |> Array.filter (fun s -> s.Length > 0)
                        |> Array.toList

                    return Result.Ok thoughts
                | Result.Error e -> return Result.Error e
            }

    // -------------------------------------------------------------------------
    // 3. CRITIC
    // -------------------------------------------------------------------------
    module Critic =
        /// Evaluates a thought, returning a score (0.0 - 1.0) and critique
        let evaluate
            (content: string)
            (reasoner: IReasoner)
            (ctx: ExecContext)
            : Async<Result<float * string, string>> =
            async {
                let instruction =
                    "Rate this thought from 0.0 to 1.0 and explain why. Format: Score: 0.X\nReason: ..."

                match! reasoner.Reason("critic", ctx, None, Some instruction, None) with
                | Result.Ok res ->
                    // Naive parse
                    let lines = res.Content.Split('\n')
                    let scoreLine = lines |> Array.tryFind (fun l -> l.StartsWith("Score:"))

                    let score =
                        match scoreLine with
                        | Some s ->
                            match Double.TryParse(s.Replace("Score:", "").Trim()) with
                            | true, v -> v
                            | _ -> 0.5
                        | None -> 0.5

                    return Result.Ok(score, res.Content)
                | Result.Error e -> return Result.Error e
            }

    // -------------------------------------------------------------------------
    // 5. ROUTER
    // -------------------------------------------------------------------------
    module Router =

        /// Decides the next action based on hypothesis state (Volume, Score)
        let decide (h: HypothesisScore) (threshold: float) (maxVolume: int) : RouterDecision =
            if h.OverallScore >= 0.9 then
                Finalize h.Id
            elif h.Volume > maxVolume then
                // Too much thought, diminishing returns -> Force a decision or backtrack
                if h.OverallScore < 0.3 then
                    Backtrack h.Id
                else
                    Finalize h.Id
            elif h.OverallScore < threshold then
                // Poor score, try to improve or abandon?
                if h.Contradictions > 0 then
                    Backtrack h.Id // Contradicted path
                else
                    Refine h.Id // Try to fix it
            else
                // Good potential, keep exploring
                Expand(h.Id, 3)

    // -------------------------------------------------------------------------
    // 6. DISTILLER
    // -------------------------------------------------------------------------
    module Distiller =
        /// Compresses a chain of thoughts into a final answer
        let distill (history: string list) (reasoner: IReasoner) (ctx: ExecContext) : Async<Result<string, string>> =
            async {
                let text = String.Join("\n---\n", history)

                let instruction =
                    "Summarize the reasoning chain above into a high-quality final answer."

                match! reasoner.Reason("distiller", ctx, None, Some instruction, None) with
                | Result.Ok res -> return Result.Ok res.Content
                | Result.Error e -> return Result.Error e
            }
