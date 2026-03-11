/// Task Prioritization for Evolution Engine
/// Implements budget-aware task scoring and queue prioritization
namespace Tars.Evolution

/// Task prioritization helpers for budget-aware scheduling
module TaskPrioritization =

    /// Estimated cost category for a task
    type CostEstimate =
        | Cheap // Simple task, < 1000 tokens
        | Moderate // Average task, 1000-3000 tokens
        | Expensive // Complex task, > 3000 tokens

    /// Estimate the cost category based on task difficulty
    let estimateCost (task: TaskDefinition) : CostEstimate =
        match task.DifficultyLevel with
        | d when d <= 2 -> Cheap
        | d when d <= 5 -> Moderate
        | _ -> Expensive

    /// Get token multiplier based on cost estimate
    let tokensForCost (cost: CostEstimate) : int =
        match cost with
        | Cheap -> 800
        | Moderate -> 2000
        | Expensive -> 4000

    /// Calculate expected value (learning potential)
    let expectedValue (task: TaskDefinition) (completedTasks: TaskResult list) : float =
        // Base value from difficulty
        let difficultyBonus = float task.DifficultyLevel * 0.1

        // Penalty for similar completed tasks (avoid repetition)
        let similarTasks =
            completedTasks
            |> List.filter (fun t -> t.TaskGoal.Contains(task.Goal.Substring(0, min 20 task.Goal.Length)))
            |> List.length

        let noveltyBonus =
            if similarTasks = 0 then 0.5
            elif similarTasks < 3 then 0.2
            else -0.3

        // Success rate bonus
        let successRate =
            if completedTasks.IsEmpty then
                0.5
            else
                let successful = completedTasks |> List.filter (fun t -> t.Success) |> List.length
                float successful / float completedTasks.Length

        let successBonus = successRate * 0.2

        1.0 + difficultyBonus + noveltyBonus + successBonus

    /// Score a task by value/cost ratio (higher = better priority)
    let scoreTask (task: TaskDefinition) (completedTasks: TaskResult list) (remainingBudgetTokens: int option) : float =

        let value = expectedValue task completedTasks
        let costEst = estimateCost task
        let costTokens = tokensForCost costEst

        // Budget feasibility check
        let feasibilityMultiplier =
            match remainingBudgetTokens with
            | Some budget when budget < costTokens -> 0.1 // Severely penalize unaffordable tasks
            | Some budget when budget < costTokens * 2 -> 0.5 // Penalize risky tasks
            | _ -> 1.0

        // Value/Cost ratio with feasibility adjustment
        let basePriority = value / (float costTokens / 1000.0)
        basePriority * feasibilityMultiplier

    /// Prioritize a task queue based on budget efficiency
    let prioritizeQueue
        (tasks: TaskDefinition list)
        (completedTasks: TaskResult list)
        (remainingBudgetTokens: int option)
        : TaskDefinition list =

        tasks
        |> List.map (fun t ->
            let score = scoreTask t completedTasks remainingBudgetTokens
            (t, score))
        |> List.sortByDescending snd
        |> List.map fst

    /// Get budget projection for a task queue
    let projectBudget (tasks: TaskDefinition list) : int =
        tasks |> List.sumBy (fun t -> t |> estimateCost |> tokensForCost)

    /// Check if queue fits within budget
    let fitsInBudget (tasks: TaskDefinition list) (budgetTokens: int) : bool = projectBudget tasks <= budgetTokens

    /// Filter queue to only affordable tasks
    let filterAffordable (tasks: TaskDefinition list) (budgetTokens: int) : TaskDefinition list =

        tasks
        |> List.fold
            (fun (acc, remaining) task ->
                let cost = task |> estimateCost |> tokensForCost

                if cost <= remaining then
                    (task :: acc, remaining - cost)
                else
                    (acc, remaining))
            ([], budgetTokens)
        |> fst
        |> List.rev

    /// Get priority report for logging
    let priorityReport
        (tasks: TaskDefinition list)
        (completedTasks: TaskResult list)
        (remainingBudgetTokens: int option)
        : string =

        if tasks.IsEmpty then
            "[Priority] No tasks in queue"
        else
            let scored =
                tasks
                |> List.map (fun t ->
                    let score = scoreTask t completedTasks remainingBudgetTokens
                    let cost = t |> estimateCost
                    $"  %.2f{score} | %A{cost} | %s{t.Goal.Substring(0, min 40 t.Goal.Length)}")
                |> String.concat "\n"

            $"[Priority] Task Queue (Score | Cost | Goal):\n%s{scored}"
