namespace Tars.Evolution

module CurriculumManager =

    /// Initialize a fresh curriculum state
    let init () =
        { CompletedProblems = Set.empty
          FailedProblems = Map.empty
          CurrentDifficulty = Beginner
          MasteryScore = 0.0 }

    /// Determine if the agent is ready for the next difficulty level
    let checkPromotion (state: CurriculumState) =
        // Simple heuristic: > 5 problems solved and Mastery > 0.8
        if state.CompletedProblems.Count > 5 && state.MasteryScore > 0.8 then
            match state.CurrentDifficulty with
            | Beginner -> Intermediate
            | Intermediate -> Advanced
            | Advanced -> Expert
            | Expert -> Expert
            | Unascertained -> Beginner
        else
            state.CurrentDifficulty

    /// Select the next best problem for the agent
    let getNextProblem (state: CurriculumState) (allProblems: Problem list) : Problem option =
        // Promotion check
        let difficulty = checkPromotion state

        // Filter out completed
        let available =
            allProblems
            |> List.filter (fun p -> not (state.CompletedProblems.Contains p.Id))

        // Strategy: Get problems of current difficulty first
        let relevant = available |> List.filter (fun p -> p.Difficulty = difficulty)

        match relevant with
        | head :: _ -> Some head
        | [] ->
            // If no problems at exact difficulty, try easier (re-enforce) then harder (challenge)
            let lower = available |> List.filter (fun p -> p.Difficulty < difficulty)

            match lower with
            | h :: _ -> Some h
            | [] -> available |> List.tryHead

    /// Record a successful problem completion
    let recordSuccess (state: CurriculumState) (problemId: ProblemId) =
        let newScore = min 1.0 (state.MasteryScore + 0.1) // Simple increment

        { state with
            CompletedProblems = state.CompletedProblems.Add problemId
            MasteryScore = newScore
            CurrentDifficulty =
                checkPromotion
                    { state with
                        CompletedProblems = state.CompletedProblems.Add problemId
                        MasteryScore = newScore } }

    /// Record a failure
    let recordFailure (state: CurriculumState) (problemId: ProblemId) =
        let currentFailures =
            state.FailedProblems |> Map.tryFind problemId |> Option.defaultValue 0

        { state with
            FailedProblems = state.FailedProblems.Add(problemId, currentFailures + 1)
            MasteryScore = max 0.0 (state.MasteryScore - 0.05) }
