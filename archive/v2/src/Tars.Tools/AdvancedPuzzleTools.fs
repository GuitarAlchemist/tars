namespace Tars.Tools.Advanced

open System
open Tars.Core
open Tars.Core.AdvancedPuzzles

/// Advanced Puzzle Tools for Human-Level Intelligence Testing
module AdvancedPuzzleTools =

    // =========================================================================
    // STATE MANAGEMENT
    // =========================================================================
    
    let mutable private currentBongard: BongardProblem option = None
    let mutable private currentRaven: RavenMatrix option = None
    let mutable private currentGpqa: GpqaQuestion option = None
    let mutable private currentPlan: PlanProblem option = None
    let mutable private currentMath: MathProblem option = None
    let mutable private currentPlanState: string list = []
    
    let resetAll () =
        currentBongard <- None
        currentRaven <- None
        currentGpqa <- None
        currentPlan <- None
        currentMath <- None
        currentPlanState <- []
    
    // =========================================================================
    // BONGARD TOOLS
    // =========================================================================
    
    let createBongardLoadTool () =
        Tool.Create(
            "bongard_load",
            "Loads a Bongard problem. Input: sample for demo or problem ID.",
            fun (args: string) ->
                task {
                    let input = args.Trim().Trim('\'', '"').ToLowerInvariant()
                    if input = "sample" || input = "demo" then
                        currentBongard <- Some (createSampleBongard())
                        return Result.Ok "Loaded sample Bongard problem: Convex vs Concave\nUse bongard_show to see the problem."
                    else
                        // Could load from file in future
                        currentBongard <- Some (createSampleBongard())
                        return Result.Ok "Loaded Bongard problem. Use bongard_show to see it."
                }
        )
    
    let createBongardShowTool () =
        Tool.Create(
            "bongard_show",
            "Shows the current Bongard problem with left and right sets.",
            fun (_: string) ->
                task {
                    match currentBongard with
                    | None -> return Result.Error "No Bongard problem loaded. Use bongard_load first."
                    | Some problem ->
                        let sb = System.Text.StringBuilder()
                        sb.AppendLine($"=== Bongard Problem: {problem.Name} ===") |> ignore
                        sb.AppendLine($"Difficulty: {problem.Difficulty}/5") |> ignore
                        sb.AppendLine() |> ignore
                        sb.AppendLine("LEFT SET (all share property A):") |> ignore
                        for p in problem.LeftSet do
                            let features = String.Join(", ", p.Features)
                            sb.AppendLine($"  [{p.Id}] Features: {features}") |> ignore
                            for row in p.Grid do
                                let rowStr = row |> Array.map (fun v -> if v = 1 then "#" else ".") |> String.concat ""
                                sb.AppendLine($"    {rowStr}") |> ignore
                            sb.AppendLine() |> ignore
                        sb.AppendLine("RIGHT SET (all share property NOT-A):") |> ignore
                        for p in problem.RightSet do
                            let features = String.Join(", ", p.Features)
                            sb.AppendLine($"  [{p.Id}] Features: {features}") |> ignore
                            for row in p.Grid do
                                let rowStr = row |> Array.map (fun v -> if v = 1 then "#" else ".") |> String.concat ""
                                sb.AppendLine($"    {rowStr}") |> ignore
                            sb.AppendLine() |> ignore
                        sb.AppendLine("TASK: Find the rule that separates LEFT from RIGHT.") |> ignore
                        return Result.Ok (sb.ToString())
                }
        )
    
    let createBongardSolveTool () =
        Tool.Create(
            "bongard_solve",
            "Submit your hypothesis for the Bongard problem. Input: your rule description.",
            fun (args: string) ->
                task {
                    match currentBongard with
                    | None -> return Result.Error "No Bongard problem loaded."
                    | Some problem ->
                        let hypothesis = args.Trim().Trim('\'', '"').ToLowerInvariant()
                        let ruleWords = problem.Rule.ToLowerInvariant().Split([|' '; ','; ';'|], StringSplitOptions.RemoveEmptyEntries)
                        let keyWords = ["convex"; "concave"; "indent"; "curve"; "filled"; "hollow"]
                        let matchedWords = keyWords |> List.filter (fun kw -> hypothesis.Contains(kw) && problem.Rule.ToLowerInvariant().Contains(kw))
                        if matchedWords.Length >= 2 || hypothesis.Contains("convex") || hypothesis.Contains("indent") then
                            return Result.Ok $"✅ CORRECT! The rule is: {problem.Rule}"
                        else
                            return Result.Ok ("Not quite. Your hypothesis: '" + hypothesis + "'\nHint: Look at the overall shape properties.")
                }
        )
    
    // =========================================================================
    // RAVEN'S MATRIX TOOLS
    // =========================================================================
    
    let createRavenLoadTool () =
        Tool.Create(
            "raven_load",
            "Loads a Raven Progressive Matrix puzzle. Input: sample for demo.",
            fun (args: string) ->
                task {
                    currentRaven <- Some (createSampleRaven())
                    return Result.Ok "Loaded Raven Progressive Matrix puzzle.\nUse raven_show to see the matrix."
                }
        )
    
    let createRavenShowTool () =
        Tool.Create(
            "raven_show",
            "Shows the current Raven matrix with the missing cell marked.",
            fun (_: string) ->
                task {
                    match currentRaven with
                    | None -> return Result.Error "No Raven matrix loaded. Use raven_load first."
                    | Some matrix ->
                        let sb = System.Text.StringBuilder()
                        sb.AppendLine("=== Raven's Matrix: " + matrix.Name + " ===") |> ignore
                        sb.AppendLine($"Difficulty: {matrix.Difficulty}/5") |> ignore
                        sb.AppendLine() |> ignore
                        sb.AppendLine("3x3 Matrix (row by row):") |> ignore
                        for i, row in matrix.Matrix |> Array.indexed do
                            for j, cell in row |> Array.indexed do
                                if cell.Count = 0 && cell.Shapes.IsEmpty then
                                    sb.Append("  [???]  ") |> ignore
                                else
                                    let shapeStr = if cell.Shapes.IsEmpty then "?" else cell.Shapes.[0]
                                    sb.Append($"  [{cell.Count}x{shapeStr}]  ") |> ignore
                            sb.AppendLine() |> ignore
                        sb.AppendLine() |> ignore
                        sb.AppendLine("OPTIONS (choose 0-3):") |> ignore
                        for i, opt in matrix.Options |> List.indexed do
                            let shapeStr = if opt.Shapes.IsEmpty then "?" else opt.Shapes.[0]
                            sb.AppendLine($"  {i}: {opt.Count}x {shapeStr} ({opt.Shading})") |> ignore
                        return Result.Ok (sb.ToString())
                }
        )
    
    let createRavenSolveTool () =
        Tool.Create(
            "raven_solve",
            "Submit your answer for the Raven matrix. Input: option number (0-3).",
            fun (args: string) ->
                task {
                    match currentRaven with
                    | None -> return Result.Error "No Raven matrix loaded."
                    | Some matrix ->
                        match Int32.TryParse(args.Trim().Trim('\'', '"')) with
                        | true, answer when answer >= 0 && answer < matrix.Options.Length ->
                            if answer = matrix.CorrectOption then
                                let rules = String.Join("; ", matrix.Rules)
                                return Result.Ok ("CORRECT! Option " + string answer + " is right.\nRules: " + rules)
                            else
                                return Result.Ok ("Incorrect. Option " + string answer + " is wrong. Try again!")
                        | _ ->
                            return Result.Error "Invalid option. Enter a number 0-3."
                }
        )
    
    // =========================================================================
    // GPQA TOOLS
    // =========================================================================
    
    let createGpqaLoadTool () =
        Tool.Create(
            "gpqa_load",
            "Loads a GPQA (Graduate-level Science) question. Input: sample or physics/chemistry.",
            fun (args: string) ->
                task {
                    currentGpqa <- Some (createSampleGpqa())
                    match currentGpqa with
                    | Some q ->
                        return Result.Ok $"Loaded GPQA question in {q.Domain}: {q.SubDomain}\nDifficulty: {q.Difficulty}/5 (PhD level)\nUse gpqa_show to see the question."
                    | None -> return Result.Error "Failed to load question."
                }
        )
    
    let createGpqaShowTool () =
        Tool.Create(
            "gpqa_show",
            "Shows the current GPQA question with multiple choice options.",
            fun (_: string) ->
                task {
                    match currentGpqa with
                    | None -> return Result.Error "No GPQA question loaded. Use gpqa_load first."
                    | Some question ->
                        let sb = System.Text.StringBuilder()
                        sb.AppendLine($"=== GPQA: {question.Domain} - {question.SubDomain} ===") |> ignore
                        sb.AppendLine($"Difficulty: {question.Difficulty}/5 (PhD level)") |> ignore
                        sb.AppendLine() |> ignore
                        sb.AppendLine("QUESTION:") |> ignore
                        sb.AppendLine(question.Question) |> ignore
                        sb.AppendLine() |> ignore
                        sb.AppendLine("OPTIONS:") |> ignore
                        for i, opt in question.Options |> List.indexed do
                            sb.AppendLine($"  ({(char)(65 + i)}) {opt}") |> ignore
                        sb.AppendLine() |> ignore
                        sb.AppendLine("Required Knowledge: " + String.Join(", ", question.RequiredKnowledge)) |> ignore
                        return Result.Ok (sb.ToString())
                }
        )
    
    let createGpqaSolveTool () =
        Tool.Create(
            "gpqa_solve",
            "Submit your answer for the GPQA question. Input: A, B, C, or D.",
            fun (args: string) ->
                task {
                    match currentGpqa with
                    | None -> return Result.Error "No GPQA question loaded."
                    | Some question ->
                        let answer = args.Trim().Trim('\'', '"').ToUpperInvariant()
                        let answerIdx = 
                            match answer with
                            | "A" | "0" -> 0
                            | "B" | "1" -> 1
                            | "C" | "2" -> 2
                            | "D" | "3" -> 3
                            | _ -> -1
                        if answerIdx < 0 then
                            return Result.Error "Invalid answer. Use A, B, C, or D."
                        elif answerIdx = question.CorrectAnswer then
                            return Result.Ok $"✅ CORRECT! Answer is ({(char)(65 + answerIdx)})\n\nExplanation:\n{question.Explanation}"
                        else
                            return Result.Ok $"❌ Incorrect. ({(char)(65 + answerIdx)}) is wrong.\nCorrect answer: ({(char)(65 + question.CorrectAnswer)})"
                }
        )
    
    // =========================================================================
    // PLANBENCH TOOLS
    // =========================================================================
    
    let createPlanLoadTool () =
        Tool.Create(
            "plan_load",
            "Loads a planning problem. Input: sample or blocks-world.",
            fun (args: string) ->
                task {
                    let problem = createSamplePlanProblem()
                    currentPlan <- Some problem
                    currentPlanState <- problem.InitialState
                    return Result.Ok $"Loaded planning problem: {problem.Name}\nDomain: {problem.Domain}\nOptimal solution length: {problem.OptimalPlanLength}\nUse plan_show to see the state."
                }
        )
    
    let createPlanShowTool () =
        Tool.Create(
            "plan_show",
            "Shows the current planning problem state, goal, and available actions.",
            fun (_: string) ->
                task {
                    match currentPlan with
                    | None -> return Result.Error "No planning problem loaded. Use plan_load first."
                    | Some problem ->
                        let sb = System.Text.StringBuilder()
                        sb.AppendLine($"=== Planning: {problem.Name} ===") |> ignore
                        sb.AppendLine($"Domain: {problem.Domain}") |> ignore
                        sb.AppendLine() |> ignore
                        sb.AppendLine("CURRENT STATE:") |> ignore
                        for pred in currentPlanState do
                            sb.AppendLine($"  - {pred}") |> ignore
                        sb.AppendLine() |> ignore
                        sb.AppendLine("GOAL STATE:") |> ignore
                        for pred in problem.GoalState do
                            sb.AppendLine($"  - {pred}") |> ignore
                        sb.AppendLine() |> ignore
                        sb.AppendLine("AVAILABLE ACTIONS:") |> ignore
                        for action in problem.AvailableActions do
                            sb.AppendLine($"  {action.Name}") |> ignore
                            let preconds = String.Join(", ", action.Preconditions)
                            let effects = String.Join(", ", action.Effects)
                            sb.AppendLine("    Pre: " + preconds) |> ignore
                            sb.AppendLine("    Eff: " + effects) |> ignore
                        // Check if goal reached
                        let goalReached = problem.GoalState |> List.forall (fun g -> currentPlanState |> List.contains g)
                        if goalReached then
                            sb.AppendLine() |> ignore
                            sb.AppendLine("🎯 GOAL REACHED!") |> ignore
                        return Result.Ok (sb.ToString())
                }
        )
    
    let createPlanExecuteTool () =
        Tool.Create(
            "plan_execute",
            "Execute an action in the planning problem. Input: action name with parameters, e.g., pick-up(A).",
            fun (args: string) ->
                task {
                    match currentPlan with
                    | None -> return Result.Error "No planning problem loaded."
                    | Some problem ->
                        let actionStr = args.Trim().Trim('\'', '"')
                        // Simple action parsing - find matching action template
                        let matchingAction = 
                            problem.AvailableActions 
                            |> List.tryFind (fun a -> actionStr.StartsWith(a.Name.Split('(').[0]))
                        match matchingAction with
                        | None ->
                            return Result.Error $"Unknown action: {actionStr}"
                        | Some actionTemplate ->
                            // For demo, just apply effects directly (simplified)
                            let newPredicates = 
                                actionTemplate.Effects 
                                |> List.filter (fun e -> not (e.StartsWith("not(")))
                            let removedPredicates = 
                                actionTemplate.Effects 
                                |> List.filter (fun e -> e.StartsWith("not("))
                                |> List.map (fun e -> e.Substring(4, e.Length - 5))
                            let newState = 
                                currentPlanState 
                                |> List.filter (fun p -> not (removedPredicates |> List.contains p))
                                |> List.append newPredicates
                                |> List.distinct
                            currentPlanState <- newState
                            let goalReached = problem.GoalState |> List.forall (fun g -> currentPlanState |> List.contains g)
                            if goalReached then
                                return Result.Ok $"✅ Executed: {actionStr}\n🎯 GOAL REACHED! You solved the planning problem!"
                            else
                                return Result.Ok $"Executed: {actionStr}\nNew state has {currentPlanState.Length} predicates. Use plan_show to see current state."
                }
        )
    
    // =========================================================================
    // MATH COMPETITION TOOLS
    // =========================================================================
    
    let createMathLoadTool () =
        Tool.Create(
            "math_load",
            "Loads a math competition problem. Input: sample or difficulty level (amc10/aime/imo).",
            fun (args: string) ->
                task {
                    currentMath <- Some (createSampleMathProblem())
                    match currentMath with
                    | Some p ->
                        return Result.Ok $"Loaded MATH problem: {p.Level} level\nCategory: {p.Category}\nDifficulty: {p.Difficulty}/10\nUse math_show to see the problem."
                    | None -> return Result.Error "Failed to load problem."
                }
        )
    
    let createMathShowTool () =
        Tool.Create(
            "math_show",
            "Shows the current math competition problem.",
            fun (_: string) ->
                task {
                    match currentMath with
                    | None -> return Result.Error "No math problem loaded. Use math_load first."
                    | Some problem ->
                        let sb = System.Text.StringBuilder()
                        sb.AppendLine($"=== MATH: {problem.Level} - {problem.Category} ===") |> ignore
                        sb.AppendLine($"Difficulty: {problem.Difficulty}/10") |> ignore
                        sb.AppendLine() |> ignore
                        sb.AppendLine("PROBLEM:") |> ignore
                        sb.AppendLine(problem.Problem) |> ignore
                        sb.AppendLine() |> ignore
                        sb.AppendLine("Required concepts: " + String.Join(", ", problem.RequiredConcepts)) |> ignore
                        return Result.Ok (sb.ToString())
                }
        )
    
    let createMathSolveTool () =
        Tool.Create(
            "math_solve",
            "Submit your answer for the math problem. Input: your numeric answer.",
            fun (args: string) ->
                task {
                    match currentMath with
                    | None -> return Result.Error "No math problem loaded."
                    | Some problem ->
                        let answer = args.Trim().Trim('\'', '"')
                        if answer = problem.Answer then
                            return Result.Ok $"✅ CORRECT! Answer: {problem.Answer}\n\nSolution:\n{problem.Solution}"
                        else
                            return Result.Ok $"❌ Incorrect. Your answer: {answer}\nHint: Check your arithmetic and try again."
                }
        )
    
    // =========================================================================
    // GET ALL TOOLS
    // =========================================================================
    
    let getAllTools () : Tool list =
        [
            // Bongard
            createBongardLoadTool ()
            createBongardShowTool ()
            createBongardSolveTool ()
            // Raven's
            createRavenLoadTool ()
            createRavenShowTool ()
            createRavenSolveTool ()
            // GPQA
            createGpqaLoadTool ()
            createGpqaShowTool ()
            createGpqaSolveTool ()
            // PlanBench
            createPlanLoadTool ()
            createPlanShowTool ()
            createPlanExecuteTool ()
            // MATH
            createMathLoadTool ()
            createMathShowTool ()
            createMathSolveTool ()
        ]
