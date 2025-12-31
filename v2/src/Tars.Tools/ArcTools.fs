namespace Tars.Tools.Arc

open System
open System.IO
open System.Text.Json
open Tars.Core
open Tars.Core.ArcTypes

/// ARC-AGI Tools for Agents
/// Provides tools for solving Abstraction and Reasoning Corpus puzzles
module ArcTools =

    /// Mutable state to hold the currently loaded ARC task
    let mutable private currentTask: ArcTask option = None
    let mutable private currentAttempts: int = 0
    let private maxAttempts = 3

    /// Reset the state for a new puzzle
    let reset () =
        currentTask <- None
        currentAttempts <- 0

    /// Create the arc_load_task tool
    let createLoadTaskTool () =
        Tool.Create(
            "arc_load_task",
            "Loads an ARC-AGI task from a JSON file. Input: path to the task JSON file.",
            fun (args: string) ->
                task {
                    try
                        let path = args.Trim().Trim('\'', '"')
                        let fullPath = 
                            if Path.IsPathRooted(path) then path
                            else Path.GetFullPath(path)
                        
                        if not (File.Exists(fullPath)) then
                            return Result.Error $"Task file not found: {path}"
                        else
                            match loadTask fullPath with
                            | FSharp.Core.Result.Ok arcTask ->
                                currentTask <- Some arcTask
                                currentAttempts <- 0
                                
                                let trainCount = arcTask.TrainingPairs.Length
                                let testCount = arcTask.TestPairs.Length
                                
                                let desc = 
                                    $"Loaded ARC task '{arcTask.Id}' with {trainCount} training examples and {testCount} test case(s).\n" +
                                    $"You have {maxAttempts} attempts to solve each test case.\n" +
                                    "Use arc_show_examples to see the training examples."
                                
                                return Result.Ok desc
                            | FSharp.Core.Result.Error e ->
                                return Result.Error $"Failed to load task: {e}"
                    with ex ->
                        return Result.Error $"Error loading task: {ex.Message}"
                }
        )

    /// Create the arc_show_examples tool
    let createShowExamplesTool () =
        Tool.Create(
            "arc_show_examples",
            "Shows the training examples for the current ARC task. Each example has an input grid and expected output grid.",
            fun (_: string) ->
                task {
                    match currentTask with
                    | None ->
                        return Result.Error "No ARC task loaded. Use arc_load_task first."
                    | Some arcTask ->
                        let sb = System.Text.StringBuilder()
                        sb.AppendLine($"=== ARC Task: {arcTask.Id} ===") |> ignore
                        sb.AppendLine($"Training Examples ({arcTask.TrainingPairs.Length}):") |> ignore
                        sb.AppendLine() |> ignore
                        
                        arcTask.TrainingPairs
                        |> List.iteri (fun i pair ->
                            sb.AppendLine($"--- Example {i + 1} ---") |> ignore
                            sb.AppendLine("Input:") |> ignore
                            sb.AppendLine(gridToString pair.Input) |> ignore
                            sb.AppendLine() |> ignore
                            sb.AppendLine("Output:") |> ignore
                            sb.AppendLine(gridToString pair.Output) |> ignore
                            sb.AppendLine() |> ignore)
                        
                        return Result.Ok (sb.ToString())
                }
        )

    /// Create the arc_show_test tool
    let createShowTestTool () =
        Tool.Create(
            "arc_show_test",
            "Shows the test input grid that needs to be solved.",
            fun (_: string) ->
                task {
                    match currentTask with
                    | None ->
                        return Result.Error "No ARC task loaded. Use arc_load_task first."
                    | Some arcTask ->
                        if arcTask.TestPairs.IsEmpty then
                            return Result.Error "No test cases in this task."
                        else
                            let testPair = arcTask.TestPairs.[0]
                            let sb = System.Text.StringBuilder()
                            sb.AppendLine($"=== Test Input ===") |> ignore
                            sb.AppendLine(gridToString testPair.Input) |> ignore
                            sb.AppendLine() |> ignore
                            
                            let (w, h) = dimensions testPair.Input
                            sb.AppendLine($"Dimensions: {w}x{h}") |> ignore
                            sb.AppendLine() |> ignore
                            sb.AppendLine(describeGrid testPair.Input |> String.concat "\n") |> ignore
                            sb.AppendLine() |> ignore
                            sb.AppendLine($"Attempts remaining: {maxAttempts - currentAttempts}") |> ignore
                            
                            return Result.Ok (sb.ToString())
                }
        )

    /// Create the arc_describe_pattern tool
    let createDescribePatternTool () =
        Tool.Create(
            "arc_describe_pattern",
            "Analyzes the training examples and describes the pattern/transformation being applied.",
            fun (_: string) ->
                task {
                    match currentTask with
                    | None ->
                        return Result.Error "No ARC task loaded. Use arc_load_task first."
                    | Some arcTask ->
                        let sb = System.Text.StringBuilder()
                        sb.AppendLine("=== Pattern Analysis ===") |> ignore
                        sb.AppendLine() |> ignore
                        
                        arcTask.TrainingPairs
                        |> List.iteri (fun i pair ->
                            sb.AppendLine($"Example {i + 1}:") |> ignore
                            
                            let inputDesc = describeGrid pair.Input
                            let outputDesc = describeGrid pair.Output
                            let transformDesc = describeTransformation pair.Input pair.Output
                            
                            sb.AppendLine("  Input properties:") |> ignore
                            inputDesc |> List.iter (fun d -> sb.AppendLine($"    - {d}") |> ignore)
                            
                            sb.AppendLine("  Output properties:") |> ignore
                            outputDesc |> List.iter (fun d -> sb.AppendLine($"    - {d}") |> ignore)
                            
                            if not transformDesc.IsEmpty then
                                sb.AppendLine("  Detected transformations:") |> ignore
                                transformDesc |> List.iter (fun d -> sb.AppendLine($"    - {d}") |> ignore)
                            
                            sb.AppendLine() |> ignore)
                        
                        // Check for consistent transformations
                        let transforms = [
                            FlipHorizontal; FlipVertical; Rotate90; Rotate180; Rotate270; 
                            Transpose; InvertColors; Identity
                        ]
                        
                        let validTransforms =
                            transforms
                            |> List.filter (fun t -> validateTransform t arcTask)
                        
                        if not validTransforms.IsEmpty then
                            sb.AppendLine("=== Valid Transformations (work on ALL examples) ===") |> ignore
                            validTransforms |> List.iter (fun t ->
                                sb.AppendLine($"  - {t}") |> ignore)
                        
                        return Result.Ok (sb.ToString())
                }
        )

    /// Create the arc_apply_transform tool
    let createApplyTransformTool () =
        Tool.Create(
            "arc_apply_transform", 
            "Applies a transformation to the test input. Valid transforms: flip_horizontal, flip_vertical, rotate_90, rotate_180, rotate_270, transpose, invert_colors, identity",
            fun (args: string) ->
                task {
                    match currentTask with
                    | None ->
                        return Result.Error "No ARC task loaded. Use arc_load_task first."
                    | Some arcTask ->
                        if arcTask.TestPairs.IsEmpty then
                            return Result.Error "No test cases in this task."
                        else
                            let transformName = args.Trim().Trim('\'', '"').ToLowerInvariant()
                            
                            let transform =
                                match transformName with
                                | "flip_horizontal" | "fliphorizontal" -> Some FlipHorizontal
                                | "flip_vertical" | "flipvertical" -> Some FlipVertical
                                | "rotate_90" | "rotate90" -> Some Rotate90
                                | "rotate_180" | "rotate180" -> Some Rotate180
                                | "rotate_270" | "rotate270" -> Some Rotate270
                                | "transpose" -> Some Transpose
                                | "invert_colors" | "invertcolors" -> Some InvertColors
                                | "identity" -> Some Identity
                                | _ -> None
                            
                            match transform with
                            | None ->
                                return Result.Error $"Unknown transformation: {transformName}. Valid: flip_horizontal, flip_vertical, rotate_90, rotate_180, rotate_270, transpose, invert_colors, identity"
                            | Some t ->
                                let testInput = arcTask.TestPairs.[0].Input
                                match applyTransform t testInput with
                                | Success result ->
                                    let sb = System.Text.StringBuilder()
                                    sb.AppendLine($"Applied {t} transformation:") |> ignore
                                    sb.AppendLine() |> ignore
                                    sb.AppendLine("Result:") |> ignore
                                    sb.AppendLine(gridToString result) |> ignore
                                    return Result.Ok (sb.ToString())
                                | Failure msg ->
                                    return Result.Error msg
                }
        )

    /// Create the arc_submit_answer tool
    let createSubmitAnswerTool () =
        Tool.Create(
            "arc_submit_answer",
            "Submit your answer grid. Input: transformation name OR grid as JSON array of arrays. You have 3 attempts.",
            fun (args: string) ->
                task {
                    match currentTask with
                    | None ->
                        return Result.Error "No ARC task loaded. Use arc_load_task first."
                    | Some arcTask ->
                        if arcTask.TestPairs.IsEmpty then
                            return Result.Error "No test cases in this task."
                        elif currentAttempts >= maxAttempts then
                            return Result.Error $"No attempts remaining. You used all {maxAttempts} attempts."
                        else
                            currentAttempts <- currentAttempts + 1
                            let testPair = arcTask.TestPairs.[0]
                            let input = args.Trim().Trim('\'', '"')
                            
                            // Try to parse as transformation name first
                            let resultGrid =
                                match input.ToLowerInvariant() with
                                | "flip_horizontal" | "fliphorizontal" -> 
                                    Some (applyTransform FlipHorizontal testPair.Input)
                                | "flip_vertical" | "flipvertical" -> 
                                    Some (applyTransform FlipVertical testPair.Input)
                                | "rotate_90" | "rotate90" -> 
                                    Some (applyTransform Rotate90 testPair.Input)
                                | "rotate_180" | "rotate180" -> 
                                    Some (applyTransform Rotate180 testPair.Input)
                                | "rotate_270" | "rotate270" -> 
                                    Some (applyTransform Rotate270 testPair.Input)
                                | "transpose" -> 
                                    Some (applyTransform Transpose testPair.Input)
                                | "invert_colors" | "invertcolors" -> 
                                    Some (applyTransform InvertColors testPair.Input)
                                | "identity" -> 
                                    Some (applyTransform Identity testPair.Input)
                                | _ ->
                                    // Try to parse as JSON grid
                                    try
                                        let doc = JsonDocument.Parse(input)
                                        let grid = parseGrid doc.RootElement
                                        Some (Success grid)
                                    with _ ->
                                        None
                            
                            match resultGrid with
                            | None ->
                                return Result.Error $"Could not parse answer. Use a transformation name or JSON grid. Attempts remaining: {maxAttempts - currentAttempts}"
                            | Some (Failure msg) ->
                                return Result.Error $"Transform failed: {msg}. Attempts remaining: {maxAttempts - currentAttempts}"
                            | Some (Success answerGrid) ->
                                if gridsEqual answerGrid testPair.Output then
                                    return Result.Ok $"✅ CORRECT! You solved the ARC puzzle in {currentAttempts} attempt(s)!"
                                else
                                    let remaining = maxAttempts - currentAttempts
                                    if remaining > 0 then
                                        let sb = System.Text.StringBuilder()
                                        sb.AppendLine($"❌ Incorrect. Attempts remaining: {remaining}") |> ignore
                                        sb.AppendLine() |> ignore
                                        sb.AppendLine("Your answer:") |> ignore
                                        sb.AppendLine(gridToString answerGrid) |> ignore
                                        return Result.Ok (sb.ToString())
                                    else
                                        let sb = System.Text.StringBuilder()
                                        sb.AppendLine($"❌ Incorrect. No attempts remaining.") |> ignore
                                        sb.AppendLine() |> ignore
                                        sb.AppendLine("Your answer:") |> ignore
                                        sb.AppendLine(gridToString answerGrid) |> ignore
                                        sb.AppendLine() |> ignore
                                        sb.AppendLine("Expected answer:") |> ignore
                                        sb.AppendLine(gridToString testPair.Output) |> ignore
                                        return Result.Ok (sb.ToString())
                }
        )

    /// Get all ARC tools
    let getAllTools () : Tool list =
        [
            createLoadTaskTool ()
            createShowExamplesTool ()
            createShowTestTool ()
            createDescribePatternTool ()
            createApplyTransformTool ()
            createSubmitAnswerTool ()
        ]
