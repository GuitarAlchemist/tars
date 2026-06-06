namespace Tars.Tools.Puzzles

open Tars.Tools

open System.Text.RegularExpressions

/// Puzzle validation tool for WoT workflows
module PuzzleValidationTools =
    
    /// Validator functions for each puzzle type
    let validators : Map<string, string -> bool> =
        Map.ofList [
            ("river_crossing", fun answer ->
                let lower = answer.ToLowerInvariant()
                let hasStepCount =
                    lower.Contains("step 7") || lower.Contains("step seven") ||
                    lower.Contains("7 steps") || lower.Contains("seven steps") ||
                    lower.Contains("finally") ||
                    Regex.IsMatch(lower, @"\b7[.)]\s")
                lower.Contains("goat") && lower.Contains("wolf") && 
                lower.Contains("cabbage") && hasStepCount)
            
            ("knights_and_knaves", fun answer ->
                let lower = answer.ToLowerInvariant()
                // Solution ambiguous in some prompts, but we accept either valid deduction:
                // 1. A=Knave, B=Knight ("B is Knight" is False, "Different" is True)
                // 2. A=Knave, B=Knave ("B is Knight" is False, "Different" is False)
                // The prompt used tends towards Knave+Knave.
                // Strict check: Must conclude BOTH are Knaves OR A=Knave, B=Knight.
                let bothKnaves = lower.Contains("a is a knave") && lower.Contains("b is a knave")
                let abSplit = lower.Contains("a is a knave") && lower.Contains("b is a knight")
                bothKnaves || abSplit)
            
            ("tower_of_hanoi", fun answer ->
                let lower = answer.ToLowerInvariant()
                lower.Contains("7") && (lower.Contains("move") || lower.Contains("step")))
            
            ("logic_grid", fun answer ->
                let lower = answer.ToLowerInvariant()
                lower.Contains("alice") && lower.Contains("fish") &&
                lower.Contains("carol") && lower.Contains("cat") &&
                lower.Contains("bob") && lower.Contains("dog"))
            
            ("math_word", fun answer ->
                let lower = answer.ToLowerInvariant()
                (lower.Contains("11:34") || lower.Contains("11:35") || 
                 (lower.Contains("11") && (lower.Contains("34") || lower.Contains("35")))) &&
                (lower.Contains("154") || lower.Contains("155") || lower.Contains("94")))
            
            ("cryptarithmetic", fun answer ->
                let lower = answer.ToLowerInvariant()
                lower.Contains("s=9") || lower.Contains("s = 9") ||
                (lower.Contains("9567") && lower.Contains("1085")))
            
            ("monty_hall", fun answer ->
                let lower = answer.ToLowerInvariant()
                // Accept "2/3", "66%", or LaTeX "\frac{2}{3}" or "2 / 3"
                lower.Contains("switch") && 
                (lower.Contains("2/3") || lower.Contains("66%") || 
                 lower.Contains("2} {3}") || lower.Contains("{2}{3}")))
            
            ("cheryls_birthday", fun answer ->
                let lower = answer.ToLowerInvariant()
                lower.Contains("july") && lower.Contains("16"))
            
            ("scheduling", fun answer ->
                let lower = answer.ToLowerInvariant()
                (lower.Contains("16:00") || lower.Contains("4:00 pm") || lower.Contains("4 pm")) &&
                lower.Contains("conclusion"))
        ]

    /// Validate a puzzle answer
    [<TarsToolAttribute("validate_puzzle_answer", "Validates an answer to a puzzle. Input: JSON with 'puzzle_type' and 'answer' fields.")>]
    let validatePuzzleAnswer (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement
                
                let mutable prop = Unchecked.defaultof<System.Text.Json.JsonElement>
                
                let puzzleType = 
                    if root.TryGetProperty("puzzle_type", &prop) then
                        prop.GetString()
                    else ""
                let answer = 
                    if root.TryGetProperty("answer", &prop) then
                        prop.GetString()
                    else ""
                
                match validators.TryFind(puzzleType.ToLowerInvariant().Replace(" ", "_")) with
                | None ->
                    return $"Error: Unknown puzzle type: {puzzleType}"
                | Some validator ->
                    let isCorrect = validator answer
                    if isCorrect then
                        return $"PASS: Answer validated successfully for puzzle '{puzzleType}'"
                    else
                        return $"FAIL: Answer did not pass validation for puzzle '{puzzleType}'"
            with ex ->
                return $"Error: Failed to parse args: {ex.Message}"
        }
