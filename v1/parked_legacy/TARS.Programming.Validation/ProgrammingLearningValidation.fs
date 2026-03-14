module TARS.Programming.Validation.ProgrammingLearning

open System

type LearnedPattern =
    { Name: string
      Description: string }

type ProgrammingLearningValidator() =

    member _.AnalyseFSharpPatterns(source: string) =
        let mutable patterns = []
        if source.Contains("type Result") then
            patterns <- { Name = "Railway-Oriented Programming"; Description = "Success/Failure discriminated union." } :: patterns
        if source.Contains(">>=") then
            patterns <- { Name = "Bind Operator"; Description = "Monadic composition operator detected." } :: patterns
        if source.Contains("match result with") then
            patterns <- { Name = "Pattern Matching"; Description = "Explicit result pattern matching found." } :: patterns
        patterns |> List.rev

    member _.AnalyseCSharpPatterns(source: string) =
        let mutable patterns = []
        if source.Contains("async Task") then
            patterns <- { Name = "Async/Await"; Description = "Async workflow with Task result." } :: patterns
        if source.Contains("<T>") then
            patterns <- { Name = "Generics"; Description = "Generic method or type detected." } :: patterns
        if source.Contains("try") && source.Contains("catch") then
            patterns <- { Name = "Exception handling"; Description = "Try/catch protection found." } :: patterns
        patterns |> List.rev

    member this.RunValidation() =
        printfn "?? PROGRAMMING PATTERN LEARNING"
        printfn "================================"

        let fsharpSource =
            """type Result<'T,'E> = Success of 'T | Failure of 'E
               let (>>=) r f = match r with | Success v -> f v | Failure e -> Failure e"""

        let csharpSource =
            """public async Task<string> ProcessAsync<T>(T input) {
                try { var result = await SomethingAsync(input); return result.ToString(); }
                catch(Exception ex) { return ex.Message; }}"""

        let fsharpPatterns = this.AnalyseFSharpPatterns fsharpSource
        let csharpPatterns = this.AnalyseCSharpPatterns csharpSource

        printfn "  F#: detected %d patterns" (List.length fsharpPatterns)
        fsharpPatterns |> List.iteri (fun idx pattern -> printfn "    %d. %s" (idx + 1) pattern.Name)

        printfn "  C#: detected %d patterns" (List.length csharpPatterns)
        csharpPatterns |> List.iteri (fun idx pattern -> printfn "    %d. %s" (idx + 1) pattern.Name)

        let success = (List.length fsharpPatterns) > 0 && (List.length csharpPatterns) > 0
        printfn "  Result: %s" (if success then "? PASSED" else "? FAILED")
        success
