namespace TarsEngineFSharp

module CodeAnalysis =
    open System
    open System.IO
    open System.Text.RegularExpressions
    
    // Define the types of code issues we can detect
    type CodeIssue =
        | MissingExceptionHandling of location:string * description:string
        | IneffectiveCode of location:string * description:string * suggestion:string
        | StyleViolation of location:string * rule:string
        | DocumentationIssue of location:string * missingElement:string
    
    // Result of analyzing a file
    type AnalysisResult = {
        FilePath: string
        Issues: CodeIssue list
        SuggestedFixes: (string * string) list // (original, replacement)
    }
    
    // Simple pattern matching for detecting common issues
    let detectDivideByZeroIssue (code: string) =
        let pattern = @"return\s+(\w+)\s*/\s*(\w+)"
        let matches = Regex.Matches(code, pattern)
        
        matches 
        |> Seq.cast<Match>
        |> Seq.map (fun m -> 
            let dividend = m.Groups.[1].Value
            let divisor = m.Groups.[2].Value
            let location = $"Line containing: {m.Value}"
            
            // Check if there's a null check before this
            let hasDivisorCheck = 
                Regex.IsMatch(code, $"if\\s*\\(\\s*{divisor}\\s*==\\s*0")
            
            if not hasDivisorCheck then
                Some (MissingExceptionHandling(location, $"No check for division by zero when dividing by {divisor}"),
                      m.Value,
                      $"if ({divisor} == 0) {{ throw new DivideByZeroException(\"Cannot divide by {divisor}\"); }}\n        {m.Value}")
            else None)
        |> Seq.choose id
        |> Seq.toList
    
    // Detect inefficient loops that could use LINQ
    let detectIneffectiveLoops (code: string) =
        let sumPattern = @"(?s)int\s+(\w+)\s*=\s*0;.*?for\s*\(\s*int\s+(\w+)\s*=\s*.*?\)\s*{\s*(\w+)\s*=\s*\3\s*\+\s*(\w+)\[.*?\];\s*}"
        let matches = Regex.Matches(code, sumPattern)
        
        matches 
        |> Seq.cast<Match>
        |> Seq.map (fun m -> 
            let sumVar = m.Groups.[1].Value
            let collection = m.Groups.[4].Value
            let location = $"Loop calculating sum of {collection}"
            
            (IneffectiveCode(location, $"Manual loop to sum {collection}", $"Use {collection}.Sum() instead"),
             m.Value,
             $"{sumVar} = {collection}.Sum();"))
        |> Seq.toList
    
    // Analyze a single file
    let analyzeFile (filePath: string) : AnalysisResult =
        try
            let code = File.ReadAllText(filePath)
            
            // Detect various issues
            let divideByZeroIssues = detectDivideByZeroIssue code
            let ineffectiveLoopIssues = detectIneffectiveLoops code
            
            // Combine all issues
            let allIssues = 
                divideByZeroIssues @ ineffectiveLoopIssues
                |> List.map (fun (issue, _, _) -> issue)
                
            // Extract suggested fixes
            let suggestedFixes =
                divideByZeroIssues @ ineffectiveLoopIssues
                |> List.map (fun (_, original, replacement) -> (original, replacement))
            
            { FilePath = filePath; Issues = allIssues; SuggestedFixes = suggestedFixes }
        with
        | ex -> 
            { FilePath = filePath; Issues = []; SuggestedFixes = [] }
    
    // Analyze multiple files in a project
    let analyzeProject (projectPath: string) (maxFiles: int) : AnalysisResult list =
        if Directory.Exists(projectPath) then
            Directory.GetFiles(projectPath, "*.cs", SearchOption.AllDirectories)
            |> Seq.truncate maxFiles
            |> Seq.map analyzeFile
            |> Seq.toList
        else
            []
