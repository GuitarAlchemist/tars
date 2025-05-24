namespace TarsEngine.FSharp.Core.CodeAnalysis

/// Module for analyzing code complexity
module ComplexityAnalysis =
    open System
    open System.IO
    open System.Text.RegularExpressions
    open Types
    
    /// Represents a complexity metric
    type ComplexityMetric = {
        /// The name of the metric
        Name: string
        /// The value of the metric
        Value: float
        /// The file path
        FilePath: string
        /// The structure name (method, class, etc.)
        StructureName: string
        /// The structure type (method, class, etc.)
        StructureType: string
        /// The location of the structure
        Location: CodeLocation
        /// The threshold for the metric
        Threshold: float option
        /// Whether the metric exceeds the threshold
        ExceedsThreshold: bool
    }
    
    /// Represents a maintainability metric
    type MaintainabilityMetric = {
        /// The name of the metric
        Name: string
        /// The value of the metric
        Value: float
        /// The file path
        FilePath: string
        /// The structure name (method, class, etc.)
        StructureName: string
        /// The structure type (method, class, etc.)
        StructureType: string
        /// The location of the structure
        Location: CodeLocation
        /// The rating (Excellent, Good, Fair, Poor)
        Rating: string
    }
    
    /// Represents Halstead complexity metrics
    type HalsteadMetric = {
        /// The name of the metric
        Name: string
        /// The value of the metric
        Value: float
        /// The file path
        FilePath: string
        /// The structure name (method, class, etc.)
        StructureName: string
        /// The structure type (method, class, etc.)
        StructureType: string
        /// The location of the structure
        Location: CodeLocation
        /// The number of unique operators
        UniqueOperators: int
        /// The number of unique operands
        UniqueOperands: int
        /// The total number of operators
        TotalOperators: int
        /// The total number of operands
        TotalOperands: int
    }
    
    /// Counts the occurrences of a pattern in a string
    let countOccurrences (text: string) (pattern: string) =
        Regex.Matches(text, pattern).Count
    
    /// Calculates the cyclomatic complexity of C# code
    let calculateCSharpCyclomaticComplexity (content: string) (structure: CodeStructure) =
        try
            // Extract the method content
            let methodContent = content.Substring(structure.Location.StartOffset, structure.Location.EndOffset - structure.Location.StartOffset)
            
            // Start with base complexity of 1
            let mutable complexity = 1.0
            
            // Add complexity for control flow statements
            complexity <- complexity + (countOccurrences methodContent @"\bif\s*\(" |> float)
            complexity <- complexity + (countOccurrences methodContent @"\belse\s+if\s*\(" |> float)
            complexity <- complexity + (countOccurrences methodContent @"\bwhile\s*\(" |> float)
            complexity <- complexity + (countOccurrences methodContent @"\bfor\s*\(" |> float)
            complexity <- complexity + (countOccurrences methodContent @"\bforeach\s*\(" |> float)
            complexity <- complexity + (countOccurrences methodContent @"\bcase\s+[^:]+:" |> float)
            complexity <- complexity + (countOccurrences methodContent @"\bcatch\s*\(" |> float)
            complexity <- complexity + (countOccurrences methodContent @"\bcatch\s*\{" |> float)
            complexity <- complexity + (countOccurrences methodContent @"\b\|\|" |> float) * 0.5
            complexity <- complexity + (countOccurrences methodContent @"\b&&" |> float) * 0.5
            complexity <- complexity + (countOccurrences methodContent @"\?.*:.*" |> float)
            
            complexity
        with
        | ex -> 
            printfn "Error calculating C# cyclomatic complexity for method %s: %s" structure.Name ex.Message
            1.0
    
    /// Calculates the cyclomatic complexity of F# code
    let calculateFSharpCyclomaticComplexity (content: string) (structure: CodeStructure) =
        try
            // Extract the function content
            let functionContent = content.Substring(structure.Location.StartOffset, structure.Location.EndOffset - structure.Location.StartOffset)
            
            // Start with base complexity of 1
            let mutable complexity = 1.0
            
            // Add complexity for pattern matching
            complexity <- complexity + (countOccurrences functionContent @"\bmatch\b" |> float)
            complexity <- complexity + (countOccurrences functionContent @"\s*\|\s+" |> float) * 0.5
            
            // Add complexity for conditional expressions
            complexity <- complexity + (countOccurrences functionContent @"\bif\b" |> float)
            complexity <- complexity + (countOccurrences functionContent @"\belse\s+if\b" |> float)
            complexity <- complexity + (countOccurrences functionContent @"\belif\b" |> float)
            
            // Add complexity for exception handling
            complexity <- complexity + (countOccurrences functionContent @"\btry\b" |> float)
            complexity <- complexity + (countOccurrences functionContent @"\bwith\b" |> float)
            
            // Add complexity for logical operators
            complexity <- complexity + (countOccurrences functionContent @"\b\|\|\b" |> float) * 0.5
            complexity <- complexity + (countOccurrences functionContent @"\b&&\b" |> float) * 0.5
            
            complexity
        with
        | ex -> 
            printfn "Error calculating F# cyclomatic complexity for function %s: %s" structure.Name ex.Message
            1.0
    
    /// Calculates the cognitive complexity of C# code
    let calculateCSharpCognitiveComplexity (content: string) (structure: CodeStructure) =
        try
            // Extract the method content
            let methodContent = content.Substring(structure.Location.StartOffset, structure.Location.EndOffset - structure.Location.StartOffset)
            
            // Start with base complexity of 0
            let mutable complexity = 0.0
            
            // Add complexity for control flow statements
            complexity <- complexity + (countOccurrences methodContent @"\bif\s*\(" |> float)
            complexity <- complexity + (countOccurrences methodContent @"\belse\s+if\s*\(" |> float) * 1.5
            complexity <- complexity + (countOccurrences methodContent @"\bwhile\s*\(" |> float)
            complexity <- complexity + (countOccurrences methodContent @"\bfor\s*\(" |> float)
            complexity <- complexity + (countOccurrences methodContent @"\bforeach\s*\(" |> float)
            complexity <- complexity + (countOccurrences methodContent @"\bswitch\s*\(" |> float)
            complexity <- complexity + (countOccurrences methodContent @"\bcase\s+[^:]+:" |> float) * 0.5
            complexity <- complexity + (countOccurrences methodContent @"\bcatch\s*\(" |> float)
            complexity <- complexity + (countOccurrences methodContent @"\bcatch\s*\{" |> float)
            
            // Add complexity for logical operators
            complexity <- complexity + (countOccurrences methodContent @"\b\|\|" |> float) * 0.5
            complexity <- complexity + (countOccurrences methodContent @"\b&&" |> float) * 0.5
            
            // Add complexity for nested structures (simplified)
            let indentationLevels = 
                methodContent.Split('\n')
                |> Array.map (fun line -> line.TrimStart())
                |> Array.filter (fun line -> not (String.IsNullOrWhiteSpace(line)))
                |> Array.map (fun line -> line.Length - line.TrimStart().Length)
                |> Array.max
                
            complexity <- complexity + (float indentationLevels / 4.0)
            
            // Add complexity for ternary operators
            complexity <- complexity + (countOccurrences methodContent @"\?.*:.*" |> float)
            
            // Add complexity for goto statements
            complexity <- complexity + (countOccurrences methodContent @"\bgoto\b" |> float) * 2.0
            
            complexity
        with
        | ex -> 
            printfn "Error calculating C# cognitive complexity for method %s: %s" structure.Name ex.Message
            0.0
    
    /// Calculates the cognitive complexity of F# code
    let calculateFSharpCognitiveComplexity (content: string) (structure: CodeStructure) =
        try
            // Extract the function content
            let functionContent = content.Substring(structure.Location.StartOffset, structure.Location.EndOffset - structure.Location.StartOffset)
            
            // Start with base complexity of 0
            let mutable complexity = 0.0
            
            // Add complexity for pattern matching
            complexity <- complexity + (countOccurrences functionContent @"\bmatch\b" |> float)
            complexity <- complexity + (countOccurrences functionContent @"\s*\|\s+" |> float) * 0.5
            
            // Add complexity for conditional expressions
            complexity <- complexity + (countOccurrences functionContent @"\bif\b" |> float)
            complexity <- complexity + (countOccurrences functionContent @"\belse\s+if\b" |> float) * 1.5
            complexity <- complexity + (countOccurrences functionContent @"\belif\b" |> float) * 1.5
            
            // Add complexity for exception handling
            complexity <- complexity + (countOccurrences functionContent @"\btry\b" |> float)
            complexity <- complexity + (countOccurrences functionContent @"\bwith\b" |> float)
            
            // Add complexity for logical operators
            complexity <- complexity + (countOccurrences functionContent @"\b\|\|\b" |> float) * 0.5
            complexity <- complexity + (countOccurrences functionContent @"\b&&\b" |> float) * 0.5
            
            // Add complexity for recursive calls (simplified detection)
            if functionContent.Contains("rec " + structure.Name) || functionContent.Contains(structure.Name + " " + structure.Name) then
                complexity <- complexity + 1.0
            
            // Add complexity for nested functions
            complexity <- complexity + (countOccurrences functionContent @"\blet\s+[a-zA-Z0-9_]+\s*=" |> float) * 0.5
            
            // Add complexity for piping operations (simplified)
            complexity <- complexity + (countOccurrences functionContent @"\|\>" |> float) * 0.2
            
            complexity
        with
        | ex -> 
            printfn "Error calculating F# cognitive complexity for function %s: %s" structure.Name ex.Message
            0.0
    
    /// Calculates the maintainability index
    let calculateMaintainabilityIndex (cyclomaticComplexity: float) (halsteadVolume: float) (linesOfCode: int) =
        try
            // Maintainability Index formula
            let mi = 171.0 - 5.2 * Math.Log(halsteadVolume) - 0.23 * cyclomaticComplexity - 16.2 * Math.Log(float linesOfCode)
            
            // Normalize to 0-100 scale
            let normalizedMi = Math.Max(0.0, Math.Min(100.0, mi))
            
            // Determine rating
            let rating =
                if normalizedMi >= 85.0 then "Excellent"
                elif normalizedMi >= 65.0 then "Good"
                elif normalizedMi >= 40.0 then "Fair"
                else "Poor"
                
            (normalizedMi, rating)
        with
        | ex -> 
            printfn "Error calculating maintainability index: %s" ex.Message
            (50.0, "Fair")
    
    /// Calculates Halstead complexity metrics
    let calculateHalsteadMetrics (content: string) (language: string) =
        try
            // Define operators and operands based on language
            let operators =
                match language.ToLowerInvariant() with
                | "csharp" -> 
                    [|"+"; "-"; "*"; "/"; "%"; "++"; "--"; "=="; "!="; ">"; "<"; ">="; "<="; 
                      "&&"; "||"; "!"; "&"; "|"; "^"; "~"; "<<"; ">>"; "="; "+="; "-="; "*="; 
                      "/="; "%="; "&="; "|="; "^="; "<<="; ">>="; "?:"; "??"; "?."; "=>"|]
                | "fsharp" -> 
                    [|"+"; "-"; "*"; "/"; "%"; "**"; "="; "<>"; ">"; "<"; ">="; "<="; "&&"; "||"; 
                      "not"; "<<<"; ">>>"; "|>"; "<|"; ">>"; "<<"; "->"; "<-"; ":="; "::"; "@"; "^"|]
                | _ -> [|"+"; "-"; "*"; "/"; "="|]
                
            // Count operators and operands
            let mutable uniqueOperators = Set.empty
            let mutable uniqueOperands = Set.empty
            let mutable totalOperators = 0
            let mutable totalOperands = 0
            
            // Simple tokenization (this is a simplified approach)
            let tokens = Regex.Split(content, @"\s+|[;,.()\[\]{}]")
            
            for token in tokens do
                if not (String.IsNullOrWhiteSpace(token)) then
                    if Array.contains token operators then
                        uniqueOperators <- uniqueOperators.Add(token)
                        totalOperators <- totalOperators + 1
                    else
                        uniqueOperands <- uniqueOperands.Add(token)
                        totalOperands <- totalOperands + 1
                        
            // Calculate Halstead metrics
            let n1 = uniqueOperators.Count
            let n2 = uniqueOperands.Count
            let N1 = totalOperators
            let N2 = totalOperands
            
            let vocabulary = n1 + n2
            let length = N1 + N2
            let volume = float length * Math.Log(float vocabulary, 2.0)
            let difficulty = (float n1 / 2.0) * (float N2 / float n2)
            let effort = difficulty * volume
            let timeToImplement = effort / 18.0 // Time in seconds
            let bugsDelivered = volume / 3000.0
            
            (n1, n2, N1, N2, vocabulary, length, volume, difficulty, effort, timeToImplement, bugsDelivered)
        with
        | ex -> 
            printfn "Error calculating Halstead metrics: %s" ex.Message
            (0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    /// Analyzes the cyclomatic complexity of a file
    let analyzeCyclomaticComplexity (filePath: string) (language: string) =
        try
            let content = File.ReadAllText(filePath)
            let structures = extractCodeStructures content language
            
            structures
            |> List.filter (fun s -> s.Type = "Method" || s.Type = "Function")
            |> List.map (fun structure ->
                let complexity =
                    match language.ToLowerInvariant() with
                    | "csharp" -> calculateCSharpCyclomaticComplexity content structure
                    | "fsharp" -> calculateFSharpCyclomaticComplexity content structure
                    | _ -> 1.0
                    
                let threshold = 
                    match language.ToLowerInvariant() with
                    | "csharp" -> 15.0
                    | "fsharp" -> 10.0
                    | _ -> 10.0
                    
                {
                    Name = "Cyclomatic Complexity"
                    Value = complexity
                    FilePath = filePath
                    StructureName = structure.Name
                    StructureType = structure.Type
                    Location = structure.Location
                    Threshold = Some threshold
                    ExceedsThreshold = complexity > threshold
                })
        with
        | ex -> 
            printfn "Error analyzing cyclomatic complexity for file %s: %s" filePath ex.Message
            []
    
    /// Analyzes the cognitive complexity of a file
    let analyzeCognitiveComplexity (filePath: string) (language: string) =
        try
            let content = File.ReadAllText(filePath)
            let structures = extractCodeStructures content language
            
            structures
            |> List.filter (fun s -> s.Type = "Method" || s.Type = "Function")
            |> List.map (fun structure ->
                let complexity =
                    match language.ToLowerInvariant() with
                    | "csharp" -> calculateCSharpCognitiveComplexity content structure
                    | "fsharp" -> calculateFSharpCognitiveComplexity content structure
                    | _ -> 0.0
                    
                let threshold = 
                    match language.ToLowerInvariant() with
                    | "csharp" -> 15.0
                    | "fsharp" -> 10.0
                    | _ -> 10.0
                    
                {
                    Name = "Cognitive Complexity"
                    Value = complexity
                    FilePath = filePath
                    StructureName = structure.Name
                    StructureType = structure.Type
                    Location = structure.Location
                    Threshold = Some threshold
                    ExceedsThreshold = complexity > threshold
                })
        with
        | ex -> 
            printfn "Error analyzing cognitive complexity for file %s: %s" filePath ex.Message
            []
    
    /// Analyzes the maintainability index of a file
    let analyzeMaintainabilityIndex (filePath: string) (language: string) =
        try
            let content = File.ReadAllText(filePath)
            let structures = extractCodeStructures content language
            
            structures
            |> List.filter (fun s -> s.Type = "Method" || s.Type = "Function")
            |> List.map (fun structure ->
                let cyclomaticComplexity =
                    match language.ToLowerInvariant() with
                    | "csharp" -> calculateCSharpCyclomaticComplexity content structure
                    | "fsharp" -> calculateFSharpCyclomaticComplexity content structure
                    | _ -> 1.0
                    
                let (_, _, _, _, _, _, halsteadVolume, _, _, _, _) = calculateHalsteadMetrics content language
                
                let linesOfCode = 
                    content.Substring(structure.Location.StartOffset, structure.Location.EndOffset - structure.Location.StartOffset)
                        .Split('\n')
                        .Length
                        
                let (mi, rating) = calculateMaintainabilityIndex cyclomaticComplexity halsteadVolume linesOfCode
                
                {
                    Name = "Maintainability Index"
                    Value = mi
                    FilePath = filePath
                    StructureName = structure.Name
                    StructureType = structure.Type
                    Location = structure.Location
                    Rating = rating
                })
        with
        | ex -> 
            printfn "Error analyzing maintainability index for file %s: %s" filePath ex.Message
            []
    
    /// Analyzes the Halstead complexity of a file
    let analyzeHalsteadComplexity (filePath: string) (language: string) =
        try
            let content = File.ReadAllText(filePath)
            let structures = extractCodeStructures content language
            
            structures
            |> List.filter (fun s -> s.Type = "Method" || s.Type = "Function")
            |> List.map (fun structure ->
                let structureContent = content.Substring(structure.Location.StartOffset, structure.Location.EndOffset - structure.Location.StartOffset)
                let (n1, n2, N1, N2, _, _, volume, difficulty, effort, timeToImplement, bugsDelivered) = calculateHalsteadMetrics structureContent language
                
                {
                    Name = "Halstead Complexity"
                    Value = volume
                    FilePath = filePath
                    StructureName = structure.Name
                    StructureType = structure.Type
                    Location = structure.Location
                    UniqueOperators = n1
                    UniqueOperands = n2
                    TotalOperators = N1
                    TotalOperands = N2
                })
        with
        | ex -> 
            printfn "Error analyzing Halstead complexity for file %s: %s" filePath ex.Message
            []
    
    /// Extracts code structures from content
    and extractCodeStructures (content: string) (language: string) =
        try
            match language.ToLowerInvariant() with
            | "csharp" -> extractCSharpStructures content
            | "fsharp" -> extractFSharpStructures content
            | _ -> []
        with
        | ex -> 
            printfn "Error extracting code structures: %s" ex.Message
            []
    
    /// Extracts C# code structures
    and extractCSharpStructures (content: string) =
        try
            let methodRegex = new Regex(@"(public|private|protected|internal|static)?\s+\w+\s+(\w+)\s*\([^)]*\)\s*{", RegexOptions.Compiled)
            let classRegex = new Regex(@"(public|private|protected|internal)?\s+class\s+(\w+)", RegexOptions.Compiled)
            
            let methodMatches = methodRegex.Matches(content)
            let classMatches = classRegex.Matches(content)
            
            let methodStructures =
                methodMatches
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let methodName = m.Groups[2].Value
                    let startOffset = m.Index
                    
                    // Find the end of the method (simplified approach)
                    let endOffset = 
                        let mutable braceCount = 1
                        let mutable currentPos = startOffset + m.Length
                        
                        while braceCount > 0 && currentPos < content.Length do
                            match content[currentPos] with
                            | '{' -> braceCount <- braceCount + 1
                            | '}' -> braceCount <- braceCount - 1
                            | _ -> ()
                            
                            currentPos <- currentPos + 1
                            
                        currentPos
                        
                    {
                        Name = methodName
                        Type = "Method"
                        Location = {
                            StartOffset = startOffset
                            EndOffset = endOffset
                            StartLine = content.Substring(0, startOffset).Split('\n').Length
                            EndLine = content.Substring(0, endOffset).Split('\n').Length
                        }
                    })
                |> Seq.toList
                
            let classStructures =
                classMatches
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let className = m.Groups[2].Value
                    let startOffset = m.Index
                    
                    // Find the end of the class (simplified approach)
                    let endOffset = 
                        let mutable braceCount = 0
                        let mutable currentPos = startOffset + m.Length
                        
                        // Find the opening brace
                        while braceCount = 0 && currentPos < content.Length do
                            if content[currentPos] = '{' then
                                braceCount <- 1
                            currentPos <- currentPos + 1
                            
                        // Find the closing brace
                        while braceCount > 0 && currentPos < content.Length do
                            match content[currentPos] with
                            | '{' -> braceCount <- braceCount + 1
                            | '}' -> braceCount <- braceCount - 1
                            | _ -> ()
                            
                            currentPos <- currentPos + 1
                            
                        currentPos
                        
                    {
                        Name = className
                        Type = "Class"
                        Location = {
                            StartOffset = startOffset
                            EndOffset = endOffset
                            StartLine = content.Substring(0, startOffset).Split('\n').Length
                            EndLine = content.Substring(0, endOffset).Split('\n').Length
                        }
                    })
                |> Seq.toList
                
            methodStructures @ classStructures
        with
        | ex -> 
            printfn "Error extracting C# structures: %s" ex.Message
            []
    
    /// Extracts F# code structures
    and extractFSharpStructures (content: string) =
        try
            let functionRegex = new Regex(@"let\s+(?:rec\s+)?(\w+)(?:\s+\w+)*\s*=", RegexOptions.Compiled)
            let moduleRegex = new Regex(@"module\s+(\w+)", RegexOptions.Compiled)
            
            let functionMatches = functionRegex.Matches(content)
            let moduleMatches = moduleRegex.Matches(content)
            
            let functionStructures =
                functionMatches
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let functionName = m.Groups[1].Value
                    let startOffset = m.Index
                    
                    // Find the end of the function (simplified approach)
                    let endOffset = 
                        let nextFunctionMatch = 
                            functionMatches
                            |> Seq.cast<Match>
                            |> Seq.tryFind (fun nextM -> nextM.Index > m.Index)
                            
                        match nextFunctionMatch with
                        | Some nextM -> nextM.Index
                        | None -> content.Length
                        
                    {
                        Name = functionName
                        Type = "Function"
                        Location = {
                            StartOffset = startOffset
                            EndOffset = endOffset
                            StartLine = content.Substring(0, startOffset).Split('\n').Length
                            EndLine = content.Substring(0, endOffset).Split('\n').Length
                        }
                    })
                |> Seq.toList
                
            let moduleStructures =
                moduleMatches
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let moduleName = m.Groups[1].Value
                    let startOffset = m.Index
                    
                    // Find the end of the module (simplified approach)
                    let endOffset = 
                        let nextModuleMatch = 
                            moduleMatches
                            |> Seq.cast<Match>
                            |> Seq.tryFind (fun nextM -> nextM.Index > m.Index)
                            
                        match nextModuleMatch with
                        | Some nextM -> nextM.Index
                        | None -> content.Length
                        
                    {
                        Name = moduleName
                        Type = "Module"
                        Location = {
                            StartOffset = startOffset
                            EndOffset = endOffset
                            StartLine = content.Substring(0, startOffset).Split('\n').Length
                            EndLine = content.Substring(0, endOffset).Split('\n').Length
                        }
                    })
                |> Seq.toList
                
            functionStructures @ moduleStructures
        with
        | ex -> 
            printfn "Error extracting F# structures: %s" ex.Message
            []
