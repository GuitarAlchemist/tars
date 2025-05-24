namespace TarsEngine.FSharp.Core.CodeAnalysis

/// Main module for code analysis
module CodeAnalyzer =
    open System
    open System.IO
    open Types
    open PatternDetector
    open CodeTransformer
    open ReportGenerator
    
    /// Creates a default configuration
    let createDefaultConfiguration () : Configuration =
        { Patterns = []
          Transformations = []
          FileExtensions = [".cs"; ".fs"; ".fsx"; ".js"; ".ts"; ".html"; ".css"; ".xml"; ".json"]
          ExcludeDirectories = ["bin"; "obj"; "node_modules"; ".git"; ".vs"]
          ExcludeFiles = [] }
    
    /// Creates a pattern
    let createPattern (name: string) (description: string) (regex: string) (severity: float) (category: string) (language: string) : Pattern =
        { Name = name
          Description = description
          Regex = regex
          Severity = severity
          Category = category
          Language = language }
    
    /// Creates a transformation
    let createTransformation (name: string) (description: string) (pattern: string) (replacement: string) (language: string) : Transformation =
        { Name = name
          Description = description
          Pattern = pattern
          Replacement = replacement
          Language = language }
    
    /// Analyzes a directory
    let analyzeDirectory (config: Configuration) (directoryPath: string) : Report =
        // Detect patterns
        let matches = detectPatternsInDirectory config directoryPath
        
        // Apply transformations
        let transformedFiles = applyTransformationsToDirectory config directoryPath
        let transformations = 
            transformedFiles
            |> List.collect (fun (_, transformations) -> transformations)
        
        // Create report
        createReport matches transformations
    
    /// Analyzes a file
    let analyzeFile (config: Configuration) (filePath: string) : Report =
        // Detect patterns
        let matches = detectPatternsInFile config.Patterns filePath
        
        // Apply transformations
        let transformations = applyTransformationsToFile config.Transformations filePath
        
        // Create report
        createReport matches transformations
    
    /// Loads patterns from a file
    let loadPatternsFromFile (filePath: string) : Pattern list =
        try
            let json = File.ReadAllText(filePath)
            Newtonsoft.Json.JsonConvert.DeserializeObject<Pattern list>(json)
        with
        | ex -> 
            printfn "Error loading patterns from %s: %s" filePath ex.Message
            []
    
    /// Loads transformations from a file
    let loadTransformationsFromFile (filePath: string) : Transformation list =
        try
            let json = File.ReadAllText(filePath)
            Newtonsoft.Json.JsonConvert.DeserializeObject<Transformation list>(json)
        with
        | ex -> 
            printfn "Error loading transformations from %s: %s" filePath ex.Message
            []
    
    /// Loads configuration from a file
    let loadConfigurationFromFile (filePath: string) : Configuration =
        try
            let json = File.ReadAllText(filePath)
            Newtonsoft.Json.JsonConvert.DeserializeObject<Configuration>(json)
        with
        | ex -> 
            printfn "Error loading configuration from %s: %s" filePath ex.Message
            createDefaultConfiguration()
    
    /// Saves configuration to a file
    let saveConfigurationToFile (config: Configuration) (filePath: string) : unit =
        try
            let json = Newtonsoft.Json.JsonConvert.SerializeObject(config, Newtonsoft.Json.Formatting.Indented)
            File.WriteAllText(filePath, json)
        with
        | ex -> 
            printfn "Error saving configuration to %s: %s" filePath ex.Message
    
    /// Creates a C# code quality configuration
    let createCSharpCodeQualityConfiguration () : Configuration =
        let patterns = [
            createPattern "UnusedVariable" "Unused variable" @"\bvar\s+([a-zA-Z0-9_]+)\s*=.*?;\s*(?!.*\b\1\b)" 0.5 "Code Quality" "C#"
            createPattern "MagicNumber" "Magic number" @"\b[0-9]+\b" 0.3 "Code Quality" "C#"
            createPattern "LongMethod" "Method is too long" @"(?s)public\s+\w+\s+\w+\s*\(.*?\)\s*\{.*?(?=\})[^\{]*\}" 0.7 "Code Quality" "C#"
            createPattern "ComplexCondition" "Complex condition" @"if\s*\(.*?&&.*?&&.*?\)" 0.6 "Code Quality" "C#"
            createPattern "CatchAll" "Catch all exceptions" @"catch\s*\(\s*\)" 0.8 "Error Handling" "C#"
            createPattern "HardcodedString" "Hardcoded string" @"""[^""]{10,}""" 0.4 "Code Quality" "C#"
            createPattern "TodoComment" "TODO comment" @"//\s*TODO" 0.2 "Documentation" "C#"
        ]
        
        let transformations = [
            createTransformation "RemoveUnusedUsings" "Remove unused using statements" @"using\s+[^;]+;\s*(?=using|namespace)" "" "C#"
            createTransformation "AddBracesToIfStatements" "Add braces to if statements" @"if\s*\(([^)]+)\)\s*([^{;].*?;)" "if ($1)\n{\n    $2\n}" "C#"
            createTransformation "ReplaceVarWithExplicitType" "Replace var with explicit type" @"var\s+([a-zA-Z0-9_]+)\s*=\s*new\s+([a-zA-Z0-9_<>]+)" "$2 $1 = new $2" "C#"
        ]
        
        { createDefaultConfiguration() with 
            Patterns = patterns
            Transformations = transformations
            FileExtensions = [".cs"] }
    
    /// Creates an F# code quality configuration
    let createFSharpCodeQualityConfiguration () : Configuration =
        let patterns = [
            createPattern "UnusedVariable" "Unused variable" @"\blet\s+([a-zA-Z0-9_]+)\s*=.*?(?!.*\b\1\b)" 0.5 "Code Quality" "F#"
            createPattern "MagicNumber" "Magic number" @"\b[0-9]+\b" 0.3 "Code Quality" "F#"
            createPattern "LongFunction" "Function is too long" @"(?s)let\s+\w+.*?=\s*.*?(?=let\s+\w+|$)" 0.7 "Code Quality" "F#"
            createPattern "ComplexCondition" "Complex condition" @"if\s*.*?&&.*?&&.*?then" 0.6 "Code Quality" "F#"
            createPattern "MutableVariable" "Mutable variable" @"\bmutable\b" 0.6 "Code Quality" "F#"
            createPattern "HardcodedString" "Hardcoded string" @"""[^""]{10,}""" 0.4 "Code Quality" "F#"
            createPattern "TodoComment" "TODO comment" @"//\s*TODO" 0.2 "Documentation" "F#"
        ]
        
        let transformations = [
            createTransformation "ReplaceImperativeForWithMap" "Replace imperative for with map" @"for\s+([a-zA-Z0-9_]+)\s+in\s+([a-zA-Z0-9_]+)\s+do\s+([a-zA-Z0-9_]+)\s*<-\s*([^;]+)" "$2 |> List.map (fun $1 -> $4)" "F#"
            createTransformation "ReplaceImperativeForWithIter" "Replace imperative for with iter" @"for\s+([a-zA-Z0-9_]+)\s+in\s+([a-zA-Z0-9_]+)\s+do\s+([^;]+)" "$2 |> List.iter (fun $1 -> $3)" "F#"
            createTransformation "ReplaceMutableWithLet" "Replace mutable with let" @"let\s+mutable\s+([a-zA-Z0-9_]+)\s*=\s*([^;]+)" "let $1 = ref $2" "F#"
        ]
        
        { createDefaultConfiguration() with 
            Patterns = patterns
            Transformations = transformations
            FileExtensions = [".fs"; ".fsx"] }
    
    /// Creates a JavaScript code quality configuration
    let createJavaScriptCodeQualityConfiguration () : Configuration =
        let patterns = [
            createPattern "UnusedVariable" "Unused variable" @"\bvar\s+([a-zA-Z0-9_]+)\s*=.*?;\s*(?!.*\b\1\b)" 0.5 "Code Quality" "JavaScript"
            createPattern "MagicNumber" "Magic number" @"\b[0-9]+\b" 0.3 "Code Quality" "JavaScript"
            createPattern "LongFunction" "Function is too long" @"(?s)function\s+\w+\s*\(.*?\)\s*\{.*?(?=\})[^\{]*\}" 0.7 "Code Quality" "JavaScript"
            createPattern "ComplexCondition" "Complex condition" @"if\s*\(.*?&&.*?&&.*?\)" 0.6 "Code Quality" "JavaScript"
            createPattern "AlertStatement" "Alert statement" @"\balert\s*\(" 0.8 "Code Quality" "JavaScript"
            createPattern "HardcodedString" "Hardcoded string" @"""[^""]{10,}""" 0.4 "Code Quality" "JavaScript"
            createPattern "TodoComment" "TODO comment" @"//\s*TODO" 0.2 "Documentation" "JavaScript"
        ]
        
        let transformations = [
            createTransformation "ReplaceVarWithLet" "Replace var with let" @"\bvar\b" "let" "JavaScript"
            createTransformation "AddSemicolons" "Add semicolons" @"([^;])\s*$" "$1;" "JavaScript"
            createTransformation "ReplaceForWithForEach" "Replace for with forEach" @"for\s*\(\s*var\s+([a-zA-Z0-9_]+)\s*=\s*0\s*;\s*\1\s*<\s*([a-zA-Z0-9_]+)\.length\s*;\s*\1\+\+\s*\)\s*\{(.*?)\}" "$2.forEach(function($1) {$3})" "JavaScript"
        ]
        
        { createDefaultConfiguration() with 
            Patterns = patterns
            Transformations = transformations
            FileExtensions = [".js"; ".ts"] }
