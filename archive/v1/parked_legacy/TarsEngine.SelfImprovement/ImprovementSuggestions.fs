namespace TarsEngine.SelfImprovement

open System
open System.Text.RegularExpressions
open System.Collections.Generic

/// <summary>
/// Represents an improvement suggestion for a code pattern
/// </summary>
type ImprovementSuggestion =
    { PatternId: string
      BeforePattern: string
      AfterTemplate: string
      Description: string
      ApplicabilityCheck: string -> bool }

/// <summary>
/// Functions for suggesting code improvements
/// </summary>
module ImprovementSuggestions =
    /// <summary>
    /// Common improvement suggestions for code patterns
    /// </summary>
    let commonSuggestions =
        [
            // C# patterns
            { PatternId = "CS003"
              BeforePattern = @"for\s*\(([^)]*)\)\s*{([^}]*?)(\w+)\s*\+=\s*([^;]+);([^}]*?)}"
              AfterTemplate = "var sb = new System.Text.StringBuilder();\nfor ($1) {$2sb.Append($4);$5}\n$3 = sb.ToString();"
              Description = "Replace string concatenation with StringBuilder for better performance"
              ApplicabilityCheck = fun code -> code.Contains("+=") && not (code.Contains("StringBuilder")) }
            
            { PatternId = "CS004"
              BeforePattern = @"for\s*\(([^)]*)\)\s*{([^}]*?)var\s+(\w+)\s*=\s*(\w+)\.Where\(([^)]+)\)([^}]*?)}"
              AfterTemplate = "for ($1) {$2var $3 = new List<var>();\nforeach (var item in $4) {\n    if ($5(item)) {\n        $3.Add(item);\n    }\n}$6}"
              Description = "Replace LINQ in tight loops with traditional loops for better performance"
              ApplicabilityCheck = fun code -> code.Contains(".Where(") && code.Contains("for (") }
            
            { PatternId = "CS005"
              BeforePattern = @"var\s+(\w+)\s*=\s*new\s+(List|Dictionary|HashSet)<([^>]*)>\(\);\s*(?:for|foreach)\s*\(([^)]*)\)\s*{([^}]*?)}"
              AfterTemplate = "var $1 = new $2<$3>(10); // Pre-allocate with estimated capacity\nfor ($4) {$5}"
              Description = "Initialize collections with capacity when used in loops"
              ApplicabilityCheck = fun code -> 
                  (code.Contains("new List<") || code.Contains("new Dictionary<") || code.Contains("new HashSet<")) && 
                  not (code.Contains("(") && code.Contains(")") && not (code.Contains("()")))}
            
            { PatternId = "CS006"
              BeforePattern = @"public\s+async\s+void\s+(\w+)\s*\(([^)]*)\)\s*{([^}]*)}"
              AfterTemplate = "public async Task $1($2) {$3}"
              Description = "Change async void to async Task for better error handling"
              ApplicabilityCheck = fun code -> code.Contains("async void") && not (code.Contains("event")) }
            
            { PatternId = "CS007"
              BeforePattern = @"var\s+(\w+)\s*=\s*new\s+(FileStream|StreamReader|StreamWriter|SqlConnection|SqlCommand|HttpClient)\(([^)]*)\);"
              AfterTemplate = "using var $1 = new $2($3);"
              Description = "Use 'using' declaration to ensure proper disposal of resources"
              ApplicabilityCheck = fun code -> 
                  (code.Contains("new FileStream") || code.Contains("new StreamReader") || 
                   code.Contains("new StreamWriter") || code.Contains("new SqlConnection") || 
                   code.Contains("new SqlCommand") || code.Contains("new HttpClient")) && 
                  not (code.Contains("using")) }
            
            // F# patterns
            { PatternId = "FS003"
              BeforePattern = @"let\s+rec\s+(\w+)([^=]*?)=\s*([^if]*?)if\s+([^then]*?)then\s+([^\s]*?)\s+else\s+([^\s]*?)\s*\*\s*\1\s+([^$]*)"
              AfterTemplate = "let $1$2=\n    let rec loop acc n =\n        if $4then acc\n        else loop ($6 * acc) $7\n    loop $5 $7"
              Description = "Convert recursive function to use tail recursion with accumulator"
              ApplicabilityCheck = fun code -> code.Contains("let rec") && not (code.Contains("loop")) }
            
            { PatternId = "FS004"
              BeforePattern = @"let\s+mutable\s+(\w+)\s*=\s*\[\]\s*for\s+([^do]*?)do\s*([^<]*?)<-\s*\1\s*@\s*\[(.*?)\]"
              AfterTemplate = "let $1 = [$2 do yield $4]"
              Description = "Replace list concatenation in loop with list comprehension"
              ApplicabilityCheck = fun code -> code.Contains("mutable") && code.Contains("@") && code.Contains("for") }
            
            { PatternId = "FS005"
              BeforePattern = @"let\s+(\w+)([^:=]*?)=\s*([^$]*)"
              AfterTemplate = "let $1$2: 'a -> 'b = $3 // Add appropriate type annotation"
              Description = "Add type annotation to function for better documentation"
              ApplicabilityCheck = fun code -> code.StartsWith("let ") && not (code.Contains(":")) && not (code.Contains("private")) }
            
            // General patterns
            { PatternId = "GEN003"
              BeforePattern = @"//\s*([a-zA-Z0-9_]+\s*[({][^$]*)"
              AfterTemplate = "/* Removed commented code:\n$1\nReason: Commented code should be removed from the codebase. */"
              Description = "Replace commented out code with explanation"
              ApplicabilityCheck = fun code -> code.StartsWith("//") && (code.Contains("{") || code.Contains("(")) }
            
            { PatternId = "GEN004"
              BeforePattern = @"([^=]*?)=\s*""([a-zA-Z0-9_\-\.]{10,})""([^$]*)"
              AfterTemplate = "private const string $1Name = \"$2\"; // Extract magic string to constant\n$1= $1Name$3"
              Description = "Extract magic string to named constant"
              ApplicabilityCheck = fun code -> code.Contains("\"") && not (code.Contains("const")) }
            
            { PatternId = "GEN005"
              BeforePattern = @"if\s*\(([^)]{50,})\)\s*{([^}]*)}"
              AfterTemplate = "// Break down complex condition\nbool conditionIsMet = $1;\nif (conditionIsMet) {$2}"
              Description = "Break down complex conditional into named boolean variable"
              ApplicabilityCheck = fun code -> code.Contains("if (") && code.Contains("&&") && code.Contains("||") }
        ]
    
    /// <summary>
    /// Gets an improvement suggestion for a pattern match
    /// </summary>
    let getSuggestion (patternId: string) (code: string) =
        commonSuggestions
        |> List.tryFind (fun s -> s.PatternId = patternId && s.ApplicabilityCheck(code))
    
    /// <summary>
    /// Applies an improvement suggestion to code
    /// </summary>
    let applySuggestion (suggestion: ImprovementSuggestion) (code: string) =
        let regex = new Regex(suggestion.BeforePattern)
        let result = regex.Replace(code, suggestion.AfterTemplate)
        result
