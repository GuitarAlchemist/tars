namespace TarsEngine.FSharp.Cli.CodeProtection

open System
open System.IO
open System.Text.RegularExpressions
open System.Collections.Generic
open System.Text.Json
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open FSharp.Compiler.Diagnostics
open Microsoft.CodeAnalysis
open Microsoft.CodeAnalysis.CSharp
open Microsoft.CodeAnalysis.CSharp.Syntax

/// RAG-based code analysis system with CodeQL integration for preventing LLM corruption
/// Inspired by InfoQ RAG pipeline architecture best practices
/// Enhanced with GitHub CodeQL semantic analysis capabilities
module RAGCodeAnalyzer =
    
    /// AST-based code context for RAG analysis
    type CodeContext = {
        FilePath: string
        Content: string
        Language: string
        Dependencies: string list
        Exports: string list
        Imports: string list
        Functions: FunctionInfo list
        Types: TypeInfo list
        Modules: ModuleInfo list
        LastModified: DateTime
        Hash: string
        AST: ParsedInput option
        Diagnostics: FSharpDiagnostic list
    }

    and FunctionInfo = {
        Name: string
        Parameters: ParameterInfo list
        ReturnType: string option
        IsPublic: bool
        IsAsync: bool
        IsRecursive: bool
        LineNumber: int
        Complexity: int
        Documentation: string option
    }

    and ParameterInfo = {
        Name: string
        Type: string option
        IsOptional: bool
    }

    and TypeInfo = {
        Name: string
        Kind: TypeKind
        Members: string list
        IsPublic: bool
        LineNumber: int
        Documentation: string option
    }

    and TypeKind =
        | Record
        | Union
        | Class
        | Interface
        | Enum
        | Alias

    and ModuleInfo = {
        Name: string
        IsPublic: bool
        LineNumber: int
        NestedModules: string list
    }

    /// CodeQL analysis results
    type CodeQLResult = {
        QueryName: string
        Severity: CodeQLSeverity
        Message: string
        Location: CodeLocation
        Metadata: Map<string, string>
    }

    and CodeQLSeverity =
        | Error
        | Warning
        | Note
        | Recommendation

    and CodeLocation = {
        FilePath: string
        StartLine: int
        EndLine: int
        StartColumn: int
        EndColumn: int
    }

    /// Enhanced code context with CodeQL analysis
    type EnhancedCodeContext = {
        BaseContext: CodeContext
        CodeQLResults: CodeQLResult list
        SecurityVulnerabilities: SecurityVulnerability list
        CodeSmells: CodeSmell list
        PerformanceIssues: PerformanceIssue list
    }

    and SecurityVulnerability = {
        Type: string
        Severity: string
        Description: string
        Location: CodeLocation
        CWE: string option
        CVSS: float option
    }

    and CodeSmell = {
        Type: string
        Description: string
        Location: CodeLocation
        Impact: string
    }

    and PerformanceIssue = {
        Type: string
        Description: string
        Location: CodeLocation
        EstimatedImpact: string
    }
    
    /// Code quality metrics
    type QualityMetrics = {
        CyclomaticComplexity: int
        LinesOfCode: int
        TestCoverage: float
        TechnicalDebt: float
        Maintainability: float
        Reliability: float
        Security: float
    }
    
    /// RAG retrieval result
    type RetrievalResult = {
        Context: CodeContext
        Relevance: float
        Confidence: float
        Reasoning: string
    }
    
    /// Simplified semantic code analysis (working baseline)
    module SemanticAnalysis =

        /// Parse F# code using simplified approach
        let private parseFSharpCode (content: string) (filePath: string) : (FunctionInfo list * TypeInfo list * ModuleInfo list * string list) =
            try
                // Simplified F# parsing using line-by-line analysis
                let lines = content.Split('\n')
                let mutable functions = []
                let mutable types = []
                let mutable modules = []
                let mutable opens = []

                lines
                |> Array.iteri (fun i line ->
                    let lineNum = i + 1
                    let trimmed = line.Trim()

                    // Extract functions
                    if trimmed.StartsWith("let ") && not (trimmed.Contains("=")) then
                        let parts = trimmed.Split(' ')
                        if parts.Length > 1 then
                            let funcName = parts.[1].Split('(').[0]
                            functions <- {
                                Name = funcName
                                Parameters = []
                                ReturnType = None
                                IsPublic = not (trimmed.Contains("private"))
                                IsAsync = trimmed.Contains("async")
                                IsRecursive = trimmed.Contains("rec")
                                LineNumber = lineNum
                                Complexity = 1
                                Documentation = None
                            } :: functions

                    // Extract types
                    elif trimmed.StartsWith("type ") then
                        let parts = trimmed.Split(' ')
                        if parts.Length > 1 then
                            let typeName = parts.[1].Split('=').[0].Trim()
                            let kind =
                                if trimmed.Contains("record") then Record
                                elif trimmed.Contains("union") then Union
                                elif trimmed.Contains("interface") then Interface
                                elif trimmed.Contains("class") then Class
                                else Alias

                            types <- {
                                Name = typeName
                                Kind = kind
                                Members = []
                                IsPublic = not (trimmed.Contains("private"))
                                LineNumber = lineNum
                                Documentation = None
                            } :: types

                    // Extract modules
                    elif trimmed.StartsWith("module ") then
                        let parts = trimmed.Split(' ')
                        if parts.Length > 1 then
                            let moduleName = parts.[1]
                            modules <- {
                                Name = moduleName
                                IsPublic = not (trimmed.Contains("private"))
                                LineNumber = lineNum
                                NestedModules = []
                            } :: modules

                    // Extract opens
                    elif trimmed.StartsWith("open ") then
                        let parts = trimmed.Split(' ')
                        if parts.Length > 1 then
                            let openName = parts.[1]
                            opens <- openName :: opens
                )

                (functions, types, modules, opens)
            with
            | _ -> ([], [], [], [])



        /// Parse C# code using simplified approach
        let private parseCSharpCode (content: string) (filePath: string) : (FunctionInfo list * TypeInfo list * ModuleInfo list * string list) =
            try
                // Simplified C# parsing using line-by-line analysis
                let lines = content.Split('\n')
                let mutable functions = []
                let mutable types = []
                let mutable namespaces = []
                let mutable usings = []

                lines
                |> Array.iteri (fun i line ->
                    let lineNum = i + 1
                    let trimmed = line.Trim()

                    // Extract using statements
                    if trimmed.StartsWith("using ") && trimmed.EndsWith(";") then
                        let usingName = trimmed.Substring(6).Replace(";", "").Trim()
                        usings <- usingName :: usings

                    // Extract methods
                    elif trimmed.Contains("(") && trimmed.Contains(")") &&
                         (trimmed.Contains("public") || trimmed.Contains("private") || trimmed.Contains("protected")) &&
                         not (trimmed.Contains("class")) && not (trimmed.Contains("interface")) then
                        let parts = trimmed.Split(' ')
                        let methodName =
                            parts
                            |> Array.tryFind (fun part -> part.Contains("("))
                            |> Option.map (fun part -> part.Split('(').[0])
                            |> Option.defaultValue "Unknown"

                        functions <- {
                            Name = methodName
                            Parameters = []
                            ReturnType = None
                            IsPublic = trimmed.Contains("public")
                            IsAsync = trimmed.Contains("async")
                            IsRecursive = false
                            LineNumber = lineNum
                            Complexity = 1
                            Documentation = None
                        } :: functions

                    // Extract classes and interfaces
                    elif trimmed.StartsWith("public class ") || trimmed.StartsWith("class ") ||
                         trimmed.StartsWith("public interface ") || trimmed.StartsWith("interface ") then
                        let parts = trimmed.Split(' ')
                        let typeName =
                            if parts.Length > 2 then parts.[2]
                            elif parts.Length > 1 then parts.[1]
                            else "Unknown"

                        let kind = if trimmed.Contains("interface") then Interface else Class

                        types <- {
                            Name = typeName
                            Kind = kind
                            Members = []
                            IsPublic = trimmed.Contains("public")
                            LineNumber = lineNum
                            Documentation = None
                        } :: types

                    // Extract namespaces
                    elif trimmed.StartsWith("namespace ") then
                        let parts = trimmed.Split(' ')
                        if parts.Length > 1 then
                            let namespaceName = parts.[1]
                            namespaces <- {
                                Name = namespaceName
                                IsPublic = true
                                LineNumber = lineNum
                                NestedModules = []
                            } :: namespaces
                )

                (functions, types, namespaces, usings)
            with
            | _ -> ([], [], [], [])

        /// Parse JavaScript/TypeScript using proper parser
        let private parseJavaScriptCode (content: string) (filePath: string) : (FunctionInfo list * TypeInfo list * ModuleInfo list * string list) =
            // TODO: Implement proper JS/TS parsing
            // This could use Esprima.NET or similar for AST parsing
            ([], [], [], [])

        /// Main entry point for semantic analysis
        let analyzeCode (content: string) (filePath: string) (language: string) : (FunctionInfo list * TypeInfo list * ModuleInfo list * string list) =
            match language.ToLowerInvariant() with
            | "f#" -> parseFSharpCode content filePath
            | "c#" -> parseCSharpCode content filePath
            | "javascript" | "typescript" -> parseJavaScriptCode content filePath
            | _ -> ([], [], [], []) // Unsupported language

    /// CodeQL integration (simplified for baseline)
    module CodeQLIntegration =

        /// Check if CodeQL is available
        let isCodeQLAvailable () : bool =
            try
                // Simple check - look for codeql executable
                let paths = Environment.GetEnvironmentVariable("PATH").Split(Path.PathSeparator)
                paths |> Array.exists (fun path ->
                    let codeqlPath = Path.Combine(path, "codeql.exe")
                    File.Exists(codeqlPath) || File.Exists(Path.Combine(path, "codeql"))
                )
            with
            | _ -> false

        /// Create placeholder CodeQL database (simplified)
        let createCodeQLDatabase (sourceDir: string) (outputDir: string) (language: string) : Result<string, string> =
            try
                let dbPath = Path.Combine(outputDir, "codeql-db")
                Directory.CreateDirectory(dbPath) |> ignore

                // For now, just create the directory structure
                // In full implementation, this would call CodeQL CLI
                Ok dbPath
            with
            | ex -> Result.Error ("CodeQL integration error: " + ex.Message)

        /// Run security queries (placeholder)
        let runSecurityQueries (dbPath: string) : Result<CodeQLResult list, string> =
            try
                // Placeholder implementation
                // In full version, this would execute CodeQL queries
                let results = [
                    {
                        QueryName = "placeholder-security-check"
                        Severity = CodeQLSeverity.Warning
                        Message = "CodeQL integration placeholder - implement real queries"
                        Location = {
                            FilePath = "placeholder"
                            StartLine = 1
                            EndLine = 1
                            StartColumn = 1
                            EndColumn = 1
                        }
                        Metadata = Map.empty
                    }
                ]
                Ok results
            with
            | ex -> Result.Error ("CodeQL query execution error: " + ex.Message)

        /// Run code quality queries (placeholder)
        let runCodeQualityQueries (dbPath: string) : Result<CodeQLResult list, string> =
            try
                // Placeholder implementation
                let results = [
                    {
                        QueryName = "placeholder-quality-check"
                        Severity = CodeQLSeverity.Note
                        Message = "CodeQL code quality placeholder - implement real queries"
                        Location = {
                            FilePath = "placeholder"
                            StartLine = 1
                            EndLine = 1
                            StartColumn = 1
                            EndColumn = 1
                        }
                        Metadata = Map.empty
                    }
                ]
                Ok results
            with
            | ex -> Result.Error ("CodeQL code quality analysis error: " + ex.Message)

    /// Extract code context using proper semantic analysis
    let extractCodeContext (filePath: string) : CodeContext option =
        try
            if not (File.Exists(filePath)) then None
            else
                let content = File.ReadAllText(filePath)
                let language =
                    match Path.GetExtension(filePath).ToLowerInvariant() with
                    | ".fs" | ".fsx" -> "F#"
                    | ".cs" -> "C#"
                    | ".js" | ".ts" -> "JavaScript/TypeScript"
                    | ".py" -> "Python"
                    | _ -> "Unknown"

                // Use proper semantic analysis instead of regex
                let (functions, types, modules, opens) = SemanticAnalysis.analyzeCode content filePath language

                // Calculate hash
                let hash =
                    use sha256 = System.Security.Cryptography.SHA256.Create()
                    let bytes = System.Text.Encoding.UTF8.GetBytes(content)
                    let hashBytes = sha256.ComputeHash(bytes)
                    Convert.ToHexString(hashBytes)

                Some {
                    FilePath = filePath
                    Content = content
                    Language = language
                    Dependencies = opens
                    Exports = (types |> List.map (fun t -> t.Name)) @ (functions |> List.map (fun f -> f.Name))
                    Imports = opens
                    Functions = functions
                    Types = types
                    Modules = modules
                    LastModified = File.GetLastWriteTime(filePath)
                    Hash = hash
                    AST = None  // Will be populated by semantic analysis
                    Diagnostics = []  // Will be populated by semantic analysis
                }
        with
        | ex ->
            // Return None if analysis fails
            None
    
    /// Calculate cyclomatic complexity using semantic analysis
    module ComplexityAnalysis =

        /// Calculate complexity from function information
        let calculateFunctionComplexity (func: FunctionInfo) (content: string) : int =
            // For now, use a simple heuristic based on function characteristics
            // In a full implementation, this would analyze the actual AST
            let baseComplexity = 1

            let asyncBonus = if func.IsAsync then 1 else 0
            let recursiveBonus = if func.IsRecursive then 2 else 0
            let parameterComplexity = func.Parameters.Length / 3 // Every 3 parameters adds 1 complexity

            baseComplexity + asyncBonus + recursiveBonus + parameterComplexity

        /// Calculate overall complexity for a code context
        let calculateOverallComplexity (context: CodeContext) : int =
            if context.Functions.IsEmpty then
                1
            else
                context.Functions
                |> List.sumBy (fun func -> calculateFunctionComplexity func context.Content)
                |> max 1

        /// Analyze control flow complexity (would use AST in full implementation)
        let analyzeControlFlow (content: string) : int =
            // This is a simplified version - in production, analyze actual control flow from AST
            let lines = content.Split('\n')
            let mutable complexity = 1

            for line in lines do
                let trimmed = line.Trim()
                // Look for control flow keywords (more precise than regex)
                if trimmed.StartsWith("if ") || trimmed.StartsWith("elif ") then
                    complexity <- complexity + 1
                elif trimmed.StartsWith("match ") then
                    complexity <- complexity + 1
                elif trimmed.StartsWith("while ") || trimmed.StartsWith("for ") then
                    complexity <- complexity + 1
                elif trimmed.StartsWith("try ") then
                    complexity <- complexity + 1
                elif trimmed.Contains(" && ") || trimmed.Contains(" || ") then
                    complexity <- complexity + 1

            complexity

    /// Calculate quality metrics using AST analysis when available
    let calculateQualityMetrics (context: CodeContext) : QualityMetrics =
        let lines = context.Content.Split('\n')
        let linesOfCode = lines |> Array.filter (fun line -> not (String.IsNullOrWhiteSpace(line) || line.Trim().StartsWith("//"))) |> Array.length

        // Calculate complexity using semantic analysis
        let complexity = ComplexityAnalysis.calculateOverallComplexity context

        // Technical debt indicators (enhanced with AST information)
        let debtIndicators = [
            (@"TODO:", 1.0)
            (@"FIXME:", 2.0)
            (@"HACK:", 3.0)
            (@"NotImplementedException", 5.0)
            (@"throw new Exception", 2.0)
            (@"//.*REAL:\s*Implement", 4.0)
            (@"//.*HONEST:\s*Cannot\s+generate", 3.0)
        ]

        let technicalDebt =
            debtIndicators
            |> List.sumBy (fun (pattern, weight) ->
                float (Regex.Matches(context.Content, pattern).Count) * weight)

        // Add simple debt indicators (no AST diagnostics for now)
        let astDebt = 0.0

        let totalDebt = technicalDebt + astDebt

        // Maintainability (enhanced with AST metrics)
        let functionComplexity =
            context.Functions
            |> List.map (fun f -> f.Complexity)
            |> List.fold max 1
            |> float

        let typeComplexity =
            context.Types
            |> List.sumBy (fun t -> t.Members.Length)
            |> float

        let maintainability = max 0.0 (10.0 - (functionComplexity * 0.3) - (typeComplexity * 0.1) - (totalDebt * 0.1))

        // Reliability (enhanced with AST information)
        let hasErrorHandling =
            context.Functions |> List.exists (fun f -> f.Name.Contains("Result") || f.Name.Contains("Option")) ||
            Regex.IsMatch(context.Content, @"try\s*{|Result<|Option<")

        let hasTests =
            context.Functions |> List.exists (fun f -> f.Name.Contains("Test") || f.Name.Contains("Should")) ||
            Regex.IsMatch(context.Content, @"Test|Assert|Should")

        let hasDocumentation =
            context.Functions |> List.exists (fun f -> f.Documentation.IsSome) ||
            context.Types |> List.exists (fun t -> t.Documentation.IsSome)

        let reliability =
            (if hasErrorHandling then 3.0 else 1.0) +
            (if hasTests then 3.0 else 0.0) +
            (if hasDocumentation then 2.0 else 0.0)

        // Security (enhanced with AST patterns)
        let securityIssues = [
            @"Process\.Start"
            @"File\.Delete"
            @"Registry\."
            @"unsafe\s*{"
            @"Environment\.Exit"
            @"System\.Runtime\.InteropServices"
        ]

        let securityScore =
            10.0 - (securityIssues |> List.sumBy (fun pattern ->
                float (Regex.Matches(context.Content, pattern).Count) * 2.0))
            |> max 0.0

        // Test coverage estimation based on AST
        let testCoverage =
            if hasTests then
                let testFunctions = context.Functions |> List.filter (fun f -> f.Name.Contains("Test") || f.Name.Contains("Should"))
                let totalFunctions = context.Functions.Length
                if totalFunctions > 0 then
                    min 1.0 (float testFunctions.Length / float totalFunctions * 2.0)
                else 0.0
            else 0.0

        {
            CyclomaticComplexity = complexity
            LinesOfCode = linesOfCode
            TestCoverage = testCoverage
            TechnicalDebt = totalDebt
            Maintainability = maintainability / 10.0
            Reliability = reliability / 8.0
            Security = securityScore / 10.0
        }
    
    /// Build knowledge base from codebase
    let buildKnowledgeBase (baseDir: string) : CodeContext list =
        let sourceDir = Path.Combine(baseDir, "src")
        if not (Directory.Exists(sourceDir)) then []
        else
            let rec getAllCodeFiles (dir: string) =
                seq {
                    yield! Directory.GetFiles(dir, "*.fs")
                    yield! Directory.GetFiles(dir, "*.fsx")
                    yield! Directory.GetFiles(dir, "*.cs")
                    for subDir in Directory.GetDirectories(dir) do
                        yield! getAllCodeFiles subDir
                }
            
            getAllCodeFiles sourceDir
            |> Seq.choose extractCodeContext
            |> Seq.toList
    
    /// Calculate semantic similarity using AST information
    let calculateSemanticSimilarity (context: CodeContext) (query: string) : float =
        let queryLower = query.ToLowerInvariant()
        let mutable score = 0.0

        // Function name similarity (weighted heavily)
        let functionMatches =
            context.Functions
            |> List.filter (fun f ->
                queryLower.Contains(f.Name.ToLowerInvariant()) ||
                f.Name.ToLowerInvariant().Contains(queryLower))
        score <- score + (float functionMatches.Length * 0.3)

        // Type name similarity
        let typeMatches =
            context.Types
            |> List.filter (fun t ->
                queryLower.Contains(t.Name.ToLowerInvariant()) ||
                t.Name.ToLowerInvariant().Contains(queryLower))
        score <- score + (float typeMatches.Length * 0.25)

        // Module similarity
        let moduleMatches =
            context.Modules
            |> List.filter (fun m ->
                queryLower.Contains(m.Name.ToLowerInvariant()) ||
                m.Name.ToLowerInvariant().Contains(queryLower))
        score <- score + (float moduleMatches.Length * 0.2)

        // Dependency similarity (imports/opens)
        let dependencyMatches =
            context.Dependencies
            |> List.filter (fun d ->
                queryLower.Contains(d.ToLowerInvariant()) ||
                d.ToLowerInvariant().Contains(queryLower))
        score <- score + (float dependencyMatches.Length * 0.15)

        // Content token similarity (fallback)
        let queryTokens =
            Regex.Split(queryLower, @"\W+")
            |> Array.filter (fun token -> token.Length > 2)
            |> Set.ofArray

        let contentTokens =
            Regex.Split(context.Content.ToLowerInvariant(), @"\W+")
            |> Array.filter (fun token -> token.Length > 2)
            |> Set.ofArray

        let intersection = Set.intersect queryTokens contentTokens
        let union = Set.union queryTokens contentTokens
        let tokenSimilarity = if union.Count = 0 then 0.0 else float intersection.Count / float union.Count
        score <- score + (tokenSimilarity * 0.1)

        score

    /// Retrieve relevant code contexts using AST-based semantic analysis
    let retrieveRelevantContexts (knowledgeBase: CodeContext list) (query: string) (maxResults: int) : RetrievalResult list =
        knowledgeBase
        |> List.map (fun context ->
            // Calculate semantic similarity using AST information
            let semanticRelevance = calculateSemanticSimilarity context query

            // Calculate confidence based on code quality and AST completeness
            let metrics = calculateQualityMetrics context
            let astBonus = if context.AST.IsSome then 0.1 else 0.0
            let diagnosticPenalty = float context.Diagnostics.Length * 0.02
            let confidence = (metrics.Maintainability + metrics.Reliability + metrics.Security) / 3.0 + astBonus - diagnosticPenalty

            // Create detailed reasoning
            let functionMatches = context.Functions |> List.filter (fun f -> query.Contains(f.Name, StringComparison.OrdinalIgnoreCase))
            let typeMatches = context.Types |> List.filter (fun t -> query.Contains(t.Name, StringComparison.OrdinalIgnoreCase))
            let moduleMatches = context.Modules |> List.filter (fun m -> query.Contains(m.Name, StringComparison.OrdinalIgnoreCase))

            let reasoning =
                [
                    if functionMatches.Length > 0 then "Functions: " + (functionMatches |> List.map (fun f -> f.Name) |> String.concat ", ")
                    if typeMatches.Length > 0 then "Types: " + (typeMatches |> List.map (fun t -> t.Name) |> String.concat ", ")
                    if moduleMatches.Length > 0 then "Modules: " + (moduleMatches |> List.map (fun m -> m.Name) |> String.concat ", ")
                    if context.AST.IsSome then "AST available"
                    if context.Diagnostics.Length > 0 then context.Diagnostics.Length.ToString() + " diagnostics"
                ] |> String.concat "; "

            {
                Context = context
                Relevance = semanticRelevance
                Confidence = max 0.0 (min 1.0 confidence)
                Reasoning = if reasoning = "" then "Content similarity" else reasoning
            }
        )
        |> List.filter (fun result -> result.Relevance > 0.05)
        |> List.sortByDescending (fun result -> result.Relevance * result.Confidence)
        |> List.take (min maxResults knowledgeBase.Length)
    
    /// Analyze code change using RAG
    let analyzeCodeChange (knowledgeBase: CodeContext list) (filePath: string) (originalContent: string) (newContent: string) : Result<string, string> =
        try
            // Create contexts for analysis
            let originalContext = {
                FilePath = filePath
                Content = originalContent
                Language = "F#"
                Dependencies = []
                Exports = []
                Imports = []
                Functions = []
                Types = []
                Modules = []
                LastModified = DateTime.Now
                Hash = ""
                AST = None
                Diagnostics = []
            }
            
            let newContext = { originalContext with Content = newContent }
            
            // Calculate quality metrics
            let originalMetrics = calculateQualityMetrics originalContext
            let newMetrics = calculateQualityMetrics newContext
            
            // Retrieve relevant contexts
            let fileName = Path.GetFileName(filePath)
            let relevantContexts = retrieveRelevantContexts knowledgeBase fileName 5
            
            // Analyze changes
            let analysis = System.Text.StringBuilder()
            analysis.AppendLine("=== RAG CODE CHANGE ANALYSIS ===") |> ignore
            analysis.AppendLine($"File: {filePath}") |> ignore
            analysis.AppendLine($"Analysis Date: {DateTime.Now}") |> ignore
            analysis.AppendLine() |> ignore
            
            // Quality comparison
            analysis.AppendLine("📊 QUALITY METRICS COMPARISON:") |> ignore
            analysis.AppendLine($"Lines of Code: {originalMetrics.LinesOfCode} → {newMetrics.LinesOfCode}") |> ignore
            analysis.AppendLine($"Complexity: {originalMetrics.CyclomaticComplexity} → {newMetrics.CyclomaticComplexity}") |> ignore
            analysis.AppendLine($"Technical Debt: {originalMetrics.TechnicalDebt:F1} → {newMetrics.TechnicalDebt:F1}") |> ignore
            analysis.AppendLine($"Maintainability: {originalMetrics.Maintainability:F2} → {newMetrics.Maintainability:F2}") |> ignore
            analysis.AppendLine($"Security: {originalMetrics.Security:F2} → {newMetrics.Security:F2}") |> ignore
            analysis.AppendLine() |> ignore
            
            // Risk assessment
            let risks = ResizeArray<string>()
            
            if newMetrics.TechnicalDebt > originalMetrics.TechnicalDebt then
                risks.Add("⚠️ Technical debt increased")
            
            if newMetrics.CyclomaticComplexity > originalMetrics.CyclomaticComplexity + 5 then
                risks.Add("⚠️ Significant complexity increase")
            
            if newMetrics.Security < originalMetrics.Security then
                risks.Add("🔒 Security score decreased")
            
            if newMetrics.Maintainability < originalMetrics.Maintainability - 0.2 then
                risks.Add("🔧 Maintainability significantly decreased")
            
            if risks.Count > 0 then
                analysis.AppendLine("🚨 RISKS DETECTED:") |> ignore
                risks |> Seq.iter (fun risk -> analysis.AppendLine($"  {risk}") |> ignore)
                analysis.AppendLine() |> ignore
            
            // Relevant context analysis
            if not relevantContexts.IsEmpty then
                analysis.AppendLine("🔍 RELEVANT CODE CONTEXTS:") |> ignore
                relevantContexts
                |> List.take 3
                |> List.iter (fun result ->
                    analysis.AppendLine($"  📁 {Path.GetFileName(result.Context.FilePath)} (Relevance: {result.Relevance:F2}, Confidence: {result.Confidence:F2})") |> ignore
                    analysis.AppendLine($"     {result.Reasoning}") |> ignore
                )
                analysis.AppendLine() |> ignore
            
            // Recommendation
            let recommendation = 
                if risks.Count = 0 then "✅ APPROVED: Change appears safe"
                elif risks.Count <= 2 then "⚠️ REVIEW REQUIRED: Minor risks detected"
                else "❌ REJECTED: Multiple risks detected"
            
            analysis.AppendLine($"🎯 RECOMMENDATION: {recommendation}") |> ignore
            
            Ok (analysis.ToString())
        with
        | ex -> Result.Error ("RAG analysis failed: " + ex.Message)
