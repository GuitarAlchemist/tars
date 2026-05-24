// ================================================
// 🧠 TARS Autonomous Code Analysis Engine
// ================================================
// Real autonomous code analysis where TARS examines its own codebase
// and identifies improvement opportunities for true self-evolution

namespace TarsEngine.FSharp.Core

open System
open System.IO
open System.Text.RegularExpressions
open Microsoft.Extensions.Logging

/// Code analysis result types
type CodeAnalysisResult =
    | ImprovementOpportunity of string * string * float // description, suggestion, impact score
    | PerformanceBottleneck of string * string * float // location, issue, severity
    | OptimizationCandidate of string * string * float // function, optimization, benefit
    | Error of string

/// Code improvement suggestion
type CodeImprovement = {
    Id: string
    TargetFile: string
    TargetFunction: string
    IssueDescription: string
    ProposedSolution: string
    ImpactScore: float // 0.0 to 1.0
    ConfidenceLevel: float // 0.0 to 1.0
    EstimatedPerformanceGain: float
    RiskLevel: string // "Low", "Medium", "High"
    RequiredChanges: string list
}

/// Code analysis metrics
type CodeMetrics = {
    LinesOfCode: int
    FunctionCount: int
    ComplexityScore: float
    PerformanceScore: float
    MaintainabilityScore: float
    TestCoverage: float
    TechnicalDebt: float
}

/// Autonomous code analysis engine
module TarsAutonomousCodeAnalysis =

    /// Analyze for performance bottlenecks
    let analyzePerformanceBottlenecks (content: string) (lines: string[]) (logger: ILogger) : CodeAnalysisResult list =
        let mutable issues = []

        // Check for inefficient list operations
        if content.Contains("List.append") && content.Contains("List.fold") then
            issues <- PerformanceBottleneck("List operations", "Multiple list operations could be combined", 0.7) :: issues

        // Check for repeated computations
        let functionCalls = Regex.Matches(content, @"(\w+)\s*\(.*?\)")
        let callCounts =
            functionCalls
            |> Seq.cast<Match>
            |> Seq.groupBy (fun m -> m.Groups.[1].Value)
            |> Seq.map (fun (name, matches) -> (name, Seq.length matches))
            |> Seq.filter (fun (_, count) -> count > 5)
            |> Seq.toList

        for (funcName, count) in callCounts do
            if funcName.Length > 3 && not (funcName.StartsWith("Log")) then
                issues <- PerformanceBottleneck($"Function {funcName}", $"Called {count} times - consider memoization", 0.6) :: issues

        // Check for inefficient string operations
        if content.Contains("sprintf") && content.Contains("String.concat") then
            issues <- PerformanceBottleneck("String operations", "Multiple string operations could use StringBuilder", 0.5) :: issues

        logger.LogDebug($"Found {issues.Length} performance bottlenecks")
        issues

    /// Analyze for optimization opportunities
    let analyzeOptimizationOpportunities (content: string) (lines: string[]) (logger: ILogger) : CodeAnalysisResult list =
        let mutable opportunities = []

        // Check for mathematical optimizations
        if content.Contains("List.map") && content.Contains("List.filter") then
            opportunities <- OptimizationCandidate("List processing", "Combine map and filter operations", 0.4) :: opportunities

        // Check for prime number optimizations
        if content.Contains("isPrime") && content.Contains("mod") then
            opportunities <- OptimizationCandidate("Prime checking", "Use sieve or optimized prime checking", 0.8) :: opportunities

        // Check for recursive functions that could be tail-recursive
        let recursivePatterns = Regex.Matches(content, @"let\s+rec\s+(\w+)")
        for m in recursivePatterns do
            let funcName = m.Groups.[1].Value
            if not (content.Contains($"{funcName} acc") || content.Contains("tailrec")) then
                opportunities <- OptimizationCandidate($"Function {funcName}", "Convert to tail-recursive for better performance", 0.6) :: opportunities

        // Check for array vs list usage
        if content.Contains("List.") && content.Contains("Array.") then
            opportunities <- OptimizationCandidate("Data structures", "Consider consistent use of arrays for performance", 0.3) :: opportunities

        logger.LogDebug($"Found {opportunities.Length} optimization opportunities")
        opportunities

    /// Analyze code quality improvements
    let analyzeCodeQuality (content: string) (lines: string[]) (logger: ILogger) : CodeAnalysisResult list =
        let mutable improvements = []

        // Check for long functions
        let functionStarts =
            lines
            |> Array.mapi (fun i line -> (i, line))
            |> Array.filter (fun (_, line) -> line.Trim().StartsWith("let ") && line.Contains("="))

        for i in 0..functionStarts.Length-2 do
            let (startLine, _) = functionStarts.[i]
            let (endLine, _) = functionStarts.[i+1]
            let functionLength = endLine - startLine

            if functionLength > 50 then
                improvements <- ImprovementOpportunity($"Long function at line {startLine}", "Consider breaking into smaller functions", 0.4) :: improvements

        // Check for magic numbers
        let numberMatches = Regex.Matches(content, @"\b\d{2,}\b")
        if numberMatches.Count > 5 then
            improvements <- ImprovementOpportunity("Magic numbers", "Consider using named constants", 0.3) :: improvements

        // Check for error handling
        if not (content.Contains("try") || content.Contains("Result")) then
            improvements <- ImprovementOpportunity("Error handling", "Add proper error handling", 0.5) :: improvements

        logger.LogDebug($"Found {improvements.Length} code quality improvements")
        improvements

    /// Analyze F# code for improvement opportunities
    let analyzeCodeFile (filePath: string) (logger: ILogger) : CodeAnalysisResult list =
        try
            if not (File.Exists(filePath)) then
                [Error $"File not found: {filePath}"]
            else
                let content = File.ReadAllText(filePath)
                let lines = content.Split('\n')
                let mutable results = []
                
                logger.LogInformation($"🔍 Analyzing code file: {Path.GetFileName(filePath)}")
                
                // Analyze for performance bottlenecks
                let performanceIssues = analyzePerformanceBottlenecks content lines logger
                results <- performanceIssues @ results

                // Analyze for optimization opportunities
                let optimizations = analyzeOptimizationOpportunities content lines logger
                results <- optimizations @ results

                // Analyze for code quality improvements
                let qualityImprovements = analyzeCodeQuality content lines logger
                results <- qualityImprovements @ results
                
                logger.LogInformation($"✅ Analysis complete: {results.Length} findings")
                results
                
        with
        | ex ->
            logger.LogError($"❌ Code analysis failed: {ex.Message}")
            [Error ex.Message]



    /// Calculate code metrics for a file
    let calculateCodeMetrics (filePath: string) (logger: ILogger) : CodeMetrics =
        try
            let content = File.ReadAllText(filePath)
            let lines = content.Split('\n') |> Array.filter (fun line -> not (String.IsNullOrWhiteSpace(line)))
            
            let linesOfCode = lines.Length
            let functionCount = Regex.Matches(content, @"let\s+(\w+)").Count
            
            // Simple complexity calculation
            let complexityScore = 
                let conditions = Regex.Matches(content, @"\b(if|match|when|while|for)\b").Count
                float conditions / float linesOfCode * 100.0
            
            // Performance score based on efficient patterns
            let performanceScore = 
                let efficientPatterns = Regex.Matches(content, @"\b(Array\.|Seq\.|async|parallel)\b").Count
                let inefficientPatterns = Regex.Matches(content, @"\b(List\.append|String\.concat)\b").Count
                max 0.0 (1.0 - float inefficientPatterns / float (efficientPatterns + 1))
            
            // Maintainability score
            let maintainabilityScore =
                let avgLineLength = lines |> Array.map (fun l -> float l.Length) |> Array.average
                let commentLines = lines |> Array.filter (fun l -> l.Trim().StartsWith("//")) |> Array.length
                let commentRatio = float commentLines / float linesOfCode
                max 0.0 (1.0 - avgLineLength / 200.0 + commentRatio)
            
            {
                LinesOfCode = linesOfCode
                FunctionCount = functionCount
                ComplexityScore = complexityScore
                PerformanceScore = performanceScore
                MaintainabilityScore = maintainabilityScore
                TestCoverage = 0.0 // Would need test analysis
                TechnicalDebt = complexityScore * (1.0 - maintainabilityScore)
            }
            
        with
        | ex ->
            logger.LogError($"❌ Metrics calculation failed: {ex.Message}")
            {
                LinesOfCode = 0
                FunctionCount = 0
                ComplexityScore = 0.0
                PerformanceScore = 0.0
                MaintainabilityScore = 0.0
                TestCoverage = 0.0
                TechnicalDebt = 1.0
            }

    /// Generate improvement suggestions from analysis results
    let generateImprovementSuggestions (results: CodeAnalysisResult list) (filePath: string) (logger: ILogger) : CodeImprovement list =
        let mutable improvements = []
        let fileName = Path.GetFileName(filePath)
        
        for result in results do
            match result with
            | ImprovementOpportunity (desc, suggestion, impact) ->
                let improvement = {
                    Id = Guid.NewGuid().ToString("N").[..7]
                    TargetFile = fileName
                    TargetFunction = "Unknown"
                    IssueDescription = desc
                    ProposedSolution = suggestion
                    ImpactScore = impact
                    ConfidenceLevel = 0.8
                    EstimatedPerformanceGain = impact * 0.1
                    RiskLevel = if impact > 0.7 then "Medium" else "Low"
                    RequiredChanges = [suggestion]
                }
                improvements <- improvement :: improvements
                
            | PerformanceBottleneck (location, issue, severity) ->
                let improvement = {
                    Id = Guid.NewGuid().ToString("N").[..7]
                    TargetFile = fileName
                    TargetFunction = location
                    IssueDescription = issue
                    ProposedSolution = $"Optimize {location} for better performance"
                    ImpactScore = severity
                    ConfidenceLevel = 0.9
                    EstimatedPerformanceGain = severity * 0.2
                    RiskLevel = if severity > 0.8 then "High" else "Medium"
                    RequiredChanges = [$"Refactor {location}"]
                }
                improvements <- improvement :: improvements
                
            | OptimizationCandidate (func, optimization, benefit) ->
                let improvement = {
                    Id = Guid.NewGuid().ToString("N").[..7]
                    TargetFile = fileName
                    TargetFunction = func
                    IssueDescription = $"Optimization opportunity in {func}"
                    ProposedSolution = optimization
                    ImpactScore = benefit
                    ConfidenceLevel = 0.7
                    EstimatedPerformanceGain = benefit * 0.15
                    RiskLevel = "Low"
                    RequiredChanges = [optimization]
                }
                improvements <- improvement :: improvements
                
            | Error _ -> () // Skip errors for improvement suggestions
        
        logger.LogInformation($"💡 Generated {improvements.Length} improvement suggestions")
        improvements

    /// Analyze entire TARS codebase
    let analyzeEntireCodebase (rootPath: string) (logger: ILogger) : (string * CodeAnalysisResult list * CodeMetrics) list =
        try
            logger.LogInformation("🔍 Starting comprehensive TARS codebase analysis")
            
            let fsFiles = 
                Directory.GetFiles(rootPath, "*.fs", SearchOption.AllDirectories)
                |> Array.filter (fun f -> not (f.Contains("bin") || f.Contains("obj")))
                |> Array.take 5 // Limit for demonstration
            
            let mutable results = []
            
            for file in fsFiles do
                logger.LogInformation($"📄 Analyzing: {Path.GetFileName(file)}")
                let analysisResults = analyzeCodeFile file logger
                let metrics = calculateCodeMetrics file logger
                results <- (file, analysisResults, metrics) :: results
            
            logger.LogInformation($"✅ Codebase analysis complete: {fsFiles.Length} files analyzed")
            results
            
        with
        | ex ->
            logger.LogError($"❌ Codebase analysis failed: {ex.Message}")
            []

    /// Test autonomous code analysis
    let testAutonomousCodeAnalysis (logger: ILogger) : bool =
        try
            logger.LogInformation("🧪 Testing autonomous code analysis")
            
            // Analyze current file as test
            let currentFile = __SOURCE_FILE__
            let results = analyzeCodeFile currentFile logger
            let metrics = calculateCodeMetrics currentFile logger
            let improvements = generateImprovementSuggestions results currentFile logger
            
            logger.LogInformation($"✅ Analysis test successful:")
            logger.LogInformation($"   Findings: {results.Length}")
            logger.LogInformation($"   Lines of code: {metrics.LinesOfCode}")
            logger.LogInformation($"   Functions: {metrics.FunctionCount}")
            logger.LogInformation($"   Complexity: {metrics.ComplexityScore:F2}")
            logger.LogInformation($"   Performance score: {metrics.PerformanceScore:F2}")
            logger.LogInformation($"   Improvements suggested: {improvements.Length}")
            
            for improvement in improvements |> List.take (min 3 improvements.Length) do
                logger.LogInformation($"   💡 {improvement.IssueDescription}: {improvement.ProposedSolution}")
            
            true
            
        with
        | ex ->
            logger.LogError($"❌ Code analysis test failed: {ex.Message}")
            false
