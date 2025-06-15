namespace TarsEngine.FSharp.Core

open System
open System.IO
open System.Reflection
open System.Text.Json
open Microsoft.Extensions.Logging

/// Real-time component analysis data
type ComponentAnalysis = {
    ComponentId: string
    Name: string
    Type: string
    FilePath: string
    AnalysisTimestamp: DateTime
    SourceCodeHash: string
    Dependencies: string[]
    PublicInterface: string[]
    PrivateMembers: string[]
    ComplexityMetrics: Map<string, float>
    DesignPatterns: string[]
    CodeQuality: Map<string, float>
    RealTimeInsights: string[]
    SuggestedImprovements: string[]
    ReactAgentTasks: string[]
    LLMPrompts: string[]
}

/// Real-time component analyzer - generates fresh analysis every time
type RealTimeComponentAnalyzer(logger: ILogger<RealTimeComponentAnalyzer>) =
    
    /// Analyze source code in real-time
    member private this.AnalyzeSourceCode(filePath: string) =
        try
            if File.Exists(filePath) then
                let content = File.ReadAllText(filePath)
                let lines = content.Split([|'\n'|], StringSplitOptions.None)
                
                // Real-time code analysis
                let publicMembers = 
                    lines 
                    |> Array.filter (fun line -> 
                        line.Trim().StartsWith("member ") || 
                        line.Trim().StartsWith("let ") ||
                        line.Trim().StartsWith("type "))
                    |> Array.map (fun line -> line.Trim())
                
                let privateMethods = 
                    lines
                    |> Array.filter (fun line -> 
                        line.Trim().StartsWith("let private ") ||
                        line.Trim().StartsWith("member private "))
                    |> Array.map (fun line -> line.Trim())
                
                // Calculate real complexity metrics
                let cyclomaticComplexity = 
                    lines
                    |> Array.sumBy (fun line ->
                        let trimmed = line.Trim()
                        if trimmed.Contains("if ") || trimmed.Contains("match ") || 
                           trimmed.Contains("while ") || trimmed.Contains("for ") then 1.0
                        else 0.0)
                
                let codeLength = float lines.Length
                let commentRatio = 
                    let commentLines = lines |> Array.filter (fun line -> line.Trim().StartsWith("//") || line.Trim().StartsWith("(*"))
                    if lines.Length > 0 then float commentLines.Length / float lines.Length else 0.0
                
                // Detect design patterns in real-time
                let detectedPatterns = [
                    if content.Contains("interface ") then "Interface Pattern"
                    if content.Contains("abstract ") then "Abstract Pattern"
                    if content.Contains("inherit ") then "Inheritance Pattern"
                    if content.Contains("module ") then "Module Pattern"
                    if content.Contains("type ") && content.Contains("=") then "Type Definition Pattern"
                    if content.Contains("member this.") then "Object-Oriented Pattern"
                    if content.Contains("let rec ") then "Recursive Pattern"
                    if content.Contains("|>") then "Pipeline Pattern"
                    if content.Contains("async ") then "Async Pattern"
                    if content.Contains("task ") then "Task Pattern"
                ]
                
                (publicMembers, privateMethods, cyclomaticComplexity, codeLength, commentRatio, detectedPatterns)
            else
                ([||], [||], 0.0, 0.0, 0.0, [])
        with
        | ex ->
            logger.LogError(ex, sprintf "Failed to analyze source code: %s" filePath)
            ([||], [||], 0.0, 0.0, 0.0, [])
    
    /// Generate real-time insights (never canned)
    member private this.GenerateRealTimeInsights(analysis: ComponentAnalysis) =
        let timestamp = DateTime.UtcNow
        let insights = ResizeArray<string>()
        
        // Generate fresh insights based on current state
        insights.Add(sprintf "Analysis performed at %s - completely fresh data" (timestamp.ToString("HH:mm:ss.fff")))
        
        if analysis.ComplexityMetrics.ContainsKey("CyclomaticComplexity") then
            let complexity = analysis.ComplexityMetrics.["CyclomaticComplexity"]
            if complexity > 10.0 then
                insights.Add(sprintf "High complexity detected: %.1f - consider refactoring into smaller functions" complexity)
            elif complexity > 5.0 then
                insights.Add(sprintf "Moderate complexity: %.1f - monitor for potential growth" complexity)
            else
                insights.Add(sprintf "Low complexity: %.1f - well-structured code" complexity)
        
        if analysis.ComplexityMetrics.ContainsKey("CommentRatio") then
            let commentRatio = analysis.ComplexityMetrics.["CommentRatio"]
            if commentRatio < 0.1 then
                insights.Add(sprintf "Low documentation: %.1f%% comments - consider adding more documentation" (commentRatio * 100.0))
            elif commentRatio > 0.3 then
                insights.Add(sprintf "Well documented: %.1f%% comments - good documentation practices" (commentRatio * 100.0))
        
        // Real-time pattern analysis
        if analysis.DesignPatterns.Length > 0 then
            insights.Add(sprintf "Design patterns detected: %s - shows good architectural awareness" (String.Join(", ", analysis.DesignPatterns)))
        
        // Dynamic dependency analysis
        if analysis.Dependencies.Length > 10 then
            insights.Add(sprintf "High dependency count: %d - consider dependency injection or modularization" analysis.Dependencies.Length)
        
        insights.ToArray()
    
    /// Generate React agent tasks dynamically
    member private this.GenerateReactAgentTasks(analysis: ComponentAnalysis) =
        let tasks = ResizeArray<string>()
        
        // Generate dynamic tasks based on real analysis
        if analysis.ComplexityMetrics.ContainsKey("CyclomaticComplexity") && analysis.ComplexityMetrics.["CyclomaticComplexity"] > 8.0 then
            tasks.Add(sprintf "CREATE_REFACTORING_PLAN: Analyze %s for complexity reduction opportunities" analysis.Name)
        
        if analysis.PublicInterface.Length > 15 then
            tasks.Add(sprintf "INTERFACE_REVIEW: Evaluate public interface of %s for potential simplification" analysis.Name)
        
        if analysis.Dependencies.Length > 8 then
            tasks.Add(sprintf "DEPENDENCY_ANALYSIS: Map dependency graph for %s and suggest optimizations" analysis.Name)
        
        if analysis.DesignPatterns |> Array.contains "Async Pattern" then
            tasks.Add(sprintf "ASYNC_OPTIMIZATION: Review async patterns in %s for performance improvements" analysis.Name)
        
        // Always add a real-time monitoring task
        tasks.Add(sprintf "REAL_TIME_MONITOR: Track changes to %s and alert on quality degradation" analysis.Name)
        
        tasks.ToArray()
    
    /// Generate LLM prompts dynamically
    member private this.GenerateLLMPrompts(analysis: ComponentAnalysis) =
        let prompts = ResizeArray<string>()
        
        // Generate fresh prompts based on actual component state
        prompts.Add(sprintf "Analyze the architectural design of component '%s' located at '%s'. Focus on: 1) Current design patterns used, 2) Potential improvements, 3) Integration opportunities with other components. Provide specific, actionable recommendations." analysis.Name analysis.FilePath)
        
        if analysis.ComplexityMetrics.ContainsKey("CyclomaticComplexity") then
            let complexity = analysis.ComplexityMetrics.["CyclomaticComplexity"]
            prompts.Add(sprintf "The component '%s' has a cyclomatic complexity of %.1f. Generate a detailed refactoring strategy that: 1) Identifies specific complex methods, 2) Suggests decomposition approaches, 3) Maintains existing functionality. Provide concrete code examples." analysis.Name complexity)
        
        if analysis.Dependencies.Length > 0 then
            prompts.Add(sprintf "Component '%s' has %d dependencies: %s. Analyze the dependency structure and suggest: 1) Opportunities for dependency injection, 2) Potential circular dependencies, 3) Strategies for reducing coupling." analysis.Name analysis.Dependencies.Length (String.Join(", ", analysis.Dependencies)))
        
        // Dynamic prompt based on detected patterns
        if analysis.DesignPatterns.Length > 0 then
            prompts.Add(sprintf "The component '%s' implements these design patterns: %s. Evaluate: 1) Pattern appropriateness for the use case, 2) Implementation quality, 3) Opportunities to combine or enhance patterns." analysis.Name (String.Join(", ", analysis.DesignPatterns)))
        
        prompts.ToArray()
    
    /// Perform real-time component analysis
    member this.AnalyzeComponent(componentName: string, filePath: string) =
        try
            let analysisStart = DateTime.UtcNow
            logger.LogInformation(sprintf "Starting real-time analysis of component: %s" componentName)
            
            // Generate unique component ID based on current state
            let componentId = sprintf "%s_%s_%d" componentName (analysisStart.ToString("yyyyMMdd_HHmmss")) (analysisStart.Millisecond)
            
            // Real-time source code analysis
            let (publicMembers, privateMethods, complexity, codeLength, commentRatio, patterns) = this.AnalyzeSourceCode(filePath)
            
            // Calculate source code hash for change detection
            let sourceHash = 
                if File.Exists(filePath) then
                    let content = File.ReadAllText(filePath)
                    content.GetHashCode().ToString("X8")
                else
                    "NO_FILE"
            
            // Extract dependencies dynamically
            let dependencies = 
                if File.Exists(filePath) then
                    let content = File.ReadAllText(filePath)
                    content.Split('\n')
                    |> Array.choose (fun line ->
                        let trimmed = line.Trim()
                        if trimmed.StartsWith("open ") then
                            Some (trimmed.Substring(5))
                        else None)
                    |> Array.distinct
                else
                    [||]
            
            // Build real-time analysis
            let analysis = {
                ComponentId = componentId
                Name = componentName
                Type = Path.GetExtension(filePath)
                FilePath = filePath
                AnalysisTimestamp = analysisStart
                SourceCodeHash = sourceHash
                Dependencies = dependencies
                PublicInterface = publicMembers
                PrivateMembers = privateMethods
                ComplexityMetrics = Map.ofList [
                    ("CyclomaticComplexity", complexity)
                    ("CodeLength", codeLength)
                    ("CommentRatio", commentRatio)
                    ("PublicInterfaceSize", float publicMembers.Length)
                    ("DependencyCount", float dependencies.Length)
                ]
                DesignPatterns = patterns |> List.toArray
                CodeQuality = Map.ofList [
                    ("Maintainability", if complexity < 5.0 then 0.9 else if complexity < 10.0 then 0.7 else 0.4)
                    ("Documentation", commentRatio)
                    ("Modularity", if dependencies.Length < 5 then 0.9 else if dependencies.Length < 10 then 0.7 else 0.5)
                ]
                RealTimeInsights = [||] // Will be populated below
                SuggestedImprovements = [||] // Will be populated below
                ReactAgentTasks = [||] // Will be populated below
                LLMPrompts = [||] // Will be populated below
            }
            
            // Generate dynamic content
            let insights = this.GenerateRealTimeInsights(analysis)
            let reactTasks = this.GenerateReactAgentTasks(analysis)
            let llmPrompts = this.GenerateLLMPrompts(analysis)
            
            // Generate improvement suggestions dynamically
            let improvements = [
                if complexity > 8.0 then sprintf "Reduce cyclomatic complexity from %.1f to under 8.0" complexity
                if commentRatio < 0.15 then sprintf "Increase documentation coverage from %.1f%% to at least 15%%" (commentRatio * 100.0)
                if dependencies.Length > 8 then sprintf "Reduce dependencies from %d to under 8" dependencies.Length
                if publicMembers.Length > 12 then sprintf "Consider splitting large interface (%d public members)" publicMembers.Length
                sprintf "Consider implementing unit tests for %s" componentName
                sprintf "Add performance benchmarks for critical paths in %s" componentName
            ]
            
            let finalAnalysis = { analysis with 
                RealTimeInsights = insights
                SuggestedImprovements = improvements |> List.toArray
                ReactAgentTasks = reactTasks
                LLMPrompts = llmPrompts
            }
            
            let analysisEnd = DateTime.UtcNow
            let analysisTime = (analysisEnd - analysisStart).TotalMilliseconds
            
            logger.LogInformation(sprintf "Real-time analysis completed for %s in %.1fms" componentName analysisTime)
            
            Some finalAnalysis
        with
        | ex ->
            logger.LogError(ex, sprintf "Real-time analysis failed for component: %s" componentName)
            None
    
    /// Get all components and analyze them in real-time
    member this.AnalyzeAllComponents(rootDirectory: string) =
        try
            let analysisStart = DateTime.UtcNow
            logger.LogInformation("Starting real-time analysis of all components")
            
            let fsFiles = Directory.GetFiles(rootDirectory, "*.fs", SearchOption.AllDirectories)
            let analyses = ResizeArray<ComponentAnalysis>()
            
            for file in fsFiles do
                let componentName = Path.GetFileNameWithoutExtension(file)
                match this.AnalyzeComponent(componentName, file) with
                | Some analysis -> analyses.Add(analysis)
                | None -> ()
            
            let analysisEnd = DateTime.UtcNow
            let totalTime = (analysisEnd - analysisStart).TotalSeconds
            
            logger.LogInformation(sprintf "Real-time analysis of %d components completed in %.1f seconds" analyses.Count totalTime)
            
            {|
                TotalComponents = analyses.Count
                AnalysisTimestamp = analysisStart
                TotalAnalysisTime = totalTime
                Components = analyses.ToArray()
                Summary = {|
                    AverageComplexity = if analyses.Count > 0 then analyses |> Seq.averageBy (fun a -> a.ComplexityMetrics.["CyclomaticComplexity"]) else 0.0
                    TotalDependencies = analyses |> Seq.sumBy (fun a -> a.Dependencies.Length)
                    PatternsDetected = analyses |> Seq.collect (fun a -> a.DesignPatterns) |> Seq.distinct |> Seq.length
                    ComponentsNeedingAttention = analyses |> Seq.filter (fun a -> a.ComplexityMetrics.["CyclomaticComplexity"] > 8.0) |> Seq.length
                |}
            |}
        with
        | ex ->
            logger.LogError(ex, "Failed to analyze all components")
            {|
                TotalComponents = 0
                AnalysisTimestamp = DateTime.UtcNow
                TotalAnalysisTime = 0.0
                Components = [||]
                Summary = {|
                    AverageComplexity = 0.0
                    TotalDependencies = 0
                    PatternsDetected = 0
                    ComponentsNeedingAttention = 0
                |}
            |}
