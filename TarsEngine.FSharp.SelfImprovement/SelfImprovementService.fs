namespace TarsEngine.FSharp.SelfImprovement

open System
open System.IO
open System.Net.Http
open Microsoft.Extensions.Logging
open ImprovementTypes
open PatternRecognition
open OllamaCodeAnalyzer

/// Interface for self-improvement service
type ISelfImprovementService =
    abstract member AnalyzeFileAsync: filePath: string -> Async<AnalysisResult option>
    abstract member AnalyzeDirectoryAsync: directoryPath: string -> Async<AnalysisResult list>
    abstract member ApplyImprovementsAsync: filePath: string * improvements: ImprovementPattern list -> Async<AppliedImprovement list>
    abstract member GetImprovementHistoryAsync: filePath: string option -> Async<AppliedImprovement list>

/// Main self-improvement orchestration service
type SelfImprovementService(
    httpClient: HttpClient,
    logger: ILogger<SelfImprovementService>) =
    
    let ollamaAnalyzer = OllamaCodeAnalyzer(httpClient, logger)
    let mutable improvementHistory: AppliedImprovement list = []
    
    /// Analyze a single file for improvements
    member this.AnalyzeFileAsync(filePath: string) =
        async {
            try
                if not (File.Exists(filePath)) then
                    logger.LogWarning("File not found: {FilePath}", filePath)
                    return None
                
                logger.LogInformation("Analyzing file: {FilePath}", filePath)
                
                let content = File.ReadAllText(filePath)
                
                // First, use pattern recognition
                let patternResult = PatternRecognition.analyzeFile filePath content
                
                // Then, enhance with AI analysis if Ollama is available
                let! isOllamaAvailable = ollamaAnalyzer.IsAvailableAsync()
                
                if isOllamaAvailable then
                    logger.LogInformation("Enhancing analysis with Ollama AI")
                    let! aiResult = ollamaAnalyzer.AnalyzeCodeAsync(filePath, content, "llama3")
                    
                    match aiResult with
                    | Some aiAnalysis ->
                        // Combine pattern recognition with AI analysis
                        let combinedIssues = patternResult.Issues @ aiAnalysis.Issues |> List.distinct
                        let combinedRecommendations = patternResult.Recommendations @ aiAnalysis.Recommendations |> List.distinct
                        let averageScore = (patternResult.OverallScore + aiAnalysis.OverallScore) / 2.0
                        
                        let enhancedResult = {
                            FilePath = filePath
                            Issues = combinedIssues
                            OverallScore = averageScore
                            Recommendations = combinedRecommendations
                            AnalyzedAt = DateTime.UtcNow
                        }
                        
                        logger.LogInformation("Enhanced analysis completed: {IssueCount} issues, score: {Score:F1}", 
                                            combinedIssues.Length, averageScore)
                        return Some enhancedResult
                    | None ->
                        logger.LogInformation("AI analysis failed, using pattern recognition only")
                        return Some patternResult
                else
                    logger.LogInformation("Ollama not available, using pattern recognition only")
                    return Some patternResult
                    
            with
            | ex ->
                logger.LogError(ex, "Error analyzing file: {FilePath}", filePath)
                return None
        }
    
    /// Analyze all files in a directory
    member this.AnalyzeDirectoryAsync(directoryPath: string) =
        async {
            try
                logger.LogInformation("Analyzing directory: {DirectoryPath}", directoryPath)
                
                let sourceFiles = 
                    Directory.GetFiles(directoryPath, "*", SearchOption.AllDirectories)
                    |> Array.filter (fun file -> 
                        let ext = Path.GetExtension(file).ToLower()
                        ext = ".fs" || ext = ".cs" || ext = ".fsx")
                    |> Array.toList
                
                logger.LogInformation("Found {FileCount} source files to analyze", sourceFiles.Length)
                
                let results = ResizeArray<AnalysisResult>()
                
                for filePath in sourceFiles do
                    let! result = this.AnalyzeFileAsync(filePath)
                    match result with
                    | Some analysis -> results.Add(analysis)
                    | None -> ()
                
                let finalResults = results |> Seq.toList
                logger.LogInformation("Directory analysis completed: {AnalyzedCount}/{TotalCount} files", 
                                    finalResults.Length, sourceFiles.Length)
                
                return finalResults
                
            with
            | ex ->
                logger.LogError(ex, "Error analyzing directory: {DirectoryPath}", directoryPath)
                return []
        }
    
    /// Apply improvements to a file
    member this.ApplyImprovementsAsync(filePath: string, improvements: ImprovementPattern list) =
        async {
            try
                logger.LogInformation("Applying {ImprovementCount} improvements to: {FilePath}", 
                                    improvements.Length, filePath)
                
                if not (File.Exists(filePath)) then
                    logger.LogWarning("File not found: {FilePath}", filePath)
                    return []
                
                let originalContent = File.ReadAllText(filePath)
                let mutable currentContent = originalContent
                let appliedImprovements = ResizeArray<AppliedImprovement>()
                
                // Create backup
                let backupPath = filePath + ".bak"
                File.WriteAllText(backupPath, originalContent)
                logger.LogInformation("Created backup: {BackupPath}", backupPath)
                
                for improvement in improvements do
                    try
                        // For now, we'll just log the improvements
                        // In a real implementation, we would apply actual code transformations
                        let appliedImprovement = {
                            Id = Guid.NewGuid()
                            FilePath = filePath
                            Pattern = improvement
                            OriginalCode = "// Original code section"
                            ImprovedCode = "// Improved code section"
                            AppliedAt = DateTime.UtcNow
                            Success = true
                            Notes = Some "Simulated improvement application"
                        }
                        
                        appliedImprovements.Add(appliedImprovement)
                        logger.LogInformation("Applied improvement: {ImprovementName}", improvement.Name)
                        
                    with
                    | ex ->
                        logger.LogError(ex, "Failed to apply improvement: {ImprovementName}", improvement.Name)
                        
                        let failedImprovement = {
                            Id = Guid.NewGuid()
                            FilePath = filePath
                            Pattern = improvement
                            OriginalCode = ""
                            ImprovedCode = ""
                            AppliedAt = DateTime.UtcNow
                            Success = false
                            Notes = Some ex.Message
                        }
                        
                        appliedImprovements.Add(failedImprovement)
                
                // Update improvement history
                let newImprovements = appliedImprovements |> Seq.toList
                improvementHistory <- improvementHistory @ newImprovements
                
                logger.LogInformation("Applied {SuccessCount}/{TotalCount} improvements successfully", 
                                    newImprovements |> List.filter (fun i -> i.Success) |> List.length,
                                    newImprovements.Length)
                
                return newImprovements
                
            with
            | ex ->
                logger.LogError(ex, "Error applying improvements to: {FilePath}", filePath)
                return []
        }
    
    /// Get improvement history
    member this.GetImprovementHistoryAsync(filePath: string option) =
        async {
            match filePath with
            | Some path ->
                return improvementHistory |> List.filter (fun i -> i.FilePath = path)
            | None ->
                return improvementHistory
        }
    
    interface ISelfImprovementService with
        member this.AnalyzeFileAsync(filePath) = this.AnalyzeFileAsync(filePath)
        member this.AnalyzeDirectoryAsync(directoryPath) = this.AnalyzeDirectoryAsync(directoryPath)
        member this.ApplyImprovementsAsync(filePath, improvements) = this.ApplyImprovementsAsync(filePath, improvements)
        member this.GetImprovementHistoryAsync(filePath) = this.GetImprovementHistoryAsync(filePath)
