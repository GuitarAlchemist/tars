namespace TarsEngine.TreeOfThought

open System
open System.IO

/// Represents the configuration for the Tree of Thought pipeline
type PipelineConfig = {
    AnalysisEnabled: bool
    FixGenerationEnabled: bool
    FixApplicationEnabled: bool
    CreateBackups: bool
    ConfidenceThreshold: float
    MaxFixesPerIssue: int
    OutputDirectory: string
}

/// Represents the complete result of running the pipeline
type PipelineResult = {
    Config: PipelineConfig
    AnalysisResult: AnalysisResult option
    FixGenerationResult: FixGenerationResult option
    FixApplicationResult: BatchFixApplicationResult option
    TotalTime: TimeSpan
    Success: bool
    ErrorMessage: string option
}

/// Tree of Thought pipeline for automated code analysis and fixing
module Pipeline =
    
    /// Default pipeline configuration
    let defaultConfig = {
        AnalysisEnabled = true
        FixGenerationEnabled = true
        FixApplicationEnabled = false  // Safe default - don't auto-apply fixes
        CreateBackups = true
        ConfidenceThreshold = 0.7
        MaxFixesPerIssue = 3
        OutputDirectory = "./pipeline_output"
    }
    
    /// Create output directory if it doesn't exist
    let ensureOutputDirectory (config: PipelineConfig) : unit =
        if not (Directory.Exists(config.OutputDirectory)) then
            Directory.CreateDirectory(config.OutputDirectory) |> ignore
    
    /// Save analysis results to file
    let saveAnalysisResults (config: PipelineConfig) (result: AnalysisResult) : string =
        ensureOutputDirectory config
        let timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss")
        let filePath = Path.Combine(config.OutputDirectory, $"analysis_{timestamp}.json")
        
        // Simple JSON-like format for now
        let content = 
            $"{{" +
            $"  \"timestamp\": \"{timestamp}\"," +
            $"  \"filesAnalyzed\": {result.FilesAnalyzed}," +
            $"  \"linesAnalyzed\": {result.LinesAnalyzed}," +
            $"  \"analysisTime\": \"{result.AnalysisTime}\"," +
            $"  \"totalIssues\": {result.Issues.Length}," +
            $"  \"issues\": [" +
            (result.Issues 
             |> List.map (fun issue ->
                 let escapedMessage = issue.Message.Replace("\"", "\\\"")
                 let escapedFile = issue.File.Replace("\\", "\\\\")
                 $"    {{" +
                 $"      \"type\": \"{issue.Type}\"," +
                 $"      \"message\": \"{escapedMessage}\"," +
                 $"      \"file\": \"{escapedFile}\"," +
                 $"      \"line\": {issue.Line}," +
                 $"      \"column\": {issue.Column}," +
                 $"      \"severity\": \"{issue.Severity}\"" +
                 $"    }}")
             |> String.concat ",\n") +
            $"  ]" +
            $"}}"
        
        File.WriteAllText(filePath, content)
        filePath
    
    /// Save fix generation results to file
    let saveFixResults (config: PipelineConfig) (result: FixGenerationResult) : string =
        ensureOutputDirectory config
        let timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss")
        let filePath = Path.Combine(config.OutputDirectory, $"fixes_{timestamp}.json")
        
        let content = 
            $"{{" +
            $"  \"timestamp\": \"{timestamp}\"," +
            $"  \"generationTime\": \"{result.GenerationTime}\"," +
            $"  \"successRate\": {result.SuccessRate}," +
            $"  \"totalFixes\": {result.Fixes.Length}," +
            $"  \"fixes\": [" +
            (result.Fixes 
             |> List.map (fun fix ->
                 let escapedDescription = fix.Description.Replace("\"", "\\\"")
                 let escapedTargetFile = fix.TargetFile.Replace("\\", "\\\\")
                 let escapedOriginalText = fix.OriginalText.Replace("\"", "\\\"")
                 let escapedReplacementText = fix.ReplacementText.Replace("\"", "\\\"")
                 $"    {{" +
                 $"      \"id\": \"{fix.Id}\"," +
                 $"      \"description\": \"{escapedDescription}\"," +
                 $"      \"targetFile\": \"{escapedTargetFile}\"," +
                 $"      \"targetLine\": {fix.TargetLine}," +
                 $"      \"targetColumn\": {fix.TargetColumn}," +
                 $"      \"originalText\": \"{escapedOriginalText}\"," +
                 $"      \"replacementText\": \"{escapedReplacementText}\"," +
                 $"      \"confidence\": {fix.Confidence}," +
                 $"      \"category\": \"{fix.Category}\"" +
                 $"    }}")
             |> String.concat ",\n") +
            $"  ]" +
            $"}}"
        
        File.WriteAllText(filePath, content)
        filePath
    
    /// Run the complete pipeline on a single file
    let runPipelineOnFile (config: PipelineConfig) (filePath: string) : PipelineResult =
        let startTime = DateTime.Now
        
        try
            // Step 1: Analysis
            let analysisResult = 
                if config.AnalysisEnabled then
                    Some (CodeAnalysis.analyzeFile filePath |> fun issues -> 
                        { Issues = issues; FilesAnalyzed = 1; LinesAnalyzed = 0; AnalysisTime = TimeSpan.Zero })
                else None
            
            // Step 2: Fix Generation
            let fixGenerationResult = 
                match analysisResult with
                | Some analysis when config.FixGenerationEnabled ->
                    let fixes = FixGeneration.generateFixes analysis.Issues
                    let filteredFixes = FixGeneration.filterByConfidence config.ConfidenceThreshold fixes
                    Some filteredFixes
                | _ -> None
            
            // Step 3: Fix Application
            let fixApplicationResult = 
                match fixGenerationResult with
                | Some fixGen when config.FixApplicationEnabled ->
                    let fixesToApply = 
                        fixGen.Fixes 
                        |> List.take (min fixGen.Fixes.Length config.MaxFixesPerIssue)
                    Some (FixApplication.applyFixesWithConflictResolution fixesToApply config.CreateBackups)
                | _ -> None
            
            let endTime = DateTime.Now
            
            // Save results
            analysisResult |> Option.iter (saveAnalysisResults config >> ignore)
            fixGenerationResult |> Option.iter (saveFixResults config >> ignore)
            
            {
                Config = config
                AnalysisResult = analysisResult
                FixGenerationResult = fixGenerationResult
                FixApplicationResult = fixApplicationResult
                TotalTime = endTime - startTime
                Success = true
                ErrorMessage = None
            }
        with
        | ex ->
            let endTime = DateTime.Now
            {
                Config = config
                AnalysisResult = None
                FixGenerationResult = None
                FixApplicationResult = None
                TotalTime = endTime - startTime
                Success = false
                ErrorMessage = Some ex.Message
            }
    
    /// Run the pipeline on multiple files
    let runPipelineOnFiles (config: PipelineConfig) (filePaths: string list) : PipelineResult =
        let startTime = DateTime.Now
        
        try
            // Step 1: Analysis
            let analysisResult = 
                if config.AnalysisEnabled then
                    Some (CodeAnalysis.analyzeFiles filePaths)
                else None
            
            // Step 2: Fix Generation
            let fixGenerationResult = 
                match analysisResult with
                | Some analysis when config.FixGenerationEnabled ->
                    let fixes = FixGeneration.generateFixes analysis.Issues
                    let filteredFixes = FixGeneration.filterByConfidence config.ConfidenceThreshold fixes
                    Some filteredFixes
                | _ -> None
            
            // Step 3: Fix Application
            let fixApplicationResult = 
                match fixGenerationResult with
                | Some fixGen when config.FixApplicationEnabled ->
                    Some (FixApplication.applyFixesWithConflictResolution fixGen.Fixes config.CreateBackups)
                | _ -> None
            
            let endTime = DateTime.Now
            
            // Save results
            analysisResult |> Option.iter (saveAnalysisResults config >> ignore)
            fixGenerationResult |> Option.iter (saveFixResults config >> ignore)
            
            {
                Config = config
                AnalysisResult = analysisResult
                FixGenerationResult = fixGenerationResult
                FixApplicationResult = fixApplicationResult
                TotalTime = endTime - startTime
                Success = true
                ErrorMessage = None
            }
        with
        | ex ->
            let endTime = DateTime.Now
            {
                Config = config
                AnalysisResult = None
                FixGenerationResult = None
                FixApplicationResult = None
                TotalTime = endTime - startTime
                Success = false
                ErrorMessage = Some ex.Message
            }
    
    /// Run the pipeline on a directory
    let runPipelineOnDirectory (config: PipelineConfig) (directoryPath: string) (pattern: string) : PipelineResult =
        if Directory.Exists(directoryPath) then
            let files = Directory.GetFiles(directoryPath, pattern, SearchOption.AllDirectories) |> Array.toList
            runPipelineOnFiles config files
        else
            {
                Config = config
                AnalysisResult = None
                FixGenerationResult = None
                FixApplicationResult = None
                TotalTime = TimeSpan.Zero
                Success = false
                ErrorMessage = Some $"Directory not found: {directoryPath}"
            }
    
    /// Get pipeline summary
    let getPipelineSummary (result: PipelineResult) : string =
        let analysisInfo = 
            match result.AnalysisResult with
            | Some analysis -> $"Analysis: {analysis.Issues.Length} issues found"
            | None -> "Analysis: Skipped"
        
        let fixGenInfo = 
            match result.FixGenerationResult with
            | Some fixGen -> $"Fix Generation: {fixGen.Fixes.Length} fixes generated"
            | None -> "Fix Generation: Skipped"
        
        let fixAppInfo = 
            match result.FixApplicationResult with
            | Some fixApp -> $"Fix Application: {fixApp.TotalApplied} applied, {fixApp.TotalFailed} failed"
            | None -> "Fix Application: Skipped"
        
        $"Pipeline completed in {result.TotalTime.TotalMilliseconds:F0}ms. " +
        $"Success: {result.Success}. {analysisInfo}. {fixGenInfo}. {fixAppInfo}."
