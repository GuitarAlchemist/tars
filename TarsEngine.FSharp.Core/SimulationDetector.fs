namespace TarsEngine.FSharp.Core

open System
open System.Text.RegularExpressions
open Microsoft.Extensions.Logging

/// <summary>
/// AI-powered simulation detection system - CRITICAL SECURITY COMPONENT
/// Prevents TARS from accepting any simulated or fake execution results
/// </summary>
type SimulationDetector(logger: ILogger<SimulationDetector>) =
    
    /// Forbidden simulation keywords that indicate fake execution
    let simulationKeywords = [
        "simulated"; "simulation"; "simulate"; "simulating"
        "placeholder"; "mock"; "fake"; "dummy"; "stub"
        "pretend"; "pretending"; "fictional"; "imaginary"
        "example"; "sample"; "demo only"; "for demonstration"
        "not real"; "not actual"; "pseudo"; "artificial"
        "test data"; "sample data"; "dummy data"
        "hardcoded"; "static"; "fixed"; "predetermined"
        "would execute"; "would run"; "would perform"
        "as if"; "pretends to"; "appears to"
        "simulates the"; "mimics the"; "emulates the"
        "fake result"; "mock result"; "dummy result"
        "placeholder result"; "example result"
        "TODO"; "FIXME"; "NOT IMPLEMENTED"
        "coming soon"; "to be implemented"
        "Thread.Sleep"; "Task.Delay"; "setTimeout"
        "return \""; "return '"; "box \""; "box '"
    ]
    
    /// Forbidden code patterns that indicate simulation
    let simulationPatterns = [
        @"return\s+""[^""]*simulated[^""]*"""
        @"return\s+'[^']*simulated[^']*'"
        @"box\s+""[^""]*simulated[^""]*"""
        @"Task\.Delay\s*\(\s*\d+\s*\)"
        @"Thread\.Sleep\s*\(\s*\d+\s*\)"
        @"//\s*simulate"
        @"//\s*mock"
        @"//\s*placeholder"
        @"//\s*TODO"
        @"//\s*FIXME"
        @"//\s*not\s+implemented"
        @"printf.*simulated"
        @"printfn.*simulated"
        @"Console\.WriteLine.*simulated"
        @"would\s+(execute|run|perform|do)"
        @"as\s+if\s+(it|we|this)"
        @"pretend\s+to"
        @"fake\s+(execution|result|output)"
        @"mock\s+(execution|result|output)"
        @"dummy\s+(execution|result|output)"
        @"placeholder\s+(execution|result|output)"
    ]
    
    /// <summary>
    /// Analyzes code content for simulation indicators using AI-powered detection
    /// </summary>
    /// <param name="content">The code content to analyze</param>
    /// <param name="filePath">The file path being analyzed</param>
    /// <returns>Detection result with detailed analysis</returns>
    member this.AnalyzeForSimulation(content: string, filePath: string) =
        logger.LogInformation("🔍 AI SIMULATION DETECTOR: Analyzing {FilePath}", filePath)
        
        let detectedKeywords = ResizeArray<string>()
        let detectedPatterns = ResizeArray<string>()
        let suspiciousLines = ResizeArray<string * int>()
        
        // Phase 1: Keyword Detection
        let contentLower = content.ToLowerInvariant()
        for keyword in simulationKeywords do
            if contentLower.Contains(keyword.ToLowerInvariant()) then
                detectedKeywords.Add(keyword)
                logger.LogWarning("🚨 SIMULATION KEYWORD DETECTED: {Keyword}", keyword)
        
        // Phase 2: Pattern Detection
        for pattern in simulationPatterns do
            let regex = Regex(pattern, RegexOptions.IgnoreCase ||| RegexOptions.Multiline)
            let matches = regex.Matches(content)
            if matches.Count > 0 then
                detectedPatterns.Add(pattern)
                for m in matches do
                    logger.LogWarning("🚨 SIMULATION PATTERN DETECTED: {Pattern} at position {Position}", pattern, m.Index)
        
        // Phase 3: Line-by-Line Analysis
        let lines = content.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
        for i = 0 to lines.Length - 1 do
            let line = lines.[i].Trim()
            let lineLower = line.ToLowerInvariant()
            
            // Check for suspicious return statements
            if lineLower.Contains("return") && (lineLower.Contains("simulated") || lineLower.Contains("mock") || lineLower.Contains("fake")) then
                suspiciousLines.Add((line, i + 1))
            
            // Check for delay/sleep patterns (simulation indicators)
            if lineLower.Contains("sleep") || lineLower.Contains("delay") then
                suspiciousLines.Add((line, i + 1))
            
            // Check for TODO/FIXME/placeholder comments
            if lineLower.Contains("todo") || lineLower.Contains("fixme") || lineLower.Contains("placeholder") then
                suspiciousLines.Add((line, i + 1))
        
        // Phase 4: AI-Powered Semantic Analysis
        let semanticScore = this.CalculateSemanticSimulationScore(content)
        
        // Phase 5: Generate Detection Result
        let isSimulation = 
            detectedKeywords.Count > 0 || 
            detectedPatterns.Count > 0 || 
            suspiciousLines.Count > 0 ||
            semanticScore > 0.7
        
        let confidenceScore = 
            let keywordScore = float detectedKeywords.Count * 0.3
            let patternScore = float detectedPatterns.Count * 0.4
            let lineScore = float suspiciousLines.Count * 0.2
            let totalScore = keywordScore + patternScore + lineScore + semanticScore
            Math.Min(1.0, totalScore)
        
        let explanation =
            if isSimulation then
                sprintf "Found %d simulation keywords, %d forbidden patterns, %d suspicious lines"
                    detectedKeywords.Count detectedPatterns.Count suspiciousLines.Count
            else
                "No simulation indicators detected - code appears to be authentic"

        {|
            IsSimulation = isSimulation
            ConfidenceScore = confidenceScore
            DetectedKeywords = detectedKeywords |> List.ofSeq
            DetectedPatterns = detectedPatterns |> List.ofSeq
            SuspiciousLines = suspiciousLines |> List.ofSeq
            SemanticScore = semanticScore
            FilePath = filePath
            AnalysisTimestamp = DateTime.UtcNow
            Verdict = if isSimulation then "FORBIDDEN - SIMULATION DETECTED" else "APPROVED - REAL EXECUTION"
            Explanation = explanation
        |}
    
    /// <summary>
    /// AI-powered semantic analysis to detect simulation intent
    /// </summary>
    /// <param name="content">Content to analyze</param>
    /// <returns>Simulation probability score (0.0 to 1.0)</returns>
    member private this.CalculateSemanticSimulationScore(content: string) =
        let mutable score = 0.0
        
        // Check for simulation-indicating phrases
        let simulationPhrases = [
            "for demonstration purposes"
            "this is just an example"
            "simulate the behavior"
            "pretend to execute"
            "mock implementation"
            "placeholder implementation"
            "would normally do"
            "in a real implementation"
            "this would actually"
            "fake the result"
            "return a dummy"
            "hardcoded response"
            "static result"
        ]
        
        let contentLower = content.ToLowerInvariant()
        for phrase in simulationPhrases do
            if contentLower.Contains(phrase) then
                score <- score + 0.2
        
        // Check for implementation gaps
        if contentLower.Contains("not implemented") then score <- score + 0.5
        if contentLower.Contains("coming soon") then score <- score + 0.4
        if contentLower.Contains("to be implemented") then score <- score + 0.5
        
        // Check for obvious fake data patterns
        if Regex.IsMatch(content, @"return\s+""[^""]*test[^""]*""", RegexOptions.IgnoreCase) then
            score <- score + 0.3
        
        Math.Min(1.0, score)
    
    /// <summary>
    /// CRITICAL: Validates execution result and STOPS if simulation detected
    /// </summary>
    /// <param name="result">Execution result to validate</param>
    /// <param name="context">Execution context</param>
    /// <returns>Validation result - FAILS if simulation detected</returns>
    member this.ValidateExecutionResult(result: obj, context: string) =
        logger.LogInformation("🔍 VALIDATING EXECUTION RESULT: {Context}", context)
        
        let resultString = result.ToString()
        let analysis = this.AnalyzeForSimulation(resultString, context)
        
        if analysis.IsSimulation then
            logger.LogCritical("🚨 CRITICAL: SIMULATION DETECTED IN EXECUTION RESULT!")
            logger.LogCritical("❌ Context: {Context}", context)
            logger.LogCritical("❌ Result: {Result}", resultString)
            logger.LogCritical("❌ Confidence: {Confidence:P}", analysis.ConfidenceScore)
            logger.LogCritical("❌ Keywords: {Keywords}", String.Join(", ", analysis.DetectedKeywords))
            logger.LogCritical("❌ EXECUTION TERMINATED - FORBIDDEN OPERATION")
            
            {|
                IsValid = false
                IsForbidden = true
                Reason = "SIMULATION DETECTED - FORBIDDEN OPERATION"
                Analysis = analysis
                Action = "TERMINATE_EXECUTION"
            |}
        else
            logger.LogInformation("✅ EXECUTION RESULT VALIDATED: Real execution confirmed")
            {|
                IsValid = true
                IsForbidden = false
                Reason = "Real execution confirmed"
                Analysis = analysis
                Action = "CONTINUE_EXECUTION"
            |}
