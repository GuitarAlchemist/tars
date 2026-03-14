namespace TarsEngine.FSharp.Core.Metascript

open System
open System.Text.RegularExpressions

/// AI-powered analysis to detect "BS" scripts that just print fake results
module AntiBSAnalyzer =
    
    type AnalysisResult = {
        IsLegitimate: bool
        Confidence: float
        Reason: string
        SuspiciousPatterns: string list
        ComputationalComplexity: int
        RealCalculationRatio: float
    }
    
    type CodePattern = {
        Pattern: string
        Weight: float
        IsSuspicious: bool
        Description: string
    }
    
    /// Patterns that indicate legitimate computational work
    let legitimatePatterns = [
        { Pattern = "mathematical_operations"; Weight = 2.0; IsSuspicious = false; Description = "Contains real mathematical operations" }
        { Pattern = "variable_calculations"; Weight = 1.5; IsSuspicious = false; Description = "Variables derived from calculations" }
        { Pattern = "conditional_logic"; Weight = 1.0; IsSuspicious = false; Description = "Contains conditional logic based on calculations" }
        { Pattern = "iterative_computation"; Weight = 2.5; IsSuspicious = false; Description = "Contains loops or recursive calculations" }
        { Pattern = "data_processing"; Weight = 1.8; IsSuspicious = false; Description = "Processes input data through transformations" }
        { Pattern = "statistical_analysis"; Weight = 2.2; IsSuspicious = false; Description = "Performs statistical calculations" }
        { Pattern = "numerical_methods"; Weight = 2.8; IsSuspicious = false; Description = "Uses numerical methods or algorithms" }
    ]
    
    /// Patterns that indicate suspicious "BS" behavior
    let suspiciousPatterns = [
        { Pattern = "hardcoded_results"; Weight = -3.0; IsSuspicious = true; Description = "Contains hardcoded result values" }
        { Pattern = "fake_calculations"; Weight = -2.5; IsSuspicious = true; Description = "Variables assigned without real calculation" }
        { Pattern = "print_only"; Weight = -2.0; IsSuspicious = true; Description = "Mostly just printing predefined values" }
        { Pattern = "no_input_dependency"; Weight = -1.5; IsSuspicious = true; Description = "Results don't depend on input parameters" }
        { Pattern = "unrealistic_precision"; Weight = -1.8; IsSuspicious = true; Description = "Claims unrealistic precision without calculation" }
        { Pattern = "magic_numbers"; Weight = -1.2; IsSuspicious = true; Description = "Uses unexplained magic numbers as results" }
        { Pattern = "simulation_keywords"; Weight = -2.8; IsSuspicious = true; Description = "Contains keywords suggesting simulation/fake data" }
    ]
    
    /// Analyze F# code for computational legitimacy
    let analyzeCode (fsharpCode: string) : AnalysisResult =
        let lines = fsharpCode.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
        let codeText = String.Join(" ", lines).ToLower()
        
        // Count different types of statements
        let mutable printStatements = 0
        let mutable calculations = 0
        let mutable assignments = 0
        let mutable hardcodedValues = 0
        let mutable mathematicalOps = 0
        let mutable conditionals = 0
        let mutable loops = 0
        let mutable functions = 0
        
        let suspiciousFindings = ResizeArray<string>()
        
        // Analyze each line
        for line in lines do
            let trimmedLine = line.Trim().ToLower()
            
            // Count print statements
            if trimmedLine.Contains("printfn") || trimmedLine.Contains("console.writeline") then
                printStatements <- printStatements + 1
            
            // Count mathematical operations
            if trimmedLine.Contains("sqrt") || trimmedLine.Contains("sin") || trimmedLine.Contains("cos") ||
               trimmedLine.Contains("log") || trimmedLine.Contains("exp") || trimmedLine.Contains("pow") ||
               trimmedLine.Contains("abs") || trimmedLine.Contains("*") || trimmedLine.Contains("/") ||
               trimmedLine.Contains("+") || trimmedLine.Contains("-") then
                mathematicalOps <- mathematicalOps + 1
            
            // Count assignments
            if trimmedLine.Contains("let ") && trimmedLine.Contains("=") then
                assignments <- assignments + 1
                
                // Check if assignment is a real calculation or hardcoded value
                let rightSide = trimmedLine.Substring(trimmedLine.IndexOf("=") + 1).Trim()
                if Regex.IsMatch(rightSide, @"^\d+\.?\d*$") then
                    hardcodedValues <- hardcodedValues + 1
                elif rightSide.Contains("sqrt") || rightSide.Contains("*") || rightSide.Contains("/") ||
                     rightSide.Contains("+") || rightSide.Contains("-") then
                    calculations <- calculations + 1
            
            // Count control structures
            if trimmedLine.Contains("if ") || trimmedLine.Contains("match ") then
                conditionals <- conditionals + 1
            
            if trimmedLine.Contains("for ") || trimmedLine.Contains("while ") || trimmedLine.Contains("|>") then
                loops <- loops + 1
            
            if trimmedLine.Contains("let ") && trimmedLine.Contains("(") && trimmedLine.Contains(")") then
                functions <- functions + 1
        
        // Detect suspicious patterns
        
        // Check for hardcoded scientific results
        if codeText.Contains("7.0") && codeText.Contains("improvement") then
            suspiciousFindings.Add("Contains suspiciously specific '7.0x improvement' claim")
        
        if codeText.Contains("85%") && codeText.Contains("promising") then
            suspiciousFindings.Add("Contains suspiciously specific '85% promising' assessment")
        
        if codeText.Contains("14.4%") && codeText.Contains("shift") then
            suspiciousFindings.Add("Contains suspiciously specific '14.4% shift' value")
        
        // Check for fake calculation patterns
        if hardcodedValues > calculations * 2 then
            suspiciousFindings.Add("More hardcoded values than real calculations")
        
        if printStatements > mathematicalOps then
            suspiciousFindings.Add("More print statements than mathematical operations")
        
        // Check for simulation/fake keywords
        let fakeKeywords = ["fake"; "simulate"; "pretend"; "mock"; "dummy"; "placeholder"; "hardcoded"]
        for keyword in fakeKeywords do
            if codeText.Contains(keyword) then
                suspiciousFindings.Add($"Contains suspicious keyword: '{keyword}'")
        
        // Check for unrealistic precision without calculation
        if Regex.IsMatch(codeText, @"\d+\.\d{3,}") && calculations < 3 then
            suspiciousFindings.Add("Claims high precision without sufficient calculations")
        
        // Calculate computational complexity score
        let complexityScore = calculations * 2 + mathematicalOps + conditionals * 2 + loops * 3 + functions * 2
        
        // Calculate real calculation ratio
        let totalStatements = assignments + printStatements + conditionals + loops
        let realCalcRatio = if totalStatements > 0 then float calculations / float totalStatements else 0.0
        
        // Determine legitimacy
        let suspiciousScore = suspiciousFindings.Count
        let isLegitimate = complexityScore >= 5 && realCalcRatio >= 0.3 && suspiciousScore <= 2
        
        // Calculate confidence
        let confidence = 
            let baseConfidence = min 1.0 (float complexityScore / 10.0)
            let suspiciousPenalty = float suspiciousScore * 0.15
            let calcBonus = realCalcRatio * 0.3
            max 0.0 (min 1.0 (baseConfidence + calcBonus - suspiciousPenalty))
        
        // Generate reason
        let reason = 
            if not isLegitimate then
                if suspiciousScore > 2 then
                    $"REJECTED: Too many suspicious patterns detected ({suspiciousScore} found)"
                elif realCalcRatio < 0.3 then
                    $"REJECTED: Insufficient real calculations (ratio: {realCalcRatio:F2})"
                elif complexityScore < 5 then
                    $"REJECTED: Computational complexity too low (score: {complexityScore})"
                else
                    "REJECTED: Multiple legitimacy criteria failed"
            else
                $"APPROVED: Legitimate computational script (complexity: {complexityScore}, calc ratio: {realCalcRatio:F2})"
        
        {
            IsLegitimate = isLegitimate
            Confidence = confidence
            Reason = reason
            SuspiciousPatterns = suspiciousFindings |> Seq.toList
            ComputationalComplexity = complexityScore
            RealCalculationRatio = realCalcRatio
        }
    
    /// Analyze FLUX metascript for overall legitimacy
    let analyzeMetascript (metascriptContent: string) : AnalysisResult =
        // Extract F# blocks from metascript
        let fsharpBlocks = ResizeArray<string>()
        let lines = metascriptContent.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
        let mutable inFSharpBlock = false
        let mutable currentBlock = ResizeArray<string>()
        
        for line in lines do
            if line.Trim() = "FSHARP {" then
                inFSharpBlock <- true
                currentBlock.Clear()
            elif line.Trim() = "}" && inFSharpBlock then
                inFSharpBlock <- false
                if currentBlock.Count > 0 then
                    fsharpBlocks.Add(String.Join("\n", currentBlock))
            elif inFSharpBlock then
                currentBlock.Add(line)
        
        // Analyze all F# blocks
        if fsharpBlocks.Count = 0 then
            {
                IsLegitimate = false
                Confidence = 0.0
                Reason = "REJECTED: No F# code blocks found"
                SuspiciousPatterns = ["No computational content"]
                ComputationalComplexity = 0
                RealCalculationRatio = 0.0
            }
        else
            let allCode = String.Join("\n", fsharpBlocks)
            analyzeCode allCode
    
    /// Generate detailed analysis report
    let generateAnalysisReport (result: AnalysisResult) : string =
        let status = if result.IsLegitimate then "✅ APPROVED" else "❌ REJECTED"
        let confidence = $"{result.Confidence * 100.0:F1}%%"
        
        let report = System.Text.StringBuilder()
        report.AppendLine("🤖 TARS Anti-BS Analysis Report") |> ignore
        report.AppendLine("================================") |> ignore
        report.AppendLine($"Status: {status}") |> ignore
        report.AppendLine($"Confidence: {confidence}") |> ignore
        report.AppendLine($"Reason: {result.Reason}") |> ignore
        report.AppendLine($"Computational Complexity: {result.ComputationalComplexity}") |> ignore
        report.AppendLine($"Real Calculation Ratio: {result.RealCalculationRatio:F2}") |> ignore
        report.AppendLine("") |> ignore
        
        if result.SuspiciousPatterns.Length > 0 then
            report.AppendLine("🚨 Suspicious Patterns Detected:") |> ignore
            for pattern in result.SuspiciousPatterns do
                report.AppendLine($"  • {pattern}") |> ignore
            report.AppendLine("") |> ignore
        
        if result.IsLegitimate then
            report.AppendLine("✅ This script appears to contain legitimate computational work.") |> ignore
        else
            report.AppendLine("❌ This script appears to be 'BS' - mostly fake output without real calculations.") |> ignore
            report.AppendLine("   Please provide a script that performs actual mathematical computations.") |> ignore
        
        report.ToString()
