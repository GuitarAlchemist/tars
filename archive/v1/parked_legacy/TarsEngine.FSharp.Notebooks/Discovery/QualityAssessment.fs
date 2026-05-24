namespace TarsEngine.FSharp.Notebooks.Discovery

open System
open System.Text.RegularExpressions
open TarsEngine.FSharp.Notebooks.Types

/// <summary>
/// Quality assessment and scoring for Jupyter notebooks
/// </summary>

/// Quality metrics
type QualityMetrics = {
    OverallScore: float
    DocumentationScore: float
    CodeQualityScore: float
    StructureScore: float
    ReproducibilityScore: float
    EducationalValueScore: float
    TechnicalDepthScore: float
    Completeness: float
    Clarity: float
    Innovation: float
}

/// Quality assessment result
type QualityAssessment = {
    Notebook: JupyterNotebook
    Metrics: QualityMetrics
    Strengths: string list
    Weaknesses: string list
    Recommendations: string list
    Grade: QualityGrade
    AssessmentDate: DateTime
}

/// Quality grades
and QualityGrade = 
    | Excellent  // 90-100
    | Good       // 80-89
    | Fair       // 70-79
    | Poor       // 60-69
    | VeryPoor   // Below 60

/// Quality assessor
type NotebookQualityAssessor() =
    
    /// Assess notebook quality
    member _.AssessQuality(notebook: JupyterNotebook) : QualityAssessment =
        let metrics = this.CalculateMetrics(notebook)
        let strengths = this.IdentifyStrengths(notebook, metrics)
        let weaknesses = this.IdentifyWeaknesses(notebook, metrics)
        let recommendations = this.GenerateRecommendations(notebook, metrics)
        let grade = this.DetermineGrade(metrics.OverallScore)
        
        {
            Notebook = notebook
            Metrics = metrics
            Strengths = strengths
            Weaknesses = weaknesses
            Recommendations = recommendations
            Grade = grade
            AssessmentDate = DateTime.UtcNow
        }
    
    /// Calculate quality metrics
    member private this.CalculateMetrics(notebook: JupyterNotebook) : QualityMetrics =
        let docScore = this.CalculateDocumentationScore(notebook)
        let codeScore = this.CalculateCodeQualityScore(notebook)
        let structureScore = this.CalculateStructureScore(notebook)
        let reproScore = this.CalculateReproducibilityScore(notebook)
        let eduScore = this.CalculateEducationalValueScore(notebook)
        let techScore = this.CalculateTechnicalDepthScore(notebook)
        let completeness = this.CalculateCompleteness(notebook)
        let clarity = this.CalculateClarity(notebook)
        let innovation = this.CalculateInnovation(notebook)
        
        let overallScore = 
            (docScore * 0.2 + codeScore * 0.2 + structureScore * 0.15 + 
             reproScore * 0.15 + eduScore * 0.1 + techScore * 0.1 + 
             completeness * 0.05 + clarity * 0.03 + innovation * 0.02)
        
        {
            OverallScore = overallScore
            DocumentationScore = docScore
            CodeQualityScore = codeScore
            StructureScore = structureScore
            ReproducibilityScore = reproScore
            EducationalValueScore = eduScore
            TechnicalDepthScore = techScore
            Completeness = completeness
            Clarity = clarity
            Innovation = innovation
        }
    
    /// Calculate documentation score
    member private _.CalculateDocumentationScore(notebook: JupyterNotebook) : float =
        let markdownCells = 
            notebook.Cells 
            |> List.choose (function | MarkdownCell md -> Some md | _ -> None)
        
        let codeCells = 
            notebook.Cells 
            |> List.choose (function | CodeCell cd -> Some cd | _ -> None)
        
        if codeCells.IsEmpty then 0.0
        else
            let markdownRatio = float markdownCells.Length / float codeCells.Length
            let hasTitle = notebook.Metadata.Title.IsSome
            let hasDescription = notebook.Metadata.Description.IsSome
            
            let baseScore = min (markdownRatio * 50.0) 70.0
            let titleBonus = if hasTitle then 15.0 else 0.0
            let descBonus = if hasDescription then 15.0 else 0.0
            
            min (baseScore + titleBonus + descBonus) 100.0
    
    /// Calculate code quality score
    member private _.CalculateCodeQualityScore(notebook: JupyterNotebook) : float =
        let codeCells = 
            notebook.Cells 
            |> List.choose (function | CodeCell cd -> Some cd | _ -> None)
        
        if codeCells.IsEmpty then 0.0
        else
            let mutable totalScore = 0.0
            let mutable cellCount = 0
            
            for cell in codeCells do
                let cellScore = this.AssessCodeCellQuality(cell)
                totalScore <- totalScore + cellScore
                cellCount <- cellCount + 1
            
            if cellCount > 0 then totalScore / float cellCount else 0.0
    
    /// Assess individual code cell quality
    member private _.AssessCodeCellQuality(cell: CodeCellData) : float =
        let code = String.Join("\n", cell.Source)
        let mutable score = 50.0 // Base score
        
        // Check for comments
        let commentLines = cell.Source |> List.filter (fun line -> line.TrimStart().StartsWith("#"))
        if commentLines.Length > 0 then
            score <- score + 15.0
        
        // Check for proper imports
        let hasImports = cell.Source |> List.exists (fun line -> line.Contains("import "))
        if hasImports then
            score <- score + 10.0
        
        // Check for error handling
        let hasErrorHandling = code.Contains("try") || code.Contains("except") || code.Contains("catch")
        if hasErrorHandling then
            score <- score + 15.0
        
        // Check for function definitions
        let hasFunctions = code.Contains("def ") || code.Contains("function ")
        if hasFunctions then
            score <- score + 10.0
        
        min score 100.0
    
    /// Calculate structure score
    member private _.CalculateStructureScore(notebook: JupyterNotebook) : float =
        let cells = notebook.Cells
        let mutable score = 50.0
        
        // Check for logical flow
        let hasIntroduction = 
            cells 
            |> List.take (min 3 cells.Length)
            |> List.exists (function | MarkdownCell _ -> true | _ -> false)
        
        if hasIntroduction then score <- score + 20.0
        
        // Check for conclusion
        let hasConclusion = 
            cells 
            |> List.rev
            |> List.take (min 3 cells.Length)
            |> List.exists (function | MarkdownCell _ -> true | _ -> false)
        
        if hasConclusion then score <- score + 20.0
        
        // Check for section headers
        let markdownCells = cells |> List.choose (function | MarkdownCell md -> Some md | _ -> None)
        let hasHeaders = 
            markdownCells 
            |> List.exists (fun md -> 
                md.Source |> List.exists (fun line -> line.StartsWith("#")))
        
        if hasHeaders then score <- score + 10.0
        
        min score 100.0
    
    /// Calculate reproducibility score
    member private _.CalculateReproducibilityScore(notebook: JupyterNotebook) : float =
        let codeCells = notebook.Cells |> List.choose (function | CodeCell cd -> Some cd | _ -> None)
        let mutable score = 50.0
        
        // Check for requirements/dependencies
        let hasRequirements = 
            codeCells |> List.exists (fun cell ->
                cell.Source |> List.exists (fun line -> 
                    line.Contains("pip install") || line.Contains("conda install") || line.Contains("import ")))
        
        if hasRequirements then score <- score + 25.0
        
        // Check for data sources
        let hasDataSources = 
            codeCells |> List.exists (fun cell ->
                cell.Source |> List.exists (fun line -> 
                    line.Contains("read_csv") || line.Contains("load_data") || line.Contains("download")))
        
        if hasDataSources then score <- score + 25.0
        
        min score 100.0
    
    /// Calculate educational value score
    member private _.CalculateEducationalValueScore(notebook: JupyterNotebook) : float =
        let markdownCells = notebook.Cells |> List.choose (function | MarkdownCell md -> Some md | _ -> None)
        let mutable score = 50.0
        
        // Check for explanations
        let totalMarkdownLength = 
            markdownCells 
            |> List.sumBy (fun md -> md.Source |> List.sumBy (fun line -> line.Length))
        
        if totalMarkdownLength > 500 then score <- score + 20.0
        elif totalMarkdownLength > 200 then score <- score + 10.0
        
        // Check for examples
        let hasExamples = 
            markdownCells |> List.exists (fun md ->
                md.Source |> List.exists (fun line -> 
                    line.ToLower().Contains("example") || line.ToLower().Contains("demo")))
        
        if hasExamples then score <- score + 15.0
        
        // Check for step-by-step approach
        let hasSteps = 
            markdownCells |> List.exists (fun md ->
                md.Source |> List.exists (fun line -> 
                    Regex.IsMatch(line, @"\d+\.")))
        
        if hasSteps then score <- score + 15.0
        
        min score 100.0
    
    /// Calculate technical depth score
    member private _.CalculateTechnicalDepthScore(notebook: JupyterNotebook) : float =
        let codeCells = notebook.Cells |> List.choose (function | CodeCell cd -> Some cd | _ -> None)
        let mutable score = 50.0
        
        // Check for advanced techniques
        let advancedKeywords = [
            "machine learning"; "deep learning"; "neural network"; "algorithm"
            "optimization"; "statistics"; "regression"; "classification"
            "clustering"; "visualization"; "analysis"
        ]
        
        let codeText = 
            codeCells 
            |> List.collect (fun cell -> cell.Source)
            |> String.concat " "
            |> fun s -> s.ToLower()
        
        let advancedCount = 
            advancedKeywords 
            |> List.sumBy (fun keyword -> if codeText.Contains(keyword) then 1 else 0)
        
        score <- score + float advancedCount * 5.0
        
        min score 100.0
    
    /// Calculate completeness
    member private _.CalculateCompleteness(notebook: JupyterNotebook) : float =
        let hasTitle = notebook.Metadata.Title.IsSome
        let hasCode = notebook.Cells |> List.exists (function | CodeCell _ -> true | _ -> false)
        let hasMarkdown = notebook.Cells |> List.exists (function | MarkdownCell _ -> true | _ -> false)
        let hasOutputs = 
            notebook.Cells 
            |> List.exists (function 
                | CodeCell cd -> cd.Outputs.IsSome && not cd.Outputs.Value.IsEmpty 
                | _ -> false)
        
        let components = [hasTitle; hasCode; hasMarkdown; hasOutputs]
        let completedComponents = components |> List.filter id |> List.length
        
        float completedComponents / float components.Length * 100.0
    
    /// Calculate clarity
    member private _.CalculateClarity(notebook: JupyterNotebook) : float =
        // Simple heuristic based on cell organization and content
        let cellCount = notebook.Cells.Length
        if cellCount = 0 then 0.0
        elif cellCount < 5 then 60.0
        elif cellCount < 20 then 80.0
        else 70.0
    
    /// Calculate innovation
    member private _.CalculateInnovation(notebook: JupyterNotebook) : float =
        // Simple heuristic - could be enhanced with more sophisticated analysis
        let codeText = 
            notebook.Cells 
            |> List.choose (function | CodeCell cd -> Some (String.Join(" ", cd.Source)) | _ -> None)
            |> String.concat " "
            |> fun s -> s.ToLower()
        
        let innovativeKeywords = [
            "novel"; "new"; "innovative"; "creative"; "original"
            "breakthrough"; "cutting-edge"; "state-of-the-art"
        ]
        
        let innovationCount = 
            innovativeKeywords 
            |> List.sumBy (fun keyword -> if codeText.Contains(keyword) then 1 else 0)
        
        min (50.0 + float innovationCount * 10.0) 100.0
    
    /// Identify strengths
    member private _.IdentifyStrengths(notebook: JupyterNotebook, metrics: QualityMetrics) : string list =
        let strengths = ResizeArray<string>()
        
        if metrics.DocumentationScore >= 80.0 then
            strengths.Add("Excellent documentation and explanations")
        
        if metrics.CodeQualityScore >= 80.0 then
            strengths.Add("High-quality, well-structured code")
        
        if metrics.StructureScore >= 80.0 then
            strengths.Add("Clear logical structure and organization")
        
        if metrics.ReproducibilityScore >= 80.0 then
            strengths.Add("Good reproducibility with clear dependencies")
        
        if metrics.EducationalValueScore >= 80.0 then
            strengths.Add("High educational value with clear explanations")
        
        strengths |> List.ofSeq
    
    /// Identify weaknesses
    member private _.IdentifyWeaknesses(notebook: JupyterNotebook, metrics: QualityMetrics) : string list =
        let weaknesses = ResizeArray<string>()
        
        if metrics.DocumentationScore < 60.0 then
            weaknesses.Add("Insufficient documentation and explanations")
        
        if metrics.CodeQualityScore < 60.0 then
            weaknesses.Add("Code quality could be improved")
        
        if metrics.StructureScore < 60.0 then
            weaknesses.Add("Poor structure and organization")
        
        if metrics.ReproducibilityScore < 60.0 then
            weaknesses.Add("Reproducibility issues - unclear dependencies")
        
        weaknesses |> List.ofSeq
    
    /// Generate recommendations
    member private _.GenerateRecommendations(notebook: JupyterNotebook, metrics: QualityMetrics) : string list =
        let recommendations = ResizeArray<string>()
        
        if metrics.DocumentationScore < 70.0 then
            recommendations.Add("Add more markdown cells with explanations")
        
        if metrics.CodeQualityScore < 70.0 then
            recommendations.Add("Improve code quality with comments and error handling")
        
        if metrics.StructureScore < 70.0 then
            recommendations.Add("Organize content with clear sections and headers")
        
        if metrics.ReproducibilityScore < 70.0 then
            recommendations.Add("Include clear dependency installation instructions")
        
        recommendations |> List.ofSeq
    
    /// Determine quality grade
    member private _.DetermineGrade(score: float) : QualityGrade =
        if score >= 90.0 then Excellent
        elif score >= 80.0 then Good
        elif score >= 70.0 then Fair
        elif score >= 60.0 then Poor
        else VeryPoor

/// Quality assessment utilities
module QualityUtils =
    
    /// Format quality assessment as text
    let formatAssessment (assessment: QualityAssessment) : string =
        let sb = System.Text.StringBuilder()
        
        sb.AppendLine($"📊 Notebook Quality Assessment") |> ignore
        sb.AppendLine($"Overall Score: {assessment.Metrics.OverallScore:F1}/100 ({assessment.Grade})") |> ignore
        sb.AppendLine() |> ignore
        
        sb.AppendLine("📈 Detailed Metrics:") |> ignore
        sb.AppendLine($"  Documentation: {assessment.Metrics.DocumentationScore:F1}/100") |> ignore
        sb.AppendLine($"  Code Quality: {assessment.Metrics.CodeQualityScore:F1}/100") |> ignore
        sb.AppendLine($"  Structure: {assessment.Metrics.StructureScore:F1}/100") |> ignore
        sb.AppendLine($"  Reproducibility: {assessment.Metrics.ReproducibilityScore:F1}/100") |> ignore
        sb.AppendLine($"  Educational Value: {assessment.Metrics.EducationalValueScore:F1}/100") |> ignore
        sb.AppendLine() |> ignore
        
        if not assessment.Strengths.IsEmpty then
            sb.AppendLine("✅ Strengths:") |> ignore
            for strength in assessment.Strengths do
                sb.AppendLine($"  • {strength}") |> ignore
            sb.AppendLine() |> ignore
        
        if not assessment.Weaknesses.IsEmpty then
            sb.AppendLine("❌ Areas for Improvement:") |> ignore
            for weakness in assessment.Weaknesses do
                sb.AppendLine($"  • {weakness}") |> ignore
            sb.AppendLine() |> ignore
        
        if not assessment.Recommendations.IsEmpty then
            sb.AppendLine("💡 Recommendations:") |> ignore
            for recommendation in assessment.Recommendations do
                sb.AppendLine($"  • {recommendation}") |> ignore
        
        sb.ToString()
    
    /// Get grade emoji
    let getGradeEmoji (grade: QualityGrade) : string =
        match grade with
        | Excellent -> "🏆"
        | Good -> "👍"
        | Fair -> "👌"
        | Poor -> "👎"
        | VeryPoor -> "💩"
