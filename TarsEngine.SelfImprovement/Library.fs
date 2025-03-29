namespace TarsEngine.SelfImprovement

open System
open System.IO
open System.Text.RegularExpressions
open FSharp.Data

type AnalysisResult =
    { FileName: string
      Issues: string list
      Recommendations: string list
      Score: float }

type ImprovementProposal =
    { FileName: string
      OriginalContent: string
      ImprovedContent: string
      Explanation: string }

type SelfImprovementResult =
    { Success: bool
      Message: string
      Proposal: ImprovementProposal option }

module SelfAnalyzer =
    let analyzeFile (filePath: string) (ollamaEndpoint: string) (model: string) =
        async {
            try
                // Read the file content
                let! content = File.ReadAllTextAsync(filePath) |> Async.AwaitTask
                let contentForPrompt = content.Replace("\r\n", "\n").Replace("\"", "\\\"")

                // Determine file type and language for pattern recognition
                let fileExtension = Path.GetExtension(filePath).ToLower()
                let language =
                    match fileExtension with
                    | ".cs" -> "csharp"
                    | ".fs" | ".fsx" -> "fsharp"
                    | _ -> "any"

                // Use pattern recognition to identify common issues
                let patternMatches = PatternRecognition.recognizePatterns content language
                let patternIssues = PatternRecognition.getIssues patternMatches
                let patternRecommendations = PatternRecognition.getRecommendations patternMatches

                // Include pattern recognition results in the prompt
                let patternAnalysis =
                    if patternIssues.Length > 0 || patternRecommendations.Length > 0 then
                        let issuesText = String.Join(",", patternIssues |> List.map (fun i -> $"\"{i}\""))
                        let recsText = String.Join(",", patternRecommendations |> List.map (fun r -> $"\"{r}\""))
                        $"\n\nStatic analysis has already identified these issues and recommendations:\n{{\n  \"static_issues\": [{issuesText}],\n  \"static_recommendations\": [{recsText}]\n}}\n\nPlease consider these in your analysis and add any additional insights."
                    else ""

                // Create a prompt for analysis
                let prompt =
                    $"You are TARS, an AI assistant specialized in code analysis.\n\nPlease analyze the following code and identify potential issues, bugs, or areas for improvement.\nFocus on code quality, performance, and maintainability.\n\nFILE: %s{filePath}\n\nCODE:\n```\n%s{contentForPrompt}\n```%s{patternAnalysis}\n\nProvide your analysis in the following JSON format:\n{{\n    \"issues\": [\"issue1\", \"issue2\", ...],\n    \"recommendations\": [\"recommendation1\", \"recommendation2\", ...],\n    \"score\": 0.0 to 10.0 (where 10 is perfect code)\n}}\n\nOnly return the JSON, no other text."

                // Call Ollama API
                let ollamaUrl = $"%s{ollamaEndpoint}/api/generate"
                // Properly escape the prompt for JSON
                let escapedPrompt =
                    prompt
                    |> fun s -> s.Replace("\\", "\\\\")
                    |> fun s -> s.Replace("\"", "\\\"")
                    |> fun s -> s.Replace("\n", "\\n")
                    |> fun s -> s.Replace("\r", "\\r")
                    |> fun s -> s.Replace("\t", "\\t")

                let requestBody =
                    sprintf "{\"model\": \"%s\", \"prompt\": \"%s\", \"stream\": false}"
                        model escapedPrompt

                let! response = Http.AsyncRequestString(ollamaUrl, httpMethod = "POST", body = TextRequest requestBody)

                // Extract the response content
                let jsonPattern = "\{[\s\S]*\}"
                let jsonMatch = Regex.Match(response, jsonPattern)

                if jsonMatch.Success then
                    let jsonResponse = jsonMatch.Value
                    let parsedJson = JsonValue.Parse(jsonResponse)

                    // Combine AI-identified issues with pattern-identified issues
                    let aiIssues =
                        try
                            parsedJson.GetProperty("issues").AsArray()
                            |> Array.map (fun x -> x.AsString())
                            |> Array.toList
                        with _ -> []

                    let issues =
                        if aiIssues.Length > 0 then
                            // Combine AI issues with pattern issues, removing duplicates
                            let allIssues = aiIssues @ patternIssues |> List.distinct
                            allIssues
                        else if patternIssues.Length > 0 then
                            patternIssues
                        else
                            ["No specific issues identified"]

                    // Combine AI-identified recommendations with pattern-identified recommendations
                    let aiRecommendations =
                        try
                            parsedJson.GetProperty("recommendations").AsArray()
                            |> Array.map (fun x -> x.AsString())
                            |> Array.toList
                        with _ -> []

                    let recommendations =
                        if aiRecommendations.Length > 0 then
                            // Combine AI recommendations with pattern recommendations, removing duplicates
                            let allRecommendations = aiRecommendations @ patternRecommendations |> List.distinct
                            allRecommendations
                        else if patternRecommendations.Length > 0 then
                            patternRecommendations
                        else
                            ["No specific recommendations"]

                    let score =
                        try
                            parsedJson.GetProperty("score").AsFloat()
                        with _ -> 5.0 // Default middle score

                    let analysisResult =
                        { FileName = filePath
                          Issues = issues
                          Recommendations = recommendations
                          Score = score }

                    // Record the analysis in the learning database
                    try
                        let fileTypeForDb = fileExtension.TrimStart('.')
                        do! LearningDatabase.recordAnalysis filePath fileTypeForDb content analysisResult |> Async.Ignore
                    with _ ->
                        // Ignore errors with the learning database
                        ()

                    return analysisResult
                else
                    return
                        { FileName = filePath
                          Issues = patternIssues @ ["Failed to parse AI response"]
                          Recommendations = patternRecommendations @ ["Try again with a different model or prompt"]
                          Score = 0.0 }
            with ex ->
                return
                    { FileName = filePath
                      Issues = [ $"Error during analysis: %s{ex.Message}"]
                      Recommendations = ["Check if the file exists and is accessible"]
                      Score = 0.0 }
        }

module SelfImprover =
    let proposeImprovement (filePath: string) (analysisResult: AnalysisResult) (ollamaEndpoint: string) (model: string) =
        async {
            try
                // Read the file content
                let! content = File.ReadAllTextAsync(filePath) |> Async.AwaitTask
                let contentForPrompt = content.Replace("\r\n", "\n").Replace("\"", "\\\"")

                // Determine file type for learning database
                let fileExtension = Path.GetExtension(filePath).ToLower()
                let fileType = fileExtension.TrimStart('.')

                // Create a prompt for improvement
                let issuesText = String.Join("\n- ", analysisResult.Issues)
                let recommendationsText = String.Join("\n- ", analysisResult.Recommendations)

                // Get previous improvements from the learning database
                let! previousImprovements =
                    async {
                        try
                            let! events = LearningDatabase.getEventsByFile(filePath)
                            let improvementEvents =
                                events
                                |> List.filter (fun e -> e.EventType = "Improvement" && e.Success)
                                |> List.sortByDescending (fun e -> e.Timestamp)
                                |> List.truncate 3

                            if improvementEvents.Length > 0 then
                                let improvementSummary =
                                    improvementEvents
                                    |> List.mapi (fun i e ->
                                        match e.ImprovementProposal with
                                        | Some p ->
                                            try
                                                let propInfo = p.GetType().GetProperty("Explanation")
                                                let explanation = propInfo.GetValue(p) :?> string
                                                $"Previous improvement {i+1}: {explanation}"
                                            with _ ->
                                                $"Previous improvement {i+1}: (details not available)"
                                        | None -> "")
                                    |> List.filter (fun s -> s <> "")
                                    |> String.concat "\n\n"

                                return $"\n\nPREVIOUS IMPROVEMENTS:\n{improvementSummary}\n\nBuild upon these previous improvements."
                            else return ""
                        with _ -> return ""
                    }

                let prompt =
                    $"You are TARS, an AI assistant specialized in code improvement.\n\nI need you to improve the following code based on the analysis:\n\nFILE: %s{filePath}\n\nANALYSIS ISSUES:\n- %s{issuesText}\n\nRECOMMENDATIONS:\n- %s{recommendationsText}%s{previousImprovements}\n\nORIGINAL CODE:\n```\n%s{contentForPrompt}\n```\n\nPlease provide an improved version of this code that addresses the issues and follows the recommendations.\nReturn ONLY the improved code in a code block, followed by a brief explanation of your changes.\n\nFormat your response as:\n```\n[IMPROVED CODE HERE]\n```\n\nEXPLANATION: [Your explanation here]"

                // Call Ollama API
                let ollamaUrl = $"%s{ollamaEndpoint}/api/generate"
                // Properly escape the prompt for JSON
                let escapedPrompt =
                    prompt
                    |> fun s -> s.Replace("\\", "\\\\")
                    |> fun s -> s.Replace("\"", "\\\"")
                    |> fun s -> s.Replace("\n", "\\n")
                    |> fun s -> s.Replace("\r", "\\r")
                    |> fun s -> s.Replace("\t", "\\t")

                let requestBody =
                    sprintf "{\"model\": \"%s\", \"prompt\": \"%s\", \"stream\": false}"
                        model escapedPrompt

                let! response = Http.AsyncRequestString(ollamaUrl, httpMethod = "POST", body = TextRequest requestBody)

                // Extract the code and explanation
                let codePattern = "```[\s\S]*?```"
                let explanationPattern = "EXPLANATION:\s*([\s\S]*)"

                let codeMatch = Regex.Match(response, codePattern)
                let explanationMatch = Regex.Match(response, explanationPattern)

                if codeMatch.Success then
                    let improvedCode = codeMatch.Value.Replace("```", "").Trim()
                    let explanation =
                        if explanationMatch.Success && explanationMatch.Groups.Count > 1 then
                            explanationMatch.Groups.[1].Value.Trim()
                        else
                            "No explanation provided"

                    let proposal =
                        { FileName = filePath
                          OriginalContent = content
                          ImprovedContent = improvedCode
                          Explanation = explanation }

                    // Record the improvement proposal in the learning database
                    try
                        do! LearningDatabase.recordImprovement filePath fileType proposal |> Async.Ignore
                    with _ ->
                        // Ignore errors with the learning database
                        ()

                    return Some proposal
                else
                    return None
            with ex ->
                return None
        }

    let applyImprovement (proposal: ImprovementProposal) =
        async {
            try
                // Create a backup of the original file
                let backupPath = sprintf "%s.bak.%s" proposal.FileName (DateTime.Now.ToString("yyyyMMddHHmmss"))
                do! File.WriteAllTextAsync(backupPath, proposal.OriginalContent) |> Async.AwaitTask

                // Write the improved content to the original file
                do! File.WriteAllTextAsync(proposal.FileName, proposal.ImprovedContent) |> Async.AwaitTask

                // Record successful application in the learning database
                try
                    let fileExtension = Path.GetExtension(proposal.FileName).ToLower()
                    let fileType = fileExtension.TrimStart('.')

                    // Get the most recent improvement event for this file
                    let! events = LearningDatabase.getEventsByFile(proposal.FileName)
                    let improvementEvents =
                        events
                        |> List.filter (fun e -> e.EventType = "Improvement")
                        |> List.sortByDescending (fun e -> e.Timestamp)

                    if improvementEvents.Length > 0 then
                        let latestEvent = improvementEvents.[0]
                        do! LearningDatabase.recordFeedback latestEvent.Id "Improvement applied successfully" true |> Async.Ignore
                with _ ->
                    // Ignore errors with the learning database
                    ()

                return true
            with ex ->
                // Record failure in the learning database
                try
                    let fileExtension = Path.GetExtension(proposal.FileName).ToLower()
                    let fileType = fileExtension.TrimStart('.')

                    // Get the most recent improvement event for this file
                    let! events = LearningDatabase.getEventsByFile(proposal.FileName)
                    let improvementEvents =
                        events
                        |> List.filter (fun e -> e.EventType = "Improvement")
                        |> List.sortByDescending (fun e -> e.Timestamp)

                    if improvementEvents.Length > 0 then
                        let latestEvent = improvementEvents.[0]
                        do! LearningDatabase.recordFeedback latestEvent.Id $"Failed to apply improvement: {ex.Message}" false |> Async.Ignore
                with _ ->
                    // Ignore errors with the learning database
                    ()

                return false
        }

module SelfImprovement =
    let analyzeAndImprove (filePath: string) (ollamaEndpoint: string) (model: string) =
        async {
            try
                // Step 1: Analyze the file
                let! analysisResult = SelfAnalyzer.analyzeFile filePath ollamaEndpoint model

                // Step 2: If the score is below 8.0, propose improvements
                if analysisResult.Score < 8.0 then
                    let! improvementProposal = SelfImprover.proposeImprovement filePath analysisResult ollamaEndpoint model

                    match improvementProposal with
                    | Some proposal ->
                        // Record successful improvement proposal in learning database
                        try
                            let fileExtension = Path.GetExtension(filePath).ToLower()
                            let fileType = fileExtension.TrimStart('.')
                            do! LearningDatabase.recordImprovement filePath fileType proposal |> Async.Ignore
                        with _ ->
                            // Ignore errors with the learning database
                            ()

                        return
                            { Success = true
                              Message = $"Analysis completed with score %.1f{analysisResult.Score}. Improvement proposal generated."
                              Proposal = Some proposal }
                    | None ->
                        return
                            { Success = false
                              Message = $"Analysis completed with score %.1f{analysisResult.Score}, but failed to generate improvement proposal."
                              Proposal = None }
                else
                    return
                        { Success = true
                          Message = $"Analysis completed with score %.1f{analysisResult.Score}. No improvements needed."
                          Proposal = None }
            with ex ->
                return
                    { Success = false
                      Message = $"Error during self-improvement process: %s{ex.Message}"
                      Proposal = None }
        }

    let analyzeAndImproveWithApply (filePath: string) (ollamaEndpoint: string) (model: string) (autoApply: bool) =
        async {
            let! result = analyzeAndImprove filePath ollamaEndpoint model

            if result.Success && result.Proposal.IsSome && autoApply then
                let! applySuccess = SelfImprover.applyImprovement result.Proposal.Value

                if applySuccess then
                    // Record successful application in learning database
                    try
                        let fileExtension = Path.GetExtension(filePath).ToLower()
                        let fileType = fileExtension.TrimStart('.')

                        // Get the most recent improvement event for this file
                        let! events = LearningDatabase.getEventsByFile(filePath)
                        let improvementEvents =
                            events
                            |> List.filter (fun e -> e.EventType = "Improvement")
                            |> List.sortByDescending (fun e -> e.Timestamp)

                        if improvementEvents.Length > 0 then
                            let latestEvent = improvementEvents.[0]
                            do! LearningDatabase.recordFeedback latestEvent.Id "Auto-improvement applied successfully" true |> Async.Ignore
                    with _ ->
                        // Ignore errors with the learning database
                        ()

                    return { result with Message = result.Message + " Improvements applied successfully." }
                else
                    // Record failed application in learning database
                    try
                        let fileExtension = Path.GetExtension(filePath).ToLower()
                        let fileType = fileExtension.TrimStart('.')

                        // Get the most recent improvement event for this file
                        let! events = LearningDatabase.getEventsByFile(filePath)
                        let improvementEvents =
                            events
                            |> List.filter (fun e -> e.EventType = "Improvement")
                            |> List.sortByDescending (fun e -> e.Timestamp)

                        if improvementEvents.Length > 0 then
                            let latestEvent = improvementEvents.[0]
                            do! LearningDatabase.recordFeedback latestEvent.Id "Failed to apply auto-improvement" false |> Async.Ignore
                    with _ ->
                        // Ignore errors with the learning database
                        ()

                    return { result with Message = result.Message + " Failed to apply improvements." }
            else
                return result
        }
