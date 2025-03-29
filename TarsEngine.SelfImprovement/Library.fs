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
                let content = content.Replace("\r\n", "\n").Replace("\"", "\\\"")

                // Create a prompt for analysis
                let prompt =
                    $"You are TARS, an AI assistant specialized in code analysis.\n\nPlease analyze the following code and identify potential issues, bugs, or areas for improvement.\nFocus on code quality, performance, and maintainability.\n\nFILE: %s{filePath}\n\nCODE:\n```\n%s{content}\n```\n\nProvide your analysis in the following JSON format:\n{{\n    \"issues\": [\"issue1\", \"issue2\", ...],\n    \"recommendations\": [\"recommendation1\", \"recommendation2\", ...],\n    \"score\": 0.0 to 10.0 (where 10 is perfect code)\n}}\n\nOnly return the JSON, no other text."

                // Call Ollama API
                let ollamaUrl = $"%s{ollamaEndpoint}/api/generate"
                let requestBody =
                    sprintf "{\"model\": \"%s\", \"prompt\": \"%s\", \"stream\": false}"
                        model (prompt.Replace("\"", "\\\"").Replace("\n", "\\n"))

                let! response = Http.AsyncRequestString(ollamaUrl, httpMethod = "POST", body = TextRequest requestBody)

                // Extract the response content
                let jsonPattern = "\{[\s\S]*\}"
                let jsonMatch = Regex.Match(response, jsonPattern)

                if jsonMatch.Success then
                    let jsonResponse = jsonMatch.Value
                    let parsedJson = JsonValue.Parse(jsonResponse)

                    let issues =
                        try
                            parsedJson.GetProperty("issues").AsArray()
                            |> Array.map (fun x -> x.AsString())
                            |> Array.toList
                        with _ -> ["No specific issues identified"]

                    let recommendations =
                        try
                            parsedJson.GetProperty("recommendations").AsArray()
                            |> Array.map (fun x -> x.AsString())
                            |> Array.toList
                        with _ -> ["No specific recommendations"]

                    let score =
                        try
                            parsedJson.GetProperty("score").AsFloat()
                        with _ -> 5.0 // Default middle score

                    return
                        { FileName = filePath
                          Issues = issues
                          Recommendations = recommendations
                          Score = score }
                else
                    return
                        { FileName = filePath
                          Issues = ["Failed to parse AI response"]
                          Recommendations = ["Try again with a different model or prompt"]
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
                let content = content.Replace("\r\n", "\n").Replace("\"", "\\\"")

                // Create a prompt for improvement
                let issuesText = String.Join("\n- ", analysisResult.Issues)
                let recommendationsText = String.Join("\n- ", analysisResult.Recommendations)

                let prompt =
                    $"You are TARS, an AI assistant specialized in code improvement.\n\nI need you to improve the following code based on the analysis:\n\nFILE: %s{filePath}\n\nANALYSIS ISSUES:\n- %s{issuesText}\n\nRECOMMENDATIONS:\n- %s{recommendationsText}\n\nORIGINAL CODE:\n```\n%s{content}\n```\n\nPlease provide an improved version of this code that addresses the issues and follows the recommendations.\nReturn ONLY the improved code in a code block, followed by a brief explanation of your changes.\n\nFormat your response as:\n```\n[IMPROVED CODE HERE]\n```\n\nEXPLANATION: [Your explanation here]"

                // Call Ollama API
                let ollamaUrl = $"%s{ollamaEndpoint}/api/generate"
                let requestBody =
                    sprintf "{\"model\": \"%s\", \"prompt\": \"%s\", \"stream\": false}"
                        model (prompt.Replace("\"", "\\\"").Replace("\n", "\\n"))

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

                    return Some
                        { FileName = filePath
                          OriginalContent = content
                          ImprovedContent = improvedCode
                          Explanation = explanation }
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

                return true
            with ex ->
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
                    return { result with Message = result.Message + " Improvements applied successfully." }
                else
                    return { result with Message = result.Message + " Failed to apply improvements." }
            else
                return result
        }
