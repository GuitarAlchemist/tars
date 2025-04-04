namespace TarsEngine.SelfImprovement

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open System.Text.Json
open System.Text

/// <summary>
/// Module for the reporting step in the autonomous improvement workflow
/// </summary>
module ReportingStep =
    /// <summary>
    /// The path to the applied improvements file
    /// </summary>
    let appliedImprovementsPath = "applied_improvements.json"

    /// <summary>
    /// The path to the feedback file
    /// </summary>
    let feedbackPath = "improvement_feedback.json"

    /// <summary>
    /// The path to the report file
    /// </summary>
    let reportPath = "improvement_report.md"

    /// <summary>
    /// Loads the applied improvements
    /// </summary>
    let loadAppliedImprovements () =
        task {
            if File.Exists(appliedImprovementsPath) then
                let! json = File.ReadAllTextAsync(appliedImprovementsPath)
                return JsonSerializer.Deserialize<ImprovementApplicationStep.AppliedImprovement[]>(json)
            else
                return [||]
        }

    /// <summary>
    /// Loads the feedback
    /// </summary>
    let loadFeedback () =
        task {
            if File.Exists(feedbackPath) then
                let! json = File.ReadAllTextAsync(feedbackPath)
                return JsonSerializer.Deserialize<FeedbackCollectionStep.ImprovementFeedback[]>(json)
            else
                return [||]
        }

    /// <summary>
    /// Generates a report of the improvements
    /// </summary>
    let generateReport (logger: ILogger) (state: WorkflowState) =
        task {
            try
                // Load the applied improvements
                let! appliedImprovements = loadAppliedImprovements()

                // Load the feedback
                let! feedback = loadFeedback()

                // Create a map of feedback by file path and pattern ID
                let feedbackMap =
                    feedback
                    |> Array.map (fun f -> (f.FilePath, f.PatternId), f)
                    |> Map.ofArray

                // Generate the report
                let report = StringBuilder()

                // Add the header
                report.AppendLine("# TARS Autonomous Improvement Report") |> ignore
                report.AppendLine() |> ignore
                report.AppendLine(sprintf "**Date:** %s" (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"))) |> ignore
                report.AppendLine(sprintf "**Workflow:** %s" state.Name) |> ignore
                report.AppendLine(sprintf "**Duration:** %.2f minutes" (DateTime.UtcNow - state.StartTime).TotalMinutes) |> ignore
                report.AppendLine() |> ignore

                // Add the summary
                report.AppendLine("## Summary") |> ignore
                report.AppendLine() |> ignore
                report.AppendLine($"- **Improvements Applied:** {appliedImprovements.Length}") |> ignore
                report.AppendLine($"- **Successful Improvements:** {feedback |> Array.filter (fun f -> f.IsSuccessful) |> Array.length}") |> ignore
                report.AppendLine($"- **Failed Improvements:** {feedback |> Array.filter (fun f -> not f.IsSuccessful) |> Array.length}") |> ignore
                report.AppendLine() |> ignore

                // Add the workflow steps
                report.AppendLine("## Workflow Steps") |> ignore
                report.AppendLine() |> ignore
                report.AppendLine("| Step | Status | Duration |") |> ignore
                report.AppendLine("|------|--------|----------|") |> ignore

                for step in state.Steps do
                    let status =
                        match step.Status with
                        | StepStatus.NotStarted -> "Not Started"
                        | StepStatus.InProgress -> "In Progress"
                        | StepStatus.Completed -> "Completed"
                        | StepStatus.Failed -> "Failed"
                        | StepStatus.Skipped -> "Skipped"

                    let duration =
                        match step.StartTime, step.EndTime with
                        | Some start, Some end' ->
                            $"{(end' - start).TotalMinutes:F2} minutes"
                        | _ ->
                            "N/A"

                    report.AppendLine($"| {step.Name} | {status} | {duration} |") |> ignore

                report.AppendLine() |> ignore

                // Add the improvements
                if appliedImprovements.Length > 0 then
                    report.AppendLine("## Applied Improvements") |> ignore
                    report.AppendLine() |> ignore

                    for improvement in appliedImprovements do
                        // Get the feedback for this improvement
                        let feedbackOption =
                            feedbackMap.TryFind (improvement.FilePath, improvement.PatternId)

                        let status =
                            match feedbackOption with
                            | Some f when f.IsSuccessful -> "✅ Successful"
                            | Some f -> "❌ Failed"
                            | None -> "⚠️ Unknown"

                        report.AppendLine($"### {improvement.PatternName} ({status})") |> ignore
                        report.AppendLine() |> ignore
                        report.AppendLine($"**File:** {improvement.FilePath}") |> ignore

                        match improvement.LineNumber with
                        | Some line -> report.AppendLine($"**Line:** {line}") |> ignore
                        | None -> ()

                        report.AppendLine() |> ignore
                        report.AppendLine("**Original Code:**") |> ignore
                        report.AppendLine() |> ignore
                        report.AppendLine("```") |> ignore
                        report.AppendLine(improvement.OriginalCode) |> ignore
                        report.AppendLine("```") |> ignore
                        report.AppendLine() |> ignore
                        report.AppendLine("**Improved Code:**") |> ignore
                        report.AppendLine() |> ignore
                        report.AppendLine("```") |> ignore
                        report.AppendLine(improvement.ImprovedCode) |> ignore
                        report.AppendLine("```") |> ignore
                        report.AppendLine() |> ignore

                        // Add the error message if the improvement failed
                        match feedbackOption with
                        | Some f when not f.IsSuccessful && f.ErrorMessage.IsSome ->
                            report.AppendLine("**Error:**") |> ignore
                            report.AppendLine() |> ignore
                            report.AppendLine("```") |> ignore
                            report.AppendLine(f.ErrorMessage.Value) |> ignore
                            report.AppendLine("```") |> ignore
                            report.AppendLine() |> ignore
                        | _ -> ()

                // Save the report
                do! File.WriteAllTextAsync(reportPath, report.ToString())

                return reportPath
            with ex ->
                logger.LogError(ex, "Error generating report")
                return null
        }

    /// <summary>
    /// Gets the reporting step handler
    /// </summary>
    let getHandler (logger: ILogger) : WorkflowEngine.StepHandler =
        fun state ->
            task {
                logger.LogInformation("Starting reporting step")

                // Generate the report
                let! reportPath = generateReport logger state

                if reportPath <> null then
                    logger.LogInformation("Report generated: {ReportPath}", reportPath)

                    // Return the result data
                    return Ok (Map.ofList [
                        "report_path", reportPath
                    ])
                else
                    return Error "Failed to generate report"
            }
