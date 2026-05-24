namespace Tars.Evolution

open System
open System.Text.Json
open System.Threading.Tasks
open Tars.Llm
open Tars.Core.MetaCognition
open Tars.Core.WorkflowOfThought

/// Post-execution meta-analysis: compares intent vs outcome and extracts lessons.
module ReflectionEngine =

    // =====================================================================
    // Pure intent vs outcome comparison
    // =====================================================================

    /// Compare planned steps (from WoT plan) with actual trace events.
    /// Pure logic — no LLM needed.
    let compareIntentVsOutcome
        (plannedStepIds: string list)
        (traceEvents: TraceEvent list)
        (goal: string)
        : IntentOutcomeComparison =

        let executedIds = traceEvents |> List.map (fun e -> e.StepId) |> Set.ofList
        let plannedSet = plannedStepIds |> Set.ofList

        let skipped =
            plannedStepIds
            |> List.filter (fun id -> not (Set.contains id executedIds))

        let failed =
            traceEvents
            |> List.filter (fun e -> e.Status = StepStatus.Error)
            |> List.map (fun e -> e.StepId)

        let unexpected =
            traceEvents
            |> List.map (fun e -> e.StepId)
            |> List.filter (fun id -> not (Set.contains id plannedSet))

        let totalPlanned = max 1 plannedStepIds.Length
        let alignment =
            let executed = Set.intersect plannedSet executedIds |> Set.count |> float
            let failPenalty = float failed.Length * 0.5
            max 0.0 (min 1.0 ((executed - failPenalty) / float totalPlanned))

        { PlannedSteps = plannedStepIds.Length
          ExecutedSteps = traceEvents.Length
          SkippedSteps = skipped
          FailedSteps = failed
          UnexpectedSteps = unexpected
          OverallAlignment = alignment }

    /// Classify the outcome from the comparison (pure logic).
    let classifyOutcome (comparison: IntentOutcomeComparison) : ReflectionOutcome =
        if comparison.FailedSteps.IsEmpty && comparison.SkippedSteps.IsEmpty && comparison.UnexpectedSteps.IsEmpty then
            if comparison.OverallAlignment >= 0.9 then
                ReflectionOutcome.AsExpected
            else
                ReflectionOutcome.BetterThanExpected "Completed with fewer steps than planned"
        elif comparison.FailedSteps.Length > comparison.PlannedSteps / 2 then
            ReflectionOutcome.WorseThanExpected
                (sprintf "%d of %d steps failed" comparison.FailedSteps.Length comparison.PlannedSteps)
        elif not comparison.UnexpectedSteps.IsEmpty then
            ReflectionOutcome.Unexpected
                (sprintf "%d unexpected steps executed" comparison.UnexpectedSteps.Length)
        else
            ReflectionOutcome.WorseThanExpected
                (sprintf "%d steps skipped, %d failed" comparison.SkippedSteps.Length comparison.FailedSteps.Length)

    // =====================================================================
    // LLM-enhanced reflection
    // =====================================================================

    /// Use LLM to generate a qualitative reflection report.
    let reflect
        (llm: ILlmService)
        (comparison: IntentOutcomeComparison)
        (goal: string)
        (result: string)
        : Task<ReflectionReport> =
        task {
            let prompt =
                sprintf """You are a meta-cognitive reflection engine for a self-improving AI system.

GOAL: %s

EXECUTION ANALYSIS:
- Planned steps: %d
- Executed steps: %d
- Skipped: %s
- Failed: %s
- Unexpected: %s
- Alignment score: %.2f

RESULT (truncated):
%s

Reflect on this execution. Output JSON:
{
  "intended_strategy": "what was the plan",
  "actual_behavior": "what actually happened",
  "surprises": ["list of unexpected things"],
  "lessons_learned": ["actionable insights"],
  "suggested_improvements": ["specific changes to try next time"]
}"""
                    goal
                    comparison.PlannedSteps
                    comparison.ExecutedSteps
                    (if comparison.SkippedSteps.IsEmpty then "none" else String.Join(", ", comparison.SkippedSteps))
                    (if comparison.FailedSteps.IsEmpty then "none" else String.Join(", ", comparison.FailedSteps))
                    (if comparison.UnexpectedSteps.IsEmpty then "none" else String.Join(", ", comparison.UnexpectedSteps))
                    comparison.OverallAlignment
                    (result.Substring(0, min 500 result.Length))

            let req =
                { ModelHint = Some "cheap"
                  Model = None
                  SystemPrompt = Some "You are a meta-cognitive reflection engine. Output JSON ONLY."
                  MaxTokens = Some 1000
                  Temperature = Some 0.2
                  Stop = []
                  Messages = [ { Role = Role.User; Content = prompt } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = None
                  ContextWindow = None }

            try
                let! resp = llm.CompleteAsync req
                let text = resp.Text.Trim()
                let firstBrace = text.IndexOf('{')
                let lastBrace = text.LastIndexOf('}')

                if firstBrace >= 0 && lastBrace > firstBrace then
                    let json = text.Substring(firstBrace, lastBrace - firstBrace + 1)
                    let doc = JsonDocument.Parse(json)
                    let root = doc.RootElement

                    let getStr (name: string) (fallback: string) =
                        let mutable p = JsonElement()
                        if root.TryGetProperty(name, &p) then p.GetString() else fallback

                    let getArr (name: string) =
                        let mutable p = JsonElement()
                        if root.TryGetProperty(name, &p) && p.ValueKind = JsonValueKind.Array then
                            p.EnumerateArray() |> Seq.map (fun e -> e.GetString()) |> Seq.toList
                        else []

                    return
                        { RunId = Guid.NewGuid().ToString("N").Substring(0, 12)
                          Goal = goal
                          IntendedStrategy = getStr "intended_strategy" "unknown"
                          ActualBehavior = getStr "actual_behavior" "unknown"
                          Outcome = classifyOutcome comparison
                          Surprises = getArr "surprises"
                          LessonsLearned = getArr "lessons_learned"
                          SuggestedImprovements = getArr "suggested_improvements"
                          Timestamp = DateTime.UtcNow }
                else
                    return
                        { RunId = Guid.NewGuid().ToString("N").Substring(0, 12)
                          Goal = goal
                          IntendedStrategy = "planned execution"
                          ActualBehavior = sprintf "%d/%d steps completed" comparison.ExecutedSteps comparison.PlannedSteps
                          Outcome = classifyOutcome comparison
                          Surprises = []
                          LessonsLearned = []
                          SuggestedImprovements = []
                          Timestamp = DateTime.UtcNow }
            with _ ->
                return
                    { RunId = Guid.NewGuid().ToString("N").Substring(0, 12)
                      Goal = goal
                      IntendedStrategy = "planned execution"
                      ActualBehavior = sprintf "%d/%d steps completed" comparison.ExecutedSteps comparison.PlannedSteps
                      Outcome = classifyOutcome comparison
                      Surprises = []
                      LessonsLearned = []
                      SuggestedImprovements = []
                      Timestamp = DateTime.UtcNow }
        }

    // =====================================================================
    // Pure reflection (no LLM)
    // =====================================================================

    /// Generate a basic reflection without LLM (for testing and offline use).
    let reflectPure
        (comparison: IntentOutcomeComparison)
        (goal: string)
        : ReflectionReport =

        let surprises = ResizeArray<string>()
        let lessons = ResizeArray<string>()
        let improvements = ResizeArray<string>()

        if not comparison.SkippedSteps.IsEmpty then
            surprises.Add(sprintf "%d planned steps were never executed" comparison.SkippedSteps.Length)
            lessons.Add("Plan contained unreachable or conditional steps")

        if not comparison.FailedSteps.IsEmpty then
            surprises.Add(sprintf "%d steps failed during execution" comparison.FailedSteps.Length)
            lessons.Add("Error handling or step prerequisites need strengthening")
            improvements.Add("Add validation before failure-prone steps")

        if not comparison.UnexpectedSteps.IsEmpty then
            surprises.Add(sprintf "%d steps executed that weren't in the plan" comparison.UnexpectedSteps.Length)
            lessons.Add("Execution diverged from plan — may indicate dynamic adaptation or errors")

        if comparison.OverallAlignment < 0.5 then
            improvements.Add("Consider switching to a different reasoning pattern for this type of goal")

        if comparison.OverallAlignment >= 0.9 && comparison.FailedSteps.IsEmpty then
            lessons.Add("Execution closely matched plan — pattern is well-suited for this goal type")

        { RunId = Guid.NewGuid().ToString("N").Substring(0, 12)
          Goal = goal
          IntendedStrategy = sprintf "Execute %d-step plan" comparison.PlannedSteps
          ActualBehavior = sprintf "Completed %d steps (%.0f%% alignment)" comparison.ExecutedSteps (comparison.OverallAlignment * 100.0)
          Outcome = classifyOutcome comparison
          Surprises = surprises |> Seq.toList
          LessonsLearned = lessons |> Seq.toList
          SuggestedImprovements = improvements |> Seq.toList
          Timestamp = DateTime.UtcNow }

    // =====================================================================
    // Synthesis
    // =====================================================================

    /// Extract common lessons from multiple reflections.
    let synthesizeLessons (reports: ReflectionReport list) : string list =
        reports
        |> List.collect (fun r -> r.LessonsLearned)
        |> List.countBy id
        |> List.sortByDescending snd
        |> List.map (fun (lesson, count) ->
            if count > 1 then sprintf "%s (observed %d times)" lesson count
            else lesson)
