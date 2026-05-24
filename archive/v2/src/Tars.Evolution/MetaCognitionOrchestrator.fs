namespace Tars.Evolution

open System
open System.Threading.Tasks
open Tars.Llm
open Tars.Core.MetaCognition
open Tars.Core.WorkflowOfThought
open Tars.Cortex

/// Top-level orchestrator that ties the five meta-cognitive components into a coherent cycle:
/// 1. Collect and cluster failures
/// 2. Detect capability gaps
/// 3. Generate targeted curriculum
/// 4. Reflect on recent executions
/// 5. Produce recommendations
module MetaCognitionOrchestrator =

    /// Run the full meta-cognitive analysis cycle.
    let runCycle
        (llm: ILlmService option)
        (config: MetaCognitionConfig)
        (recentOutcomes: PatternOutcomeStore.PatternOutcome list)
        (recentCycles: RetroactionLoop.CycleResult list)
        (recentTraces: (string list * TraceEvent list * string * string) list)  // (plannedStepIds, trace, goal, result)
        : Task<MetaCognitionResult> =
        task {
            // =========================================================
            // Step 1: Collect and cluster failures
            // =========================================================
            let! clusters =
                FailureAnalyzer.analyzeFailures llm config.FailureClusterThreshold recentOutcomes recentCycles

            // =========================================================
            // Step 2: Detect capability gaps
            // =========================================================
            let successes =
                recentOutcomes
                |> List.filter (fun o -> o.Success)
                |> List.map (fun o -> o.Goal, GapDetection.extractDomainTags o.Goal)

            let failures = FailureAnalyzer.collectFailures recentOutcomes recentCycles

            let gaps =
                GapDetection.detectGaps config.GapDetectionThreshold clusters successes failures
                |> GapDetection.rankGaps

            // =========================================================
            // Step 3: Generate targeted curriculum
            // =========================================================
            let! tasks =
                match llm with
                | Some llmService when config.EnableLlmRefinement ->
                    CurriculumPlanner.generateTasksWithLlm llmService gaps config.MaxTargetedTasks
                | _ ->
                    Task.FromResult(CurriculumPlanner.generateTasksFromTemplates gaps config.MaxTargetedTasks)

            // =========================================================
            // Step 4: Reflect on recent executions
            // =========================================================
            let mutable reflections = []

            for (plannedIds, trace, goal, result) in recentTraces do
                let comparison = ReflectionEngine.compareIntentVsOutcome plannedIds trace goal

                let report =
                    match llm with
                    | Some llmService when config.EnableLlmRefinement ->
                        ReflectionEngine.reflect llmService comparison goal result
                        |> Async.AwaitTask |> Async.RunSynchronously
                    | _ ->
                        ReflectionEngine.reflectPure comparison goal

                reflections <- report :: reflections

            let reflections = reflections |> List.rev

            // =========================================================
            // Step 5: Synthesize recommendations
            // =========================================================
            let recommendations = ResizeArray<string>()

            // From gaps
            for gap in gaps |> List.truncate 3 do
                recommendations.Add(
                    sprintf "GAP: %s — %.0f%% failure rate. Remedy: %A"
                        gap.Domain (gap.FailureRate * 100.0) gap.SuggestedRemedy)

            // From reflections
            let lessons = ReflectionEngine.synthesizeLessons reflections
            for lesson in lessons |> List.truncate 3 do
                recommendations.Add(sprintf "LESSON: %s" lesson)

            // From clusters
            for cluster in clusters |> List.sortByDescending (fun c -> c.Frequency) |> List.truncate 2 do
                recommendations.Add(
                    sprintf "RECURRING: %d failures from %A"
                        cluster.Frequency cluster.RootCause)

            return
                { FailureClusters = clusters
                  DetectedGaps = gaps
                  GeneratedTasks = tasks
                  Reflections = reflections
                  NewBeliefs = gaps.Length + reflections.Length
                  Recommendations = recommendations |> Seq.toList }
        }

    /// Quick summary for display.
    let summarize (result: MetaCognitionResult) : string =
        let lines = ResizeArray<string>()
        lines.Add(sprintf "Meta-Cognitive Analysis: %d clusters, %d gaps, %d tasks, %d reflections"
            result.FailureClusters.Length
            result.DetectedGaps.Length
            result.GeneratedTasks.Length
            result.Reflections.Length)
        for r in result.Recommendations do
            lines.Add(sprintf "  > %s" r)
        String.Join("\n", lines)
