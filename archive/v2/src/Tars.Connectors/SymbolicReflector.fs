namespace Tars.Connectors

open System
open Tars.Core
open Tars.Core.WorkflowOfThought
open Tars.LinkedData

/// Performs Symbolic Reflection over execution traces stored in the Knowledge Graph.
type SymbolicReflector(endpointUri: Uri, ?authOpt: string) =

    let queryUri = Uri(endpointUri.ToString() + "/query")
    let systemAgentId = AgentId Guid.Empty

    /// Fetch recent runs from the Knowledge Graph
    member this.GetRecentRunsAsync(limit: int) : Async<Result<RunEntity list, string>> =
        async {
            let query =
                sprintf
                    "PREFIX tars: <http://tars.ai/ns#> SELECT ?run ?goal ?pattern ?timestamp WHERE { ?run a tars:Run ; tars:goal ?goal ; tars:pattern ?pattern ; tars:timestamp ?timestamp . } ORDER BY DESC(?timestamp) LIMIT %d"
                    limit

            let! res = SparqlQueryRunner.query queryUri authOpt query

            match res with
            | Result.Ok results -> return Result.Ok(RdfReconstructor.toRunEntity results)
            | Result.Error err -> return Result.Error err
        }

    /// Fetch all steps for a specific run
    member this.GetStepsForRunAsync(runId: Guid) : Async<Result<StepEntity list, string>> =
        async {
            let query =
                sprintf
                    "PREFIX tars: <http://tars.ai/ns#> SELECT ?s ?stepId ?runId ?nodeType ?content ?timestamp WHERE { ?s a tars:Step ; tars:runId <http://tars.ai/resource/run/%s> ; tars:runId ?runId ; tars:stepId ?stepId ; tars:nodeType ?nodeType ; tars:content ?content ; tars:timestamp ?timestamp . } ORDER BY ?timestamp"
                    (runId.ToString())

            let! res = SparqlQueryRunner.query queryUri authOpt query

            match res with
            | Result.Ok results -> return Result.Ok(RdfReconstructor.toStepEntity results)
            | Result.Error err -> return Result.Error err
        }

    /// Pure logic to analyze steps and produce a reflection, separated for testing.
    static member AnalyzeSteps(runId: Guid, steps: StepEntity list) : SymbolicReflection =
        let systemAgentId = AgentId Guid.Empty

        let trigger =
            TaskCompleted(runId, ReflectionTaskResult.TaskSuccess "Trace Analyzed")

        let reflection = SymbolicReflection.Create(systemAgentId, trigger)

        let mutable updatedReflection = reflection

        // Check for duplicate step IDs (retries or loops)
        let stepCounts =
            steps
            |> List.countBy (fun s -> s.StepId)
            |> List.filter (fun (_, count) -> count > 1)

        for (stepId, count) in stepCounts do
            updatedReflection <-
                updatedReflection.WithObservation(PatternObserved(sprintf "StepRepetition:%s" stepId, count, 0.8))

        // Check for abnormal duration
        if steps.Length > 20 then
            updatedReflection <-
                updatedReflection.WithObservation(
                    AnomalyObserved(sprintf "HighStepCount: %d steps" steps.Length, AnomalySeverity.Medium)
                )

        // 1. Tool Usage Analysis
        let toolRegex = System.Text.RegularExpressions.Regex(@"ToolCall\(\s*([^,)]+)")

        let toolUsage =
            steps
            |> List.choose (fun s ->
                let m = toolRegex.Match(s.Content)

                if m.Success then
                    Some(m.Groups.[1].Value.Trim().Trim('"').Trim('\''))
                else
                    None)
            |> List.countBy id

        for (tool, count) in toolUsage do
            if count >= 3 then
                updatedReflection <-
                    updatedReflection.WithObservation(PatternObserved(sprintf "FrequentToolUsage: %s" tool, count, 0.7))

        // 2. Error detection in content
        let errorSteps =
            steps
            |> List.filter (fun s ->
                s.Content.ToLowerInvariant().Contains("error")
                || s.Content.ToLowerInvariant().Contains("fail")
                || s.Content.ToLowerInvariant().Contains("exception"))
            |> List.length

        if errorSteps > 0 then
            updatedReflection <-
                updatedReflection.WithObservation(
                    AnomalyObserved(sprintf "ErrorKeywordsCaptured: %d steps" errorSteps, AnomalySeverity.Low)
                )

        updatedReflection

    /// Perform reflection on a single run to detect anomalies or patterns
    member this.ReflectOnRunAsync(runId: Guid) : Async<Result<SymbolicReflection, string>> =
        async {
            let! res = this.GetStepsForRunAsync(runId)

            match res with
            | Result.Error err -> return Result.Error err
            | Result.Ok steps ->
                let reflection = SymbolicReflector.AnalyzeSteps(runId, steps)
                return Result.Ok reflection
        }

    interface ISymbolicReflector with
        member this.ReflectOnRunAsync(runId: Guid, traces: CanonicalTraceEvent list) =
            // Legacy connector implementation ignores traces and fetches from graph
            this.ReflectOnRunAsync(runId) |> Async.StartAsTask
