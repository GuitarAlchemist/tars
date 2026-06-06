namespace Tars.Evolution

open System
open System.Threading.Tasks
open System.Text.Json
open Tars.Core
open Tars.Core.WorkflowOfThought
open Tars.Llm

/// <summary>
/// Phase 15.1: Symbolic Reflector implementation.
/// Uses LLM to analyze execution traces and produce structured symbolic reflections.
/// </summary>
type SymbolicReflector(llm: ILlmService, graphService: IGraphService, agentId: AgentId) =

    interface ISymbolicReflector with
        member this.ReflectOnRunAsync(runId: Guid, traces: CanonicalTraceEvent list) =
            task {
                let! result = this.ReflectOnTrace(runId, traces)
                return Result.Ok result
            }

    /// <summary>
    /// Analyzes a specific trace list directly
    /// </summary>
    member this.ReflectOnTrace(runId: Guid, traces: CanonicalTraceEvent list) =
        task {
            let options = JsonSerializerOptions()
            options.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())
            let traceJson = JsonSerializer.Serialize(traces, options)

            // Heuristic: Check for obvious failure in traces to guide the prompt
            let failure =
                traces
                |> List.tryFind (fun t -> t.Status = StepStatus.Error || t.Error.IsSome)
                |> Option.map (fun t -> t.Error |> Option.defaultValue "Unknown error")

            let contextPrompt =
                match failure with
                | Some err -> $"The run FAILED with error: {err}. Focus on the root cause."
                | None -> "The run SUCCEEDED. Focus on optimization and pattern recognition."

            let prompt =
                $"""You are TARS's Meta-Cognitive Reflector.
Your job is to analyze the execution trace of an agent run and produce specific, structured OBSERVATIONS.

CONTEXT:
{contextPrompt}

TRACE:
{traceJson}

INSTRUCTIONS:
Analyze the trace and output a JSON object with the following structure:
{{
  "trigger_type": "TaskFailed" | "TaskCompleted" | "ContradictionDetected",
  "trigger_details": "Description of the trigger event",
  "observations": [
    {{
      "type": "Performance" | "Anomaly" | "Contradiction" | "Pattern",
      "description": "One sentence description",
      "severity": "Low" | "Medium" | "High" (optional, for Anomaly)
    }}
  ],
  "summary": "Brief summary of the run's outcome"
}}

Output ONLY valid JSON.
"""

            let req =
                { ModelHint = Some "reasoning" // Use a smart model for reflection
                  Model = None
                  SystemPrompt = Some "You are a meta-cognitive analysis engine. Output structured JSON."
                  MaxTokens = Some 2000
                  Temperature = Some 0.0
                  Stop = []
                  Messages = [ { Role = Role.User; Content = prompt } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = true
                  Seed = None
                  ContextWindow = None }

            try
                let! resp = llm.CompleteAsync req

                // Robust JSON extraction (same as SelfImprovement)
                let mutable text = resp.Text.Trim()
                let firstBrace = text.IndexOf('{')
                let lastBrace = text.LastIndexOf('}')

                if firstBrace >= 0 && lastBrace > firstBrace then
                    text <- text.Substring(firstBrace, lastBrace - firstBrace + 1)

                use doc = JsonDocument.Parse(text)
                let root = doc.RootElement

                // Parse Trigger
                let triggerType = root.GetProperty("trigger_type").GetString()
                let triggerDetails = root.GetProperty("trigger_details").GetString()

                let trigger =
                    match triggerType with
                    | "TaskFailed" -> ReflectionTrigger.TaskFailed(Guid.Empty, triggerDetails)
                    | "ContradictionDetected" -> ReflectionTrigger.ContradictionDetected("Unknown", triggerDetails)
                    | _ -> ReflectionTrigger.TaskCompleted(Guid.Empty, ReflectionTaskResult.TaskSuccess(triggerDetails))

                let reflection = SymbolicReflection.Create(agentId, trigger)

                // Parse Observations
                let obsProp = root.GetProperty("observations")

                let reflected =
                    if obsProp.ValueKind = JsonValueKind.Array then
                        let obsList =
                            [ for obs in obsProp.EnumerateArray() do
                                  let typeStr = obs.GetProperty("type").GetString()
                                  let desc = obs.GetProperty("description").GetString()

                                  yield
                                      match typeStr with
                                      | "Anomaly" ->
                                          let sev =
                                              if
                                                  obs.TryGetProperty("severity", ref (Unchecked.defaultof<_>)) // rough check
                                              then
                                                  AnomalySeverity.Medium
                                              else
                                                  AnomalySeverity.Medium

                                          ReflectionObservation.AnomalyObserved(desc, sev)
                                      | "Contradiction" ->
                                          ReflectionObservation.ContradictionObserved("Unknown", "Unknown", desc)
                                      | "Pattern" -> ReflectionObservation.PatternObserved(desc, 1, 0.8)
                                      | _ ->
                                          ReflectionObservation.PerformanceObserved(
                                              desc,
                                              0.0,
                                              ReflectionTrend.TrendUnknown
                                          ) ]

                        // Add observations to reflection
                        obsList
                        |> List.fold (fun (acc: SymbolicReflection) o -> acc.WithObservation(o)) reflection
                    else
                        reflection

                return reflected
            with ex ->
                // Fallback for parsing errors
                return
                    SymbolicReflection.Create(
                        agentId,
                        ReflectionTrigger.TaskFailed(Guid.Empty, $"Reflection failed: {ex.Message}")
                    )
        }
