namespace Tars.Evolution

open System
open System.Text.Json
open Tars.Llm
open Tars.Core.WorkflowOfThought

type PatternDefinition =
    { Name: string
      Goal: string
      Description: string
      Template: string
      Score: float
      CreatedFromRunId: Guid option
      /// The lower-level expansion this pattern was abstracted from.
      /// Used by the Grammar Governor for round-trip validation.
      RollbackExpansion: string option }

/// <summary>
/// Phase 15.6: Compiles execution traces into reusable reasoning patterns.
/// </summary>
module TraceCompiler =

    /// <summary>
    /// Analyzes a successful trace and extracts a generalized pattern.
    /// </summary>
    let compileFromTrace
        (llm: ILlmService)
        (runId: Guid)
        (trace: CanonicalTraceEvent list)
        (goal: string)
        : Async<Result<PatternDefinition, string>> =
        async {
            let options = JsonSerializerOptions()
            options.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())
            let traceJson = JsonSerializer.Serialize(trace, options)

            let prompt =
                $"""You are TARS's Pattern Compiler.
Your task is to generalize a specific execution trace into a REUSABLE REASONING PATTERN.

GOAL: {goal}

TRACE:
{traceJson}

INSTRUCTIONS:
1. Analyze the trace to understand the successful strategy used.
2. Abstract away specific data (e.g., "goat", "cabbage") into variables or generic descriptions.
3. Create a JSON template for a Workflow of Thought (WoT) that replicates this strategy.
4. The template should be a valid JSON object describing the steps.

OUTPUT FORMAT (JSON ONLY):
{{
  "synthesis": {{
    "strategy_name": "Name of the strategy (e.g. IterativeConstraintVerification)",
    "description": "How the strategy works",
    "applicability": "When to use this pattern"
  }},
  "template": {{
    "kind": "custom",
    "nodes": [
       {{ "id": "step1", "kind": "reason", "prompt": "Analyze {{input}}..." }},
       {{ "id": "step2", "kind": "tool", "tool": "search_web", "args": {{ "q": "{{query}}" }} }}
    ]
  }}
}}
"""

            let req =
                { ModelHint = None
                  Model = None
                  SystemPrompt = Some "You are a meta-cognitive compiler. Output valid JSON."
                  MaxTokens = Some 2000
                  Temperature = Some 0.0
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
                let! resp = llm.CompleteAsync req |> Async.AwaitTask

                // Robust extraction
                let mutable text = resp.Text.Trim()
                let firstBrace = text.IndexOf('{')
                let lastBrace = text.LastIndexOf('}')

                if firstBrace >= 0 && lastBrace > firstBrace then
                    text <- text.Substring(firstBrace, lastBrace - firstBrace + 1)

                use doc = JsonDocument.Parse(text)
                let root = doc.RootElement

                let synth = root.GetProperty("synthesis")
                let name = synth.GetProperty("strategy_name").GetString()
                let desc = synth.GetProperty("description").GetString()
                let template = root.GetProperty("template").GetRawText()

                // The trace is the lower-level expansion this pattern was abstracted from.
                // Capture it as the rollback path for round-trip validation.
                let rollbackExpansion =
                    trace
                    |> List.collect (fun evt ->
                        let args =
                            evt.ResolvedArgs
                            |> Option.defaultValue []
                            |> List.map (fun (k, v) -> $"  {k}: {v}")
                        let outputs = evt.Outputs |> List.map (fun o -> $"  output: {o}")
                        let toolInfo =
                            match evt.ToolName with
                            | Some t -> $"  tool: {t}"
                            | None -> ""
                        [ $"step: {evt.StepId} ({evt.Kind})"
                          toolInfo ]
                        @ args @ outputs)
                    |> List.filter (fun s -> s.Length > 0)
                    |> String.concat "\n"

                let pattern =
                    { Name = name
                      Goal = goal
                      Description = desc
                      Template = template
                      Score = 0.5 // Initial score
                      CreatedFromRunId = Some runId
                      RollbackExpansion = if rollbackExpansion.Length > 10 then Some rollbackExpansion else None }

                return Result.Ok pattern
            with ex ->
                return Result.Error $"Pattern Compilation Failed: {ex.Message}"
        }
