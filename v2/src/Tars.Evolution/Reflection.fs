namespace Tars.Evolution

open System
open System.Threading.Tasks
open Tars.Core
open Tars.Llm
open Tars.Llm.LlmService

module Reflection =

    type TraceItem =
        { Step: string
          Input: string
          Output: string
          DurationMs: int64 }

    type FeedbackType =
        | Success
        | Failure
        | Optimization

    type Feedback =
        { Type: FeedbackType
          Score: float // 0.0 to 1.0
          Comment: string
          Suggestion: string option }

    type IReflectionAgent =
        abstract member ReflectAsync: taskGoal: string * output: string * trace: TraceItem list -> Task<Feedback>

    type LlmReflectionAgent(llm: ILlmService) =
        interface IReflectionAgent with
            member _.ReflectAsync(goal, output, trace) =
                task {
                    let traceStr =
                        trace
                        |> List.map (fun t ->
                            $"[%s{t.Step}] -> %s{t.Output.Substring(0, min t.Output.Length 50)} (took %d{t.DurationMs} ms)"
                        )
                        |> String.concat "\n"

                    let prompt =
                        $"""You are a generic optimization critic (Reflection Agent).
                            
GOAL: %s{goal}

ACTUAL OUTPUT:
%s{output}

EXECUTION TRACE:
%s{traceStr}

Analyze the performance.
1. Did it achieve the goal?
2. Was it efficient?
3. Any obvious bugs or improvemens?

Output JSON only:
{{
  "type": "Success" | "Failure" | "Optimization",
  "score": 0.0 to 1.0,
  "comment": "Short analysis",
  "suggestion": "Specific instruction on how to change the workflow or prompt to improve"
}}"""

                    let req =
                        { ModelHint = Some "reasoning" // Use smart model
                          Model = None
                          SystemPrompt = None
                          MaxTokens = None
                          Temperature = Some 0.0
                          Stop = []
                          Messages = [ { Role = Role.User; Content = prompt } ]
                          Tools = []
                          ToolChoice = None
                          ResponseFormat = Some ResponseFormat.Json
                          Stream = false
                          JsonMode = true
                          Seed = None

                          ContextWindow = None }

                    let! response = llm.CompleteAsync req

                    let tryGetProp name (elem: System.Text.Json.JsonElement) =
                        if elem.ValueKind = System.Text.Json.JsonValueKind.Object then
                            elem.EnumerateObject()
                            |> Seq.tryFind (fun p -> p.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
                            |> Option.map (fun p -> p.Value)
                        else
                            None

                    match JsonParsing.tryParseElement response.Text with
                    | Result.Ok root ->
                        let typeStr =
                            tryGetProp "type" root
                            |> Option.filter (fun p -> p.ValueKind = System.Text.Json.JsonValueKind.String)
                            |> Option.map (fun p -> p.GetString())
                            |> Option.defaultValue "optimization"

                        let fType =
                            match typeStr.ToLower() with
                            | "success" -> Success
                            | "failure" -> Failure
                            | _ -> Optimization

                        let score =
                            tryGetProp "score" root
                            |> Option.map (fun p ->
                                match p.ValueKind with
                                | System.Text.Json.JsonValueKind.Number -> p.GetDouble()
                                | System.Text.Json.JsonValueKind.String ->
                                    match Double.TryParse(p.GetString()) with
                                    | true, v -> v
                                    | _ -> 0.5
                                | _ -> 0.5)
                            |> Option.defaultValue 0.5

                        let comment =
                            tryGetProp "comment" root
                            |> Option.filter (fun p -> p.ValueKind = System.Text.Json.JsonValueKind.String)
                            |> Option.map (fun p -> p.GetString())
                            |> Option.defaultValue "No comment provided."

                        let suggestion =
                            tryGetProp "suggestion" root
                            |> Option.filter (fun p -> p.ValueKind = System.Text.Json.JsonValueKind.String)
                            |> Option.map (fun p -> p.GetString())

                        return
                            { Type = fType
                              Score = score
                              Comment = comment
                              Suggestion = suggestion }
                    | Result.Error err ->
                        return
                            { Type = Failure
                              Score = 0.0
                              Comment = $"Parsing failed: {err}"
                              Suggestion = None }
                }
