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
                            sprintf
                                "[%s] -> %s (took %d ms)"
                                t.Step
                                (t.Output.Substring(0, min t.Output.Length 50))
                                t.DurationMs)
                        |> String.concat "\n"

                    let prompt =
                        sprintf
                            """You are a generic optimization critic (Reflection Agent).
                            
GOAL: %s

ACTUAL OUTPUT:
%s

EXECUTION TRACE:
%s

Analyze the performance.
1. Did it achieve the goal?
2. Was it efficient?
3. Any obvious bugs or improvemens?

Output JSON only:
{
  "type": "Success" | "Failure" | "Optimization",
  "score": 0.0 to 1.0,
  "comment": "Short analysis",
  "suggestion": "Specific instruction on how to change the workflow or prompt to improve"
}"""
                            goal
                            output
                            traceStr

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
                          Seed = None }

                    let! response = llm.CompleteAsync req

                    try
                        let doc = System.Text.Json.JsonDocument.Parse(response.Text)
                        let root = doc.RootElement

                        let typeStr = root.GetProperty("type").GetString()

                        let fType =
                            match typeStr.ToLower() with
                            | "success" -> Success
                            | "failure" -> Failure
                            | _ -> Optimization

                        let mutable scoreProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                        let score =
                            if root.TryGetProperty("score", &scoreProp) then
                                scoreProp.GetDouble()
                            else
                                0.5

                        let comment = root.GetProperty("comment").GetString()

                        let mutable suggestionProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                        let suggestion =
                            if root.TryGetProperty("suggestion", &suggestionProp) then
                                Some(suggestionProp.GetString())
                            else
                                None

                        return
                            { Type = fType
                              Score = score
                              Comment = comment
                              Suggestion = suggestion }
                    with ex ->
                        // Fallback
                        return
                            { Type = Failure
                              Score = 0.0
                              Comment = $"Parsing failed: {ex.Message}"
                              Suggestion = None }
                }
