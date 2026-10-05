namespace Tars.Evolution

open System
open System.Text.Json
open System.Threading.Tasks
open Tars.Llm

type IEvaluationStrategy =
    abstract member Evaluate: taskDef: TaskDefinition * result: TaskResult -> Task<EvaluationResult>

type SemanticEvaluation(llm: ILlmService, ?minConfidence: float, ?logger: string -> unit) =
    let minConfidence = defaultArg minConfidence 0.6
    let log = defaultArg logger (fun _ -> ())

    let tryGetProperty (name: string) (elem: JsonElement) =
        let mutable prop = Unchecked.defaultof<JsonElement>
        if elem.TryGetProperty(name, &prop) then
            Some prop
        elif elem.ValueKind = JsonValueKind.Object then
            elem.EnumerateObject()
            |> Seq.tryFind (fun p -> p.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
            |> Option.map (fun p -> p.Value)
        else
            None

    let getBool name (elem: JsonElement) =
        match tryGetProperty name elem with
        | Some prop ->
            match prop.ValueKind with
            | JsonValueKind.True -> Some true
            | JsonValueKind.False -> Some false
            | JsonValueKind.String ->
                match Boolean.TryParse(prop.GetString()) with
                | true, value -> Some value
                | _ -> None
            | _ -> None
        | None -> None

    let getString name (elem: JsonElement) =
        match tryGetProperty name elem with
        | Some prop when prop.ValueKind = JsonValueKind.String -> prop.GetString()
        | _ -> ""

    let getDouble name (elem: JsonElement) =
        match tryGetProperty name elem with
        | Some prop ->
            match prop.ValueKind with
            | JsonValueKind.Number -> prop.GetDouble()
            | JsonValueKind.String ->
                match Double.TryParse(prop.GetString()) with
                | true, value -> value
                | _ -> 0.0
            | _ -> 0.0
        | None -> 0.0

    let getStringList name (elem: JsonElement) =
        match tryGetProperty name elem with
        | Some prop ->
            match prop.ValueKind with
            | JsonValueKind.Array ->
                prop.EnumerateArray()
                |> Seq.choose (fun item ->
                    if item.ValueKind = JsonValueKind.String then
                        Some(item.GetString())
                    else
                        None)
                |> Seq.toList
            | JsonValueKind.String -> [ prop.GetString() ]
            | _ -> []
        | None -> []

    interface IEvaluationStrategy with
        member _.Evaluate(taskDef, result) =
            task {
                let constraints =
                    if taskDef.Constraints.IsEmpty then
                        "None"
                    else
                        String.concat "; " taskDef.Constraints

                let jsonTemplate =
                    "{\"passed\":true,\"confidence\":0.0,\"summary\":\"...\",\"issues\":[\"...\"],\"suggested_fixes\":[\"...\"]}"

                // With --run-code the examples in the validation criteria may have run with the code and
                // passed (result.Evaluation). They check its values, not what else the goal asks.
                let examplesPassed =
                    match result.Evaluation with
                    | Some examples when examples.Passed ->
                        [ ""
                          $"The code in the output was run with the examples in the validation criteria: {examples.Summary}"
                          "Do not judge its results again. Judge only what the examples do not check, such as how the goal or the constraints say the code must be written. If they ask nothing more, it passes." ]
                    | _ -> []

                let prompt =
                    String.concat
                        "\n"
                        [ "You are a strict evaluator of task outputs."
                          $"Goal: {taskDef.Goal}"
                          $"Constraints: {constraints}"
                          $"Validation Criteria: {taskDef.ValidationCriteria}"
                          yield! examplesPassed
                          "Judge the code, not how the answer presents it: the agent starts its answers with an `ACT:` line and may write text around its code. Ignore both, and any constraint on that presentation (the `ACT:` line, text around the code, the number of code blocks). Constraints on what the code returns or prints still apply."
                          ""
                          "Output:"
                          "```"
                          result.Output
                          "```"
                          ""
                          "Return ONLY this JSON object:"
                          jsonTemplate ]

                let req: LlmRequest =
                    { ModelHint = Some "reasoning"
                      Model = None
                      SystemPrompt = Some "Evaluate task output for semantic correctness."
                      MaxTokens = Some 400
                      Temperature = Some 0.2
                      Stop = []
                      Messages = [ { Role = Role.User; Content = prompt } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = Some (ResponseFormat.Constrained (Grammar.JsonSchema EvolutionSchemas.evaluationSchema))
                      Stream = false
                      JsonMode = true
                      Seed = None

                      ContextWindow = None }

                try
                    let! response = llm.CompleteAsync(req)

                    match JsonParsing.tryParseElement response.Text with
                    | Ok elem ->
                        let rawPassed = getBool "passed" elem |> Option.defaultValue false
                        let confidence = getDouble "confidence" elem |> max 0.0 |> min 1.0
                        let summary = getString "summary" elem |> fun s -> if String.IsNullOrWhiteSpace s then "No summary provided." else s
                        let issues = getStringList "issues" elem
                        let fixes = getStringList "suggested_fixes" elem

                        let passed =
                            if rawPassed && confidence >= minConfidence then
                                true
                            elif rawPassed then
                                false
                            else
                                false

                        let summary =
                            if rawPassed && confidence < minConfidence then
                                summary + $" (Confidence {confidence:F2} below threshold {minConfidence:F2}.)"
                            else
                                summary

                        return
                            { Passed = passed
                              Confidence = confidence
                              Summary = summary
                              Issues = issues
                              SuggestedFixes = fixes
                              EvaluatedAt = DateTime.UtcNow }
                    | Error err ->
                        log $"[Evaluation] Failed to parse JSON: {err}"

                        return
                            { Passed = false
                              Confidence = 0.0
                              Summary = $"Failed to parse evaluation JSON: {err}"
                              Issues = [ "llm_output_parse_error" ]
                              SuggestedFixes = []
                              EvaluatedAt = DateTime.UtcNow }
                with ex ->
                    log $"[Evaluation] Error during evaluation: {ex.Message}"

                    return
                        { Passed = false
                          Confidence = 0.0
                          Summary = $"Evaluation failed: {ex.Message}"
                          Issues = [ "evaluation_error" ]
                          SuggestedFixes = []
                          EvaluatedAt = DateTime.UtcNow }
            }
