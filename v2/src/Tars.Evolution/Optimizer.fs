namespace Tars.Evolution

open System
open System.Threading.Tasks
open Tars.Core
open Tars.Llm
open Tars.Llm.LlmService
open Reflection
open Tars.Metascript.Domain
open Tars.Metascript

module Optimizer =

    type IWorkflowOptimizer =
        abstract member OptimizeAsync: workflow: Workflow * feedback: Feedback -> Task<Workflow option>

    type LlmWorkflowOptimizer(llm: ILlmService) =
        interface IWorkflowOptimizer with
            member _.OptimizeAsync(workflow, feedback) =
                task {
                    // Serialize current workflow to JSON for the prompt
                    let workflowJson =
                        System.Text.Json.JsonSerializer.Serialize(
                            workflow,
                            System.Text.Json.JsonSerializerOptions(WriteIndented = true)
                        )

                    let suggestion =
                        feedback.Suggestion
                        |> Option.defaultValue "Improve the workflow logic based on the feedback."

                    let prompt =
                        $"""You are an Expert Metascript Optimizer.
                            
CURRENT WORKFLOW (JSON):
%s{workflowJson}

FEEDBACK:
Score: %.2f{feedback.Score}
Comment: %s{feedback.Comment}
Suggestion: %s{suggestion}

TASK:
Modify the workflow JSON to implement the suggestion and improve the workflow.
- You can add steps, remove steps, or change instructions.
- Ensure the JSON is valid and matches the Metascript schema.
- Maintain the same Inputs unless strictly necessary to change.
- OUTPUT ONLY THE NEW JSON.

METASCRIPT SCHEMA EXAMPLES:
{{ "Type": "agent", "Id": "step1", "Instruction": "...", "Agent": "...", "Outputs": ["out1"] }}
{{ "Type": "decision", "Id": "check", "Params": {{ "condition": "{{{{step1.out1}}}} == 'yes'", "trueOutput": "ok", "falseOutput": "retry" }} }}
"""

                    let req =
                        { ModelHint = Some "code" // Use coding model
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
                        let json =
                            match JsonParsing.tryParseElement response.Text with
                            | Result.Ok elem -> elem.GetRawText()
                            | Result.Error _ -> response.Text

                        match Parser.parseJson json with
                        | Parser.Success newWorkflow -> return Some newWorkflow
                        | _ -> return None
                    with _ ->
                        return None
                }
