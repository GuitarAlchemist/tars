namespace Tars.Metascript

open System
open System.Threading.Tasks
open System.Text.RegularExpressions
open Tars.Core
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Tools
open Domain

module Engine =

    type MetascriptContext =
        { Llm: ILlmService
          Kernel: KernelContext
          Tools: ToolRegistry }

    let private resolveVariables (text: string) (state: WorkflowState) =
        let pattern = "\{\{([^}]+)\}\}"

        Regex.Replace(
            text,
            pattern,
            fun m ->
                let key = m.Groups.[1].Value.Trim()

                if key.Contains(".") then
                    let parts = key.Split('.')
                    let stepId = parts.[0]
                    let outputName = parts.[1]

                    match state.StepOutputs.TryFind stepId with
                    | Some outputs ->
                        match outputs.TryFind outputName with
                        | Some value -> string value
                        | None -> m.Value
                    | None -> m.Value
                else
                    match state.Variables.TryFind key with
                    | Some value -> string value
                    | None -> m.Value
        )

    let executeStep (ctx: MetascriptContext) (step: WorkflowStep) (state: WorkflowState) : Task<Map<string, obj>> =
        task {
            match step.Type.ToLower() with
            | "agent" ->
                let instruction = resolveVariables (defaultArg step.Instruction "") state

                // Gather context
                let contextStr =
                    defaultArg step.Context []
                    |> List.map (fun c ->
                        match state.StepOutputs.TryFind c.StepId with
                        | Some outputs ->
                            match outputs.TryFind c.OutputName with
                            | Some value -> sprintf "Context from %s (%s):\n%s" c.StepId c.OutputName (string value)
                            | None -> ""
                        | None -> "")
                    |> String.concat "\n\n"

                let prompt =
                    sprintf
                        """You are %s.
Instruction: %s

%s

Output only the requested result."""
                        (defaultArg step.Agent "Assistant")
                        instruction
                        contextStr

                let req =
                    { ModelHint = Some "reasoning"
                      MaxTokens = None
                      Temperature = None
                      Messages = [ { Role = Role.User; Content = prompt } ] }

                let! response = ctx.Llm.CompleteAsync req

                // Assume the first output is the main text response
                let outputName =
                    match defaultArg step.Outputs [] with
                    | head :: _ -> head
                    | [] -> "output"

                return Map [ outputName, box response.Text ]

            | "tool" ->
                match step.Tool with
                | Some toolName ->
                    match ctx.Tools.Get(toolName) with
                    | Some tool ->
                        let args =
                            step.Params
                            |> Option.defaultValue Map.empty
                            |> Map.map (fun _ v -> resolveVariables v state)

                        let input =
                            if args.ContainsKey("input") then
                                args["input"]
                            elif args.ContainsKey("command") then
                                args["command"]
                            else
                                // Fallback: Serialize to JSON
                                System.Text.Json.JsonSerializer.Serialize(args)

                        let! result = tool.Execute(input)

                        match result with
                        | Result.Ok s -> return Map [ "stdout", box s ]
                        | Result.Error e -> return Map [ "error", box e ]
                    | None -> return Map [ "error", box (sprintf "Tool '%s' not found" toolName) ]
                | None -> return Map.empty

            | _ -> return Map.empty
        }

    let run (ctx: MetascriptContext) (workflow: Workflow) (inputs: Map<string, obj>) =
        task {
            let mutable state =
                { Workflow = workflow
                  CurrentStepIndex = 0
                  Variables = inputs
                  StepOutputs = Map.empty
                  ExecutionTrace = [] }

            for step in workflow.Steps do
                let! outputs = executeStep ctx step state

                state <-
                    { state with
                        StepOutputs = state.StepOutputs.Add(step.Id, outputs) }

            return state
        }
