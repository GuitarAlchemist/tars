namespace Tars.Core.WorkflowOfThought

open System
open System.Threading.Tasks
open VariableResolution

type VerifyResult = { Passed: bool; Errors: string list }

module WotExecutor =

  let private storeSingleOutput (ctx: ExecContext) (outputs: string list) (value: obj) : Result<ExecContext,string> =
    match outputs with
    | [] -> Ok ctx
    | [name] -> Ok { ctx with Vars = ctx.Vars.Add(name, value) }
    | many -> 
        let joined = String.Join(", ", many)
        Error $"v0 supports at most one output var per step. Got: {joined}"

  let private runVerify (ctx: ExecContext) (checks: WotCheck list) : VerifyResult =
    let errs =
      checks
      |> List.choose (function
        | WotCheck.NonEmpty v ->
            match ctx.Vars.TryFind v with
            | Some (:? string as s) when not (String.IsNullOrWhiteSpace s) -> None
            | Some (:? string) -> Some $"NonEmpty failed: '{v}' is empty."
            | Some _ -> Some $"NonEmpty failed: '{v}' is not a string."
            | None -> Some $"NonEmpty failed: missing var '{v}'."

        | WotCheck.Contains (v, needle) ->
            match ctx.Vars.TryFind v with
            | Some (:? string as s) when s.Contains needle -> None
            | Some (:? string) -> Some $"Contains failed: '{v}' missing needle '{needle}'."
            | Some _ -> Some $"Contains failed: '{v}' is not a string."
            | None -> Some $"Contains failed: missing var '{v}'."

        | other ->
            Some $"Check not implemented in v0: {other}"
      )

    { Passed = errs.IsEmpty; Errors = errs }

  type ExecutionPolicy = {
    AllowedTools: Set<string>
    MaxToolCalls: int
  }

  let executePlanV0
      (toolInvoker: IToolInvoker)
      (reasoner: IReasoner)
      (mode: ReasonStepMode)
      (policy: ExecutionPolicy)
      (inputs: Map<string,string>)
      (planSteps: Step list) 
      : Async<Result<ExecContext * VerifyResult option * TraceEvent list, string * TraceEvent list>> =
    async {

      let mutable ctx = { Inputs = inputs; Vars = Map.empty }
      let mutable lastVerify : VerifyResult option = None
      let mutable toolCallCount = 0
      let traces = ResizeArray<TraceEvent>()

      let addTrace stepId kind (startUtc: DateTime) (endUtc: DateTime) toolName args outputs status err =
          let duration = int64 (endUtc - startUtc).TotalMilliseconds
          traces.Add({
              StepId = stepId
              Kind = kind
              StartedAtUtc = startUtc
              EndedAtUtc = endUtc
              DurationMs = duration
              ToolName = toolName
              ResolvedArgs = args
              Outputs = outputs
              Status = status
              Error = err
          })

      // Using a recursive loop for clean early exit
      let rec loop steps currentCtx = 
          async {
              match steps with
              | [] -> return Ok (currentCtx, lastVerify, Seq.toList traces)
              | step :: rest ->
                  let startUtc = DateTime.UtcNow
                  
                  // Helper to record error and return
                  let fail (msg: string) (kind: string) (tool: string option) (args: Map<string,string> option) =
                      let endUtc = DateTime.UtcNow
                      addTrace step.Id kind startUtc endUtc tool args step.Outputs StepStatus.Error (Some msg)
                      Error (msg, Seq.toList traces)

                  // Helper to record success
                  let succeed (kind: string) (tool: string option) (args: Map<string,string> option) =
                      let endUtc = DateTime.UtcNow
                      addTrace step.Id kind startUtc endUtc tool args step.Outputs StepStatus.Ok None

                  match step.Action with
                  | StepAction.Work (WorkOperation.ToolCall (toolName, args)) ->
                      // Policy Enforcement
                      if not (policy.AllowedTools.Contains toolName) then
                          return fail $"PolicyViolation: Tool '{toolName}' is not in the allowed list." "tool" (Some toolName) None
                      elif toolCallCount >= policy.MaxToolCalls then
                          return fail $"PolicyViolation: Maximum tool calls ({policy.MaxToolCalls}) exceeded." "tool" (Some toolName) None
                      else
                          toolCallCount <- toolCallCount + 1
                          
                          match resolveToolArgs currentCtx args with
                          | Error e -> return fail $"Step '{step.Id}' resolve args failed: {e}" "tool" (Some toolName) None
                          | Ok resolvedArgs ->
                              let! r = toolInvoker.Invoke(toolName, resolvedArgs)
                              match r with
                              | Error e -> return fail $"Tool '{toolName}' failed: {e}" "tool" (Some toolName) (Some resolvedArgs)
                              | Ok value ->
                                  match storeSingleOutput currentCtx step.Outputs value with
                                  | Error e -> return fail $"Step '{step.Id}' output failed: {e}" "tool" (Some toolName) (Some resolvedArgs)
                                  | Ok ctx2 -> 
                                      succeed "tool" (Some toolName) (Some resolvedArgs)
                                      return! loop rest ctx2

                  | StepAction.Work (WorkOperation.Verify checks) ->
                      let vr = runVerify currentCtx checks
                      lastVerify <- Some vr
                      // store verification output too (optional but handy)
                      match storeSingleOutput currentCtx step.Outputs (box vr) with
                      | Error e -> return fail $"Step '{step.Id}' output failed: {e}" "verify" None None
                      | Ok ctx2 -> 
                          // Verification failure is NOT execution failure in v0
                          succeed "verify" None None
                          return! loop rest ctx2

                  | StepAction.Reason op ->
                      let goal, instruction =
                          match op with
                          | ReasonOperation.Plan g -> Some g, None
                          | ReasonOperation.Critique t -> Some $"Critique {t}", None
                          | ReasonOperation.Synthesize src -> Some $"Synthesize {src}", None
                          | ReasonOperation.Explain t -> Some $"Explain {t}", None
                          | ReasonOperation.Rewrite (t, i) -> Some $"Rewrite {t}", Some i

                      match mode with
                      | ReasonStepMode.Stub ->
                          let stubValue = "<reason-step-stub>"
                          match storeSingleOutput currentCtx step.Outputs stubValue with
                          | Error e -> return fail $"Step '{step.Id}' output failed: {e}" "reason" None None
                          | Ok ctx2 -> 
                              succeed "reason" None None
                              return! loop rest ctx2
                      | ReasonStepMode.Llm | ReasonStepMode.Replay ->
                          // Call reasoner with stepId - Llm calls LLM, Replay reads journal
                          let! r = reasoner.Reason(step.Id, currentCtx, goal, instruction)
                          match r with
                          | Error e -> return fail $"Reason step '{step.Id}' failed: {e}" "reason" None None
                          | Ok valStr ->
                              match storeSingleOutput currentCtx step.Outputs valStr with
                              | Error e -> return fail $"Step '{step.Id}' output failed: {e}" "reason" None None
                              | Ok ctx2 ->
                                  succeed "reason" None None
                                  return! loop rest ctx2

                  | _ ->
                       return fail $"Action not implemented in v0: {step.Action}" "unknown" None None
          }
      
      return! loop planSteps ctx
    }
