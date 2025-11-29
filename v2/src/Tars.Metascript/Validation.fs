namespace Tars.Metascript

open Tars.Metascript.Domain

module Validation =
    let private error msg (errors: string list) = msg :: errors

    let private validateStepIds (steps: WorkflowStep list) =
        let duplicates =
            steps
            |> List.groupBy (fun s -> s.Id.Trim())
            |> List.choose (fun (id, items) -> if id = "" || List.length items > 1 then Some id else None)

        if duplicates.IsEmpty then
            []
        else
            let joined = System.String.Join(", ", duplicates)
            [ $"Duplicate or empty step ids: {joined}" ]

    let private validateStep (step: WorkflowStep) (priorSteps: Set<string>) =
        let mutable errors = []

        if System.String.IsNullOrWhiteSpace step.Id then
            errors <- error "Step Id is required" errors

        if System.String.IsNullOrWhiteSpace step.Type then
            errors <- error $"Step '{step.Id}': Type is required" errors

        match step.Type.ToLower() with
        | "agent" ->
            if step.Agent |> Option.forall System.String.IsNullOrWhiteSpace then
                errors <- error $"Step '{step.Id}': Agent name is required for agent steps" errors
        | "tool" ->
            if step.Tool |> Option.forall System.String.IsNullOrWhiteSpace then
                errors <- error $"Step '{step.Id}': Tool name is required for tool steps" errors
        | "loop"
        | "decision"
        | "retrieval" -> () // optional specifics validated in execution
        | other ->
            errors <- error $"Step '{step.Id}': Unsupported type '{other}'" errors

        match step.Context with
        | Some ctxList ->
            for ctx in ctxList do
                if priorSteps.Contains ctx.StepId |> not then
                    errors <- error $"Step '{step.Id}': Context references unknown step '{ctx.StepId}'" errors
        | None -> ()

        match step.Outputs with
        | Some outputs when outputs |> List.exists (System.String.IsNullOrWhiteSpace >> not) -> ()
        | Some _ -> errors <- error $"Step '{step.Id}': Outputs must contain at least one non-empty name" errors
        | None -> () // optional

        errors

    let validateWorkflow (workflow: Workflow) : Result<Workflow, string list> =
        let mutable errors = []

        if System.String.IsNullOrWhiteSpace workflow.Name then
            errors <- error "Workflow name is required" errors

        if System.String.IsNullOrWhiteSpace workflow.Version then
            errors <- error "Workflow version is required" errors

        if workflow.Steps.IsEmpty then
            errors <- error "Workflow must contain at least one step" errors

        errors <- errors @ (validateStepIds workflow.Steps)

        // validate steps with knowledge of prior steps for context references
        let mutable seen = Set.empty
        for step in workflow.Steps do
            errors <- errors @ (validateStep step seen)
            seen <- seen.Add step.Id

        if errors.IsEmpty then Ok workflow else Error(List.rev errors)
