namespace TarsEngine.FSharp.SelfImprovement

open System
open PersistentAdaptiveMemory

/// Derives per-role directives and policy hints from stored feedback aggregates.
module AgentPolicyTuning =

    type RoleDirective =
        { Role: string
          Summary: AgentFeedbackAggregate
          PromptDirectives: string list
          PolicyHints: string list }

    let private confidenceDescriptor (value: float option) =
        match value with
        | None -> None
        | Some v when v >= 0.85 -> Some "High confidence feedback observed."
        | Some v when v <= 0.35 -> Some "Feedback carries low confidence; corroborate with additional signals."
        | Some v -> Some $"Average confidence %.2f{v} across samples."

    let private buildDirectives (aggregate: AgentFeedbackAggregate) =
        let prompts = ResizeArray<string>()
        let policyHints = ResizeArray<string>()

        if aggregate.escalate > 0 then
            prompts.Add("Escalations detected; request explicit sign-off before closing tasks.")
            policyHints.Add("Force consensus approval and enable critic gate when escalations appear.")

        if aggregate.reject > aggregate.approve then
            prompts.Add("Rejections outweigh approvals; prioritize remediation guidance in upcoming proposals.")
            policyHints.Add("Favor conservative policy genome (consensus required, critic approval enabled).")

        if aggregate.needsWork > 0 && aggregate.approve = 0 then
            prompts.Add("Deliver step-by-step improvement plan addressing outstanding issues.")

        match aggregate.role.ToLowerInvariant() with
        | role when role.Contains("implementer") ->
            if aggregate.needsWork > 0 then
                policyHints.Add("Allow iterative retries without halting the entire workflow.")
        | role when role.Contains("auditor") || role.Contains("reviewer") ->
            if aggregate.escalate + aggregate.reject > 0 then
                policyHints.Add("Ensure reviewer verdicts can gate execution until addressed.")
        | _ -> ()

        confidenceDescriptor aggregate.averageConfidence
        |> Option.iter prompts.Add

        { Role = aggregate.role
          Summary = aggregate
          PromptDirectives = prompts |> List.ofSeq
          PolicyHints = policyHints |> List.ofSeq }

    /// Builds role directives from feedback aggregates.
    let deriveRoleDirectives (aggregates: AgentFeedbackAggregate list) =
        aggregates |> List.map buildDirectives


