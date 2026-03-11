namespace Tars.Core

// =============================================================================
// PHASE 14.2: RUNTIME CONTRACT ENFORCEMENT
// =============================================================================
//
// Implements the validation logic for agent constitutions.
// Ensures agents adhere to their symbolic contracts.
// Reference: docs/3_Roadmap/2_Phases/phase_14_agent_constitutions.md

module ContractEnforcement =

    /// Validates an intended action against the agent's constitution
    let validateAction (constitution: AgentConstitution) (action: AgentAction) : Result<unit, Violation> =
        // 1. Check Prohibitions (Negative Constraints - Highest Priority)
        let prohibitionViolation =
            constitution.Prohibitions
            |> List.tryPick (fun rule ->
                let violates =
                    match rule, action with
                    | CannotModifyCore, WriteFile path -> path.Replace("\\", "/").Contains("src/Tars.Core")
                    | CannotDeleteData, ExecuteTool(name, _) ->
                        name.ToLowerInvariant().Contains("delete")
                        || name.ToLowerInvariant().Contains("remove")
                    | CannotDeleteData, GenericAction(name, _) when name.Contains("delete") -> true
                    | CannotAccessNetwork, NetworkRequest _ -> true
                    | CannotUseTool forbidden, ExecuteTool(name, _) -> name = forbidden
                    | CannotAccessPath forbidden, ReadFile path -> path.StartsWith(forbidden)
                    | CannotAccessPath forbidden, WriteFile path -> path.StartsWith(forbidden)
                    | _ -> false

                if violates then
                    Some(ProhibitionViolated(rule, sprintf "Action %A violates prohibition %A" action rule))
                else
                    None)

        match prohibitionViolation with
        | Some violation -> FSharp.Core.Error violation
        | None ->
            // 2. Check Permissions (Positive Constraints)
            // Policy:
            // - If Permissions list is empty -> Permissive Mode (Allow everything not prohibited)
            // - If Permissions list is present -> Restrictive Mode (Allow only what is explicitly permitted)
            if constitution.Permissions.IsEmpty then
                FSharp.Core.Ok()
            else
                let isPermitted =
                    constitution.Permissions
                    |> List.exists (fun perm ->
                        match perm, action with
                        | All, _ -> true
                        | ReadCode pattern, ReadFile path ->
                            // Simple substring match for now, could be regex
                            path.Contains(pattern) || pattern = "*"
                        | ModifyCode pattern, WriteFile path -> path.Contains(pattern) || pattern = "*"
                        | CallTool toolName, ExecuteTool(target, _) -> toolName = target || toolName = "*"
                        | SpawnAgent role, SpawnChild _ ->
                            // TODO: Check role match
                            true
                        | ExecuteShellCommand pattern, ExecuteTool(name, args) when name = "run_command" ->
                            // Check args against pattern roughly
                            args.Contains(pattern) || pattern = "*"
                        | _ -> false)

                if isPermitted then
                    FSharp.Core.Ok()
                else
                    FSharp.Core.Error(PermissionDenied(action, "Action not explicitly permitted by constitution"))

    /// Checks if a resource usage update violates quotas
    let checkResources
        (constitution: AgentConstitution)
        (resource: string)
        (currentValue: decimal)
        : Result<unit, Violation> =
        let bound =
            constitution.SymbolicContract.ResourceBounds @ constitution.HardResourceBounds
            |> List.tryPick (fun limit ->
                match limit, resource with
                | MaxTokens max, "tokens" when currentValue > decimal max -> Some limit
                | MaxCost max, "cost" when currentValue > max -> Some limit
                // Add other mappings
                | _ -> None)

        match bound with
        | Some limit -> FSharp.Core.Error(ResourceQuotaExceeded(limit, currentValue, limit))
        | None -> FSharp.Core.Ok()

    /// Checks dependencies before agent start/spawn
    let checkDependencies (constitution: AgentConstitution) (activeAgents: AgentId list) : Result<unit, Violation> =
        let missing =
            constitution.SymbolicContract.Dependencies
            |> List.tryFind (fun dep -> not (List.contains dep activeAgents))

        match missing with
        | Some agentId -> FSharp.Core.Error(DependencyMissing agentId)
        | None -> FSharp.Core.Ok()
