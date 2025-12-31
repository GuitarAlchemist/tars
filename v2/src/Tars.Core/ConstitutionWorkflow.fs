namespace Tars.Core

open System
open Tars.Core.Metrics

// =============================================================================
// PHASE 14.3: CONSTITUTION-AWARE WORKFLOW INTEGRATION
// =============================================================================
//
// Provides workflow helpers that integrate constitution enforcement into
// the agent computation expression. Enables declarative governance.
// Reference: docs/3_Roadmap/2_Phases/phase_14_agent_constitutions.md

module ConstitutionWorkflow =

    /// Resource usage tracker for a single agent run
    type ResourceUsage =
        { TokensUsed: int
          CostIncurred: decimal
          ToolCalls: int
          FilesModified: int
          StartTime: DateTimeOffset }

    module ResourceUsage =
        let empty =
            { TokensUsed = 0
              CostIncurred = 0m
              ToolCalls = 0
              FilesModified = 0
              StartTime = DateTimeOffset.UtcNow }

        let addTokens n usage =
            { usage with
                TokensUsed = usage.TokensUsed + n }

        let addCost c usage =
            { usage with
                CostIncurred = usage.CostIncurred + c }

        let incToolCalls usage =
            { usage with
                ToolCalls = usage.ToolCalls + 1 }

        let incFilesModified usage =
            { usage with
                FilesModified = usage.FilesModified + 1 }

        /// Elapsed time since start
        let elapsed usage = DateTimeOffset.UtcNow - usage.StartTime

    /// Mutable tracker for runtime resource accumulation
    type ResourceTracker(constitution: AgentConstitution) =
        let mutable usage = ResourceUsage.empty

        member _.Constitution = constitution
        member _.Usage = usage

        member _.RecordTokens(count: int) =
            usage <- ResourceUsage.addTokens count usage
            Metrics.recordSimple "constitution.tokens" "recorded" None (Some(float count)) None

        member _.RecordCost(cost: decimal) =
            usage <- ResourceUsage.addCost cost usage

        member _.RecordToolCall() =
            usage <- ResourceUsage.incToolCalls usage

        member _.RecordFileModification() =
            usage <- ResourceUsage.incFilesModified usage

        /// Check if any resource limits are exceeded
        member this.CheckLimits() : Result<unit, Violation> =
            let c = constitution
            let limits = c.SymbolicContract.ResourceBounds @ c.HardResourceBounds

            let violation =
                limits
                |> List.tryPick (fun limit ->
                    match limit with
                    | MaxTokens max when usage.TokensUsed > max ->
                        Some(ResourceQuotaExceeded(limit, box usage.TokensUsed, box max))
                    | MaxCost max when usage.CostIncurred > max ->
                        Some(ResourceQuotaExceeded(limit, box usage.CostIncurred, box max))
                    | MaxTimeMinutes max when (ResourceUsage.elapsed usage).TotalMinutes > float max ->
                        Some(
                            TimeConstraintViolated(
                                MustCompleteWithin(TimeSpan.FromMinutes(float max)),
                                ResourceUsage.elapsed usage
                            )
                        )
                    | _ -> None)

            match violation with
            | Some v -> FSharp.Core.Error v
            | None -> FSharp.Core.Ok()

    // =========================================================================
    // Workflow Helpers
    // =========================================================================

    /// Validate an action before execution within a workflow
    let validateAction (action: AgentAction) : AgentWorkflow<unit> =
        fun ctx ->
            async {
                let constitution = ctx.Self.Constitution

                match ContractEnforcement.validateAction constitution action with
                | FSharp.Core.Ok() ->
                    Metrics.recordSimple "constitution.validate" "ok" (Some ctx.Self.Id) None None
                    return ExecutionOutcome.Success()
                | FSharp.Core.Error violation ->
                    Metrics.recordSimple "constitution.validate" "violation" (Some ctx.Self.Id) None None
                    let msg = sprintf "Constitutional violation: %A" violation
                    ctx.Logger msg
                    return ExecutionOutcome.Failure [ PartialFailure.Error msg ]
            }

    /// Wrap a file write action with constitutional validation
    let guardedWriteFile (path: string) (write: unit -> Async<Result<'T, string>>) : AgentWorkflow<'T> =
        fun ctx ->
            async {
                let action = AgentAction.WriteFile path

                match ContractEnforcement.validateAction ctx.Self.Constitution action with
                | FSharp.Core.Error violation ->
                    return ExecutionOutcome.Failure [ PartialFailure.Error(sprintf "Write blocked: %A" violation) ]
                | FSharp.Core.Ok() ->
                    let! result = write ()

                    match result with
                    | Result.Ok v -> return ExecutionOutcome.Success v
                    | Result.Error e -> return ExecutionOutcome.Failure [ PartialFailure.Error e ]
            }

    /// Wrap a tool call with constitutional validation
    let guardedToolCall
        (toolName: string)
        (args: string)
        (execute: unit -> Async<Result<'T, string>>)
        : AgentWorkflow<'T> =
        fun ctx ->
            async {
                let action = AgentAction.ExecuteTool(toolName, args)

                match ContractEnforcement.validateAction ctx.Self.Constitution action with
                | FSharp.Core.Error violation ->
                    return ExecutionOutcome.Failure [ PartialFailure.Error(sprintf "Tool call blocked: %A" violation) ]
                | FSharp.Core.Ok() ->
                    let! result = execute ()

                    match result with
                    | Result.Ok v -> return ExecutionOutcome.Success v
                    | Result.Error e -> return ExecutionOutcome.Failure [ PartialFailure.Error e ]
            }

    /// Check if spawning a child agent is permitted
    let canSpawnChild (role: NeuralRole) : AgentWorkflow<bool> =
        fun ctx ->
            async {
                let action = AgentAction.SpawnChild role

                match ContractEnforcement.validateAction ctx.Self.Constitution action with
                | FSharp.Core.Ok() -> return ExecutionOutcome.Success true
                | FSharp.Core.Error _ -> return ExecutionOutcome.Success false
            }

    /// Check that all dependencies are active before proceeding
    let checkDependencies (activeAgents: AgentId list) : AgentWorkflow<unit> =
        fun ctx ->
            async {
                match ContractEnforcement.checkDependencies ctx.Self.Constitution activeAgents with
                | FSharp.Core.Ok() -> return ExecutionOutcome.Success()
                | FSharp.Core.Error violation ->
                    return ExecutionOutcome.Failure [ PartialFailure.Error(sprintf "Dependency missing: %A" violation) ]
            }

    // =========================================================================
    // Spawn-time Validation
    // =========================================================================

    /// Validates an agent's constitution at spawn time
    /// Returns Ok if the constitution is valid and all dependencies are met
    let validateAtSpawn (agent: Agent) (registry: IAgentRegistry) : Async<Result<unit, string>> =
        async {
            let c = agent.Constitution

            // 1. Check dependencies
            let! allAgents = registry.GetAllAgents()
            let activeIds = allAgents |> List.map (fun a -> a.Id)

            match ContractEnforcement.checkDependencies c activeIds with
            | FSharp.Core.Error(DependencyMissing(AgentId id)) ->
                return Result.Error(sprintf "Cannot spawn: dependency agent %A not active" id)
            | FSharp.Core.Error v -> return Result.Error(sprintf "Cannot spawn: %A" v)
            | FSharp.Core.Ok() ->

                // 2. Check for conflicting agents
                let conflicts =
                    c.SymbolicContract.ConflictsWith
                    |> List.filter (fun id -> List.contains id activeIds)

                if not conflicts.IsEmpty then
                    return Result.Error(sprintf "Cannot spawn: conflicts with active agents %A" conflicts)
                else
                    return Result.Ok()
        }

    // =========================================================================
    // Constitution Amendment Protocol
    // =========================================================================

    /// Amendment request for changing a constitution
    type AmendmentRequest =
        { AgentId: AgentId
          ProposedChanges: AgentConstitution
          Reason: string
          RequestedBy: string
          RequestedAt: DateTimeOffset }

    /// Amendment decision
    type AmendmentDecision =
        | Approved of newConstitution: AgentConstitution
        | Rejected of reason: string
        | RequiresHumanReview of reason: string

    /// Evaluates whether a constitutional amendment should be allowed
    /// Core modifications require explicit human approval
    let evaluateAmendment (request: AmendmentRequest) : AmendmentDecision =
        let proposed = request.ProposedChanges

        // Check if this removes core protections
        let removesCoreSafety =
            proposed.Prohibitions |> List.exists (fun p -> p = CannotModifyCore) |> not

        if removesCoreSafety then
            RequiresHumanReview "Amendment would remove core modification protection"
        else
            // Check if adding dangerous permissions
            let grantsDangerousPerms =
                proposed.Permissions
                |> List.exists (fun p ->
                    match p with
                    | All -> true // God mode
                    | ExecuteShellCommand "*" -> true // Unrestricted shell
                    | _ -> false)

            if grantsDangerousPerms then
                RequiresHumanReview "Amendment grants dangerous permissions (All or unrestricted shell)"
            else
                Approved proposed

    // =========================================================================
    // Decorators for Constitutional Enforcement
    // =========================================================================

    /// Wraps a workflow with pre-action validation
    let withConstitutionCheck (action: AgentAction) (workflow: AgentWorkflow<'T>) : AgentWorkflow<'T> =
        fun ctx ->
            async {
                match ContractEnforcement.validateAction ctx.Self.Constitution action with
                | FSharp.Core.Error violation ->
                    return
                        ExecutionOutcome.Failure
                            [ PartialFailure.Error(sprintf "Blocked by constitution: %A" violation) ]
                | FSharp.Core.Ok() -> return! workflow ctx
            }

    /// Wraps a workflow with resource limit checking after each step
    let withResourceTracking (tracker: ResourceTracker) (workflow: AgentWorkflow<'T>) : AgentWorkflow<'T> =
        fun ctx ->
            async {
                let! result = workflow ctx

                // Check limits after execution
                match tracker.CheckLimits() with
                | FSharp.Core.Error violation ->
                    // Convert to warning rather than failure (already executed)
                    match result with
                    | ExecutionOutcome.Success v ->
                        return
                            ExecutionOutcome.PartialSuccess(
                                v,
                                [ PartialFailure.Warning(sprintf "Resource limit exceeded: %A" violation) ]
                            )
                    | ExecutionOutcome.PartialSuccess(v, w) ->
                        return
                            ExecutionOutcome.PartialSuccess(
                                v,
                                w @ [ PartialFailure.Warning(sprintf "Resource limit: %A" violation) ]
                            )
                    | ExecutionOutcome.Failure e -> return ExecutionOutcome.Failure e
                | FSharp.Core.Ok() -> return result
            }
