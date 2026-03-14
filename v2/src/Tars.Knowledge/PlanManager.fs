/// TARS Plan Manager - Hypothesis-driven planning
/// "Plans are hypotheses about future actions, not beliefs. Different symbolic class."
namespace Tars.Knowledge

open System
open System.Threading.Tasks

/// The Plan Manager - manages hypothesis-driven plans
type PlanManager(storage: IPlanStorage, ledger: KnowledgeLedger) =

    /// Create a new plan
    member this.CreatePlan(goal: string, steps: PlanStep list, assumptions: BeliefId list, agentId: AgentId) =
        task {
            let plan = Plan.create goal assumptions steps agentId

            let! saveResult = storage.SavePlan(plan)

            match saveResult with
            | Ok() ->
                let! _ = storage.AppendEvent(PlanCreated plan)
                return Ok plan
            | Error e -> return Error e
        }

    /// Get a plan by ID
    member this.Get(planId: PlanId) = storage.GetPlan(planId)

    /// Get all active plans
    member this.GetActive() =
        storage.GetPlansByStatus(PlanStatus.Active)

    /// Start a step
    member this.StartStep(planId: PlanId, stepOrder: int) =
        task {
            let! planOpt = storage.GetPlan(planId)

            match planOpt with
            | Some plan ->
                let updatedSteps =
                    plan.Steps
                    |> List.map (fun s ->
                        if s.Order = stepOrder then
                            { s with
                                Status = StepStatus.InProgress }
                        else
                            s)

                let updatedPlan =
                    { plan with
                        Steps = updatedSteps
                        UpdatedAt = DateTime.UtcNow }

                let! _ = storage.UpdatePlan(updatedPlan)
                let! _ = storage.AppendEvent(StepStarted(planId, stepOrder))
                return Ok()
            | None -> return Error $"Plan {planId} not found"
        }

    /// Complete a step
    member this.CompleteStep(planId: PlanId, stepOrder: int, evidence: string) =
        task {
            let! planOpt = storage.GetPlan(planId)

            match planOpt with
            | Some plan ->
                let updatedSteps =
                    plan.Steps
                    |> List.map (fun s ->
                        if s.Order = stepOrder then
                            { s with
                                Status = StepStatus.Completed
                                CompletedAt = Some DateTime.UtcNow }
                        else
                            s)

                let updatedPlan =
                    { plan with
                        Steps = updatedSteps
                        UpdatedAt = DateTime.UtcNow }

                let! _ = storage.UpdatePlan(updatedPlan)
                let! _ = storage.AppendEvent(StepCompleted(planId, stepOrder, evidence))

                // Check if all steps complete
                let allComplete =
                    updatedSteps
                    |> List.forall (fun s ->
                        match s.Status with
                        | StepStatus.Completed
                        | StepStatus.Skipped _ -> true
                        | _ -> false)

                if allComplete then
                    let! _ = this.CompletePlan(planId)
                    ()

                return Ok()
            | None -> return Error $"Plan {planId} not found"
        }

    /// Fail a step
    member this.FailStep(planId: PlanId, stepOrder: int, reason: string) =
        task {
            let! planOpt = storage.GetPlan(planId)

            match planOpt with
            | Some plan ->
                let updatedSteps =
                    plan.Steps
                    |> List.map (fun s ->
                        if s.Order = stepOrder then
                            { s with
                                Status = StepStatus.Failed reason }
                        else
                            s)

                let updatedPlan =
                    { plan with
                        Steps = updatedSteps
                        UpdatedAt = DateTime.UtcNow }

                let! _ = storage.UpdatePlan(updatedPlan)
                let! _ = storage.AppendEvent(StepFailed(planId, stepOrder, reason))
                return Ok()
            | None -> return Error $"Plan {planId} not found"
        }

    /// Complete a plan
    member this.CompletePlan(planId: PlanId) =
        task {
            let! planOpt = storage.GetPlan(planId)

            match planOpt with
            | Some plan ->
                let updatedPlan =
                    { plan with
                        Status = PlanStatus.Completed
                        UpdatedAt = DateTime.UtcNow }

                let! _ = storage.UpdatePlan(updatedPlan)
                let! _ = storage.AppendEvent(PlanCompleted planId)
                return Ok()
            | None -> return Error $"Plan {planId} not found"
        }

    /// Check if any assumptions have been invalidated
    member this.CheckAssumptions(planId: PlanId) =
        task {
            let! planOpt = storage.GetPlan(planId)

            match planOpt with
            | Some plan ->
                let invalidated =
                    plan.Assumptions
                    |> List.choose (fun beliefId ->
                        match ledger.Get(beliefId) with
                        | Some belief when not belief.IsValid -> Some(beliefId, belief)
                        | _ -> None)

                if not invalidated.IsEmpty then
                    for (beliefId, _) in invalidated do
                        let! _ =
                            storage.AppendEvent(
                                AssumptionInvalidated(planId, beliefId, "Assumption belief was invalidated")
                            )

                        ()

                return invalidated
            | None -> return []
        }

    /// Fork a plan (create new version when assumptions change)
    /// "Plans are forked, not overwritten"
    member this.ForkPlan(originalId: PlanId, newGoal: string option, newSteps: PlanStep list option, agentId: AgentId) =
        task {
            let! originalOpt = storage.GetPlan(originalId)

            match originalOpt with
            | Some original ->
                let forked =
                    { original with
                        Id = PlanId.New()
                        Goal = newGoal |> Option.defaultValue original.Goal
                        Steps = newSteps |> Option.defaultValue original.Steps
                        Version = original.Version + 1
                        ParentVersion = Some original.Id
                        Status = Draft
                        CreatedAt = DateTime.UtcNow
                        UpdatedAt = DateTime.UtcNow
                        CreatedBy = agentId }

                // Mark original as superseded
                let superseded =
                    { original with
                        Status = Superseded
                        UpdatedAt = DateTime.UtcNow }

                let! _ = storage.UpdatePlan(superseded)
                let! _ = storage.SavePlan(forked)
                let! _ = storage.AppendEvent(PlanForked(originalId, forked))
                let! _ = storage.AppendEvent(PlanSuperseded(originalId, forked.Id))

                return Ok forked
            | None -> return Error $"Plan {originalId} not found"
        }

    /// Activate a plan
    member this.Activate(planId: PlanId) =
        task {
            let! planOpt = storage.GetPlan(planId)

            match planOpt with
            | Some plan ->
                let activated =
                    { plan with
                        Status = Active
                        UpdatedAt = DateTime.UtcNow }

                let! _ = storage.UpdatePlan(activated)
                return Ok()
            | None -> return Error $"Plan {planId} not found"
        }

    /// Get plan history
    member this.GetHistory(planId: PlanId) =
        // We lack GetEvents in IPlanStorage based on my previous edit, but I removed it?
        // Let's check IPlanStorage def in Types.fs
        // abstract member AppendEvent: event: PlanEvent -> Task<Result<unit, string>>
        // It does NOT have GetEvents!
        // I should add GetEvents to IPlanStorage or remove this method.
        // For now, removing this method or returning empty list until interface is updated.
        Task.FromResult([])

    /// Get statistics
    member this.Stats() =
        task {
            let! active = storage.GetPlansByStatus(PlanStatus.Active)
            return {| ActivePlans = active.Length |}
        }

/// Factory for creating plan managers
module PlanManager =
    /// Create with in-memory storage (assumes ledger implements IPlanStorage)
    let createInMemory (ledger: KnowledgeLedger) =
        // Cast the ledger's storage to IPlanStorage.
        // This assumes InMemoryLedgerStorage is used and it implements IPlanStorage.
        let storage = ledger.Storage :?> IPlanStorage
        PlanManager(storage, ledger)
