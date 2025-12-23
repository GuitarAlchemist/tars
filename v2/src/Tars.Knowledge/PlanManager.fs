/// TARS Plan Manager - Hypothesis-driven planning
/// "Plans are hypotheses about future actions, not beliefs. Different symbolic class."
namespace Tars.Knowledge

open System
open System.Collections.Generic

/// Interface for plan storage
type IPlanStorage =
    abstract member Save: Plan -> Async<Result<unit, string>>
    abstract member Get: PlanId -> Async<Plan option>
    abstract member GetActive: unit -> Async<Plan list>
    abstract member AppendEvent: PlanEvent -> Async<Result<unit, string>>
    abstract member GetEvents: PlanId -> Async<PlanEvent list>

/// In-memory plan storage (for development/testing)
type InMemoryPlanStorage() =
    let plans = Dictionary<PlanId, Plan>()
    let events = ResizeArray<PlanId * PlanEvent>()

    interface IPlanStorage with
        member _.Save(plan) =
            async {
                plans.[plan.Id] <- plan
                return Ok()
            }

        member _.Get(id) =
            async {
                match plans.TryGetValue(id) with
                | true, p -> return Some p
                | false, _ -> return None
            }

        member _.GetActive() =
            async { return plans.Values |> Seq.filter (fun p -> p.IsActive) |> Seq.toList }

        member _.AppendEvent(event) =
            async {
                let planId =
                    match event with
                    | PlanCreated p -> p.Id
                    | StepStarted(id, _) -> id
                    | StepCompleted(id, _, _) -> id
                    | StepFailed(id, _, _) -> id
                    | AssumptionInvalidated(id, _, _) -> id
                    | PlanForked(_, newPlan) -> newPlan.Id
                    | PlanCompleted id -> id
                    | PlanFailed(id, _) -> id
                    | PlanSuperseded(id, _) -> id

                events.Add((planId, event))
                return Ok()
            }

        member _.GetEvents(planId) =
            async { return events |> Seq.filter (fun (id, _) -> id = planId) |> Seq.map snd |> Seq.toList }

/// The Plan Manager - manages hypothesis-driven plans
type PlanManager(storage: IPlanStorage, ledger: KnowledgeLedger) =

    /// Create a new plan
    member this.CreatePlan(goal: string, steps: PlanStep list, assumptions: BeliefId list, agentId: AgentId) =
        async {
            let plan = Plan.create goal assumptions steps agentId

            let! saveResult = storage.Save(plan)

            match saveResult with
            | Ok() ->
                let! _ = storage.AppendEvent(PlanCreated plan)
                return Ok plan
            | Error e -> return Error e
        }

    /// Get a plan by ID
    member this.Get(planId: PlanId) = storage.Get(planId)

    /// Get all active plans
    member this.GetActive() = storage.GetActive()

    /// Start a step
    member this.StartStep(planId: PlanId, stepOrder: int) =
        async {
            let! planOpt = storage.Get(planId)

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

                let! _ = storage.Save(updatedPlan)
                let! _ = storage.AppendEvent(StepStarted(planId, stepOrder))
                return Ok()
            | None -> return Error $"Plan {planId} not found"
        }

    /// Complete a step
    member this.CompleteStep(planId: PlanId, stepOrder: int, evidence: string) =
        async {
            let! planOpt = storage.Get(planId)

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

                let! _ = storage.Save(updatedPlan)
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
        async {
            let! planOpt = storage.Get(planId)

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

                let! _ = storage.Save(updatedPlan)
                let! _ = storage.AppendEvent(StepFailed(planId, stepOrder, reason))
                return Ok()
            | None -> return Error $"Plan {planId} not found"
        }

    /// Complete a plan
    member this.CompletePlan(planId: PlanId) =
        async {
            let! planOpt = storage.Get(planId)

            match planOpt with
            | Some plan ->
                let updatedPlan =
                    { plan with
                        Status = PlanStatus.Completed
                        UpdatedAt = DateTime.UtcNow }

                let! _ = storage.Save(updatedPlan)
                let! _ = storage.AppendEvent(PlanCompleted planId)
                return Ok()
            | None -> return Error $"Plan {planId} not found"
        }

    /// Check if any assumptions have been invalidated
    member this.CheckAssumptions(planId: PlanId) =
        async {
            let! planOpt = storage.Get(planId)

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
        async {
            let! originalOpt = storage.Get(originalId)

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

                let! _ = storage.Save(superseded)
                let! _ = storage.Save(forked)
                let! _ = storage.AppendEvent(PlanForked(originalId, forked))
                let! _ = storage.AppendEvent(PlanSuperseded(originalId, forked.Id))

                return Ok forked
            | None -> return Error $"Plan {originalId} not found"
        }

    /// Activate a plan
    member this.Activate(planId: PlanId) =
        async {
            let! planOpt = storage.Get(planId)

            match planOpt with
            | Some plan ->
                let activated =
                    { plan with
                        Status = Active
                        UpdatedAt = DateTime.UtcNow }

                let! _ = storage.Save(activated)
                return Ok()
            | None -> return Error $"Plan {planId} not found"
        }

    /// Get plan history
    member this.GetHistory(planId: PlanId) = storage.GetEvents(planId)

    /// Get statistics
    member this.Stats() =
        async {
            let! active = storage.GetActive()
            return {| ActivePlans = active.Length |}
        }

/// Factory for creating plan managers
module PlanManager =
    /// Create with in-memory storage
    let createInMemory (ledger: KnowledgeLedger) =
        PlanManager(InMemoryPlanStorage(), ledger)
