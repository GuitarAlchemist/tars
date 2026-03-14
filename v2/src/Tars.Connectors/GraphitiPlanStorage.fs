/// Graphiti-based Plan Storage for Temporal Knowledge Graph
/// "Track when plans existed, when they changed, who invalidated them"
namespace Tars.Connectors

open System
open System.Text.Json
open Tars.Knowledge
open Tars.Connectors.Graphiti

/// Graphiti implementation of IPlanStorage
/// Stores plans as temporal entities in the knowledge graph
type GraphitiPlanStorage(graphitiUrl: string, ?groupId: string) =
    let client = new GraphitiClient(Uri(graphitiUrl))
    let gid = defaultArg groupId "tars_plans"
    let jsonOptions = JsonSerializerOptions(WriteIndented = false)

    /// Convert plan to Graphiti message content
    let planToMessage (plan: Plan) : MessageDto =
        let stepsDesc =
            plan.Steps
            |> List.map (fun s -> $"  {s.Order}. {s.Description} [{s.Status}]")
            |> String.concat "\n"

        let assumptionsDesc =
            if plan.Assumptions.IsEmpty then
                "None"
            else
                plan.Assumptions |> List.map (fun a -> a.Value.ToString()) |> String.concat ", "

        let metricsText = String.concat "\n" plan.SuccessMetrics
        let risksText = String.concat "\n" plan.RiskFactors
        let createdDate = plan.CreatedAt.ToString("yyyy-MM-dd HH:mm:ss")
        let updatedDate = plan.UpdatedAt.ToString("yyyy-MM-dd HH:mm:ss")

        let content =
            $"""PLAN: {plan.Goal}
Status: {plan.Status}
Version: {plan.Version}
Created: {createdDate} by {plan.CreatedBy.Value}
Updated: {updatedDate}

ASSUMPTIONS (BeliefIds):
{assumptionsDesc}

STEPS:
{stepsDesc}

SUCCESS METRICS:
{metricsText}

RISK FACTORS:
{risksText}
"""

        { Content = content
          RoleType = "system"
          Role = "plan_manager"
          Timestamp = Some plan.CreatedAt
          SourceDescription = Some $"Plan v{plan.Version}"
          Uuid = Some(plan.Id.Value.ToString()) }

    /// Convert plan event to Graphiti message
    let eventToMessage (event: PlanEvent) : MessageDto =
        let planId, eventType, content =
            match event with
            | PlanCreated p -> (p.Id, "PlanCreated", $"Created plan: {p.Goal}")
            | StepStarted(id, step) -> (id, "StepStarted", $"Started step {step}")
            | StepCompleted(id, step, evidence) -> (id, "StepCompleted", $"Completed step {step}: {evidence}")
            | StepFailed(id, step, reason) -> (id, "StepFailed", $"Failed step {step}: {reason}")
            | AssumptionInvalidated(id, beliefId, reason) ->
                (id, "AssumptionInvalidated", $"Assumption {beliefId} invalidated: {reason}")
            | PlanForked(original, newPlan) -> (newPlan.Id, "PlanForked", $"Forked from {original} to {newPlan.Id}")
            | PlanCompleted id -> (id, "PlanCompleted", $"Plan completed")
            | PlanFailed(id, reason) -> (id, "PlanFailed", $"Plan failed: {reason}")
            | PlanSuperseded(id, by) -> (id, "PlanSuperseded", $"Superseded by {by}")

        { Content = content
          RoleType = "event"
          Role = "plan_event"
          Timestamp = Some DateTime.UtcNow
          SourceDescription = Some eventType
          Uuid = None }

    interface IPlanStorage with
        member _.SavePlan(plan) =
            task {
                try
                    let message = planToMessage plan
                    let! result = client.AddMessagesAsync(gid, [| message |])

                    match result with
                    | FSharp.Core.Result.Ok _ -> return FSharp.Core.Result.Ok()
                    | FSharp.Core.Result.Error e -> return FSharp.Core.Result.Error $"Graphiti SavePlan failed: {e}"
                with ex ->
                    return FSharp.Core.Result.Error $"Graphiti SavePlan exception: {ex.Message}"
            }

        member _.UpdatePlan(plan) =
            task {
                // In Graphiti, updates are new temporal states
                // We add a new message with the updated plan state
                try
                    let message = planToMessage plan
                    let! result = client.AddMessagesAsync(gid, [| message |])

                    match result with
                    | FSharp.Core.Result.Ok _ -> return FSharp.Core.Result.Ok()
                    | FSharp.Core.Result.Error e -> return FSharp.Core.Result.Error $"Graphiti UpdatePlan failed: {e}"
                with ex ->
                    return FSharp.Core.Result.Error $"Graphiti UpdatePlan exception: {ex.Message}"
            }

        member _.GetPlan(planId) =
            task {
                // Graphiti doesn't have direct UUID lookup
                // This would require querying by UUID in content
                // For now, return None (Graphiti is write-only for plans)
                // In production, we'd use hybrid storage with PostgreSQL as primary
                return None
            }

        member _.GetPlansByStatus(status) =
            task {
                // Graphiti is optimized for semantic search, not status filtering
                // Return empty list (use PostgreSQL for queries)
                return []
            }

        member _.AppendEvent(event) =
            task {
                try
                    let message = eventToMessage event
                    let! result = client.AddMessagesAsync(gid, [| message |])

                    match result with
                    | FSharp.Core.Result.Ok _ -> return FSharp.Core.Result.Ok()
                    | FSharp.Core.Result.Error e -> return FSharp.Core.Result.Error $"Graphiti AppendEvent failed: {e}"
                with ex ->
                    return FSharp.Core.Result.Error $"Graphiti AppendEvent exception: {ex.Message}"
            }

/// Module for creating Graphiti plan storage
module GraphitiPlanStorage =
    let create (url: string) = GraphitiPlanStorage(url)

    let createWithGroup (url: string) (groupId: string) = GraphitiPlanStorage(url, groupId)
