namespace Tars.Cortex

/// Multi-agent orchestrator for composing TARS agents.
///
/// Supports:
///   - Agent registration with capability descriptions
///   - Goal-based routing: picks the best agent for a task
///   - Sequential pipelines: chain agents where each feeds the next
///   - Parallel fan-out: run multiple agents and merge results

open System
open System.Collections.Generic
open System.Threading
open System.Threading.Tasks
open Microsoft.Agents.AI
open Microsoft.Extensions.AI

// ─────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────

/// A registered agent with its routing metadata.
type AgentRegistration = {
    Agent: AIAgent
    Capabilities: string list
    Priority: int
}

/// Result from an orchestrated multi-agent run.
type OrchestrationResult = {
    AgentId: string
    AgentName: string
    Response: string
    DurationMs: int64
    Success: bool
}

// ─────────────────────────────────────────────────────────────────────
// Orchestrator
// ─────────────────────────────────────────────────────────────────────

/// Routes tasks to registered agents based on capability matching.
type AgentOrchestrator() =
    let agents = Dictionary<string, AgentRegistration>()

    /// Register an agent with its capabilities.
    member _.Register(agent: AIAgent, capabilities: string list, ?priority: int) =
        let reg = {
            Agent = agent
            Capabilities = capabilities |> List.map (fun s -> s.ToLowerInvariant())
            Priority = defaultArg priority 0
        }
        agents.[agent.Name] <- reg

    /// Find the best agent for a goal based on keyword matching.
    member _.Route(goal: string) : AgentRegistration option =
        let goalLower = goal.ToLowerInvariant()
        let scored =
            agents.Values
            |> Seq.map (fun reg ->
                let matchCount =
                    reg.Capabilities
                    |> List.sumBy (fun cap ->
                        if goalLower.Contains(cap) then 1 else 0)
                let score = matchCount * 10 + reg.Priority
                (reg, score))
            |> Seq.filter (fun (_, score) -> score > 0)
            |> Seq.sortByDescending snd
            |> Seq.toList
        match scored with
        | (reg, _) :: _ -> Some reg
        | [] ->
            // Fall back to highest-priority agent
            agents.Values
            |> Seq.sortByDescending (fun r -> r.Priority)
            |> Seq.tryHead

    /// Run a single agent on a goal.
    member private _.RunAgent
        (agent: AIAgent, goal: string, ct: CancellationToken)
        : Task<OrchestrationResult> =
        task {
            let sw = System.Diagnostics.Stopwatch.StartNew()
            try
                let messages = [ ChatMessage(ChatRole.User, goal) ] |> Seq.ofList
                let! response = agent.RunAsync(messages, cancellationToken = ct)
                sw.Stop()
                let lastMsg = response.Messages |> Seq.tryLast
                let text =
                    lastMsg
                    |> Option.bind (fun m -> m.Text |> Option.ofObj)
                    |> Option.defaultValue ""
                return {
                    AgentId = agent.Id
                    AgentName = agent.Name
                    Response = text
                    DurationMs = sw.ElapsedMilliseconds
                    Success = true
                }
            with ex ->
                sw.Stop()
                return {
                    AgentId = agent.Id
                    AgentName = agent.Name
                    Response = $"Error: {ex.Message}"
                    DurationMs = sw.ElapsedMilliseconds
                    Success = false
                }
        }

    /// Route a goal to the best agent and execute it.
    member this.Execute(goal: string, ?ct: CancellationToken) : Task<OrchestrationResult> =
        let ct = defaultArg ct CancellationToken.None
        task {
            match this.Route(goal) with
            | Some reg ->
                return! this.RunAgent(reg.Agent, goal, ct)
            | None ->
                return {
                    AgentId = ""
                    AgentName = "none"
                    Response = "No agent registered that can handle this goal."
                    DurationMs = 0L
                    Success = false
                }
        }

    /// Run a sequential pipeline: each agent's output becomes the next agent's input.
    member this.Pipeline
        (agentNames: string list, initialGoal: string, ?ct: CancellationToken)
        : Task<OrchestrationResult list> =
        let ct = defaultArg ct CancellationToken.None
        task {
            let mutable results = []
            let mutable currentInput = initialGoal
            for name in agentNames do
                match agents.TryGetValue(name) with
                | true, reg ->
                    let! result = this.RunAgent(reg.Agent, currentInput, ct)
                    results <- results @ [ result ]
                    if result.Success then
                        currentInput <- result.Response
                | false, _ ->
                    results <- results @ [{
                        AgentId = ""; AgentName = name
                        Response = $"Agent '{name}' not found."
                        DurationMs = 0L; Success = false
                    }]
            return results
        }

    /// Run multiple agents in parallel on the same goal (fan-out).
    member this.FanOut
        (agentNames: string list, goal: string, ?ct: CancellationToken)
        : Task<OrchestrationResult list> =
        let ct = defaultArg ct CancellationToken.None
        task {
            let tasks =
                agentNames
                |> List.choose (fun name ->
                    match agents.TryGetValue(name) with
                    | true, reg -> Some (this.RunAgent(reg.Agent, goal, ct))
                    | false, _ -> None)
            let! results = Task.WhenAll(tasks)
            return results |> Array.toList
        }

    /// Get all registered agent names.
    member _.GetRegisteredAgents() : string list =
        agents.Keys |> Seq.toList
