namespace Tars.Core

open System

/// The immutable context of the TARS Kernel
type KernelContext =
    {
        Agents: Map<AgentId, Agent>
        /// Map<Name, Map<Version, AgentId>>
        Registry: Map<string, Map<string, AgentId>>
        /// Active routing strategies for logical names
        Routes: Map<string, RoutingStrategy>
    }

module Kernel =
    /// Initialize a new empty kernel
    let init () =
        { Agents = Map.empty
          Registry = Map.empty
          Routes = Map.empty }

    /// Register a new agent in the kernel
    let registerAgent (agent: Agent) (ctx: KernelContext) =
        let versionMap =
            match ctx.Registry.TryFind agent.Name with
            | Some m -> m
            | None -> Map.empty

        let newRegistry =
            ctx.Registry.Add(agent.Name, versionMap.Add(agent.Version, agent.Id))

        { ctx with
            Agents = ctx.Agents.Add(agent.Id, agent)
            Registry = newRegistry }

    /// Create a new agent with default state
    let createAgent (id: Guid) name version model systemPrompt tools =
        { Id = AgentId id
          Name = name
          Version = version
          ParentVersion = None
          CreatedAt = DateTime.UtcNow
          Model = model
          SystemPrompt = systemPrompt
          Tools = tools
          State = Idle
          Memory = [] }

    /// Retrieve an agent by ID
    let getAgent id ctx = ctx.Agents.TryFind id

    /// Retrieve a specific version of an agent by name
    let getAgentVersion name version ctx =
        match ctx.Registry.TryFind name with
        | Some versions ->
            match versions.TryFind version with
            | Some id -> getAgent id ctx
            | None -> None
        | None -> None

    /// Update an agent's state
    let updateAgent (agent: Agent) (ctx: KernelContext) =
        if ctx.Agents.ContainsKey agent.Id then
            { ctx with
                Agents = ctx.Agents.Add(agent.Id, agent) }
        else
            ctx // Or return error? For now, idempotent.

    /// Set a routing strategy for a logical name
    let setRoute name strategy ctx =
        { ctx with
            Routes = ctx.Routes.Add(name, strategy) }

    /// Add a message to an agent's memory
    let receiveMessage (msg: Message) (agent: Agent) =
        { agent with
            Memory = agent.Memory @ [ msg ] }
