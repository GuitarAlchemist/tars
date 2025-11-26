namespace Tars.Core

open System

/// The immutable context of the TARS Kernel
type KernelContext = { Agents: Map<AgentId, Agent> }

module Kernel =
    /// Initialize a new empty kernel
    let init () = { Agents = Map.empty }

    /// Register a new agent in the kernel
    let registerAgent (agent: Agent) (ctx: KernelContext) =
        { ctx with
            Agents = ctx.Agents.Add(agent.Id, agent) }

    /// Create a new agent with default state
    let createAgent (id: Guid) name model systemPrompt tools =
        { Id = AgentId id
          Name = name
          Model = model
          SystemPrompt = systemPrompt
          Tools = tools
          State = Idle
          Memory = [] }

    /// Retrieve an agent by ID
    let getAgent id ctx = ctx.Agents.TryFind id

    /// Update an agent's state
    let updateAgent (agent: Agent) (ctx: KernelContext) =
        if ctx.Agents.ContainsKey agent.Id then
            { ctx with
                Agents = ctx.Agents.Add(agent.Id, agent) }
        else
            ctx // Or return error? For now, idempotent.

    /// Add a message to an agent's memory
    let receiveMessage (msg: Message) (agent: Agent) =
        { agent with
            Memory = agent.Memory @ [ msg ] }
