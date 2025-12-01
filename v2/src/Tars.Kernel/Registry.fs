namespace Tars.Kernel

open System
open System.Collections.Concurrent
open Tars.Core

/// Thread-safe implementation of IAgentRegistry
type AgentRegistry() =
    let agents = ConcurrentDictionary<AgentId, Agent>()
    
    /// Register or update an agent in the registry
    member this.Register(agent: Agent) =
        agents.AddOrUpdate(agent.Id, agent, (fun _ _ -> agent)) |> ignore
        
    interface IAgentRegistry with
        member _.GetAgent(id) = 
            async {
                match agents.TryGetValue(id) with
                | true, agent -> return Some agent
                | false, _ -> return None
            }
            
        member _.FindAgents(capability) =
            async {
                return 
                    agents.Values 
                    |> Seq.filter (fun a -> a.Capabilities |> List.exists (fun c -> c.Kind = capability))
                    |> Seq.toList
            }
            
        member _.GetAllAgents() =
            async {
                return agents.Values |> Seq.toList
            }
