namespace TarsEngine.DSL

open Ast

/// Module containing interfaces for the agent framework
module AgentInterfaces =
    /// Result of executing an agent function
    type AgentResult =
        | Success of PropertyValue
        | Error of string

    /// Interface for agent registry
    type IAgentRegistry =
        /// Register an agent
        abstract member RegisterAgent: agent: obj -> unit
        
        /// Execute an agent task
        abstract member ExecuteTask: agentName: string * taskName: string * functionName: string option * parameters: Map<string, PropertyValue> * env: Map<string, PropertyValue> -> AgentResult
