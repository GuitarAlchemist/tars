namespace TarsEngine.DSL

open System
open System.Collections.Generic
open Ast

/// Module containing the agent execution framework for the TARS DSL
module AgentFramework =
    /// Agent capability
    type Capability = string

    /// Result of executing an agent function
    type AgentResult =
        | Success of PropertyValue
        | Error of string

    /// Agent function
    type AgentFunction = {
        Name: string
        Parameters: string list
        Block: TarsBlock
    }

    /// Agent task
    type AgentTask = {
        Name: string
        Description: string
        Functions: Map<string, AgentFunction>
    }

    /// Agent definition
    type Agent = {
        Name: string
        Description: string
        Capabilities: Capability list
        Tasks: Map<string, AgentTask>
    }

    // Using AgentResult defined above

    /// Agent registry
    type AgentRegistry() =
        let agents = Dictionary<string, Agent>()

        /// Register an agent
        member this.RegisterAgent(agent: Agent) =
            agents.[agent.Name] <- agent

        /// Get an agent by name
        member this.GetAgent(name: string) =
            match agents.TryGetValue(name) with
            | true, agent -> Some agent
            | _ -> None

        /// Check if an agent has a capability
        member this.HasCapability(agentName: string, capability: string) =
            match this.GetAgent(agentName) with
            | Some agent -> List.contains capability agent.Capabilities
            | None -> false

        /// Execute an agent task
        member this.ExecuteTask(agentName: string, taskName: string, functionName: string option, parameters: Map<string, PropertyValue>, env: Map<string, PropertyValue>) =
            match this.GetAgent(agentName) with
            | Some agent ->
                match agent.Tasks.TryFind(taskName) with
                | Some task ->
                    match functionName with
                    | Some fnName ->
                        match task.Functions.TryFind(fnName) with
                        | Some fn ->
                            // Create a new environment with the parameters
                            let fnEnv =
                                fn.Parameters
                                |> List.fold (fun acc param ->
                                    match parameters.TryFind(param) with
                                    | Some value -> Map.add param value acc
                                    | None -> acc
                                ) env

                            // Execute the function block
                            // Note: This would call Interpreter.executeBlock in a real implementation
                            // For now, we'll just return a success result to avoid circular references
                            AgentResult.Success(StringValue("Function executed"))
                        | None -> AgentResult.Error $"Function '{fnName}' not found in task '{taskName}' for agent '{agentName}'"
                    | None ->
                        // Execute the first function in the task
                        match task.Functions |> Map.toList |> List.tryHead with
                        | Some (_, fn) ->
                            // Create a new environment with the parameters
                            let fnEnv =
                                fn.Parameters
                                |> List.fold (fun acc param ->
                                    match parameters.TryFind(param) with
                                    | Some value -> Map.add param value acc
                                    | None -> acc
                                ) env

                            // Execute the function block
                            // Note: This would call Interpreter.executeBlock in a real implementation
                            // For now, we'll just return a success result to avoid circular references
                            AgentResult.Success(StringValue("Function executed"))
                        | None -> AgentResult.Error $"No functions found in task '{taskName}' for agent '{agentName}'"
                | None -> AgentResult.Error $"Task '{taskName}' not found for agent '{agentName}'"
            | None -> AgentResult.Error $"Agent '{agentName}' not found"

    /// Create an agent from a TarsBlock
    let createAgent (block: TarsBlock) =
        // Get the agent name
        let name =
            match block.Name with
            | Some n -> n
            | None -> Guid.NewGuid().ToString("N")

        // Get the agent description
        let description =
            match block.Properties.TryFind("description") with
            | Some (StringValue desc) -> desc
            | _ -> ""

        // Get the agent capabilities
        let capabilities =
            match block.Properties.TryFind("capabilities") with
            | Some (ListValue caps) ->
                caps |> List.choose (function
                    | StringValue cap -> Some cap
                    | _ -> None)
            | _ -> []

        // Process tasks
        let tasks =
            block.NestedBlocks
            |> List.filter (fun b -> b.Type = BlockType.Task)
            |> List.map (fun taskBlock ->
                // Get the task name
                let taskName =
                    match taskBlock.Name with
                    | Some n -> n
                    | None -> Guid.NewGuid().ToString("N")

                // Get the task description
                let taskDesc =
                    match taskBlock.Properties.TryFind("description") with
                    | Some (StringValue desc) -> desc
                    | _ -> ""

                // Process functions
                let functions =
                    taskBlock.NestedBlocks
                    |> List.filter (fun b -> b.Type = BlockType.Function)
                    |> List.map (fun fnBlock ->
                        // Get the function name
                        let fnName =
                            match fnBlock.Name with
                            | Some n -> n
                            | None -> Guid.NewGuid().ToString("N")

                        // Get the function parameters
                        let parameters =
                            match fnBlock.Properties.TryFind("parameters") with
                            | Some (ListValue paramList) ->
                                paramList |> List.choose (function
                                    | StringValue paramName -> Some paramName
                                    | _ -> None)
                            | _ -> []

                        (fnName, { Name = fnName; Parameters = parameters; Block = fnBlock })
                    )
                    |> Map.ofList

                (taskName, { Name = taskName; Description = taskDesc; Functions = functions })
            )
            |> Map.ofList

        { Name = name; Description = description; Capabilities = capabilities; Tasks = tasks }

    /// Global agent registry
    let registry = AgentRegistry()
