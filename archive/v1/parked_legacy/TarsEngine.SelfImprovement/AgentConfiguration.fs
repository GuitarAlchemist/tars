namespace TarsEngine.SelfImprovement

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open System.Collections.Generic

/// <summary>
/// Represents the configuration for an AI agent
/// </summary>
type AgentConfig = {
    Id: string
    Name: string
    Role: string
    Model: string
    Temperature: float
    MaxTokens: int option
    TopP: float option
    FrequencyPenalty: float option
    PresencePenalty: float option
    SystemPrompt: string option
    Description: string option
    Capabilities: string list
    Metadata: Dictionary<string, string>
}

/// <summary>
/// Represents a collection of agent configurations
/// </summary>
type AgentRegistry = {
    Agents: AgentConfig list
    DefaultAgentId: string option
}

/// <summary>
/// Functions for working with agent configurations
/// </summary>
module AgentConfiguration =
    /// <summary>
    /// Creates a new agent configuration
    /// </summary>
    let createAgentConfig (id: string) (name: string) (role: string) (model: string) (temperature: float) =
        {
            Id = id
            Name = name
            Role = role
            Model = model
            Temperature = temperature
            MaxTokens = None
            TopP = None
            FrequencyPenalty = None
            PresencePenalty = None
            SystemPrompt = None
            Description = None
            Capabilities = []
            Metadata = new Dictionary<string, string>()
        }
    
    /// <summary>
    /// Creates a new agent registry
    /// </summary>
    let createAgentRegistry () =
        {
            Agents = []
            DefaultAgentId = None
        }
    
    /// <summary>
    /// Adds an agent to the registry
    /// </summary>
    let addAgent (registry: AgentRegistry) (agent: AgentConfig) =
        { registry with Agents = registry.Agents @ [agent] }
    
    /// <summary>
    /// Sets the default agent in the registry
    /// </summary>
    let setDefaultAgent (registry: AgentRegistry) (agentId: string) =
        { registry with DefaultAgentId = Some agentId }
    
    /// <summary>
    /// Gets an agent by ID
    /// </summary>
    let getAgentById (registry: AgentRegistry) (agentId: string) =
        registry.Agents
        |> List.tryFind (fun a -> a.Id = agentId)
    
    /// <summary>
    /// Gets agents by role
    /// </summary>
    let getAgentsByRole (registry: AgentRegistry) (role: string) =
        registry.Agents
        |> List.filter (fun a -> a.Role = role)
    
    /// <summary>
    /// Gets the default agent
    /// </summary>
    let getDefaultAgent (registry: AgentRegistry) =
        match registry.DefaultAgentId with
        | Some id -> getAgentById registry id
        | None -> None
    
    /// <summary>
    /// Creates standard agent configurations
    /// </summary>
    let createStandardAgents () =
        let registry = createAgentRegistry()
        
        let planner = createAgentConfig "planner" "Planner" "planner" "llama3" 0.7
        let plannerWithPrompt = 
            { planner with 
                SystemPrompt = Some "You are a planning agent for the TARS system. Your role is to break down tasks into smaller steps."
                Description = Some "Plans the overall approach and breaks down tasks"
                Capabilities = ["planning"; "task-decomposition"; "strategy"]
            }
        
        let coder = createAgentConfig "coder" "Coder" "coder" "codellama:13b-code" 0.2
        let coderWithPrompt = 
            { coder with 
                SystemPrompt = Some "You are a coding agent for the TARS system. Your role is to write code based on requirements."
                Description = Some "Writes and refines code based on the plan"
                Capabilities = ["coding"; "debugging"; "refactoring"]
            }
        
        let critic = createAgentConfig "critic" "Critic" "critic" "llama3" 0.5
        let criticWithPrompt = 
            { critic with 
                SystemPrompt = Some "You are a critic agent for the TARS system. Your role is to review and provide feedback."
                Description = Some "Reviews and critiques code and plans"
                Capabilities = ["code-review"; "feedback"; "quality-assessment"]
            }
        
        let executor = createAgentConfig "executor" "Executor" "executor" "llama3" 0.3
        let executorWithPrompt = 
            { executor with 
                SystemPrompt = Some "You are an executor agent for the TARS system. Your role is to execute plans and report results."
                Description = Some "Executes plans and reports results"
                Capabilities = ["execution"; "testing"; "reporting"]
            }
        
        registry
        |> addAgent plannerWithPrompt
        |> addAgent coderWithPrompt
        |> addAgent criticWithPrompt
        |> addAgent executorWithPrompt
        |> setDefaultAgent "planner"
    
    /// <summary>
    /// Serializes an agent configuration to JSON
    /// </summary>
    let serializeAgentConfig (agent: AgentConfig) =
        let options = JsonSerializerOptions()
        options.WriteIndented <- true
        options.Converters.Add(JsonFSharpConverter())
        JsonSerializer.Serialize(agent, options)
    
    /// <summary>
    /// Deserializes an agent configuration from JSON
    /// </summary>
    let deserializeAgentConfig (json: string) =
        let options = JsonSerializerOptions()
        options.Converters.Add(JsonFSharpConverter())
        JsonSerializer.Deserialize<AgentConfig>(json, options)
    
    /// <summary>
    /// Serializes an agent registry to JSON
    /// </summary>
    let serializeAgentRegistry (registry: AgentRegistry) =
        let options = JsonSerializerOptions()
        options.WriteIndented <- true
        options.Converters.Add(JsonFSharpConverter())
        JsonSerializer.Serialize(registry, options)
    
    /// <summary>
    /// Deserializes an agent registry from JSON
    /// </summary>
    let deserializeAgentRegistry (json: string) =
        let options = JsonSerializerOptions()
        options.Converters.Add(JsonFSharpConverter())
        JsonSerializer.Deserialize<AgentRegistry>(json, options)
    
    /// <summary>
    /// Saves an agent registry to a file
    /// </summary>
    let saveAgentRegistryToFile (registry: AgentRegistry) (filePath: string) =
        let json = serializeAgentRegistry registry
        File.WriteAllText(filePath, json)
    
    /// <summary>
    /// Loads an agent registry from a file
    /// </summary>
    let loadAgentRegistryFromFile (filePath: string) =
        let json = File.ReadAllText(filePath)
        deserializeAgentRegistry json
