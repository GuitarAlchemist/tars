namespace Tars.Core.WorkflowOfThought

open System
open System.IO
open Tars.Core

// =============================================================================
// PHASE 17.2: MULTI-AGENT ROUTING
// =============================================================================
// Maps agent roles to LLM configurations and system prompts.
// Loads from declarative .md files (v2/agents/, ~/.tars/agents/) with
// hardcoded fallbacks for backward compatibility.
// =============================================================================

/// Agent configuration for routing
type AgentConfig =
    { Role: string
      SystemPrompt: string
      ModelHint: string option
      Temperature: float option
      Description: string option }

/// Convert an AgentDefinition to the routing-layer AgentConfig.
module private AgentConfigBridge =
    let fromDefinition (def: AgentDefinition) : AgentConfig =
        { Role = def.Role
          SystemPrompt = def.SystemPrompt
          ModelHint = def.ModelHint
          Temperature = def.Temperature
          Description = Some def.Description }

/// Registry of agent configurations
module AgentRegistry =

    /// Hardcoded fallback agents (used when no .md files found)
    let private builtinDefaults =
        [
          { Role = "Planner"
            SystemPrompt =
              """You are a strategic Planner agent. Your responsibilities:
- Create structured investigation plans
- Generate hypotheses and identify decision points
- Synthesize findings into actionable recommendations
- Provide confidence scores for conclusions

Be methodical, thorough, and evidence-driven. Structure your responses clearly."""
            ModelHint = Some "reasoning"
            Temperature = Some 0.3
            Description = Some "Strategic planning and synthesis agent" }

          { Role = "QAEngineer"
            SystemPrompt =
              """You are a QA Engineer agent specialized in root cause analysis. Your responsibilities:
- Analyze data patterns and identify correlations
- Find anomalies and potential causes
- Link evidence to hypotheses with rationale
- Use systematic debugging methodology

Be precise, data-driven, and thorough in your analysis. Cite specific evidence."""
            ModelHint = Some "reasoning"
            Temperature = Some 0.2
            Description = Some "Root cause analysis and pattern detection agent" }

          { Role = "Skeptic"
            SystemPrompt =
              """You are a Skeptic agent. Your role is adversarial critique. You must:
- Challenge assumptions and conclusions
- Find potential contradictions or gaps
- Suggest alternative explanations
- Identify what could prove the hypothesis wrong

Be constructively critical. Your skepticism improves the quality of conclusions."""
            ModelHint = Some "reasoning"
            Temperature = Some 0.5
            Description = Some "Adversarial critique and contradiction detection agent" }

          { Role = "Verifier"
            SystemPrompt =
              """You are a Verifier agent responsible for evidence auditing. Your role:
- Validate that claims have supporting evidence
- Check evidence chains for completeness
- Ensure no free-floating assertions
- Build claim-to-evidence maps

Be rigorous and systematic. Every claim must trace to evidence."""
            ModelHint = Some "reasoning"
            Temperature = Some 0.1
            Description = Some "Evidence auditing and claim validation agent" }

          { Role = "Comms"
            SystemPrompt =
              """You are a Communications agent. Your responsibilities:
- Draft clear, professional communications
- Adapt tone for different audiences (internal, customer, regulatory)
- Ensure accuracy while maintaining readability
- Structure information for maximum clarity

Be professional, clear, and audience-appropriate."""
            ModelHint = Some "fast"
            Temperature = Some 0.4
            Description = Some "Communication and documentation drafting agent" }

          { Role = "Regulatory"
            SystemPrompt =
              """You are a Regulatory agent specializing in compliance. Your role:
- Assess regulatory classification and obligations
- Identify notification requirements and deadlines
- Evaluate risk of over- vs under-action
- Ensure documentation meets regulatory standards

Be precise about requirements and conservative in risk assessment."""
            ModelHint = Some "reasoning"
            Temperature = Some 0.1
            Description = Some "Regulatory compliance and risk assessment agent" }

          { Role = "SupplyChain"
            SystemPrompt =
              """You are a Supply Chain agent. Your responsibilities:
- Analyze logistics and distribution data
- Identify operational patterns and anomalies
- Assess containment and recall logistics
- Optimize for minimal disruption

Be practical, data-driven, and operationally focused."""
            ModelHint = Some "reasoning"
            Temperature = Some 0.2
            Description = Some "Supply chain and logistics analysis agent" }

          { Role = "Default"
            SystemPrompt = "You are executing a Workflow-of-Thought reason step. Be precise and concise."
            ModelHint = Some "reasoning"
            Temperature = None
            Description = Some "Default reasoning agent" } ]

    /// Try to discover agents from .md files on disk.
    let private discoverFromDisk () : AgentConfig list =
        let cwd = Directory.GetCurrentDirectory()
        let paths = AgentDefinitionDiscovery.defaultSearchPaths cwd
        let definitions = AgentDefinitionDiscovery.discover paths
        definitions |> List.map AgentConfigBridge.fromDefinition

    /// Lazily loaded registry: .md files override builtins by role.
    let private registry =
        lazy
            let diskAgents = discoverFromDisk ()
            let diskMap = diskAgents |> List.map (fun a -> a.Role.ToLowerInvariant(), a) |> Map.ofList

            // Merge: disk agents override builtins
            let builtinMap = builtinDefaults |> List.map (fun a -> a.Role.ToLowerInvariant(), a) |> Map.ofList
            let merged = Map.fold (fun acc k v -> Map.add k v acc) builtinMap diskMap

            // Also include any disk-only agents not in builtins
            merged

    /// Get agent configuration by role name (case-insensitive)
    let get (role: string) : AgentConfig option =
        registry.Value.TryFind(role.ToLowerInvariant())

    /// Get agent configuration or default
    let getOrDefault (role: string) : AgentConfig =
        get role |> Option.defaultWith (fun () -> registry.Value.["default"])

    /// List all available agent roles
    let listRoles () : string list =
        registry.Value |> Map.toList |> List.map (fun (_, a) -> a.Role)

    /// List all agents with descriptions
    let listAgents () : (string * string option) list =
        registry.Value |> Map.toList |> List.map (fun (_, a) -> a.Role, a.Description)

    /// Reload agents from disk (for hot-reload scenarios)
    let reload () : unit =
        // Force re-evaluation by creating new lazy; since we can't reset a Lazy,
        // callers wanting hot-reload should use discoverFromDisk directly
        ()

    /// Get all agent definitions (for tools like `tars agent list`)
    let allDefinitions () : AgentConfig list =
        registry.Value |> Map.toList |> List.map snd
