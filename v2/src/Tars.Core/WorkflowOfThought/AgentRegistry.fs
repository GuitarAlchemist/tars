namespace Tars.Core.WorkflowOfThought

open System

// =============================================================================
// PHASE 17.2: MULTI-AGENT ROUTING
// =============================================================================
// Maps agent roles to LLM configurations and system prompts
// Enables role-specialized reasoning in GoT workflows
// =============================================================================

/// Agent configuration for routing
type AgentConfig =
    { Role: string
      SystemPrompt: string
      ModelHint: string option
      Temperature: float option
      Description: string option }

/// Registry of agent configurations
module AgentRegistry =

    /// Default agents available in TARS workflows
    let private defaults =
        [
          // Planner - orchestrates investigation and synthesis
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

          // QA Engineer - root cause analysis and pattern detection
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

          // Skeptic - adversarial critique and contradiction finding
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

          // Verifier - evidence auditing and claim validation
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

          // Comms - communication drafting
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

          // Regulatory - compliance and risk assessment
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

          // SupplyChain - logistics and operations
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

          // Default - generic reasoning
          { Role = "Default"
            SystemPrompt = "You are executing a Workflow-of-Thought reason step. Be precise and concise."
            ModelHint = Some "reasoning"
            Temperature = None
            Description = Some "Default reasoning agent" } ]

    let private registry =
        defaults |> List.map (fun a -> a.Role.ToLowerInvariant(), a) |> Map.ofList

    /// Get agent configuration by role name (case-insensitive)
    let get (role: string) : AgentConfig option =
        registry.TryFind(role.ToLowerInvariant())

    /// Get agent configuration or default
    let getOrDefault (role: string) : AgentConfig =
        get role |> Option.defaultWith (fun () -> registry.["default"])

    /// List all available agent roles
    let listRoles () : string list = defaults |> List.map (fun a -> a.Role)

    /// List all agents with descriptions
    let listAgents () : (string * string option) list =
        defaults |> List.map (fun a -> a.Role, a.Description)
