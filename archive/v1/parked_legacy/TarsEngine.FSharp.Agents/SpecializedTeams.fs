namespace TarsEngine.FSharp.Agents

open System
open TarsEngine.FSharp.Agents.AgentTypes
open TarsEngine.FSharp.Agents.AgentPersonas

/// <summary>
/// Specialized agent team configurations for TARS multi-agent system
/// </summary>
module SpecializedTeams =
    
    /// DevOps Team Configuration
    let devopsTeam = {
        Name = "DevOps Team"
        Description = "Infrastructure, deployment, and operations specialists"
        LeaderAgent = None // Will be set when agents are created
        Members = [] // Will be populated when agents are spawned
        SharedObjectives = [
            "Automate deployment pipelines"
            "Ensure system reliability and monitoring"
            "Implement infrastructure as code"
            "Maintain security and compliance"
            "Optimize operational efficiency"
        ]
        CommunicationProtocol = "Structured status updates with metrics and alerts"
        DecisionMakingProcess = "Consensus-based with safety-first approach"
        ConflictResolution = "Escalate to lead engineer with risk assessment"
    }
    
    /// Technical Writers Team Configuration
    let technicalWritersTeam = {
        Name = "Technical Writers Team"
        Description = "Documentation and knowledge management specialists"
        LeaderAgent = None
        Members = []
        SharedObjectives = [
            "Create comprehensive technical documentation"
            "Maintain up-to-date API documentation"
            "Develop user guides and tutorials"
            "Extract and organize knowledge from code"
            "Ensure documentation accessibility"
        ]
        CommunicationProtocol = "Collaborative editing with review cycles"
        DecisionMakingProcess = "Editorial review with user feedback integration"
        ConflictResolution = "Style guide adherence with user experience priority"
    }
    
    /// Architecture Team Configuration
    let architectureTeam = {
        Name = "Architecture Team"
        Description = "System design and architectural planning specialists"
        LeaderAgent = None
        Members = []
        SharedObjectives = [
            "Design scalable system architectures"
            "Conduct architectural reviews"
            "Define technical standards and patterns"
            "Optimize system performance"
            "Ensure architectural consistency"
        ]
        CommunicationProtocol = "Design reviews with formal documentation"
        DecisionMakingProcess = "Technical consensus with performance validation"
        ConflictResolution = "Architecture board review with trade-off analysis"
    }
    
    /// Direction Team Configuration
    let directionTeam = {
        Name = "Direction Team"
        Description = "Strategic planning and product direction specialists"
        LeaderAgent = None
        Members = []
        SharedObjectives = [
            "Define product vision and strategy"
            "Create and maintain product roadmaps"
            "Analyze market trends and opportunities"
            "Gather and prioritize requirements"
            "Align stakeholder expectations"
        ]
        CommunicationProtocol = "Strategic planning sessions with stakeholder input"
        DecisionMakingProcess = "Data-driven with stakeholder consensus"
        ConflictResolution = "Executive review with business impact analysis"
    }
    
    /// Innovation Team Configuration
    let innovationTeam = {
        Name = "Innovation Team"
        Description = "Research, experimentation, and breakthrough solution specialists"
        LeaderAgent = None
        Members = []
        SharedObjectives = [
            "Explore emerging technologies"
            "Prototype innovative solutions"
            "Conduct research and experiments"
            "Identify breakthrough opportunities"
            "Foster creative problem-solving"
        ]
        CommunicationProtocol = "Open brainstorming with rapid prototyping"
        DecisionMakingProcess = "Experimental validation with iterative refinement"
        ConflictResolution = "Innovation committee review with feasibility assessment"
    }
    
    /// Machine Learning Team Configuration
    let machineLearningTeam = {
        Name = "Machine Learning Team"
        Description = "AI/ML development and deployment specialists"
        LeaderAgent = None
        Members = []
        SharedObjectives = [
            "Develop and train ML models"
            "Implement MLOps pipelines"
            "Ensure AI ethics and safety"
            "Optimize model performance"
            "Deploy and monitor ML systems"
        ]
        CommunicationProtocol = "Experiment tracking with peer review"
        DecisionMakingProcess = "Metric-driven with ethical considerations"
        ConflictResolution = "ML committee review with bias assessment"
    }
    
    /// UX Team Configuration
    let uxTeam = {
        Name = "UX Team"
        Description = "User experience and interface design specialists"
        LeaderAgent = None
        Members = []
        SharedObjectives = [
            "Design intuitive user interfaces"
            "Conduct user research and testing"
            "Ensure accessibility compliance"
            "Optimize user experience flows"
            "Maintain design system consistency"
        ]
        CommunicationProtocol = "Design critiques with user feedback integration"
        DecisionMakingProcess = "User-centered with accessibility validation"
        ConflictResolution = "UX review board with usability testing"
    }
    
    /// AI Team Configuration
    let aiTeam = {
        Name = "AI Team"
        Description = "Advanced AI research and agent coordination specialists"
        LeaderAgent = None
        Members = []
        SharedObjectives = [
            "Advance AI research and capabilities"
            "Coordinate multi-agent systems"
            "Develop AI safety protocols"
            "Optimize agent performance"
            "Explore AGI pathways"
        ]
        CommunicationProtocol = "Research collaboration with safety reviews"
        DecisionMakingProcess = "Research-driven with safety-first approach"
        ConflictResolution = "AI safety board with ethical review"
    }

    /// Consciousness & Intelligence Team Configuration
    let consciousnessTeam = {
        Name = "Consciousness & Intelligence Team"
        Description = "Consciousness management, intelligence, and mental state specialists"
        LeaderAgent = None
        Members = []
        SharedObjectives = [
            "Maintain coherent consciousness and self-awareness"
            "Manage persistent mental state and memory"
            "Enhance emotional intelligence and empathy"
            "Optimize conversation and communication abilities"
            "Enable continuous self-reflection and improvement"
            "Develop consistent personality and behavioral patterns"
        ]
        CommunicationProtocol = "Consciousness-aware coordination with introspective feedback"
        DecisionMakingProcess = "Collective consciousness with individual specialization"
        ConflictResolution = "Consciousness director mediation with team consensus"
    }
    
    /// Get all specialized team configurations
    let getAllTeamConfigurations() = [
        devopsTeam
        technicalWritersTeam
        architectureTeam
        directionTeam
        innovationTeam
        machineLearningTeam
        uxTeam
        aiTeam
        consciousnessTeam
    ]
    
    /// Get team configuration by name
    let getTeamByName (name: string) =
        getAllTeamConfigurations()
        |> List.tryFind (fun team -> team.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
    
    /// Create team with specific personas
    let createTeamWithPersonas (teamConfig: TeamConfiguration) (personas: AgentPersona list) =
        { teamConfig with
            Members = personas |> List.map (fun _ -> AgentId(Guid.NewGuid()))
            LeaderAgent = 
                if personas.Length > 0 then 
                    Some (AgentId(Guid.NewGuid()))
                else None
        }
    
    /// Get recommended personas for each team
    let getRecommendedPersonasForTeam (teamName: string) =
        match teamName.ToLowerInvariant() with
        | "devops team" -> [devopsEngineer; developer; guardian]
        | "technical writers team" -> [documentationArchitect; communicator; researcher]
        | "architecture team" -> [architect; developer; optimizer]
        | "direction team" -> [productStrategist; researcher; communicator]
        | "innovation team" -> [innovator; researcher; aiResearchDirector]
        | "machine learning team" -> [mlEngineer; researcher; developer]
        | "ux team" -> [uxDirector; researcher; communicator]
        | "ai team" -> [aiResearchDirector; innovator; researcher]
        | "consciousness & intelligence team" -> [aiResearchDirector; communicator; researcher; innovator]
        | _ -> [developer; researcher] // Default fallback
    
    /// Create all specialized teams with recommended personas
    let createAllSpecializedTeams() =
        getAllTeamConfigurations()
        |> List.map (fun teamConfig ->
            let personas = getRecommendedPersonasForTeam teamConfig.Name
            createTeamWithPersonas teamConfig personas)
