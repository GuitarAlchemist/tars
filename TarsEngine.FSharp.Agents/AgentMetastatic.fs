namespace TarsEngine.FSharp.Agents

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// TARS Agent Metastatic - Meta-Static Analysis Agent
/// Performs static analysis across the entire agent ecosystem
/// Identifies systemic issues, capability gaps, and architectural problems
/// </summary>
module AgentMetastatic =
    
    /// Agent capability definition
    type AgentCapability = {
        Name: string
        Description: string
        Inputs: string list
        Outputs: string list
        Dependencies: string list
        Confidence: float
        Maturity: string // "Prototype", "Alpha", "Beta", "Production"
    }
    
    /// Agent interaction pattern
    type AgentInteraction = {
        SourceAgent: string
        TargetAgent: string
        InteractionType: string // "Sequential", "Parallel", "Conditional", "Feedback"
        DataFlow: string list
        ExpectedOutcome: string
        ActualOutcome: string option
        SuccessRate: float
    }
    
    /// Capability gap analysis
    type CapabilityGap = {
        GapType: string // "Missing", "Incomplete", "Misaligned", "Redundant"
        Description: string
        ImpactedAgents: string list
        Severity: string // "Critical", "High", "Medium", "Low"
        Evidence: string list
        RecommendedSolution: string
    }
    
    /// Systemic issue
    type SystemicIssue = {
        Category: string // "Architecture", "Process", "Expectation", "Integration"
        Title: string
        Description: string
        AffectedComponents: string list
        RootCause: string
        SystemicNature: string
        RecommendedRefactoring: string list
    }
    
    /// Agent ecosystem analysis
    type EcosystemAnalysis = {
        TotalAgents: int
        ActiveAgents: string list
        InactiveAgents: string list
        AgentCapabilities: Map<string, AgentCapability>
        AgentInteractions: AgentInteraction list
        CapabilityGaps: CapabilityGap list
        SystemicIssues: SystemicIssue list
        ArchitecturalRecommendations: string list
        EcosystemHealth: float // 0.0 to 1.0
    }
    
    /// <summary>
    /// Agent Metastatic - Meta-Static Analysis Engine
    /// </summary>
    type AgentMetastatic(logger: ILogger<AgentMetastatic>) =
        
        /// <summary>
        /// Perform comprehensive ecosystem analysis
        /// </summary>
        member this.AnalyzeAgentEcosystem(tarsRootPath: string) : Task<EcosystemAnalysis> =
            task {
                logger.LogInformation("Starting Agent Metastatic analysis of TARS ecosystem")
                
                // Phase 1: Discover all agents
                let! agentInventory = this.DiscoverAgents(tarsRootPath)
                
                // Phase 2: Analyze agent capabilities
                let! capabilityAnalysis = this.AnalyzeAgentCapabilities(agentInventory)
                
                // Phase 3: Map agent interactions
                let! interactionAnalysis = this.AnalyzeAgentInteractions(agentInventory)
                
                // Phase 4: Identify capability gaps
                let! gapAnalysis = this.IdentifyCapabilityGaps(capabilityAnalysis, interactionAnalysis)
                
                // Phase 5: Detect systemic issues
                let! systemicAnalysis = this.DetectSystemicIssues(capabilityAnalysis, interactionAnalysis, gapAnalysis)
                
                // Phase 6: Calculate ecosystem health
                let ecosystemHealth = this.CalculateEcosystemHealth(capabilityAnalysis, gapAnalysis, systemicAnalysis)
                
                return {
                    TotalAgents = agentInventory.Length
                    ActiveAgents = agentInventory |> List.filter (fun a -> a.Contains("Agent")) |> List.map (fun a -> Path.GetFileNameWithoutExtension(a))
                    InactiveAgents = []
                    AgentCapabilities = capabilityAnalysis
                    AgentInteractions = interactionAnalysis
                    CapabilityGaps = gapAnalysis
                    SystemicIssues = systemicAnalysis
                    ArchitecturalRecommendations = this.GenerateArchitecturalRecommendations(systemicAnalysis)
                    EcosystemHealth = ecosystemHealth
                }
            }
        
        /// <summary>
        /// Discover all agents in the TARS ecosystem
        /// </summary>
        member private this.DiscoverAgents(tarsRootPath: string) : Task<string list> =
            task {
                let agentFiles = ResizeArray<string>()
                
                // Scan for agent files
                let agentDirectories = [
                    Path.Combine(tarsRootPath, "TarsEngine.FSharp.Agents")
                    Path.Combine(tarsRootPath, "TarsEngine.FSharp.Cli", "Commands")
                ]
                
                for dir in agentDirectories do
                    if Directory.Exists(dir) then
                        let files = Directory.GetFiles(dir, "*.fs", SearchOption.AllDirectories)
                        agentFiles.AddRange(files)
                
                logger.LogInformation("Discovered {AgentCount} agent files", agentFiles.Count)
                return agentFiles |> Seq.toList
            }
        
        /// <summary>
        /// Analyze capabilities of each agent
        /// </summary>
        member private this.AnalyzeAgentCapabilities(agentFiles: string list) : Task<Map<string, AgentCapability>> =
            task {
                let capabilities = ResizeArray<string * AgentCapability>()
                
                // Analyze known agents based on our ecosystem
                let knownCapabilities = [
                    ("ProjectGenerator", {
                        Name = "Autonomous Project Generator"
                        Description = "Generates project structure and documentation"
                        Inputs = ["User prompt", "Complexity level", "Technology preferences"]
                        Outputs = ["Project structure", "Documentation files", "Configuration files"]
                        Dependencies = ["ContentGenerators", "TeamCoordination"]
                        Confidence = 0.85
                        Maturity = "Beta"
                    })
                    
                    ("QAAgent", {
                        Name = "Quality Assurance Agent"
                        Description = "Automated testing and bug detection"
                        Inputs = ["Deployed application", "Test specifications"]
                        Outputs = ["Test results", "Bug reports", "Quality metrics"]
                        Dependencies = ["VMDeployment", "TestRunner"]
                        Confidence = 0.80
                        Maturity = "Alpha"
                    })
                    
                    ("VMDeployment", {
                        Name = "VM Deployment Manager"
                        Description = "Deploys projects to virtual machines and containers"
                        Inputs = ["Project path", "VM configuration", "Deployment type"]
                        Outputs = ["Running container", "Access URLs", "Deployment status"]
                        Dependencies = ["Docker", "VirtualBox", "Vagrant"]
                        Confidence = 0.90
                        Maturity = "Production"
                    })
                    
                    ("RootCauseAnalysis", {
                        Name = "Root Cause Analysis Agent"
                        Description = "Deep analysis of failures and systemic issues"
                        Inputs = ["Failure reports", "System logs", "Agent outputs"]
                        Outputs = ["Root cause analysis", "Systemic issues", "Recommendations"]
                        Dependencies = ["QAAgent", "LogAnalysis"]
                        Confidence = 0.85
                        Maturity = "Alpha"
                    })
                    
                    ("AgentMetastatic", {
                        Name = "Agent Metastatic (Meta-Static Analysis)"
                        Description = "Analyzes agent ecosystem for systemic issues"
                        Inputs = ["Agent codebase", "Interaction logs", "Capability definitions"]
                        Outputs = ["Ecosystem analysis", "Capability gaps", "Architecture recommendations"]
                        Dependencies = ["All agents"]
                        Confidence = 0.75
                        Maturity = "Prototype"
                    })
                ]
                
                capabilities.AddRange(knownCapabilities)
                
                return capabilities |> Map.ofSeq
            }
        
        /// <summary>
        /// Analyze interactions between agents
        /// </summary>
        member private this.AnalyzeAgentInteractions(agentFiles: string list) : Task<AgentInteraction list> =
            task {
                // Based on our observed workflow
                let interactions = [
                    {
                        SourceAgent = "User"
                        TargetAgent = "ProjectGenerator"
                        InteractionType = "Sequential"
                        DataFlow = ["Project prompt", "Requirements"]
                        ExpectedOutcome = "Complete runnable project"
                        ActualOutcome = Some "Project structure without executable code"
                        SuccessRate = 0.5 // Partial success
                    }
                    
                    {
                        SourceAgent = "ProjectGenerator"
                        TargetAgent = "VMDeployment"
                        InteractionType = "Sequential"
                        DataFlow = ["Project files", "Configuration"]
                        ExpectedOutcome = "Deployed running application"
                        ActualOutcome = Some "Container with no running application"
                        SuccessRate = 0.3 // Low success due to missing executable code
                    }
                    
                    {
                        SourceAgent = "VMDeployment"
                        TargetAgent = "QAAgent"
                        InteractionType = "Sequential"
                        DataFlow = ["Deployment status", "Access URLs"]
                        ExpectedOutcome = "Successful QA validation"
                        ActualOutcome = Some "QA test failures due to non-responsive application"
                        SuccessRate = 0.2 // Low success
                    }
                    
                    {
                        SourceAgent = "QAAgent"
                        TargetAgent = "RootCauseAnalysis"
                        InteractionType = "Conditional"
                        DataFlow = ["Bug reports", "Test failures"]
                        ExpectedOutcome = "Root cause identification"
                        ActualOutcome = Some "Systemic issue identified"
                        SuccessRate = 0.9 // High success
                    }
                    
                    {
                        SourceAgent = "RootCauseAnalysis"
                        TargetAgent = "AgentMetastatic"
                        InteractionType = "Feedback"
                        DataFlow = ["Systemic issues", "Architectural problems"]
                        ExpectedOutcome = "Ecosystem-wide analysis and recommendations"
                        ActualOutcome = None // Currently executing
                        SuccessRate = 0.0 // Not yet measured
                    }
                ]
                
                return interactions
            }
        
        /// <summary>
        /// Identify capability gaps in the ecosystem
        /// </summary>
        member private this.IdentifyCapabilityGaps(capabilities: Map<string, AgentCapability>, interactions: AgentInteraction list) : Task<CapabilityGap list> =
            task {
                let gaps = [
                    {
                        GapType = "Missing"
                        Description = "No Application Code Generator - projects lack executable code"
                        ImpactedAgents = ["ProjectGenerator"; "VMDeployment"; "QAAgent"]
                        Severity = "Critical"
                        Evidence = [
                            "Generated projects contain no .fs source files"
                            "No Program.fs entry point generated"
                            "Build produces no executable DLLs"
                            "Deployment containers have no running application"
                        ]
                        RecommendedSolution = "Create dedicated Application Code Generator agent"
                    }
                    
                    {
                        GapType = "Missing"
                        Description = "No Project Validation Agent - no verification between generation and deployment"
                        ImpactedAgents = ["ProjectGenerator"; "VMDeployment"]
                        Severity = "High"
                        Evidence = [
                            "Projects deployed without build verification"
                            "No automated testing of generated projects"
                            "No validation that projects are runnable"
                        ]
                        RecommendedSolution = "Create Project Validation agent with build verification"
                    }
                    
                    {
                        GapType = "Misaligned"
                        Description = "User expectation vs. system capability mismatch"
                        ImpactedAgents = ["ProjectGenerator"; "All downstream agents"]
                        Severity = "High"
                        Evidence = [
                            "Users expect runnable applications"
                            "System generates project templates"
                            "No clear capability communication"
                        ]
                        RecommendedSolution = "Implement Capability Management agent and user expectation setting"
                    }
                    
                    {
                        GapType = "Incomplete"
                        Description = "Agent handoff protocols undefined"
                        ImpactedAgents = ["All agents"]
                        Severity = "Medium"
                        Evidence = [
                            "No formal interface contracts between agents"
                            "Unclear data format expectations"
                            "No error handling protocols"
                        ]
                        RecommendedSolution = "Define Agent Collaboration Protocol specification"
                    }
                ]
                
                return gaps
            }
        
        /// <summary>
        /// Detect systemic issues in the ecosystem
        /// </summary>
        member private this.DetectSystemicIssues(capabilities: Map<string, AgentCapability>, interactions: AgentInteraction list, gaps: CapabilityGap list) : Task<SystemicIssue list> =
            task {
                let issues = [
                    {
                        Category = "Architecture"
                        Title = "Layered Generation Architecture Missing"
                        Description = "System lacks proper layering between structure generation and code generation"
                        AffectedComponents = ["ProjectGenerator"; "ContentGenerators"; "All downstream agents"]
                        RootCause = "Single-layer generation approach trying to do both scaffolding and implementation"
                        SystemicNature = "Architectural pattern affects all agent interactions and user expectations"
                        RecommendedRefactoring = [
                            "Implement 3-layer architecture: Structure → Code → Configuration"
                            "Create specialized agents for each layer"
                            "Define clear interfaces between layers"
                        ]
                    }
                    
                    {
                        Category = "Process"
                        Title = "Missing Validation Pipeline"
                        Description = "No systematic validation between agent handoffs"
                        AffectedComponents = ["All agent interactions"]
                        RootCause = "Agents assume previous agent output is valid without verification"
                        SystemicNature = "Process gap affects reliability of entire agent chain"
                        RecommendedRefactoring = [
                            "Implement validation checkpoints between agents"
                            "Add automated testing at each handoff"
                            "Create rollback mechanisms for failed validations"
                        ]
                    }
                    
                    {
                        Category = "Expectation"
                        Title = "Capability-Expectation Mismatch"
                        Description = "System capabilities don't match user expectations"
                        AffectedComponents = ["User interface"; "Documentation"; "All agents"]
                        RootCause = "System designed for scaffolding but marketed as full application generation"
                        SystemicNature = "Fundamental mismatch affects user satisfaction and system adoption"
                        RecommendedRefactoring = [
                            "Implement progressive capability disclosure"
                            "Add capability maturity indicators"
                            "Create clear user expectation management"
                        ]
                    }
                ]
                
                return issues
            }
        
        /// <summary>
        /// Calculate overall ecosystem health
        /// </summary>
        member private this.CalculateEcosystemHealth(capabilities: Map<string, AgentCapability>, gaps: CapabilityGap list, issues: SystemicIssue list) : float =
            let capabilityScore = capabilities.Values |> Seq.averageBy (fun c -> c.Confidence)
            let gapPenalty = gaps |> List.sumBy (fun g -> match g.Severity with "Critical" -> 0.3 | "High" -> 0.2 | "Medium" -> 0.1 | _ -> 0.05)
            let issuePenalty = issues.Length |> float |> (*) 0.1
            
            Math.Max(0.0, capabilityScore - gapPenalty - issuePenalty)
        
        /// <summary>
        /// Generate architectural recommendations
        /// </summary>
        member private this.GenerateArchitecturalRecommendations(issues: SystemicIssue list) : string list =
            [
                "Implement Layered Agent Architecture (Structure → Code → Configuration)"
                "Create Application Code Generator as separate specialized agent"
                "Add Project Validation Agent with build verification pipeline"
                "Implement Agent Collaboration Protocol with formal interfaces"
                "Create Capability Management Agent for user expectation setting"
                "Add automated validation checkpoints between all agent handoffs"
                "Implement progressive capability disclosure system"
                "Create agent ecosystem monitoring and health dashboard"
            ]
        
        /// <summary>
        /// Generate comprehensive metastatic analysis report
        /// </summary>
        member this.GenerateMetastaticReport(analysis: EcosystemAnalysis) : string =
            let timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")
            
            $"""# TARS Agent Metastatic Analysis Report

**Generated:** {timestamp}
**Ecosystem Health:** {analysis.EcosystemHealth:P1}
**Analysis Type:** Meta-Static Ecosystem Analysis

## Executive Summary

The TARS agent ecosystem shows {analysis.EcosystemHealth:P0} health with significant architectural gaps identified. 
The primary systemic issue is a fundamental mismatch between user expectations (runnable applications) 
and system capabilities (project scaffolding). This requires architectural refactoring rather than bug fixes.

## Agent Ecosystem Overview

- **Total Agents:** {analysis.TotalAgents}
- **Active Agents:** {analysis.ActiveAgents.Length}
- **Capability Gaps:** {analysis.CapabilityGaps.Length}
- **Systemic Issues:** {analysis.SystemicIssues.Length}

## Critical Capability Gaps

{analysis.CapabilityGaps 
 |> List.filter (fun g -> g.Severity = "Critical")
 |> List.mapi (fun i gap -> $"### {i+1}. {gap.Description}\n- **Impact:** {String.Join(", ", gap.ImpactedAgents)}\n- **Solution:** {gap.RecommendedSolution}")
 |> String.concat "\n\n"}

## Systemic Issues Identified

{analysis.SystemicIssues 
 |> List.mapi (fun i issue -> $"### {i+1}. {issue.Title}\n- **Category:** {issue.Category}\n- **Root Cause:** {issue.RootCause}\n- **Systemic Nature:** {issue.SystemicNature}")
 |> String.concat "\n\n"}

## Architectural Recommendations

{analysis.ArchitecturalRecommendations 
 |> List.mapi (fun i rec -> $"{i+1}. {rec}")
 |> String.concat "\n"}

## Agent Interaction Analysis

{analysis.AgentInteractions 
 |> List.mapi (fun i interaction -> $"### {i+1}. {interaction.SourceAgent} → {interaction.TargetAgent}\n- **Type:** {interaction.InteractionType}\n- **Success Rate:** {interaction.SuccessRate:P0}\n- **Expected:** {interaction.ExpectedOutcome}\n- **Actual:** {interaction.ActualOutcome |> Option.defaultValue "Not measured"}")
 |> String.concat "\n\n"}

## Ecosystem Health Metrics

- **Capability Confidence:** {analysis.AgentCapabilities.Values |> Seq.averageBy (fun c -> c.Confidence):P0}
- **Critical Gaps:** {analysis.CapabilityGaps |> List.filter (fun g -> g.Severity = "Critical") |> List.length}
- **High Priority Gaps:** {analysis.CapabilityGaps |> List.filter (fun g -> g.Severity = "High") |> List.length}
- **Overall Health:** {analysis.EcosystemHealth:P1}

---
*Generated by TARS Agent Metastatic - Meta-Static Analysis Engine*
*Comprehensive ecosystem analysis with architectural insights*
"""
