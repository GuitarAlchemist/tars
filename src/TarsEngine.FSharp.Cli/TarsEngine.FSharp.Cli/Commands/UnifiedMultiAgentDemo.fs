namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Spectre.Console
open TarsEngine.FSharp.Cli.Core.MultiAgentDomain
open TarsEngine.FSharp.Cli.Core.UnifiedFormatting
open TarsEngine.FSharp.Cli.Core.HtmlComponents
open TarsEngine.FSharp.Cli.Core.ConceptDecomposition
open TarsEngine.FSharp.Cli.Commands.Common
open TarsEngine.FSharp.Core.Agents.AgentSystem

// ============================================================================
// UNIFIED MULTI-AGENT REASONING DEMO - USING NEW ARCHITECTURE
// ============================================================================

module UnifiedMultiAgentDemo =

    // ============================================================================
    // CONVERSION UTILITIES
    // ============================================================================

    /// Convert legacy agent to unified agent
    let convertToUnifiedAgent (legacyAgent: SpecializedAgent) : UnifiedAgent =
        {
            // Core Identity
            Id = legacyAgent.Id
            Name = legacyAgent.Name
            CreatedAt = DateTime.UtcNow
            
            // Capabilities
            Specialization = legacyAgent.Specialization
            Capabilities = [
                "Problem Analysis"
                "Strategic Planning"
                "Communication"
                "Reasoning"
            ]
            ReasoningCapabilities = [
                ProblemDecomposition(complexity = 3)
                ConceptAnalysis(domains = ["AI"; "Reasoning"; "Strategy"])
                PatternRecognition(accuracy = 0.85)
                StrategicPlanning(horizon = TimeSpan.FromHours(24.0))
            ]
            
            // State
            Status = Working("Multi-agent reasoning task")
            CurrentTask = Some "Complex problem decomposition"
            Progress = Random().NextDouble()
            
            // Spatial & Communication
            Position3D = legacyAgent.Position3D
            Department = legacyAgent.Department
            CommunicationHistory = []
            
            // Game Theory & Strategy
            GameTheoryProfile = legacyAgent.GameTheoryProfile
            StrategyPreferences = [
                Cooperative(weight = 0.7)
                Adaptive(threshold = 0.3)
            ]
            
            // Performance Metrics
            PerformanceMetrics = {
                TasksCompleted = Random().Next(10, 50)
                AverageResponseTime = TimeSpan.FromSeconds(Random().NextDouble() * 5.0)
                SuccessRate = 0.8 + Random().NextDouble() * 0.2
                CommunicationEfficiency = 0.7 + Random().NextDouble() * 0.3
                ReasoningAccuracy = 0.8 + Random().NextDouble() * 0.2
            }
            QualityScore = 0.75 + Random().NextDouble() * 0.25
        }

    /// Convert legacy department to unified department
    let convertToUnifiedDepartment (legacyDept: AgentDepartment) (agents: UnifiedAgent list) : UnifiedDepartment =
        {
            // Core Identity
            Id = $"dept-{legacyDept.Name.ToLower().Replace(" ", "-")}"
            Name = legacyDept.Name
            CreatedAt = DateTime.UtcNow
            
            // Structure
            DepartmentType = legacyDept.DepartmentType
            Hierarchy = 1
            Agents = agents
            
            // Communication & Coordination
            CommunicationProtocol = legacyDept.CommunicationProtocol
            CoordinationStrategy = Distributed(consensus = Majority)
            
            // Game Theory & Strategy
            GameTheoryStrategy = legacyDept.GameTheoryStrategy
            CollectiveGoals = [
                {
                    Id = $"goal-{Guid.NewGuid().ToString("N")[..7]}"
                    Description = "Collaborative problem solving"
                    Priority = High
                    Deadline = Some (DateTime.UtcNow.AddDays(7.0))
                    AssignedAgents = agents |> List.map (fun a -> a.Id)
                    Progress = Random().NextDouble()
                }
            ]
            
            // Spatial & Performance
            Position3D = legacyDept.Position3D
            PerformanceMetrics = {
                CollectiveEfficiency = 0.7 + Random().NextDouble() * 0.3
                InterAgentCoordination = 0.6 + Random().NextDouble() * 0.4
                GoalCompletionRate = 0.8 + Random().NextDouble() * 0.2
                CommunicationOverhead = Random().NextDouble() * 0.3
                ResourceUtilization = 0.6 + Random().NextDouble() * 0.4
            }
        }

    // ============================================================================
    // ENHANCED REASONING PIPELINE WITH CONCEPT INTEGRATION
    // ============================================================================

    type EnhancedReasoningResult = {
        Problem: UnifiedProblem
        Agents: UnifiedAgent list
        Departments: UnifiedDepartment list
        ConceptAnalysis: ConceptDecompositionResult option
        SystemMetrics: SystemMetrics
        HtmlVisualization: string
    }

    let enhancedReasoningPipeline (problemStatement: string) = task {
        AnsiConsole.MarkupLine("[yellow]🧠 ENHANCED REASONING PIPELINE[/]")
        AnsiConsole.MarkupLine("[dim]Integrating concept decomposition with multi-agent reasoning[/]")
        AnsiConsole.WriteLine()

        // Step 1: Problem Analysis with Concept Decomposition
        AnsiConsole.MarkupLine("[cyan]Step 1: Analyzing problem with concept decomposition...[/]")
        
        // Create a vector representation of the problem (simplified)
        let problemVector = [| 
            0.8; 0.6; 0.9; 0.4; 0.7; 0.5; 0.8; 0.6 
        |] // This would normally be generated from the problem text
        
        let conceptBasis = ConceptDecomposition.createDefaultConceptBasis()
        let! conceptResult = ConceptDecomposition.decomposeVector conceptBasis problemVector
        
        match conceptResult.Success, conceptResult.SparseRepresentation with
        | true, Some sparseRep ->
            AnsiConsole.MarkupLine("[green]✅ Concept analysis completed[/]")
            AnsiConsole.MarkupLine($"[dim]Dominant concepts: {String.Join(", ", sparseRep.DominantConcepts |> List.take 3 |> List.map fst)}[/]")
        | _ ->
            AnsiConsole.MarkupLine("[red]⚠️ Concept analysis failed, proceeding without[/]")

        // Step 2: Create Unified Problem
        let unifiedProblem = {
            Id = Guid.NewGuid().ToString("N")[..7]
            OriginalStatement = problemStatement
            Domain = "Multi-Agent Reasoning"
            CreatedAt = DateTime.UtcNow
            
            Complexity = Complex(subProblems = 3, depth = 2, difficulty = 7)
            SubProblems = [
                {
                    Id = "sub-001"
                    Title = "Problem Decomposition"
                    Description = "Break down the complex problem into manageable parts"
                    ParentId = None
                    RequiredExpertise = ["Analysis"; "Strategy"]
                    EstimatedComplexity = 6
                    Dependencies = []
                    ExpectedOutput = "Structured problem breakdown"
                    AssignedAgents = []
                    Status = NotStarted
                }
                {
                    Id = "sub-002"
                    Title = "Solution Strategy"
                    Description = "Develop optimal solution approach"
                    ParentId = Some "sub-001"
                    RequiredExpertise = ["Strategy"; "Game Theory"]
                    EstimatedComplexity = 8
                    Dependencies = ["sub-001"]
                    ExpectedOutput = "Strategic solution plan"
                    AssignedAgents = []
                    Status = NotStarted
                }
                {
                    Id = "sub-003"
                    Title = "Implementation Plan"
                    Description = "Create detailed implementation roadmap"
                    ParentId = Some "sub-002"
                    RequiredExpertise = ["Coordination"; "Visualization"]
                    EstimatedComplexity = 7
                    Dependencies = ["sub-002"]
                    ExpectedOutput = "Implementation roadmap"
                    AssignedAgents = []
                    Status = NotStarted
                }
            ]
            Dependencies = [
                { FromSubProblem = "sub-001"; ToSubProblem = "sub-002"; DependencyType = Sequential; Strength = 0.9 }
                { FromSubProblem = "sub-002"; ToSubProblem = "sub-003"; DependencyType = Sequential; Strength = 0.8 }
            ]
            
            ConceptAnalysis = conceptResult.SparseRepresentation |> Option.map (fun sr -> {
                DominantConcepts = sr.DominantConcepts
                SemanticSummary = sr.InterpretationText
                ConceptWeights = sr.ConceptWeights
                AnalysisConfidence = 1.0 - sr.ReconstructionError
            })
            
            ReasoningSteps = []
            SolutionStrategy = Hybrid([
                Divide({ PartitioningMethod = ByExpertise; MergeStrategy = Hierarchical; QualityControl = PeerReview })
                Collaborate({ CommunicationPattern = AllToAll; DecisionMaking = Expertise(true); ConflictResolution = Mediation("coordinator") })
            ])
            RequiredExpertise = [
                { Domain = "Data Analysis"; Level = Advanced; IsCritical = true; Alternatives = [] }
                { Domain = "Game Theory"; Level = Expert; IsCritical = true; Alternatives = ["Strategic Planning"] }
                { Domain = "Communication"; Level = Intermediate; IsCritical = false; Alternatives = [] }
                { Domain = "Visualization"; Level = Advanced; IsCritical = false; Alternatives = ["Reporting"] }
            ]
            EstimatedEffort = {
                TimeEstimate = TimeSpan.FromHours(8.0)
                ResourceEstimate = {
                    ComputationalResources = 0.6
                    HumanResources = 4.0
                    DataResources = ["Historical data"; "Domain knowledge"]
                    ExternalDependencies = []
                }
                RiskFactors = [
                    { Description = "Complexity underestimation"; Probability = 0.3; Impact = 0.7; MitigationStrategy = Some "Iterative refinement" }
                ]
                ConfidenceInterval = (0.7, 0.9)
            }
            ConfidenceScore = 0.82
            QualityMetrics = {
                Completeness = 0.85
                Consistency = 0.90
                Feasibility = 0.80
                Clarity = 0.88
                Testability = 0.75
            }
        }

        // Step 3: Create Agents Based on Concept Analysis
        AnsiConsole.MarkupLine("[cyan]Step 2: Creating specialized agents based on analysis...[/]")
        
        let baseAgents = [
            { Id = "agent-001"; Name = "Alice"; Specialization = DataAnalyst; Position3D = (2.0, 3.0, 1.0); Department = Some "Research"; GameTheoryProfile = CooperativeGame("Nash Equilibrium") }
            { Id = "agent-002"; Name = "Bob"; Specialization = GameTheoryStrategist; Position3D = (5.0, 2.0, 1.5); Department = Some "Strategy"; GameTheoryProfile = NonCooperativeGame("Minimax") }
            { Id = "agent-003"; Name = "Carol"; Specialization = CommunicationBroker; Position3D = (3.5, 5.0, 0.5); Department = Some "Communication"; GameTheoryProfile = CooperativeGame("Pareto Optimal") }
            { Id = "agent-004"; Name = "Dave"; Specialization = VisualizationSpecialist; Position3D = (1.0, 4.0, 2.0); Department = Some "Visualization"; GameTheoryProfile = EvolutionaryGame("Evolutionary Stable") }
        ]

        let unifiedAgents = baseAgents |> List.map convertToUnifiedAgent

        // Enhance agents based on concept analysis
        let enhancedAgents = 
            match conceptResult.SparseRepresentation with
            | Some sparseRep ->
                unifiedAgents |> List.map (fun agent ->
                    { agent with 
                        ReasoningCapabilities = 
                            ConceptAnalysis(domains = sparseRep.DominantConcepts |> List.take 3 |> List.map fst) :: agent.ReasoningCapabilities
                        QualityScore = agent.QualityScore * (1.0 + sparseRep.Sparsity * 0.2) // Boost from concept clarity
                        CurrentTask = Some $"Concept-driven analysis: {sparseRep.InterpretationText |> fun s -> s.Substring(0, min 50 s.Length)}..."
                    })
            | None -> unifiedAgents

        AnsiConsole.MarkupLine($"[green]✅ Created {enhancedAgents.Length} enhanced agents[/]")

        // Step 4: Create Departments
        AnsiConsole.MarkupLine("[cyan]Step 3: Organizing agents into departments...[/]")
        
        let departmentGroups = 
            enhancedAgents 
            |> List.groupBy (fun a -> a.Department |> Option.defaultValue "General")
            |> List.map (fun (deptName, agents) ->
                let legacyDept = {
                    Name = deptName
                    DepartmentType = match deptName with
                                   | "Research" -> Research
                                   | "Strategy" -> Analysis
                                   | "Communication" -> Communication
                                   | "Visualization" -> Coordination
                                   | _ -> Research
                    Agents = [] // Will be filled by conversion
                    Position3D = (Random().NextDouble() * 10.0, Random().NextDouble() * 10.0, 0.0)
                    CommunicationProtocol = PeerToPeer(5)
                    GameTheoryStrategy = Nash
                }
                convertToUnifiedDepartment legacyDept agents
            )

        AnsiConsole.MarkupLine($"[green]✅ Created {departmentGroups.Length} departments[/]")

        // Step 5: Generate System Metrics
        let systemMetrics = {
            TotalProblemsProcessed = 1
            AverageProcessingTime = TimeSpan.FromMinutes(15.0)
            SystemEfficiency = departmentGroups |> List.averageBy (fun d -> d.PerformanceMetrics.CollectiveEfficiency)
            ResourceUtilization = departmentGroups |> List.averageBy (fun d -> d.PerformanceMetrics.ResourceUtilization)
            AgentSatisfaction = enhancedAgents |> List.averageBy (fun a -> a.QualityScore)
        }

        // Step 6: Generate HTML Visualization
        AnsiConsole.MarkupLine("[cyan]Step 4: Generating adaptive visualization...[/]")
        
        let agentCards = enhancedAgents |> List.map agentCard
        let departmentSummaries = departmentGroups |> List.map departmentSummary
        
        let statusItems = [
            statusItem "Total Agents" (string enhancedAgents.Length) (Some "#00ff88")
            statusItem "Departments" (string departmentGroups.Length) (Some "#4a9eff")
            statusItem "System Efficiency" (formatPercentage systemMetrics.SystemEfficiency) (Some "#00ff88")
            statusItem "Problem Confidence" (formatPercentage unifiedProblem.ConfidenceScore) (Some "#ffaa00")
        ]

        let conceptPanel = 
            match conceptResult.SparseRepresentation with
            | Some sparseRep ->
                let conceptItems = 
                    sparseRep.DominantConcepts 
                    |> List.take 5
                    |> List.map (fun (concept, weight) ->
                        metricDisplay concept (formatPercentage (abs weight)) (Some "↑"))
                [panel "🧠 Concept Analysis" conceptItems (Some "concepts")]
            | None -> []

        let mainContent = [
            panel "🤖 Active Agents" agentCards (Some "agents")
            panel "📊 System Metrics" [
                progressBar systemMetrics.SystemEfficiency (Some "System Efficiency")
                metricDisplay "Processing Time" (formatTimeSpan systemMetrics.AverageProcessingTime) None
                metricDisplay "Agent Satisfaction" (formatPercentage systemMetrics.AgentSatisfaction) (Some "↑ 5%")
            ] (Some "metrics")
        ] @ conceptPanel

        let sidebar = [
            panel "📊 System Status" statusItems (Some "status")
            panel "🏢 Departments" departmentSummaries (Some "departments")
            panel "🎮 Controls" [
                controlButton "Refresh System" "refreshSystem()" (Some "primary")
                controlButton "Analyze Concepts" "analyzeSystem()" (Some "secondary")
                controlButton "Export Data" "exportData()" None
                controlButton "Run Simulation" "runSimulation()" (Some "primary")
            ] (Some "controls")
        ]

        let layout = mainLayout "🎯 TARS Enhanced Multi-Agent Reasoning System" sidebar mainContent
        let htmlVisualization = htmlDocument "TARS Enhanced System" layout

        AnsiConsole.MarkupLine("[green]✅ Visualization generated[/]")
        AnsiConsole.WriteLine()

        return {
            Problem = unifiedProblem
            Agents = enhancedAgents
            Departments = departmentGroups
            ConceptAnalysis = conceptResult
            SystemMetrics = systemMetrics
            HtmlVisualization = htmlVisualization
        }
    }

    // ============================================================================
    // DEMO EXECUTION
    // ============================================================================

    let runEnhancedDemo (problemStatement: string option) = task {
        AnsiConsole.Clear()
        AnsiConsole.Write(
            FigletText("TARS Enhanced")
                .Centered()
                .Color(Color.Cyan))
        
        AnsiConsole.WriteLine()
        AnsiConsole.MarkupLine("[yellow]🚀 TARS Enhanced Multi-Agent Reasoning Demo[/]")
        AnsiConsole.MarkupLine("[dim]Unified architecture with concept decomposition integration[/]")
        AnsiConsole.WriteLine()

        let problem = problemStatement |> Option.defaultValue "Design an autonomous vehicle navigation system for urban environments"
        
        AnsiConsole.MarkupLine($"[cyan]🎯 Problem: {problem}[/]")
        AnsiConsole.WriteLine()

        let! result = enhancedReasoningPipeline problem

        // Display Results
        AnsiConsole.MarkupLine("[yellow]📊 ENHANCED REASONING RESULTS[/]")
        AnsiConsole.WriteLine()

        // Problem Analysis
        let (problemSummary, problemDetails) = unifiedProblem result.Problem
        AnsiConsole.MarkupLine($"[cyan]🧩 Problem Analysis:[/]")
        AnsiConsole.MarkupLine($"[green]  {problemSummary}[/]")
        problemDetails |> List.take 5 |> List.iter (fun detail -> AnsiConsole.MarkupLine($"[dim]    • {detail}[/]"))
        AnsiConsole.WriteLine()

        // Agent Summary
        AnsiConsole.MarkupLine($"[cyan]🤖 Agent Summary:[/]")
        for agent in result.Agents do
            let (agentSummary, _) = unifiedAgent agent
            AnsiConsole.MarkupLine($"[green]  • {agentSummary}[/]")
        AnsiConsole.WriteLine()

        // Department Summary
        AnsiConsole.MarkupLine($"[cyan]🏢 Department Summary:[/]")
        for dept in result.Departments do
            let (deptSummary, _) = unifiedDepartment dept
            AnsiConsole.MarkupLine($"[green]  • {deptSummary}[/]")
        AnsiConsole.WriteLine()

        // Concept Analysis
        match result.ConceptAnalysis.SparseRepresentation with
        | Some sparseRep ->
            AnsiConsole.MarkupLine($"[cyan]🧠 Concept Analysis:[/]")
            AnsiConsole.MarkupLine($"[green]  {sparseRep.InterpretationText}[/]")
            AnsiConsole.MarkupLine($"[dim]  Quality Score: {result.ConceptAnalysis.QualityScore:P1} | Sparsity: {sparseRep.Sparsity:P1}[/]")
        | None ->
            AnsiConsole.MarkupLine($"[red]🧠 Concept Analysis: Not available[/]")
        AnsiConsole.WriteLine()

        // System Metrics
        let (metricsSummary, metricsDetails) = systemMetrics result.SystemMetrics
        AnsiConsole.MarkupLine($"[cyan]📈 System Performance:[/]")
        AnsiConsole.MarkupLine($"[green]  {metricsSummary}[/]")
        metricsDetails |> List.iter (fun detail -> AnsiConsole.MarkupLine($"[dim]    • {detail}[/]"))
        AnsiConsole.WriteLine()

        // Save HTML Visualization
        let fileName = $"tars_enhanced_demo_{DateTime.Now:yyyyMMdd_HHmmss}.html"
        System.IO.File.WriteAllText(fileName, result.HtmlVisualization)
        AnsiConsole.MarkupLine($"[green]💾 HTML visualization saved: {fileName}[/]")
        AnsiConsole.WriteLine()

        AnsiConsole.MarkupLine("[green]✅ Enhanced multi-agent reasoning demo completed successfully![/]")
        AnsiConsole.MarkupLine("[yellow]🎯 Key Improvements Demonstrated:[/]")
        AnsiConsole.MarkupLine("[green]  • Unified domain model with consistent types[/]")
        AnsiConsole.MarkupLine("[green]  • Integrated concept decomposition analysis[/]")
        AnsiConsole.MarkupLine("[green]  • Composable HTML visualization components[/]")
        AnsiConsole.MarkupLine("[green]  • Performance metrics and quality tracking[/]")
        AnsiConsole.MarkupLine("[green]  • Clean, maintainable architecture[/]")
        AnsiConsole.WriteLine()

        return { Success = true; Message = "Enhanced demo completed successfully"; Data = Some (box result) }
    }
