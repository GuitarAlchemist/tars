namespace TarsEngine.FSharp.Cli.Core

open System
open TarsEngine.FSharp.Cli.Core.MultiAgentDomain
open TarsEngine.FSharp.Core.Agents.AgentSystem

// ============================================================================
// UNIFIED FORMATTING MODULE - ELIMINATES CODE DUPLICATION
// ============================================================================

module UnifiedFormatting =

    // ============================================================================
    // CORE FORMATTING TYPES
    // ============================================================================

    type Formatter<'T> = 'T -> string
    type ColorFormatter<'T> = 'T -> string * string // (text, color)
    type DetailedFormatter<'T> = 'T -> string * string list // (summary, details)

    // ============================================================================
    // AGENT FORMATTING
    // ============================================================================

    let agentSpecialization: Formatter<AgentSpecialization> = function
        | DataAnalyst -> "Data Analyst"
        | GameTheoryStrategist -> "Game Theory Strategist"
        | CommunicationBroker -> "Communication Broker"
        | VisualizationSpecialist -> "Visualization Specialist"
        | Coordinator -> "Coordinator"
        | DomainExpert(domain) -> $"Domain Expert ({domain})"

    let agentSpecializationWithColor: ColorFormatter<AgentSpecialization> = function
        | DataAnalyst -> ("Data Analyst", "green")
        | GameTheoryStrategist -> ("Game Theory Strategist", "blue")
        | CommunicationBroker -> ("Communication Broker", "red")
        | VisualizationSpecialist -> ("Visualization Specialist", "purple")
        | Coordinator -> ("Coordinator", "yellow")
        | DomainExpert(domain) -> ($"Domain Expert ({domain})", "cyan")

    let agentStatus: Formatter<AgentStatus> = function
        | Idle -> "Idle"
        | Working(task) -> $"Working: {task}"
        | Communicating(target) -> $"Communicating with {target}"
        | Reasoning(problem) -> $"Reasoning about: {problem}"
        | Completed(result) -> $"Completed: {result}"

    let agentStatusWithColor: ColorFormatter<AgentStatus> = function
        | Idle -> ("Idle", "dim")
        | Working(task) -> ($"Working: {task}", "yellow")
        | Communicating(target) -> ($"Communicating with {target}", "blue")
        | Reasoning(problem) -> ($"Reasoning about: {problem}", "purple")
        | Completed(result) -> ($"Completed: {result}", "green")

    let reasoningCapability: Formatter<ReasoningCapability> = function
        | ProblemDecomposition(complexity) -> sprintf "Problem Decomposition (Level %d)" complexity
        | ConceptAnalysis(domains) -> "Concept Analysis (" + String.Join(", ", domains) + ")"
        | PatternRecognition(accuracy) -> sprintf "Pattern Recognition (%.1f%%)" (accuracy * 100.0)
        | StrategicPlanning(horizon) -> sprintf "Strategic Planning (%.1fh horizon)" horizon.TotalHours
        | KnowledgeIntegration(sources) -> sprintf "Knowledge Integration (%d sources)" sources.Length

    let performanceMetrics: DetailedFormatter<PerformanceMetrics> = fun metrics ->
        let summary = sprintf "Quality: %.1f%% | Success: %.1f%%" (metrics.ReasoningAccuracy * 100.0) (metrics.SuccessRate * 100.0)
        let details = [
            sprintf "Tasks Completed: %d" metrics.TasksCompleted
            sprintf "Avg Response Time: %.0fms" metrics.AverageResponseTime.TotalMilliseconds
            sprintf "Success Rate: %.1f%%" (metrics.SuccessRate * 100.0)
            sprintf "Communication Efficiency: %.1f%%" (metrics.CommunicationEfficiency * 100.0)
            sprintf "Reasoning Accuracy: %.1f%%" (metrics.ReasoningAccuracy * 100.0)
        ]
        (summary, details)

    // ============================================================================
    // SPATIAL & GEOMETRIC FORMATTING
    // ============================================================================

    let position3D: Formatter<float * float * float> = fun (x, y, z) ->
        sprintf "(%.1f, %.1f, %.1f)" x y z

    let position3DDetailed: DetailedFormatter<float * float * float> = fun (x, y, z) ->
        let summary = sprintf "(%.1f, %.1f, %.1f)" x y z
        let details = [
            sprintf "X: %.2f" x
            sprintf "Y: %.2f" y
            sprintf "Z: %.2f" z
            sprintf "Distance from origin: %.2f" (sqrt(x*x + y*y + z*z))
        ]
        (summary, details)

    // ============================================================================
    // GAME THEORY FORMATTING
    // ============================================================================

    let gameTheoryModel: Formatter<GameTheoryModel> = function
        | CooperativeGame(strategy) -> $"Cooperative: {strategy}"
        | NonCooperativeGame(strategy) -> $"Non-Cooperative: {strategy}"
        | EvolutionaryGame(strategy) -> $"Evolutionary: {strategy}"
        | AuctionMechanism(mechanism) -> $"Auction: {mechanism}"

    let gameTheoryModelWithColor: ColorFormatter<GameTheoryModel> = function
        | CooperativeGame(strategy) -> ($"Cooperative: {strategy}", "green")
        | NonCooperativeGame(strategy) -> ($"Non-Cooperative: {strategy}", "blue")
        | EvolutionaryGame(strategy) -> ($"Evolutionary: {strategy}", "purple")
        | AuctionMechanism(mechanism) -> ($"Auction: {mechanism}", "red")

    let gameTheoryStrategy: Formatter<GameTheoryStrategy> = function
        | Nash -> "Nash Equilibrium"
        | Pareto -> "Pareto Optimal"
        | Minimax -> "Minimax"
        | Cooperative -> "Cooperative"
        | Evolutionary -> "Evolutionary Stable"

    let strategyPreference: Formatter<StrategyPreference> = function
        | Cooperative(weight) -> $"Cooperative ({weight:P0})"
        | Competitive(weight) -> $"Competitive ({weight:P0})"
        | Adaptive(threshold) -> $"Adaptive (threshold: {threshold:F2})"
        | Specialized(domain) -> $"Specialized in {domain}"

    // ============================================================================
    // DEPARTMENT FORMATTING
    // ============================================================================

    let departmentType: Formatter<DepartmentType> = function
        | Research -> "Research"
        | Analysis -> "Analysis"
        | Coordination -> "Coordination"
        | Communication -> "Communication"
        | Specialized(domain) -> $"Specialized ({domain})"

    let communicationProtocol: Formatter<CommunicationProtocol> = function
        | Hierarchical(levels) -> $"Hierarchical ({levels} levels)"
        | PeerToPeer(maxConnections) -> $"Peer-to-Peer (max {maxConnections})"
        | Broadcast(scope) -> $"Broadcast ({scope})"
        | GameTheoretic(mechanism) -> $"Game Theoretic ({mechanism})"
        | Mesh(redundancy) -> $"Mesh (redundancy: {redundancy})"

    let coordinationStrategy: Formatter<CoordinationStrategy> = function
        | Centralized(coordinator) -> $"Centralized (coordinator: {coordinator})"
        | Distributed(consensus) -> $"Distributed ({consensus})"
        | Hybrid(primary, backup) -> $"Hybrid (primary: {primary}, backup: {backup.Length})"

    let departmentPerformanceMetrics: DetailedFormatter<DepartmentPerformanceMetrics> = fun metrics ->
        let summary = $"Efficiency: {metrics.CollectiveEfficiency:P1} | Coordination: {metrics.InterAgentCoordination:P1}"
        let details = [
            $"Collective Efficiency: {metrics.CollectiveEfficiency:P1}"
            $"Inter-Agent Coordination: {metrics.InterAgentCoordination:P1}"
            $"Goal Completion Rate: {metrics.GoalCompletionRate:P1}"
            $"Communication Overhead: {metrics.CommunicationOverhead:P1}"
            $"Resource Utilization: {metrics.ResourceUtilization:P1}"
        ]
        (summary, details)

    // ============================================================================
    // PROBLEM FORMATTING
    // ============================================================================

    let problemComplexity: Formatter<ProblemComplexity> = function
        | Simple(difficulty) -> $"Simple (difficulty: {difficulty})"
        | Moderate(subProblems, difficulty) -> $"Moderate ({subProblems} sub-problems, difficulty: {difficulty})"
        | Complex(subProblems, depth, difficulty) -> $"Complex ({subProblems} sub-problems, depth: {depth}, difficulty: {difficulty})"
        | Adaptive(baseComplexity, factors) -> $"Adaptive (base: {problemComplexity baseComplexity}, {factors.Length} factors)"

    let subProblemStatus: ColorFormatter<SubProblemStatus> = function
        | NotStarted -> ("Not Started", "dim")
        | InProgress(progress) -> ($"In Progress ({progress:P0})", "yellow")
        | Blocked(reason) -> ($"Blocked: {reason}", "red")
        | Completed(result) -> ($"Completed: {result}", "green")
        | Failed(error) -> ($"Failed: {error}", "red")

    let dependencyType: Formatter<DependencyType> = function
        | Sequential -> "Sequential"
        | Parallel -> "Parallel"
        | Conditional(condition) -> $"Conditional ({condition})"
        | Resource(resource) -> $"Resource ({resource})"

    let reasoningType: Formatter<ReasoningType> = function
        | Deductive -> "Deductive"
        | Inductive -> "Inductive"
        | Abductive -> "Abductive"
        | Analogical -> "Analogical"
        | Causal -> "Causal"

    let solutionStrategy: Formatter<SolutionStrategy> = function
        | Divide(approach) -> $"Divide & Conquer ({approach.PartitioningMethod})"
        | Collaborate(coordination) -> $"Collaborative ({coordination.CommunicationPattern})"
        | Iterate(cycles) -> $"Iterative (max {cycles.MaxIterations} cycles)"
        | Hybrid(strategies) -> $"Hybrid ({strategies.Length} strategies)"

    let expertiseLevel: ColorFormatter<ExpertiseLevel> = function
        | Novice -> ("Novice", "red")
        | Intermediate -> ("Intermediate", "yellow")
        | Advanced -> ("Advanced", "blue")
        | Expert -> ("Expert", "green")

    // ============================================================================
    // SYSTEM FORMATTING
    // ============================================================================

    let systemStatus: ColorFormatter<SystemStatus> = function
        | Initializing -> ("Initializing", "yellow")
        | Ready -> ("Ready", "green")
        | Processing(problems) -> ($"Processing {problems.Length} problems", "blue")
        | Optimizing -> ("Optimizing", "purple")
        | Error(error) -> ($"Error: {error}", "red")

    let systemMetrics: DetailedFormatter<SystemMetrics> = fun metrics ->
        let summary = $"Efficiency: {metrics.SystemEfficiency:P1} | Processed: {metrics.TotalProblemsProcessed}"
        let details = [
            $"Total Problems Processed: {metrics.TotalProblemsProcessed}"
            $"Average Processing Time: {metrics.AverageProcessingTime.TotalMinutes:F1} minutes"
            $"System Efficiency: {metrics.SystemEfficiency:P1}"
            $"Resource Utilization: {metrics.ResourceUtilization:P1}"
            $"Agent Satisfaction: {metrics.AgentSatisfaction:P1}"
        ]
        (summary, details)

    // ============================================================================
    // UNIFIED AGENT FORMATTING
    // ============================================================================

    let unifiedAgent: DetailedFormatter<UnifiedAgent> = fun agent ->
        let summary = $"{agent.Name} ({agentSpecialization agent.Specialization}) - {fst (agentStatusWithColor agent.Status)}"
        let details = [
            $"ID: {agent.Id}"
            $"Specialization: {agentSpecialization agent.Specialization}"
            $"Status: {agentStatus agent.Status}"
            $"Position: {position3D agent.Position3D}"
            sprintf "Department: %s" (agent.Department |> Option.defaultValue "None")
            $"Game Theory: {gameTheoryModel agent.GameTheoryProfile}"
            $"Quality Score: {agent.QualityScore:P1}"
            $"Progress: {agent.Progress:P0}"
        ]
        (summary, details)

    let unifiedDepartment: DetailedFormatter<UnifiedDepartment> = fun dept ->
        let summary = $"{dept.Name} ({departmentType dept.DepartmentType}) - {dept.Agents.Length} agents"
        let details = [
            $"ID: {dept.Id}"
            $"Type: {departmentType dept.DepartmentType}"
            $"Agents: {dept.Agents.Length}"
            $"Position: {position3D dept.Position3D}"
            $"Communication: {communicationProtocol dept.CommunicationProtocol}"
            $"Coordination: {coordinationStrategy dept.CoordinationStrategy}"
            $"Game Theory: {gameTheoryStrategy dept.GameTheoryStrategy}"
            $"Goals: {dept.CollectiveGoals.Length}"
        ]
        (summary, details)

    let unifiedProblem: DetailedFormatter<UnifiedProblem> = fun problem ->
        let summary = $"{problem.OriginalStatement} (Complexity: {problemComplexity problem.Complexity})"
        let details = [
            $"ID: {problem.Id}"
            $"Domain: {problem.Domain}"
            $"Complexity: {problemComplexity problem.Complexity}"
            $"Sub-problems: {problem.SubProblems.Length}"
            $"Dependencies: {problem.Dependencies.Length}"
            $"Strategy: {solutionStrategy problem.SolutionStrategy}"
            $"Confidence: {problem.ConfidenceScore:P1}"
            $"Required Expertise: {problem.RequiredExpertise.Length} areas"
        ]
        (summary, details)

    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================

    let formatWithColor (text: string) (color: string) : string =
        $"[{color}]{text}[/]"

    let formatList (items: string list) (separator: string) : string =
        String.Join(separator, items)

    let formatProgress (progress: float) : string =
        let percentage = progress * 100.0
        let barLength = 20
        let filledLength = int (progress * float barLength)
        let bar = String.replicate filledLength "█" + String.replicate (barLength - filledLength) "░"
        $"{bar} {percentage:F1}%"

    let formatTimeSpan (timeSpan: TimeSpan) : string =
        if timeSpan.TotalDays >= 1.0 then
            $"{timeSpan.TotalDays:F1} days"
        elif timeSpan.TotalHours >= 1.0 then
            $"{timeSpan.TotalHours:F1} hours"
        elif timeSpan.TotalMinutes >= 1.0 then
            $"{timeSpan.TotalMinutes:F1} minutes"
        else
            $"{timeSpan.TotalSeconds:F1} seconds"

    let formatPercentage (value: float) : string =
        $"{value:P1}"

    let formatNumber (value: float) (decimals: int) : string =
        Math.Round(value, decimals).ToString($"F{decimals}")

    let truncateText (text: string) (maxLength: int) : string =
        if text.Length <= maxLength then text
        else text.Substring(0, maxLength - 3) + "..."
