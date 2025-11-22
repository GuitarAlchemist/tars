namespace TarsEngine.FSharp.Notebooks.Generation

open System
open System.IO
open System.Text.RegularExpressions
open TarsEngine.FSharp.Notebooks.Types
open TarsEngine.FSharp.Core
open TarsEngine.FSharp.Metascript

/// <summary>
/// Analyzes TARS metascripts to extract information for notebook generation
/// </summary>

/// Metascript analysis result
type MetascriptAnalysis = {
    FilePath: string
    Agents: AgentDefinition list
    Variables: VariableDefinition list
    Actions: ActionDefinition list
    DataSources: DataSourceDefinition list
    Dependencies: DependencyGraph
    Narrative: NarrativeStructure
    Complexity: ComplexityMetrics
}

/// Agent definition extracted from metascript
and AgentDefinition = {
    Name: string
    Type: string
    Capabilities: string list
    Configuration: Map<string, obj>
    Dependencies: string list
    Description: string option
}

/// Variable definition
and VariableDefinition = {
    Name: string
    Type: string
    Value: obj option
    Description: string option
    Usage: VariableUsage list
}

/// Variable usage context
and VariableUsage = {
    Context: string
    Operation: string
    LineNumber: int
}

/// Action definition
and ActionDefinition = {
    Name: string
    Type: string
    Parameters: Map<string, obj>
    Dependencies: string list
    ExpectedOutput: string option
    Description: string option
}

/// Data source definition
and DataSourceDefinition = {
    Name: string
    Type: string
    Location: string
    Schema: Map<string, string> option
    AccessMethod: string
    Description: string option
}

/// Dependency graph
and DependencyGraph = {
    Nodes: string list
    Edges: (string * string) list
    Levels: Map<string, int>
}

/// Narrative structure for documentation
and NarrativeStructure = {
    Title: string
    Objective: string
    Sections: NarrativeSection list
    Conclusion: string option
}

/// Narrative section
and NarrativeSection = {
    Title: string
    Content: string
    CodeBlocks: string list
    Order: int
}

/// Complexity metrics
and ComplexityMetrics = {
    AgentCount: int
    ActionCount: int
    VariableCount: int
    DependencyDepth: int
    EstimatedExecutionTime: TimeSpan
    ComplexityScore: float
}

/// Metascript analyzer
module MetascriptAnalyzer =
    
    /// Analyze a TARS metascript file
    let analyzeMetascript (metascriptPath: string) : Async<MetascriptAnalysis> = async {
        try
            if not (File.Exists(metascriptPath)) then
                failwith $"Metascript file not found: {metascriptPath}"
            
            let content = File.ReadAllText(metascriptPath)
            let lines = content.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
            
            // Extract different components
            let agents = extractAgents lines
            let variables = extractVariables lines
            let actions = extractActions lines
            let dataSources = extractDataSources lines
            let dependencies = buildDependencyGraph agents actions variables
            let narrative = extractNarrative content agents actions
            let complexity = calculateComplexity agents actions variables dependencies
            
            return {
                FilePath = metascriptPath
                Agents = agents
                Variables = variables
                Actions = actions
                DataSources = dataSources
                Dependencies = dependencies
                Narrative = narrative
                Complexity = complexity
            }
            
        with
        | ex ->
            failwith $"Failed to analyze metascript {metascriptPath}: {ex.Message}"
    }
    
    /// Extract agent definitions from metascript
    let private extractAgents (lines: string[]) : AgentDefinition list =
        let agentPattern = Regex(@"AGENT\s+(\w+)\s*\{", RegexOptions.IgnoreCase)
        let typePattern = Regex(@"type:\s*""([^""]+)""", RegexOptions.IgnoreCase)
        let capabilityPattern = Regex(@"capabilities:\s*\[(.*?)\]", RegexOptions.IgnoreCase)
        
        let agents = ResizeArray<AgentDefinition>()
        let mutable currentAgent: AgentDefinition option = None
        let mutable inAgentBlock = false
        let mutable braceCount = 0
        
        for i, line in lines |> Array.indexed do
            let trimmedLine = line.Trim()
            
            // Check for agent start
            let agentMatch = agentPattern.Match(trimmedLine)
            if agentMatch.Success then
                let agentName = agentMatch.Groups.[1].Value
                currentAgent <- Some {
                    Name = agentName
                    Type = "unknown"
                    Capabilities = []
                    Configuration = Map.empty
                    Dependencies = []
                    Description = None
                }
                inAgentBlock <- true
                braceCount <- 1
            
            elif inAgentBlock && currentAgent.IsSome then
                // Count braces to track block end
                braceCount <- braceCount + (trimmedLine |> Seq.filter ((=) '{') |> Seq.length)
                braceCount <- braceCount - (trimmedLine |> Seq.filter ((=) '}') |> Seq.length)
                
                let agent = currentAgent.Value
                
                // Extract type
                let typeMatch = typePattern.Match(trimmedLine)
                if typeMatch.Success then
                    currentAgent <- Some { agent with Type = typeMatch.Groups.[1].Value }
                
                // Extract capabilities
                let capabilityMatch = capabilityPattern.Match(trimmedLine)
                if capabilityMatch.Success then
                    let capabilitiesStr = capabilityMatch.Groups.[1].Value
                    let capabilities = 
                        capabilitiesStr.Split([|','; ';'|], StringSplitOptions.RemoveEmptyEntries)
                        |> Array.map (fun s -> s.Trim().Trim('"'))
                        |> Array.toList
                    currentAgent <- Some { agent with Capabilities = capabilities }
                
                // Check for block end
                if braceCount <= 0 then
                    agents.Add(currentAgent.Value)
                    currentAgent <- None
                    inAgentBlock <- false
        
        agents |> List.ofSeq
    
    /// Extract variable definitions
    let private extractVariables (lines: string[]) : VariableDefinition list =
        let variablePattern = Regex(@"VARIABLE\s+(\w+)\s*=\s*(.+)", RegexOptions.IgnoreCase)
        let variables = ResizeArray<VariableDefinition>()
        
        for i, line in lines |> Array.indexed do
            let trimmedLine = line.Trim()
            let varMatch = variablePattern.Match(trimmedLine)
            
            if varMatch.Success then
                let varName = varMatch.Groups.[1].Value
                let varValue = varMatch.Groups.[2].Value.Trim()
                
                // Determine type from value
                let varType = 
                    if varValue.StartsWith("\"") && varValue.EndsWith("\"") then "string"
                    elif varValue.Contains(".") && Double.TryParse(varValue) |> fst then "float"
                    elif Int32.TryParse(varValue) |> fst then "int"
                    elif varValue.ToLower() = "true" || varValue.ToLower() = "false" then "bool"
                    elif varValue.StartsWith("[") && varValue.EndsWith("]") then "array"
                    elif varValue.StartsWith("{") && varValue.EndsWith("}") then "object"
                    else "unknown"
                
                variables.Add({
                    Name = varName
                    Type = varType
                    Value = Some (box varValue)
                    Description = None
                    Usage = []
                })
        
        variables |> List.ofSeq
    
    /// Extract action definitions
    let private extractActions (lines: string[]) : ActionDefinition list =
        let actionPattern = Regex(@"ACTION\s+(\w+)\s*\{", RegexOptions.IgnoreCase)
        let actions = ResizeArray<ActionDefinition>()
        let mutable currentAction: ActionDefinition option = None
        let mutable inActionBlock = false
        let mutable braceCount = 0
        
        for i, line in lines |> Array.indexed do
            let trimmedLine = line.Trim()
            
            // Check for action start
            let actionMatch = actionPattern.Match(trimmedLine)
            if actionMatch.Success then
                let actionName = actionMatch.Groups.[1].Value
                currentAction <- Some {
                    Name = actionName
                    Type = "unknown"
                    Parameters = Map.empty
                    Dependencies = []
                    ExpectedOutput = None
                    Description = None
                }
                inActionBlock <- true
                braceCount <- 1
            
            elif inActionBlock && currentAction.IsSome then
                // Count braces to track block end
                braceCount <- braceCount + (trimmedLine |> Seq.filter ((=) '{') |> Seq.length)
                braceCount <- braceCount - (trimmedLine |> Seq.filter ((=) '}') |> Seq.length)
                
                // Check for block end
                if braceCount <= 0 then
                    actions.Add(currentAction.Value)
                    currentAction <- None
                    inActionBlock <- false
        
        actions |> List.ofSeq
    
    /// Extract data source definitions
    let private extractDataSources (lines: string[]) : DataSourceDefinition list =
        let dataSourcePattern = Regex(@"DATA_SOURCE\s+(\w+)\s*\{", RegexOptions.IgnoreCase)
        let dataSources = ResizeArray<DataSourceDefinition>()
        
        for i, line in lines |> Array.indexed do
            let trimmedLine = line.Trim()
            let dsMatch = dataSourcePattern.Match(trimmedLine)
            
            if dsMatch.Success then
                let dsName = dsMatch.Groups.[1].Value
                dataSources.Add({
                    Name = dsName
                    Type = "unknown"
                    Location = ""
                    Schema = None
                    AccessMethod = "unknown"
                    Description = None
                })
        
        dataSources |> List.ofSeq
    
    /// Build dependency graph
    let private buildDependencyGraph (agents: AgentDefinition list) (actions: ActionDefinition list) (variables: VariableDefinition list) : DependencyGraph =
        let nodes = 
            (agents |> List.map (fun a -> a.Name)) @
            (actions |> List.map (fun a -> a.Name)) @
            (variables |> List.map (fun v -> v.Name))
        
        let edges = 
            agents
            |> List.collect (fun agent -> 
                agent.Dependencies |> List.map (fun dep -> (dep, agent.Name)))
        
        // Calculate dependency levels (topological sort)
        let levels = 
            nodes
            |> List.mapi (fun i node -> (node, i))
            |> Map.ofList
        
        {
            Nodes = nodes
            Edges = edges
            Levels = levels
        }
    
    /// Extract narrative structure
    let private extractNarrative (content: string) (agents: AgentDefinition list) (actions: ActionDefinition list) : NarrativeStructure =
        // Extract title from first comment or filename
        let titlePattern = Regex(@"//\s*Title:\s*(.+)", RegexOptions.IgnoreCase)
        let titleMatch = titlePattern.Match(content)
        let title = 
            if titleMatch.Success then titleMatch.Groups.[1].Value.Trim()
            else "TARS Metascript Analysis"
        
        // Extract objective
        let objectivePattern = Regex(@"//\s*Objective:\s*(.+)", RegexOptions.IgnoreCase)
        let objectiveMatch = objectivePattern.Match(content)
        let objective = 
            if objectiveMatch.Success then objectiveMatch.Groups.[1].Value.Trim()
            else "Automated analysis and execution of TARS metascript"
        
        // Create sections based on agents and actions
        let sections = [
            {
                Title = "Agent Overview"
                Content = $"This metascript defines {agents.Length} agents for automated processing."
                CodeBlocks = agents |> List.map (fun a -> $"AGENT {a.Name} // Type: {a.Type}")
                Order = 1
            }
            {
                Title = "Action Workflow"
                Content = $"The workflow consists of {actions.Length} actions executed in sequence."
                CodeBlocks = actions |> List.map (fun a -> $"ACTION {a.Name}")
                Order = 2
            }
        ]
        
        {
            Title = title
            Objective = objective
            Sections = sections
            Conclusion = Some "This analysis provides the foundation for automated notebook generation."
        }
    
    /// Calculate complexity metrics
    let private calculateComplexity (agents: AgentDefinition list) (actions: ActionDefinition list) (variables: VariableDefinition list) (dependencies: DependencyGraph) : ComplexityMetrics =
        let agentCount = agents.Length
        let actionCount = actions.Length
        let variableCount = variables.Length
        let dependencyDepth = if dependencies.Levels.IsEmpty then 0 else dependencies.Levels.Values |> Seq.max
        
        // Estimate execution time based on complexity
        let baseTime = TimeSpan.FromMinutes(1.0)
        let agentTime = TimeSpan.FromMinutes(float agentCount * 0.5)
        let actionTime = TimeSpan.FromMinutes(float actionCount * 0.3)
        let estimatedTime = baseTime + agentTime + actionTime
        
        // Calculate complexity score (0-100)
        let complexityScore = 
            let agentScore = Math.Min(float agentCount * 10.0, 40.0)
            let actionScore = Math.Min(float actionCount * 5.0, 30.0)
            let dependencyScore = Math.Min(float dependencyDepth * 5.0, 20.0)
            let variableScore = Math.Min(float variableCount * 2.0, 10.0)
            agentScore + actionScore + dependencyScore + variableScore
        
        {
            AgentCount = agentCount
            ActionCount = actionCount
            VariableCount = variableCount
            DependencyDepth = dependencyDepth
            EstimatedExecutionTime = estimatedTime
            ComplexityScore = complexityScore
        }
    
    /// Generate notebook requirements from analysis
    let generateNotebookRequirements (analysis: MetascriptAnalysis) (strategy: NotebookGenerationStrategy) : NotebookRequirements =
        let kernelType = 
            match strategy with
            | ExploratoryDataAnalysis -> Python { Version = "3.9"; Packages = ["pandas"; "numpy"; "matplotlib"; "seaborn"]; VirtualEnv = None }
            | MachineLearningPipeline -> Python { Version = "3.9"; Packages = ["scikit-learn"; "tensorflow"; "pandas"; "numpy"]; VirtualEnv = None }
            | ResearchNotebook -> Python { Version = "3.9"; Packages = ["scipy"; "pandas"; "matplotlib"; "jupyter"]; VirtualEnv = None }
            | TutorialNotebook -> Python { Version = "3.9"; Packages = ["pandas"; "matplotlib"]; VirtualEnv = None }
            | DocumentationNotebook -> Python { Version = "3.9"; Packages = ["pandas"]; VirtualEnv = None }
            | BusinessReport -> Python { Version = "3.9"; Packages = ["pandas"; "plotly"; "openpyxl"]; VirtualEnv = None }
            | AcademicPaper -> Python { Version = "3.9"; Packages = ["scipy"; "pandas"; "matplotlib"; "seaborn"]; VirtualEnv = None }
        
        {
            Title = analysis.Narrative.Title
            Description = analysis.Narrative.Objective
            Strategy = strategy
            KernelType = kernelType
            EstimatedCells = analysis.Complexity.AgentCount + analysis.Complexity.ActionCount + 3 // +3 for intro, conclusion, imports
            DataSources = analysis.DataSources |> List.map (fun ds -> ds.Name)
            RequiredPackages = []
            Complexity = analysis.Complexity.ComplexityScore
        }

/// Notebook requirements for generation
and NotebookRequirements = {
    Title: string
    Description: string
    Strategy: NotebookGenerationStrategy
    KernelType: SupportedKernel
    EstimatedCells: int
    DataSources: string list
    RequiredPackages: string list
    Complexity: float
}
