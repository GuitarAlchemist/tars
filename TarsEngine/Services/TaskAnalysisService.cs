using System.Text;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for analyzing TODO tasks and generating implementation plans
/// </summary>
public class TaskAnalysisService : ITaskAnalysisService
{
    private readonly ILogger<TaskAnalysisService> _logger;
    private readonly ILlmService _llmService;
    private readonly ICodeAnalysisService _codeAnalysisService;
    private readonly IProjectAnalysisService _projectAnalysisService;

    /// <summary>
    /// Initializes a new instance of the <see cref="TaskAnalysisService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="llmService">The LLM service</param>
    /// <param name="codeAnalysisService">The code analysis service</param>
    /// <param name="projectAnalysisService">The project analysis service</param>
    public TaskAnalysisService(
        ILogger<TaskAnalysisService> logger,
        ILlmService llmService,
        ICodeAnalysisService codeAnalysisService,
        IProjectAnalysisService projectAnalysisService)
    {
        _logger = logger;
        _llmService = llmService;
        _codeAnalysisService = codeAnalysisService;
        _projectAnalysisService = projectAnalysisService;
    }

    /// <summary>
    /// Analyze a TODO task and generate an implementation plan
    /// </summary>
    /// <param name="taskDescription">The task description</param>
    /// <returns>The implementation plan</returns>
    public async Task<ImplementationPlan> AnalyzeTaskAsync(string taskDescription)
    {
        _logger.LogInformation($"Analyzing task: {taskDescription}");

        try
        {
            // Step 1: Extract key requirements from the task description
            var requirements = await ExtractRequirementsAsync(taskDescription);
            
            // Step 2: Identify affected components and files
            var affectedComponents = await IdentifyAffectedComponentsAsync(taskDescription, requirements);
            
            // Step 3: Generate implementation steps
            var implementationSteps = await GenerateImplementationStepsAsync(taskDescription, requirements, affectedComponents);
            
            // Step 4: Estimate complexity and effort
            var complexity = EstimateComplexity(requirements, affectedComponents, implementationSteps);
            
            // Create the implementation plan
            var plan = new ImplementationPlan
            {
                TaskDescription = taskDescription,
                Requirements = requirements,
                AffectedComponents = affectedComponents,
                ImplementationSteps = implementationSteps,
                Complexity = complexity,
                CreatedAt = DateTime.Now
            };
            
            _logger.LogInformation($"Generated implementation plan for task: {taskDescription}");
            return plan;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error analyzing task: {taskDescription}");
            throw;
        }
    }

    /// <summary>
    /// Extract key requirements from a task description
    /// </summary>
    /// <param name="taskDescription">The task description</param>
    /// <returns>The list of requirements</returns>
    private async Task<List<string>> ExtractRequirementsAsync(string taskDescription)
    {
        _logger.LogDebug($"Extracting requirements from task: {taskDescription}");
        
        var prompt = new StringBuilder();
        prompt.AppendLine("Extract the key requirements from the following task description.");
        prompt.AppendLine("Format each requirement as a separate bullet point.");
        prompt.AppendLine("Focus on functional requirements, technical constraints, and expected outcomes.");
        prompt.AppendLine();
        prompt.AppendLine($"Task: {taskDescription}");
        
        var response = await _llmService.GenerateTextAsync(prompt.ToString());
        
        // Parse the response into a list of requirements
        var requirements = new List<string>();
        foreach (var line in response.Split('\n'))
        {
            var trimmedLine = line.Trim();
            if (trimmedLine.StartsWith("-") || trimmedLine.StartsWith("*"))
            {
                requirements.Add(trimmedLine.Substring(1).Trim());
            }
        }
        
        _logger.LogDebug($"Extracted {requirements.Count} requirements from task");
        return requirements;
    }

    /// <summary>
    /// Identify components and files affected by the task
    /// </summary>
    /// <param name="taskDescription">The task description</param>
    /// <param name="requirements">The list of requirements</param>
    /// <returns>The list of affected components</returns>
    private async Task<List<AffectedComponent>> IdentifyAffectedComponentsAsync(string taskDescription, List<string> requirements)
    {
        _logger.LogDebug($"Identifying affected components for task: {taskDescription}");
        
        // Get project structure
        var projectStructure = await _projectAnalysisService.GetProjectStructureAsync();
        
        // Build a prompt to identify affected components
        var prompt = new StringBuilder();
        prompt.AppendLine("Identify the components and files that need to be modified to implement the following task:");
        prompt.AppendLine($"Task: {taskDescription}");
        prompt.AppendLine();
        prompt.AppendLine("Requirements:");
        foreach (var requirement in requirements)
        {
            prompt.AppendLine($"- {requirement}");
        }
        prompt.AppendLine();
        prompt.AppendLine("Project Structure:");
        prompt.AppendLine(projectStructure.ToString());
        prompt.AppendLine();
        prompt.AppendLine("For each affected component, specify:");
        prompt.AppendLine("1. The component name");
        prompt.AppendLine("2. The file path");
        prompt.AppendLine("3. The type of change needed (Create, Modify, Delete)");
        prompt.AppendLine("4. A brief description of the changes needed");
        
        var response = await _llmService.GenerateTextAsync(prompt.ToString());
        
        // Parse the response into a list of affected components
        var affectedComponents = ParseAffectedComponents(response);
        
        _logger.LogDebug($"Identified {affectedComponents.Count} affected components for task");
        return affectedComponents;
    }

    /// <summary>
    /// Parse the LLM response into a list of affected components
    /// </summary>
    /// <param name="response">The LLM response</param>
    /// <returns>The list of affected components</returns>
    private List<AffectedComponent> ParseAffectedComponents(string response)
    {
        var components = new List<AffectedComponent>();
        var lines = response.Split('\n');
        
        AffectedComponent? currentComponent = null;
        
        foreach (var line in lines)
        {
            var trimmedLine = line.Trim();
            
            // Check for component name (usually starts with a number or has "Component:" in it)
            if (trimmedLine.StartsWith("Component:") || 
                (trimmedLine.Length > 2 && char.IsDigit(trimmedLine[0]) && trimmedLine[1] == '.'))
            {
                // Save previous component if exists
                if (currentComponent != null)
                {
                    components.Add(currentComponent);
                }
                
                // Create new component
                currentComponent = new AffectedComponent
                {
                    Name = ExtractValue(trimmedLine, "Component:"),
                    ChangeType = ChangeType.Modify // Default
                };
            }
            // Check for file path
            else if (trimmedLine.StartsWith("File:") || trimmedLine.StartsWith("Path:"))
            {
                if (currentComponent != null)
                {
                    currentComponent.FilePath = ExtractValue(trimmedLine, "File:").Replace("Path:", "").Trim();
                }
            }
            // Check for change type
            else if (trimmedLine.StartsWith("Change:") || trimmedLine.StartsWith("Type:"))
            {
                if (currentComponent != null)
                {
                    var changeTypeStr = ExtractValue(trimmedLine, "Change:").Replace("Type:", "").Trim();
                    if (Enum.TryParse<ChangeType>(changeTypeStr, true, out var changeType))
                    {
                        currentComponent.ChangeType = changeType;
                    }
                }
            }
            // Check for description
            else if (trimmedLine.StartsWith("Description:") || trimmedLine.StartsWith("Changes:"))
            {
                if (currentComponent != null)
                {
                    currentComponent.Description = ExtractValue(trimmedLine, "Description:").Replace("Changes:", "").Trim();
                }
            }
            // If line contains a file path pattern
            else if (trimmedLine.Contains(".cs") || trimmedLine.Contains(".fs") || 
                     trimmedLine.Contains(".csproj") || trimmedLine.Contains(".fsproj"))
            {
                if (currentComponent != null && string.IsNullOrEmpty(currentComponent.FilePath))
                {
                    currentComponent.FilePath = trimmedLine;
                }
            }
        }
        
        // Add the last component if exists
        if (currentComponent != null)
        {
            components.Add(currentComponent);
        }
        
        return components;
    }

    /// <summary>
    /// Extract a value from a line that may contain a label
    /// </summary>
    /// <param name="line">The line of text</param>
    /// <param name="label">The label to look for</param>
    /// <returns>The extracted value</returns>
    private string ExtractValue(string line, string label)
    {
        if (line.StartsWith(label, StringComparison.OrdinalIgnoreCase))
        {
            return line.Substring(label.Length).Trim();
        }
        
        // If the line starts with a number (like "1. Component Name")
        if (line.Length > 2 && char.IsDigit(line[0]) && line[1] == '.')
        {
            return line.Substring(2).Trim();
        }
        
        return line;
    }

    /// <summary>
    /// Generate implementation steps for the task
    /// </summary>
    /// <param name="taskDescription">The task description</param>
    /// <param name="requirements">The list of requirements</param>
    /// <param name="affectedComponents">The list of affected components</param>
    /// <returns>The list of implementation steps</returns>
    private async Task<List<ImplementationStep>> GenerateImplementationStepsAsync(
        string taskDescription, 
        List<string> requirements, 
        List<AffectedComponent> affectedComponents)
    {
        _logger.LogDebug($"Generating implementation steps for task: {taskDescription}");
        
        // Build a prompt to generate implementation steps
        var prompt = new StringBuilder();
        prompt.AppendLine("Generate a detailed step-by-step implementation plan for the following task:");
        prompt.AppendLine($"Task: {taskDescription}");
        prompt.AppendLine();
        prompt.AppendLine("Requirements:");
        foreach (var requirement in requirements)
        {
            prompt.AppendLine($"- {requirement}");
        }
        prompt.AppendLine();
        prompt.AppendLine("Affected Components:");
        foreach (var component in affectedComponents)
        {
            prompt.AppendLine($"- {component.Name} ({component.FilePath}) - {component.ChangeType}");
            prompt.AppendLine($"  Description: {component.Description}");
        }
        prompt.AppendLine();
        prompt.AppendLine("For each step, provide:");
        prompt.AppendLine("1. A clear description of what needs to be done");
        prompt.AppendLine("2. The specific file(s) that need to be modified");
        prompt.AppendLine("3. The type of change (e.g., add method, modify class, create file)");
        prompt.AppendLine("4. Any dependencies on other steps");
        prompt.AppendLine("5. Estimated complexity (Low, Medium, High)");
        
        var response = await _llmService.GenerateTextAsync(prompt.ToString());
        
        // Parse the response into a list of implementation steps
        var implementationSteps = ParseImplementationSteps(response);
        
        _logger.LogDebug($"Generated {implementationSteps.Count} implementation steps for task");
        return implementationSteps;
    }

    /// <summary>
    /// Parse the LLM response into a list of implementation steps
    /// </summary>
    /// <param name="response">The LLM response</param>
    /// <returns>The list of implementation steps</returns>
    private List<ImplementationStep> ParseImplementationSteps(string response)
    {
        var steps = new List<ImplementationStep>();
        var lines = response.Split('\n');
        
        ImplementationStep? currentStep = null;
        int stepNumber = 1;
        
        foreach (var line in lines)
        {
            var trimmedLine = line.Trim();
            
            // Check for step number (usually starts with "Step X:" or just a number)
            if (trimmedLine.StartsWith("Step ") || 
                (trimmedLine.Length > 2 && char.IsDigit(trimmedLine[0]) && trimmedLine[1] == '.'))
            {
                // Save previous step if exists
                if (currentStep != null)
                {
                    steps.Add(currentStep);
                }
                
                // Create new step
                currentStep = new ImplementationStep
                {
                    StepNumber = stepNumber++,
                    Description = ExtractStepDescription(trimmedLine)
                };
            }
            // Check for file information
            else if (trimmedLine.StartsWith("File:") || trimmedLine.Contains(".cs") || 
                     trimmedLine.Contains(".fs") || trimmedLine.Contains(".csproj"))
            {
                if (currentStep != null)
                {
                    currentStep.FilePath = trimmedLine.Replace("File:", "").Trim();
                }
            }
            // Check for change type
            else if (trimmedLine.StartsWith("Change:") || trimmedLine.StartsWith("Type:"))
            {
                if (currentStep != null)
                {
                    currentStep.ChangeType = trimmedLine.Replace("Change:", "").Replace("Type:", "").Trim();
                }
            }
            // Check for dependencies
            else if (trimmedLine.StartsWith("Dependencies:") || trimmedLine.StartsWith("Depends on:"))
            {
                if (currentStep != null)
                {
                    currentStep.Dependencies = trimmedLine
                        .Replace("Dependencies:", "")
                        .Replace("Depends on:", "")
                        .Split(',')
                        .Select(d => d.Trim())
                        .Where(d => !string.IsNullOrEmpty(d))
                        .ToList();
                }
            }
            // Check for complexity
            else if (trimmedLine.StartsWith("Complexity:"))
            {
                if (currentStep != null)
                {
                    var complexityStr = trimmedLine.Replace("Complexity:", "").Trim();
                    if (Enum.TryParse<TaskComplexity>(complexityStr, true, out var complexity))
                    {
                        currentStep.Complexity = complexity;
                    }
                }
            }
            // If none of the above, add to the description of the current step
            else if (!string.IsNullOrWhiteSpace(trimmedLine) && currentStep != null)
            {
                currentStep.Description += Environment.NewLine + trimmedLine;
            }
        }
        
        // Add the last step if exists
        if (currentStep != null)
        {
            steps.Add(currentStep);
        }
        
        return steps;
    }

    /// <summary>
    /// Extract the description from a step line
    /// </summary>
    /// <param name="line">The line of text</param>
    /// <returns>The extracted description</returns>
    private string ExtractStepDescription(string line)
    {
        // If line starts with "Step X:"
        if (line.StartsWith("Step "))
        {
            var colonIndex = line.IndexOf(':');
            if (colonIndex > 0)
            {
                return line.Substring(colonIndex + 1).Trim();
            }
        }
        
        // If line starts with a number (like "1. Description")
        if (line.Length > 2 && char.IsDigit(line[0]) && line[1] == '.')
        {
            return line.Substring(2).Trim();
        }
        
        return line;
    }

    /// <summary>
    /// Estimate the overall complexity of the task
    /// </summary>
    /// <param name="requirements">The list of requirements</param>
    /// <param name="affectedComponents">The list of affected components</param>
    /// <param name="implementationSteps">The list of implementation steps</param>
    /// <returns>The estimated complexity</returns>
    private TaskComplexity EstimateComplexity(
        List<string> requirements, 
        List<AffectedComponent> affectedComponents, 
        List<ImplementationStep> implementationSteps)
    {
        // Count high complexity steps
        var highComplexitySteps = implementationSteps.Count(s => s.Complexity == TaskComplexity.High);
        
        // Count medium complexity steps
        var mediumComplexitySteps = implementationSteps.Count(s => s.Complexity == TaskComplexity.Medium);
        
        // Count the number of affected components
        var componentCount = affectedComponents.Count;
        
        // Count the number of requirements
        var requirementCount = requirements.Count;
        
        // Calculate a complexity score
        var complexityScore = 
            (highComplexitySteps * 3) + 
            (mediumComplexitySteps * 2) + 
            (componentCount * 1.5) + 
            (requirementCount * 1);
        
        // Determine the overall complexity based on the score
        if (complexityScore >= 10)
        {
            return TaskComplexity.High;
        }
        else if (complexityScore >= 5)
        {
            return TaskComplexity.Medium;
        }
        else
        {
            return TaskComplexity.Low;
        }
    }
}
