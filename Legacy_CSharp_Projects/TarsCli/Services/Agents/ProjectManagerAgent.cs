using System.Text.Json;
using Microsoft.Extensions.Logging;
using TarsEngine.Services;

namespace TarsCli.Services.Agents;

/// <summary>
/// Agent for managing projects and coordinating the self-improvement process
/// </summary>
public class ProjectManagerAgent
{
    private readonly ILogger<ProjectManagerAgent> _logger;
    private readonly LlmService _llmService;
    private readonly FileService _fileService;
    private readonly TarsEngine.Services.Interfaces.IProjectAnalysisService _projectAnalysisService;
    private readonly ImprovementPrioritizer _improvementPrioritizer;

    /// <summary>
    /// Initializes a new instance of the ProjectManagerAgent class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="llmService">LLM service</param>
    /// <param name="fileService">File service</param>
    /// <param name="projectAnalysisService">Project analysis service</param>
    /// <param name="improvementPrioritizer">Improvement prioritizer</param>
    public ProjectManagerAgent(
        ILogger<ProjectManagerAgent> logger,
        LlmService llmService,
        FileService fileService,
        TarsEngine.Services.Interfaces.IProjectAnalysisService projectAnalysisService,
        ImprovementPrioritizer improvementPrioritizer)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _llmService = llmService ?? throw new ArgumentNullException(nameof(llmService));
        _fileService = fileService ?? throw new ArgumentNullException(nameof(fileService));
        _projectAnalysisService = projectAnalysisService ?? throw new ArgumentNullException(nameof(projectAnalysisService));
        _improvementPrioritizer = improvementPrioritizer ?? throw new ArgumentNullException(nameof(improvementPrioritizer));
    }

    /// <summary>
    /// Handles an MCP request
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    public async Task<JsonElement> HandleRequestAsync(JsonElement request)
    {
        try
        {
            // Extract operation from the request
            string operation = "analyze";
            if (request.TryGetProperty("operation", out var operationElement))
            {
                operation = operationElement.GetString() ?? "analyze";
            }

            // Handle the operation
            return operation switch
            {
                "analyze" => await AnalyzeProjectAsync(request),
                "prioritize" => await PrioritizeImprovementsAsync(request),
                "plan" => await CreateImprovementPlanAsync(request),
                "track" => await TrackImprovementProgressAsync(request),
                "report" => await GenerateImprovementReportAsync(request),
                _ => CreateErrorResponse($"Unknown operation: {operation}")
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error handling request");
            return CreateErrorResponse(ex.Message);
        }
    }

    /// <summary>
    /// Analyzes a project
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    private async Task<JsonElement> AnalyzeProjectAsync(JsonElement request)
    {
        // Extract project path from the request
        if (!request.TryGetProperty("project_path", out var projectPathElement))
        {
            return CreateErrorResponse("Missing required parameter: project_path");
        }

        var projectPath = projectPathElement.GetString();

        if (string.IsNullOrEmpty(projectPath))
        {
            return CreateErrorResponse("Invalid parameter: project_path");
        }

        // Analyze the project
        var projectAnalysis = await _projectAnalysisService.AnalyzeProjectAsync(projectPath);

        // Create the response
        var responseObj = new
        {
            success = true,
            project_path = projectPath,
            project_name = projectAnalysis.ProjectName,
            project_full_path = projectAnalysis.ProjectPath
        };

        return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(responseObj));
    }

    /// <summary>
    /// Prioritizes improvements
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    private async Task<JsonElement> PrioritizeImprovementsAsync(JsonElement request)
    {
        // Extract improvement opportunities from the request
        if (!request.TryGetProperty("improvement_opportunities", out var opportunitiesElement))
        {
            return CreateErrorResponse("Missing required parameter: improvement_opportunities");
        }

        var opportunitiesJson = opportunitiesElement.ToString();

        if (string.IsNullOrEmpty(opportunitiesJson))
        {
            return CreateErrorResponse("Invalid parameter: improvement_opportunities");
        }

        // Deserialize the improvement opportunities
        var opportunities = JsonSerializer.Deserialize<List<TarsEngine.Models.ImprovementOpportunity>>(opportunitiesJson);

        if (opportunities == null)
        {
            return CreateErrorResponse("Failed to deserialize improvement opportunities");
        }

        // Create prioritization options
        var options = new PrioritizationOptions
        {
            PriorityWeight = 1.0,
            ImpactWeight = 1.0,
            EffortWeight = 0.5
        };

        // Prioritize the improvements
        var prioritizedOpportunities = _improvementPrioritizer.PrioritizeOpportunities(opportunities, options);

        // Create the response
        var responseObj = new
        {
            success = true,
            prioritized_opportunities = prioritizedOpportunities,
            opportunity_count = prioritizedOpportunities.Count
        };

        return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(responseObj));
    }

    /// <summary>
    /// Creates an improvement plan
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    private async Task<JsonElement> CreateImprovementPlanAsync(JsonElement request)
    {
        // Extract prioritized opportunities from the request
        if (!request.TryGetProperty("prioritized_opportunities", out var opportunitiesElement))
        {
            return CreateErrorResponse("Missing required parameter: prioritized_opportunities");
        }

        var opportunitiesJson = opportunitiesElement.ToString();

        if (string.IsNullOrEmpty(opportunitiesJson))
        {
            return CreateErrorResponse("Invalid parameter: prioritized_opportunities");
        }

        // Extract model from the request (optional)
        string model = "llama3";
        if (request.TryGetProperty("model", out var modelElement))
        {
            model = modelElement.GetString() ?? "llama3";
        }

        // Deserialize the prioritized opportunities
        var opportunities = JsonSerializer.Deserialize<List<TarsEngine.Models.ImprovementOpportunity>>(opportunitiesJson);

        if (opportunities == null)
        {
            return CreateErrorResponse("Failed to deserialize prioritized opportunities");
        }

        // Create the improvement plan
        var plan = await CreateImprovementPlanAsync(opportunities, model);

        // Create the response
        var responseObj = new
        {
            success = true,
            improvement_plan = plan,
            opportunity_count = opportunities.Count,
            plan_steps = plan.Steps.Count
        };

        return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(responseObj));
    }

    /// <summary>
    /// Tracks improvement progress
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    private async Task<JsonElement> TrackImprovementProgressAsync(JsonElement request)
    {
        // Extract improvement plan and completed steps from the request
        if (!request.TryGetProperty("improvement_plan", out var planElement) ||
            !request.TryGetProperty("completed_steps", out var completedStepsElement))
        {
            return CreateErrorResponse("Missing required parameters: improvement_plan, completed_steps");
        }

        var planJson = planElement.ToString();
        var completedStepsJson = completedStepsElement.ToString();

        if (string.IsNullOrEmpty(planJson) || string.IsNullOrEmpty(completedStepsJson))
        {
            return CreateErrorResponse("Invalid parameters: improvement_plan, completed_steps");
        }

        // Deserialize the improvement plan and completed steps
        var plan = JsonSerializer.Deserialize<TarsEngine.Models.ImprovementPlan>(planJson);
        var completedSteps = JsonSerializer.Deserialize<List<string>>(completedStepsJson);

        if (plan == null || completedSteps == null)
        {
            return CreateErrorResponse("Failed to deserialize improvement plan or completed steps");
        }

        // Update the plan with completed steps
        foreach (var step in plan.Steps)
        {
            if (completedSteps.Contains(step.Id))
            {
                step.Status = TarsEngine.Models.ImprovementStepStatus.Completed;
                step.CompletedAt = DateTime.UtcNow;
            }
        }

        // Calculate progress
        var totalSteps = plan.Steps.Count;
        var completedStepCount = plan.Steps.Count(s => s.Status == TarsEngine.Models.ImprovementStepStatus.Completed);
        var progress = totalSteps > 0 ? (double)completedStepCount / totalSteps : 0;

        // Create the response
        var responseObj = new
        {
            success = true,
            updated_plan = plan,
            total_steps = totalSteps,
            completed_steps = completedStepCount,
            progress = progress,
            is_complete = completedStepCount == totalSteps
        };

        return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(responseObj));
    }

    /// <summary>
    /// Generates an improvement report
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    private async Task<JsonElement> GenerateImprovementReportAsync(JsonElement request)
    {
        // Extract improvement plan from the request
        if (!request.TryGetProperty("improvement_plan", out var planElement))
        {
            return CreateErrorResponse("Missing required parameter: improvement_plan");
        }

        var planJson = planElement.ToString();

        if (string.IsNullOrEmpty(planJson))
        {
            return CreateErrorResponse("Invalid parameter: improvement_plan");
        }

        // Extract model from the request (optional)
        string model = "llama3";
        if (request.TryGetProperty("model", out var modelElement))
        {
            model = modelElement.GetString() ?? "llama3";
        }

        // Deserialize the improvement plan
        var plan = JsonSerializer.Deserialize<TarsEngine.Models.ImprovementPlan>(planJson);

        if (plan == null)
        {
            return CreateErrorResponse("Failed to deserialize improvement plan");
        }

        // Generate the improvement report
        var report = await GenerateImprovementReportAsync(plan, model);

        // Create the response
        var responseObj = new
        {
            success = true,
            improvement_report = report,
            plan_name = plan.Name,
            total_steps = plan.Steps.Count,
            completed_steps = plan.Steps.Count(s => s.Status == TarsEngine.Models.ImprovementStepStatus.Completed),
            report_length = report.Length
        };

        return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(responseObj));
    }

    /// <summary>
    /// Creates an improvement plan
    /// </summary>
    /// <param name="opportunities">The prioritized opportunities</param>
    /// <param name="model">The model</param>
    /// <returns>The improvement plan</returns>
    private async Task<TarsEngine.Models.ImprovementPlan> CreateImprovementPlanAsync(List<TarsEngine.Models.ImprovementOpportunity> opportunities, string model)
    {
        // Create a prompt for the LLM
        var opportunitiesText = string.Join("\n", opportunities.Select(o => $"- {o.Description} (Priority: {o.Priority})"));
        var prompt = $@"Create a detailed improvement plan for the following opportunities:

{opportunitiesText}

The plan should include:
1. A name for the plan
2. A description of the plan
3. A list of steps to implement the improvements
4. Dependencies between steps
5. Estimated effort for each step

Format the response as a JSON object with the following structure:
{{
  ""name"": ""Plan name"",
  ""description"": ""Plan description"",
  ""steps"": [
    {{
      ""id"": ""step1"",
      ""name"": ""Step name"",
      ""description"": ""Step description"",
      ""dependencies"": [""step2"", ""step3""],
      ""estimatedEffort"": ""2 hours"",
      ""status"": ""NotStarted"",
      ""opportunityIds"": [""opportunity1"", ""opportunity2""]
    }}
  ]
}}";

        // Generate the plan
        var planJson = await _llmService.GetCompletionAsync(prompt, temperature: 0.2, maxTokens: 2000, model: model);

        // Deserialize the plan
        var plan = JsonSerializer.Deserialize<TarsEngine.Models.ImprovementPlan>(planJson);

        if (plan == null)
        {
            // Create a default plan
            plan = new TarsEngine.Models.ImprovementPlan
            {
                Name = "Default Improvement Plan",
                Description = "A plan to implement the prioritized improvements",
                CreatedAt = DateTime.UtcNow,
                Steps = opportunities.Select(o => new TarsEngine.Models.ImprovementStep
                {
                    Id = Guid.NewGuid().ToString(),
                    Name = o.Description,
                    Description = o.Description,
                    Dependencies = [],
                    EstimatedEffort = "Unknown",
                    Status = TarsEngine.Models.ImprovementStepStatus.NotStarted,
                    OpportunityIds = [o.Id]
                }).ToList()
            };
        }

        return plan;
    }

    /// <summary>
    /// Generates an improvement report
    /// </summary>
    /// <param name="plan">The improvement plan</param>
    /// <param name="model">The model</param>
    /// <returns>The improvement report</returns>
    private async Task<string> GenerateImprovementReportAsync(TarsEngine.Models.ImprovementPlan plan, string model)
    {
        // Create a prompt for the LLM
        var stepsText = string.Join("\n", plan.Steps.Select(s =>
            $"- {s.Name} (Status: {s.Status}, Effort: {s.EstimatedEffort})" +
            (s.CompletedAt.HasValue ? $", Completed: {s.CompletedAt.Value.ToString("yyyy-MM-dd HH:mm:ss")}" : "")));

        var prompt = $@"Generate a comprehensive improvement report for the following plan:

Plan Name: {plan.Name}
Plan Description: {plan.Description}
Created At: {plan.CreatedAt.ToString("yyyy-MM-dd HH:mm:ss")}

Steps:
{stepsText}

The report should include:
1. An executive summary
2. Progress overview
3. Completed improvements
4. Pending improvements
5. Challenges and blockers
6. Next steps
7. Recommendations

Format the report in Markdown.";

        // Generate the report
        return await _llmService.GetCompletionAsync(prompt, temperature: 0.3, maxTokens: 2000, model: model);
    }

    /// <summary>
    /// Creates an error response
    /// </summary>
    /// <param name="message">The error message</param>
    /// <returns>The error response</returns>
    private JsonElement CreateErrorResponse(string message)
    {
        var responseObj = new
        {
            success = false,
            error = message
        };

        return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(responseObj));
    }
}
