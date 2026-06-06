namespace TarsCli.Services;

/// <summary>
/// Service for coordinating multiple agents in code analysis and transformation
/// </summary>
public class MultiAgentCollaborationService(
    ILogger<MultiAgentCollaborationService> logger,
    DynamicFSharpCompilerService fsharpCompiler,
    MetascriptEngine _metascriptEngine,  // Not used but required for dependency injection
    TransformationLearningService _transformationLearningService)  // Not used but required for dependency injection
{
    private readonly List<ICodeAgent> _registeredAgents = [];

    /// <summary>
    /// Registers an agent with the collaboration service
    /// </summary>
    /// <param name="agent">The agent to register</param>
    public void RegisterAgent(ICodeAgent agent)
    {
        _registeredAgents.Add(agent);
        logger.LogInformation($"Registered agent: {agent.Name} ({agent.Role})");
    }

    /// <summary>
    /// Gets all registered agents
    /// </summary>
    public IReadOnlyList<ICodeAgent> GetRegisteredAgents()
    {
        return _registeredAgents.AsReadOnly();
    }

    /// <summary>
    /// Gets agents with a specific role
    /// </summary>
    /// <param name="role">The role to filter by</param>
    public IEnumerable<ICodeAgent> GetAgentsByRole(AgentRole role)
    {
        return _registeredAgents.Where(a => a.Role == role);
    }

    /// <summary>
    /// Analyzes a file using all registered analysis agents
    /// </summary>
    /// <param name="filePath">Path to the file to analyze</param>
    public async Task<CollaborativeAnalysisResult> AnalyzeFileAsync(string filePath)
    {
        try
        {
            logger.LogInformation($"Analyzing file: {filePath}");

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            // Read the file
            var code = await File.ReadAllTextAsync(filePath);

            // Get all analysis agents
            var analysisAgents = GetAgentsByRole(AgentRole.Analysis).ToList();

            if (analysisAgents.Count == 0)
            {
                logger.LogWarning("No analysis agents registered");
                return new CollaborativeAnalysisResult
                {
                    FilePath = filePath,
                    Issues = [],
                    Suggestions = []
                };
            }

            // Run all analysis agents in parallel
            var analysisResults = await Task.WhenAll(
                analysisAgents.Select(agent => RunAnalysisAgentAsync(agent, filePath, code)));

            // Combine results
            var combinedIssues = new List<CodeIssue>();
            var combinedSuggestions = new List<CodeSuggestion>();

            foreach (var result in analysisResults)
            {
                combinedIssues.AddRange(result.Issues);
                // No suggestions in AnalysisResult
                // combinedSuggestions.AddRange(result.Suggestions);
            }

            // Remove duplicate issues and suggestions
            var uniqueIssues = combinedIssues
                .GroupBy(i => $"{i.Type}:{i.Location}")
                .Select(g => g.First())
                .ToList();

            var uniqueSuggestions = combinedSuggestions
                .GroupBy(s => $"{s.Type}:{s.Location}")
                .Select(g => g.OrderByDescending(s => s.Confidence).First())
                .ToList();

            return new CollaborativeAnalysisResult
            {
                FilePath = filePath,
                Issues = uniqueIssues,
                Suggestions = uniqueSuggestions
            };
        }
        catch (Exception ex)
        {
            logger.LogError(ex, $"Error analyzing file: {filePath}");
            throw;
        }
    }

    /// <summary>
    /// Transforms a file using all registered transformation agents
    /// </summary>
    /// <param name="filePath">Path to the file to transform</param>
    /// <param name="outputPath">Path to save the transformed file (optional)</param>
    public async Task<CollaborativeTransformationResult> TransformFileAsync(string filePath, string outputPath = null)
    {
        try
        {
            logger.LogInformation($"Transforming file: {filePath}");

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            // First analyze the file to identify issues
            var analysisResult = await AnalyzeFileAsync(filePath);

            // Get all transformation agents
            var transformationAgents = GetAgentsByRole(AgentRole.Transformation).ToList();

            if (transformationAgents.Count == 0)
            {
                logger.LogWarning("No transformation agents registered");
                return new CollaborativeTransformationResult
                {
                    FilePath = filePath,
                    TransformedFilePath = null,
                    AppliedTransformations = [],
                    Success = false
                };
            }

            // Read the file
            var originalCode = await File.ReadAllTextAsync(filePath);
            var currentCode = originalCode;

            // Track applied transformations
            var appliedTransformations = new List<AppliedTransformation>();

            // Apply transformations from each agent in sequence
            foreach (var agent in transformationAgents)
            {
                var transformationResult = await RunTransformationAgentAsync(agent, filePath, currentCode, analysisResult);

                if (transformationResult.Success && transformationResult.TransformedCode != currentCode)
                {
                    // Update the current code
                    currentCode = transformationResult.TransformedCode;

                    // Add to applied transformations
                    appliedTransformations.Add(new AppliedTransformation
                    {
                        AgentName = agent.Name,
                        TransformationType = transformationResult.TransformationType,
                        Location = transformationResult.Location
                    });
                }
            }

            // Check if any transformations were applied
            var success = appliedTransformations.Count > 0;

            // Save the transformed file if requested
            string transformedFilePath = null;
            if (success)
            {
                if (!string.IsNullOrEmpty(outputPath))
                {
                    transformedFilePath = outputPath;
                }
                else
                {
                    // Create a new file with _transformed suffix
                    var directory = Path.GetDirectoryName(filePath);
                    var fileName = Path.GetFileNameWithoutExtension(filePath);
                    var extension = Path.GetExtension(filePath);
                    transformedFilePath = Path.Combine(directory, $"{fileName}_transformed{extension}");
                }

                // Ensure directory exists
                Directory.CreateDirectory(Path.GetDirectoryName(transformedFilePath));

                // Save the file
                await File.WriteAllTextAsync(transformedFilePath, currentCode);
            }

            return new CollaborativeTransformationResult
            {
                FilePath = filePath,
                TransformedFilePath = transformedFilePath,
                AppliedTransformations = appliedTransformations,
                Success = success
            };
        }
        catch (Exception ex)
        {
            logger.LogError(ex, $"Error transforming file: {filePath}");
            throw;
        }
    }

    /// <summary>
    /// Validates a transformation using validation agents
    /// </summary>
    /// <param name="originalCode">The original code</param>
    /// <param name="transformedCode">The transformed code</param>
    public async Task<ValidationResult> ValidateTransformationAsync(string originalCode, string transformedCode)
    {
        try
        {
            // Get all validation agents
            var validationAgents = GetAgentsByRole(AgentRole.Validation).ToList();

            if (validationAgents.Count == 0)
            {
                logger.LogWarning("No validation agents registered");
                return new ValidationResult
                {
                    IsValid = true,
                    Issues = []
                };
            }

            // Run all validation agents in parallel
            var validationResults = await Task.WhenAll(
                validationAgents.Select(agent => RunValidationAgentAsync(agent, originalCode, transformedCode)));

            // Combine results
            var allIssues = validationResults.SelectMany(r => r.Issues).ToList();
            var isValid = !allIssues.Any(i => i.Severity == ValidationSeverity.Error);

            return new ValidationResult
            {
                IsValid = isValid,
                Issues = allIssues
            };
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error validating transformation");
            throw;
        }
    }

    /// <summary>
    /// Creates a new agent from F# code
    /// </summary>
    /// <param name="fsharpCode">The F# code defining the agent</param>
    /// <param name="agentName">Name for the agent</param>
    public async Task<ICodeAgent> CreateAgentFromFSharpAsync(string fsharpCode, string agentName)
    {
        try
        {
            logger.LogInformation($"Creating agent from F# code: {agentName}");

            // Compile the F# code
            var assembly = await fsharpCompiler.CompileFSharpCodeAsync(fsharpCode, $"Agent_{agentName}");

            // Find the agent type
            var agentType = assembly.GetTypes()
                .FirstOrDefault(t => typeof(ICodeAgent).IsAssignableFrom(t));

            if (agentType == null)
            {
                throw new Exception("No ICodeAgent implementation found in the compiled assembly");
            }

            // Create an instance of the agent
            var agent = (ICodeAgent)Activator.CreateInstance(agentType);

            // Register the agent
            RegisterAgent(agent);

            return agent;
        }
        catch (Exception ex)
        {
            logger.LogError(ex, $"Error creating agent from F# code: {agentName}");
            throw;
        }
    }

    /// <summary>
    /// Runs an analysis agent on a file
    /// </summary>
    private async Task<AnalysisResult> RunAnalysisAgentAsync(ICodeAgent agent, string filePath, string code)
    {
        try
        {
            logger.LogInformation($"Running analysis agent: {agent.Name}");

            if (agent is IAnalysisAgent analysisAgent)
            {
                return await analysisAgent.AnalyzeAsync(filePath, code);
            }

            throw new InvalidOperationException($"Agent {agent.Name} is not an analysis agent");
        }
        catch (Exception ex)
        {
            logger.LogError(ex, $"Error running analysis agent {agent.Name}");
            return new AnalysisResult
            {
                Issues = []
            };
        }
    }

    /// <summary>
    /// Runs a transformation agent on a file
    /// </summary>
    private async Task<TransformationResult> RunTransformationAgentAsync(
        ICodeAgent agent,
        string filePath,
        string code,
        CollaborativeAnalysisResult analysisResult)
    {
        try
        {
            logger.LogInformation($"Running transformation agent: {agent.Name}");

            if (agent is ITransformationAgent transformationAgent)
            {
                return await transformationAgent.TransformAsync(filePath, code, analysisResult);
            }

            throw new InvalidOperationException($"Agent {agent.Name} is not a transformation agent");
        }
        catch (Exception ex)
        {
            logger.LogError(ex, $"Error running transformation agent {agent.Name}");
            return new TransformationResult
            {
                TransformedCode = code,
                Success = false
            };
        }
    }

    /// <summary>
    /// Runs a validation agent on a transformation
    /// </summary>
    private async Task<ValidationResult> RunValidationAgentAsync(
        ICodeAgent agent,
        string originalCode,
        string transformedCode)
    {
        try
        {
            logger.LogInformation($"Running validation agent: {agent.Name}");

            if (agent is IValidationAgent validationAgent)
            {
                return await validationAgent.ValidateAsync(originalCode, transformedCode);
            }

            throw new InvalidOperationException($"Agent {agent.Name} is not a validation agent");
        }
        catch (Exception ex)
        {
            logger.LogError(ex, $"Error running validation agent {agent.Name}");
            return new ValidationResult
            {
                IsValid = false,
                Issues =
                [
                    new ValidationIssue
                    {
                        Message = $"Error running validation agent: {ex.Message}",
                        Severity = ValidationSeverity.Error
                    }
                ]
            };
        }
    }
}

/// <summary>
/// Roles that agents can play in the collaboration
/// </summary>
public enum AgentRole
{
    Analysis,
    Transformation,
    Validation,
    Learning
}

/// <summary>
/// Interface for all code agents
/// </summary>
public interface ICodeAgent
{
    string Name { get; }
    string Description { get; }
    AgentRole Role { get; }
}

/// <summary>
/// Interface for agents that analyze code
/// </summary>
public interface IAnalysisAgent : ICodeAgent
{
    Task<AnalysisResult> AnalyzeAsync(string filePath, string code);
}

/// <summary>
/// Interface for agents that transform code
/// </summary>
public interface ITransformationAgent : ICodeAgent
{
    Task<TransformationResult> TransformAsync(string filePath, string code, CollaborativeAnalysisResult analysisResult);
}

/// <summary>
/// Interface for agents that validate transformations
/// </summary>
public interface IValidationAgent : ICodeAgent
{
    Task<ValidationResult> ValidateAsync(string originalCode, string transformedCode);
}

/// <summary>
/// Interface for agents that learn from transformations
/// </summary>
public interface ILearningAgent : ICodeAgent
{
    Task LearnFromTransformationAsync(string originalCode, string transformedCode, bool wasAccepted);
    Task<List<MetascriptEngine.TransformationRule>> SuggestRulesAsync();
}



/// <summary>
/// Result of code analysis for collaboration
/// </summary>
public class CollaborationAnalysisResult
{
    public List<CodeIssue> Issues { get; set; } = [];
    public List<CodeSuggestion> Suggestions { get; set; } = [];
}



/// <summary>
/// Result of collaborative code analysis
/// </summary>
public class CollaborativeAnalysisResult
{
    public string FilePath { get; set; }
    public List<CodeIssue> Issues { get; set; } = [];
    public List<CodeSuggestion> Suggestions { get; set; } = [];
}

/// <summary>
/// Result of code transformation
/// </summary>
public class TransformationResult
{
    public string TransformedCode { get; set; }
    public bool Success { get; set; }
    public string TransformationType { get; set; }
    public string Location { get; set; }
}

/// <summary>
/// Result of collaborative code transformation
/// </summary>
public class CollaborativeTransformationResult
{
    public string FilePath { get; set; }
    public string TransformedFilePath { get; set; }
    public List<AppliedTransformation> AppliedTransformations { get; set; } = [];
    public bool Success { get; set; }
}

/// <summary>
/// Result of transformation validation
/// </summary>
public class ValidationResult
{
    public bool IsValid { get; set; }
    public List<ValidationIssue> Issues { get; set; } = [];
}

/// <summary>
/// Severity of an issue
/// </summary>
public enum IssueSeverity
{
    Info,
    Warning,
    Error
}



/// <summary>
/// A code suggestion identified by analysis
/// </summary>
public class CodeSuggestion
{
    public string Type { get; set; }
    public string Location { get; set; }
    public string Description { get; set; }
    public double Confidence { get; set; }
}

/// <summary>
/// A transformation that was applied to code
/// </summary>
public class AppliedTransformation
{
    public string AgentName { get; set; }
    public string TransformationType { get; set; }
    public string Location { get; set; }
}

/// <summary>
/// An issue identified during validation
/// </summary>
public class ValidationIssue
{
    public string Message { get; set; }
    public ValidationSeverity Severity { get; set; }
    public string Location { get; set; }
}



/// <summary>
/// Severity of a validation issue
/// </summary>
public enum ValidationSeverity
{
    Info,
    Warning,
    Error
}