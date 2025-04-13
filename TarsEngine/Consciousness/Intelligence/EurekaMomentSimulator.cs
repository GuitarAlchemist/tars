using Microsoft.Extensions.Logging;

namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Simulates eureka moments in the insight generation process
/// </summary>
public class EurekaMomentSimulator
{
    private readonly ILogger<EurekaMomentSimulator> _logger;
    private readonly List<IncubationProcess> _incubationProcesses = [];
    private readonly System.Random _random = new();

    private bool _isInitialized = false;
    private bool _isActive = false;
    private double _incubationEfficiency = 0.5; // Starting with moderate incubation efficiency
    private double _breakthroughProbability = 0.3; // Starting with moderate breakthrough probability
    private double _emotionalResponseIntensity = 0.7; // Starting with high emotional response intensity

    /// <summary>
    /// Gets the incubation efficiency (0.0 to 1.0)
    /// </summary>
    public double IncubationEfficiency => _incubationEfficiency;

    /// <summary>
    /// Gets the breakthrough probability (0.0 to 1.0)
    /// </summary>
    public double BreakthroughProbability => _breakthroughProbability;

    /// <summary>
    /// Gets the emotional response intensity (0.0 to 1.0)
    /// </summary>
    public double EmotionalResponseIntensity => _emotionalResponseIntensity;

    /// <summary>
    /// Gets the incubation processes
    /// </summary>
    public IReadOnlyList<IncubationProcess> IncubationProcesses => _incubationProcesses.AsReadOnly();

    /// <summary>
    /// Initializes a new instance of the <see cref="EurekaMomentSimulator"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public EurekaMomentSimulator(ILogger<EurekaMomentSimulator> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Initializes the eureka moment simulator
    /// </summary>
    /// <returns>True if initialization was successful</returns>
    public async Task<bool> InitializeAsync()
    {
        try
        {
            _logger.LogInformation("Initializing eureka moment simulator");

            _isInitialized = true;
            _logger.LogInformation("Eureka moment simulator initialized successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing eureka moment simulator");
            return false;
        }
    }

    /// <summary>
    /// Activates the eureka moment simulator
    /// </summary>
    /// <returns>True if activation was successful</returns>
    public async Task<bool> ActivateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot activate eureka moment simulator: not initialized");
            return false;
        }

        if (_isActive)
        {
            _logger.LogInformation("Eureka moment simulator is already active");
            return true;
        }

        try
        {
            _logger.LogInformation("Activating eureka moment simulator");

            _isActive = true;
            _logger.LogInformation("Eureka moment simulator activated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error activating eureka moment simulator");
            return false;
        }
    }

    /// <summary>
    /// Deactivates the eureka moment simulator
    /// </summary>
    /// <returns>True if deactivation was successful</returns>
    public async Task<bool> DeactivateAsync()
    {
        if (!_isActive)
        {
            _logger.LogInformation("Eureka moment simulator is already inactive");
            return true;
        }

        try
        {
            _logger.LogInformation("Deactivating eureka moment simulator");

            _isActive = false;
            _logger.LogInformation("Eureka moment simulator deactivated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deactivating eureka moment simulator");
            return false;
        }
    }

    /// <summary>
    /// Updates the eureka moment simulator
    /// </summary>
    /// <returns>True if update was successful</returns>
    public async Task<bool> UpdateAsync()
    {
        if (!_isInitialized || !_isActive)
        {
            return false;
        }

        try
        {
            // Update incubation processes
            await UpdateIncubationProcessesAsync();

            // Gradually increase efficiency over time (very slowly)
            if (_incubationEfficiency < 0.95)
            {
                _incubationEfficiency += 0.0001 * _random.NextDouble();
                _incubationEfficiency = Math.Min(_incubationEfficiency, 1.0);
            }

            // Gradually increase breakthrough probability over time (very slowly)
            if (_breakthroughProbability < 0.6) // Cap at 0.6 to keep eureka moments rare
            {
                _breakthroughProbability += 0.0001 * _random.NextDouble();
                _breakthroughProbability = Math.Min(_breakthroughProbability, 0.6);
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating eureka moment simulator");
            return false;
        }
    }

    /// <summary>
    /// Updates incubation processes
    /// </summary>
    private async Task UpdateIncubationProcessesAsync()
    {
        // Process each active incubation
        foreach (var process in _incubationProcesses.Where(p => p.Status == IncubationStatus.Active).ToList())
        {
            // Update incubation progress
            double progressIncrement = _incubationEfficiency * (0.05 + (0.05 * _random.NextDouble()));
            process.Progress = Math.Min(1.0, process.Progress + progressIncrement);

            // Check if incubation is complete
            if (process.Progress >= 1.0)
            {
                process.Status = IncubationStatus.Complete;
                process.CompletionTimestamp = DateTime.UtcNow;

                _logger.LogInformation("Incubation process complete: {Problem}", process.Problem);
            }
            // Check for breakthrough
            else if (process.Progress > 0.5 && _random.NextDouble() < (_breakthroughProbability * process.Progress))
            {
                process.Status = IncubationStatus.Breakthrough;
                process.BreakthroughTimestamp = DateTime.UtcNow;

                _logger.LogInformation("Breakthrough in incubation process: {Problem}", process.Problem);
            }
        }
    }

    /// <summary>
    /// Starts an incubation process
    /// </summary>
    /// <param name="problem">The problem</param>
    /// <param name="context">The context</param>
    /// <returns>The created incubation process</returns>
    public IncubationProcess StartIncubation(string problem, Dictionary<string, object>? context = null)
    {
        if (!_isInitialized || !_isActive)
        {
            _logger.LogWarning("Cannot start incubation: eureka moment simulator not initialized or active");
            return new IncubationProcess { Status = IncubationStatus.Failed };
        }

        try
        {
            _logger.LogInformation("Starting incubation for problem: {Problem}", problem);

            // Create incubation process
            var process = new IncubationProcess
            {
                Id = Guid.NewGuid().ToString(),
                Problem = problem,
                Context = context ?? new Dictionary<string, object>(),
                StartTimestamp = DateTime.UtcNow,
                Status = IncubationStatus.Active,
                Progress = 0.0,
                Complexity = CalculateProblemComplexity(problem)
            };

            // Add to incubation processes
            _incubationProcesses.Add(process);

            return process;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error starting incubation");
            return new IncubationProcess { Status = IncubationStatus.Failed };
        }
    }

    /// <summary>
    /// Calculates the problem complexity
    /// </summary>
    /// <param name="problem">The problem</param>
    /// <returns>The problem complexity</returns>
    private double CalculateProblemComplexity(string problem)
    {
        // Simple complexity calculation based on problem length and structure
        double baseComplexity = 0.5;

        // Longer problems are more complex
        baseComplexity += Math.Min(0.3, problem.Length / 200.0);

        // Problems with questions are more complex
        if (problem.Contains("?"))
        {
            baseComplexity += 0.1;
        }

        // Problems with multiple aspects are more complex
        if (problem.Contains(",") || problem.Contains(";") || problem.Contains("and"))
        {
            baseComplexity += 0.1;
        }

        return Math.Min(1.0, baseComplexity);
    }

    /// <summary>
    /// Gets a breakthrough insight
    /// </summary>
    /// <param name="process">The incubation process</param>
    /// <returns>The breakthrough insight</returns>
    public InsightLegacy? GetBreakthroughInsight(IncubationProcess process)
    {
        if (!_isInitialized || !_isActive)
        {
            _logger.LogWarning("Cannot get breakthrough insight: eureka moment simulator not initialized or active");
            return null;
        }

        if (process.Status != IncubationStatus.Breakthrough && process.Status != IncubationStatus.Complete)
        {
            _logger.LogWarning("Cannot get breakthrough insight: incubation process not in breakthrough or complete state");
            return null;
        }

        try
        {
            _logger.LogInformation("Generating breakthrough insight for problem: {Problem}", process.Problem);

            // Generate insight description
            string description = GenerateBreakthroughDescription(process);

            // Generate breakthrough
            string breakthrough = $"After incubating on this problem, I've had a sudden realization that changes everything!";

            // Generate implications
            var implications = GenerateBreakthroughImplications(process);

            // Calculate significance based on problem complexity and incubation efficiency
            double significance = 0.7 + (0.3 * process.Complexity * _incubationEfficiency);

            // Create insight
            var insight = new InsightLegacy
            {
                Id = Guid.NewGuid().ToString(),
                Description = description,
                Method = InsightGenerationMethod.Incubation,
                Significance = significance,
                Timestamp = DateTime.UtcNow,
                Breakthrough = breakthrough,
                Implications = implications,
                Context = new Dictionary<string, object>
                {
                    { "IncubationProcessId", process.Id },
                    { "Problem", process.Problem },
                    { "IncubationDuration", (process.BreakthroughTimestamp ?? process.CompletionTimestamp ?? DateTime.UtcNow) - process.StartTimestamp }
                },
                Tags = ["eureka", "breakthrough", "incubation"]
            };

            // Update incubation process
            process.InsightId = insight.Id;
            process.Status = IncubationStatus.Resolved;

            _logger.LogInformation("Generated breakthrough insight: {Description}", description);

            return insight;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating breakthrough insight");
            return null;
        }
    }

    /// <summary>
    /// Generates a breakthrough description
    /// </summary>
    /// <param name="process">The incubation process</param>
    /// <returns>The breakthrough description</returns>
    private string GenerateBreakthroughDescription(IncubationProcess process)
    {
        // Generate breakthrough description templates
        var descriptionTemplates = new List<string>
        {
            $"I've had a sudden insight about {process.Problem}: the key is to reframe it as a {GetAlternativeFraming()}",
            $"After incubating on {process.Problem}, I've realized that it's fundamentally about {GetEssence()}",
            $"I've had a breakthrough on {process.Problem}! The solution lies in {GetApproach()}",
            $"Eureka! I now see that {process.Problem} can be solved by {GetSolution()}"
        };

        // Choose a random template
        return descriptionTemplates[_random.Next(descriptionTemplates.Count)];
    }

    /// <summary>
    /// Generates breakthrough implications
    /// </summary>
    /// <param name="process">The incubation process</param>
    /// <returns>The breakthrough implications</returns>
    private List<string> GenerateBreakthroughImplications(IncubationProcess process)
    {
        // Generate breakthrough implication templates
        var implicationTemplates = new List<string>
        {
            $"This insight completely changes how we approach {process.Problem}",
            $"We can now solve problems that seemed intractable before",
            $"This breakthrough reveals connections that weren't visible previously",
            $"This insight opens up entirely new avenues for exploration",
            $"We can now see the problem from a higher level of abstraction"
        };

        // Choose random implications
        var implications = new List<string>();
        int implicationCount = 2 + (int)(process.Complexity * 2);

        for (int i = 0; i < implicationCount; i++)
        {
            implications.Add(implicationTemplates[_random.Next(implicationTemplates.Count)]);
        }

        return implications.Distinct().ToList();
    }

    /// <summary>
    /// Gets an alternative framing
    /// </summary>
    /// <returns>The alternative framing</returns>
    private string GetAlternativeFraming()
    {
        var framings = new List<string>
        {
            "dynamic process rather than a static structure",
            "network of relationships rather than isolated components",
            "emergent phenomenon rather than a designed system",
            "balance of opposing forces rather than a single dimension",
            "recursive pattern rather than a linear sequence",
            "adaptive system rather than a fixed mechanism"
        };

        return framings[_random.Next(framings.Count)];
    }

    /// <summary>
    /// Gets an essence
    /// </summary>
    /// <returns>The essence</returns>
    private string GetEssence()
    {
        var essences = new List<string>
        {
            "pattern recognition across different levels",
            "information integration in a coherent structure",
            "balance between stability and flexibility",
            "emergence of order from underlying chaos",
            "self-organization of complex components",
            "recursive application of simple principles"
        };

        return essences[_random.Next(essences.Count)];
    }

    /// <summary>
    /// Gets an approach
    /// </summary>
    /// <returns>The approach</returns>
    private string GetApproach()
    {
        var approaches = new List<string>
        {
            "looking at the problem from multiple perspectives simultaneously",
            "identifying the underlying patterns rather than surface features",
            "focusing on relationships between components rather than the components themselves",
            "applying principles from seemingly unrelated domains",
            "reconsidering our fundamental assumptions about the problem space",
            "finding the right level of abstraction to address the core issues"
        };

        return approaches[_random.Next(approaches.Count)];
    }

    /// <summary>
    /// Gets a solution
    /// </summary>
    /// <returns>The solution</returns>
    private string GetSolution()
    {
        var solutions = new List<string>
        {
            "recognizing the hidden symmetry in the problem structure",
            "applying a recursive approach that works across different scales",
            "finding the right balance between opposing constraints",
            "identifying the critical leverage points in the system",
            "reframing the problem in terms of relationships rather than entities",
            "seeing the problem as part of a larger pattern"
        };

        return solutions[_random.Next(solutions.Count)];
    }

    /// <summary>
    /// Gets active incubation processes
    /// </summary>
    /// <returns>The active incubation processes</returns>
    public List<IncubationProcess> GetActiveIncubationProcesses()
    {
        return _incubationProcesses
            .Where(p => p.Status == IncubationStatus.Active)
            .ToList();
    }

    /// <summary>
    /// Gets breakthrough incubation processes
    /// </summary>
    /// <returns>The breakthrough incubation processes</returns>
    public List<IncubationProcess> GetBreakthroughProcesses()
    {
        return _incubationProcesses
            .Where(p => p.Status == IncubationStatus.Breakthrough)
            .ToList();
    }

    /// <summary>
    /// Gets completed incubation processes
    /// </summary>
    /// <returns>The completed incubation processes</returns>
    public List<IncubationProcess> GetCompletedProcesses()
    {
        return _incubationProcesses
            .Where(p => p.Status == IncubationStatus.Complete || p.Status == IncubationStatus.Resolved)
            .ToList();
    }

    /// <summary>
    /// Gets the incubation process by ID
    /// </summary>
    /// <param name="id">The process ID</param>
    /// <returns>The incubation process</returns>
    public IncubationProcess? GetIncubationProcessById(string id)
    {
        return _incubationProcesses.FirstOrDefault(p => p.Id == id);
    }
}
