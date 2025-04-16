using Microsoft.Extensions.Logging;
using TarsEngine.Consciousness.Intelligence.Divergent;
using TarsEngine.Consciousness.Intelligence.Conceptual;
using TarsEngine.Consciousness.Intelligence.Pattern;
using TarsEngine.Monads;

namespace TarsEngine.Consciousness.Intelligence.Solution;

/// <summary>
/// Implements creative solution generation capabilities
/// </summary>
public class CreativeSolutionGeneration
{
    private readonly ILogger<CreativeSolutionGeneration> _logger;
    private readonly DivergentThinking _divergentThinking;
    private readonly ConceptualBlending _conceptualBlending;
    private readonly PatternDisruption _patternDisruption;
    private readonly System.Random _random = new();
    private double _creativityLevel = 0.5; // Starting with moderate creativity
    private readonly List<ProblemModel> _problemHistory = [];
    private readonly List<SolutionStrategy> _solutionStrategies = [];

    /// <summary>
    /// Gets the creativity level (0.0 to 1.0)
    /// </summary>
    public double CreativityLevel => _creativityLevel;

    /// <summary>
    /// Initializes a new instance of the <see cref="CreativeSolutionGeneration"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="divergentThinking">The divergent thinking service</param>
    /// <param name="conceptualBlending">The conceptual blending service</param>
    /// <param name="patternDisruption">The pattern disruption service</param>
    public CreativeSolutionGeneration(
        ILogger<CreativeSolutionGeneration> logger,
        DivergentThinking divergentThinking,
        ConceptualBlending conceptualBlending,
        PatternDisruption patternDisruption)
    {
        _logger = logger;
        _divergentThinking = divergentThinking;
        _conceptualBlending = conceptualBlending;
        _patternDisruption = patternDisruption;

        InitializeSolutionStrategies();
    }

    /// <summary>
    /// Initializes the solution strategies
    /// </summary>
    private void InitializeSolutionStrategies()
    {
        _solutionStrategies.Add(new SolutionStrategy
        {
            Name = "Divergent Exploration",
            Description = "Generate multiple alternative approaches to the problem",
            ProcessType = CreativeProcessType.DivergentThinking,
            GenerateFunction = (problem, constraints) =>
            {
                var alternatives = _divergentThinking.GenerateAlternatives(problem, 3);
                var bestAlternative = alternatives.OrderByDescending(a => a.QualityScore).FirstOrDefault();
                var result = bestAlternative ?? new CreativeIdea
                {
                    Id = Guid.NewGuid().ToString(),
                    Description = "A divergent approach to the problem",
                    Originality = 0.5,
                    Value = 0.5,
                    Timestamp = DateTime.UtcNow,
                    ProcessType = CreativeProcessType.DivergentThinking,
                    Problem = problem,
                    Constraints = constraints?.ToList() ?? []
                };
                return AsyncMonad.Return(result);
            }
        });

        _solutionStrategies.Add(new SolutionStrategy
        {
            Name = "Conceptual Blending",
            Description = "Create a hybrid solution by blending concepts",
            ProcessType = CreativeProcessType.ConceptualBlending,
            GenerateFunction = (problem, constraints) =>
            {
                var blendedSolution = _conceptualBlending.GenerateBlendedSolution(problem, constraints);
                var result = blendedSolution ?? new CreativeIdea
                {
                    Id = Guid.NewGuid().ToString(),
                    Description = "A conceptual blending approach to the problem",
                    Originality = 0.6,
                    Value = 0.5,
                    Timestamp = DateTime.UtcNow,
                    ProcessType = CreativeProcessType.ConceptualBlending,
                    Problem = problem,
                    Constraints = constraints?.ToList() ?? []
                };
                return AsyncMonad.Return(result);
            }
        });

        _solutionStrategies.Add(new SolutionStrategy
        {
            Name = "Pattern Disruption",
            Description = "Challenge assumptions and break patterns",
            ProcessType = CreativeProcessType.PatternDisruption,
            GenerateFunction = (problem, constraints) =>
            {
                var disruptiveSolution = _patternDisruption.GenerateDisruptiveSolution(problem, constraints);
                var result = disruptiveSolution ?? new CreativeIdea
                {
                    Id = Guid.NewGuid().ToString(),
                    Description = "A pattern-breaking approach to the problem",
                    Originality = 0.7,
                    Value = 0.5,
                    Timestamp = DateTime.UtcNow,
                    ProcessType = CreativeProcessType.PatternDisruption,
                    Problem = problem,
                    Constraints = constraints?.ToList() ?? [],
                    ImplementationSteps =
                    [
                        "1. Identify existing patterns",
                        "2. Challenge core assumptions",
                        "3. Generate alternative perspectives"
                    ]
                };
                return AsyncMonad.Return(result);
            }
        });

        _solutionStrategies.Add(new SolutionStrategy
        {
            Name = "Comprehensive Approach",
            Description = "Combine multiple creative processes for a comprehensive solution",
            ProcessType = CreativeProcessType.CombinatorialCreativity,
            GenerateFunction = (problem, constraints) =>
            {
                // Generate solutions using each approach
                var divergentSolution = _divergentThinking.GenerateAlternatives(problem, 1).FirstOrDefault() ?? new CreativeIdea
                {
                    Id = Guid.NewGuid().ToString(),
                    Description = "Divergent thinking approach",
                    Originality = 0.5,
                    Value = 0.5,
                    ProcessType = CreativeProcessType.DivergentThinking,
                    Problem = problem,
                    Constraints = constraints?.ToList() ?? []
                };

                var blendedSolution = _conceptualBlending.GenerateBlendedSolution(problem, constraints) ?? new CreativeIdea
                {
                    Id = Guid.NewGuid().ToString(),
                    Description = "Conceptual blending approach",
                    Originality = 0.6,
                    Value = 0.5,
                    ProcessType = CreativeProcessType.ConceptualBlending,
                    Problem = problem,
                    Constraints = constraints?.ToList() ?? []
                };

                var disruptiveSolution = _patternDisruption.GenerateDisruptiveSolution(problem, constraints) ?? new CreativeIdea
                {
                    Id = Guid.NewGuid().ToString(),
                    Description = "Pattern disruption approach",
                    Originality = 0.7,
                    Value = 0.5,
                    ProcessType = CreativeProcessType.PatternDisruption,
                    Problem = problem,
                    Constraints = constraints?.ToList() ?? []
                };

                // Combine the solutions
                var solutions = new List<CreativeIdea> { divergentSolution, blendedSolution, disruptiveSolution };

                // Get the best solution based on quality score
                var bestSolution = solutions.OrderByDescending(s => s.QualityScore).First();

                // Combine implementation steps from all solutions
                var allSteps = solutions
                    .SelectMany(s => s.ImplementationSteps)
                    .Distinct()
                    .ToList();

                // Create combined solution
                return AsyncMonad.Return(new CreativeIdea
                {
                    Id = Guid.NewGuid().ToString(),
                    Description = $"Comprehensive solution approach: {bestSolution.Description}",
                    Originality = solutions.Average(s => s.Originality),
                    Value = solutions.Average(s => s.Value),
                    Timestamp = DateTime.UtcNow,
                    ProcessType = CreativeProcessType.CombinatorialCreativity,
                    Concepts = solutions.SelectMany(s => s.Concepts).Distinct().ToList(),
                    Problem = problem,
                    Constraints = constraints?.ToList() ?? [],
                    ImplementationSteps = allSteps,
                    PotentialImpact = bestSolution.PotentialImpact,
                    Limitations = solutions.SelectMany(s => s.Limitations).Distinct().ToList()
                });
            }
        });

        _logger.LogInformation("Initialized solution strategies: {Count}", _solutionStrategies.Count);
    }

    /// <summary>
    /// Updates the creativity level
    /// </summary>
    /// <returns>True if the update was successful, false otherwise</returns>
    public bool Update()
    {
        try
        {
            // Gradually increase creativity level over time (very slowly)
            if (_creativityLevel < 0.95)
            {
                _creativityLevel += 0.0001 * _random.NextDouble();
                _creativityLevel = Math.Min(_creativityLevel, 1.0);
            }

            // Update component creativity levels
            _divergentThinking.Update();
            _conceptualBlending.Update();
            _patternDisruption.Update();

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating creative solution generation");
            return false;
        }
    }

    /// <summary>
    /// Generates a solution for a problem
    /// </summary>
    /// <param name="problem">The problem description</param>
    /// <param name="constraints">Optional constraints for the solution</param>
    /// <returns>The generated solution</returns>
    public async Task<CreativeIdea?> GenerateSolutionAsync(string problem, List<string>? constraints = null)
    {
        try
        {
            _logger.LogInformation("Generating solution for problem: {Problem}", problem);

            // Create problem model
            var problemModel = new ProblemModel
            {
                Id = Guid.NewGuid().ToString(),
                Description = problem,
                Constraints = constraints?.ToList() ?? [],
                Timestamp = DateTime.UtcNow
            };

            _problemHistory.Add(problemModel);

            // Choose a solution strategy based on problem characteristics
            var strategy = ChooseSolutionStrategy(problem, constraints);

            _logger.LogInformation("Chosen solution strategy: {StrategyName}", strategy.Name);

            // Generate solution using the chosen strategy
            // Ensure constraints is never null when passed to GenerateFunction
            var solution = await strategy.GenerateFunction(problem, constraints ?? []);

            if (solution != null)
            {
                // Update problem model with solution
                problemModel.SolutionId = solution.Id;
                problemModel.StrategyName = strategy.Name;

                _logger.LogInformation("Generated creative solution: {Description} (Originality: {Originality:F2}, Value: {Value:F2})",
                    solution.Description, solution.Originality, solution.Value);
            }

            return solution;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating solution");
            return null;
        }
    }

    /// <summary>
    /// Chooses a solution strategy based on problem characteristics
    /// </summary>
    /// <param name="problem">The problem description</param>
    /// <param name="constraints">The constraints</param>
    /// <returns>The chosen solution strategy</returns>
    private SolutionStrategy ChooseSolutionStrategy(string problem, List<string>? constraints)
    {
        // If problem contains keywords suggesting innovation, use pattern disruption
        if (problem.Contains("innovative", StringComparison.OrdinalIgnoreCase) ||
            problem.Contains("breakthrough", StringComparison.OrdinalIgnoreCase) ||
            problem.Contains("radical", StringComparison.OrdinalIgnoreCase) ||
            problem.Contains("disrupt", StringComparison.OrdinalIgnoreCase))
        {
            return _solutionStrategies.First(s => s.ProcessType == CreativeProcessType.PatternDisruption);
        }

        // If problem contains keywords suggesting combination, use conceptual blending
        if (problem.Contains("combine", StringComparison.OrdinalIgnoreCase) ||
            problem.Contains("integrate", StringComparison.OrdinalIgnoreCase) ||
            problem.Contains("merge", StringComparison.OrdinalIgnoreCase) ||
            problem.Contains("hybrid", StringComparison.OrdinalIgnoreCase))
        {
            return _solutionStrategies.First(s => s.ProcessType == CreativeProcessType.ConceptualBlending);
        }

        // If problem contains keywords suggesting exploration, use divergent thinking
        if (problem.Contains("explore", StringComparison.OrdinalIgnoreCase) ||
            problem.Contains("alternative", StringComparison.OrdinalIgnoreCase) ||
            problem.Contains("options", StringComparison.OrdinalIgnoreCase) ||
            problem.Contains("possibilities", StringComparison.OrdinalIgnoreCase))
        {
            return _solutionStrategies.First(s => s.ProcessType == CreativeProcessType.DivergentThinking);
        }

        // If many constraints, use conceptual blending
        if (constraints != null && constraints.Count > 2)
        {
            return _solutionStrategies.First(s => s.ProcessType == CreativeProcessType.ConceptualBlending);
        }

        // For complex problems, use comprehensive approach
        if (problem.Length > 100)
        {
            return _solutionStrategies.First(s => s.ProcessType == CreativeProcessType.CombinatorialCreativity);
        }

        // Choose randomly based on creativity level
        var rand = _random.NextDouble();

        if (rand < 0.25 * _creativityLevel)
        {
            return _solutionStrategies.First(s => s.ProcessType == CreativeProcessType.PatternDisruption);
        }
        else if (rand < 0.5 * _creativityLevel)
        {
            return _solutionStrategies.First(s => s.ProcessType == CreativeProcessType.ConceptualBlending);
        }
        else if (rand < 0.75 * _creativityLevel)
        {
            return _solutionStrategies.First(s => s.ProcessType == CreativeProcessType.CombinatorialCreativity);
        }
        else
        {
            return _solutionStrategies.First(s => s.ProcessType == CreativeProcessType.DivergentThinking);
        }
    }

    /// <summary>
    /// Generates multiple solution alternatives for a problem
    /// </summary>
    /// <param name="problem">The problem description</param>
    /// <param name="constraints">Optional constraints for the solutions</param>
    /// <param name="count">The number of alternatives to generate</param>
    /// <returns>The solution alternatives</returns>
    public async Task<List<CreativeIdea>> GenerateSolutionAlternativesAsync(string problem, List<string>? constraints = null, int count = 3)
    {
        var alternatives = new List<CreativeIdea>();

        try
        {
            _logger.LogInformation("Generating {Count} solution alternatives for problem: {Problem}", count, problem);

            // Ensure constraints is never null when passed to strategies
            var safeConstraints = constraints ?? [];

            // Generate alternatives using different strategies
            foreach (var strategy in _solutionStrategies)
            {
                var solution = await strategy.GenerateFunction(problem, safeConstraints);
                if (solution != null)
                {
                    alternatives.Add(solution);

                    if (alternatives.Count >= count)
                    {
                        break;
                    }
                }
            }

            // If we need more alternatives, use divergent thinking to generate more
            if (alternatives.Count < count)
            {
                var divergentAlternatives = _divergentThinking.GenerateAlternatives(problem, count - alternatives.Count);
                alternatives.AddRange(divergentAlternatives.Where(a => a != null));
            }

            _logger.LogInformation("Generated {Count} solution alternatives", alternatives.Count);
            return alternatives;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating solution alternatives");
            return alternatives;
        }
    }

    /// <summary>
    /// Refines a solution based on feedback
    /// </summary>
    /// <param name="solution">The solution to refine</param>
    /// <param name="feedback">The feedback</param>
    /// <returns>The refined solution</returns>
    public async Task<CreativeIdea> RefineSolutionAsync(CreativeIdea solution, string feedback)
    {
        try
        {
            _logger.LogInformation("Refining solution based on feedback: {Feedback}", feedback);

            // Create a new problem description incorporating the feedback
            var refinementProblem = $"Refine the solution '{solution.Description}' based on this feedback: {feedback}";

            // Use the original constraints
            var constraints = solution.Constraints;

            // Choose a strategy based on the feedback
            var strategy = ChooseSolutionStrategy(refinementProblem, constraints);

            // Generate refined solution
            var refinedSolution = await strategy.GenerateFunction(refinementProblem, constraints);

            if (refinedSolution != null)
            {
                // Preserve the original problem
                refinedSolution.Problem = solution.Problem;

                // Add refinement context
                refinedSolution.Context["OriginalSolutionId"] = solution.Id;
                refinedSolution.Context["Feedback"] = feedback;

                _logger.LogInformation("Refined solution: {Description}", refinedSolution.Description);
            }

            return refinedSolution;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error refining solution");

            // Return the original solution with a note
            solution.Description += $" (Refinement attempted based on feedback: {feedback})";
            return solution;
        }
    }

    /// <summary>
    /// Evaluates a solution against criteria
    /// </summary>
    /// <param name="solution">The solution to evaluate</param>
    /// <param name="criteria">The evaluation criteria</param>
    /// <returns>The evaluation score (0.0 to 1.0)</returns>
    public double EvaluateSolution(CreativeIdea solution, Dictionary<string, double> criteria)
    {
        try
        {
            _logger.LogInformation("Evaluating solution against criteria");

            var score = 0.0;
            var totalWeight = 0.0;

            // Evaluate based on criteria
            foreach (var (criterion, weight) in criteria)
            {
                var criterionScore = criterion.ToLowerInvariant() switch
                {
                    "originality" => solution.Originality,
                    "value" => solution.Value,
                    "feasibility" => 0.8 - (0.3 * solution.Originality), // More original ideas might be less feasible
                    "impact" => solution.Value * 1.2, // Impact related to value but potentially higher
                    "novelty" => solution.Originality * 1.1, // Novelty related to originality but slightly different
                    "relevance" => solution.Problem.Length > 0 ? 0.8 : 0.5, // More relevant if addressing a specific problem
                    "completeness" => solution.ImplementationSteps.Count > 0 ? 0.7 : 0.3, // More complete if implementation steps provided
                    _ => 0.5 // Default score for unknown criteria
                };

                score += criterionScore * weight;
                totalWeight += weight;
            }

            // Calculate final score
            var finalScore = totalWeight > 0 ? score / totalWeight : 0.5;

            _logger.LogInformation("Solution evaluation score: {Score:F2}", finalScore);

            return finalScore;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error evaluating solution");
            return 0.5; // Default score
        }
    }

    /// <summary>
    /// Gets similar problems from history
    /// </summary>
    /// <param name="problem">The problem description</param>
    /// <param name="count">The number of similar problems to return</param>
    /// <returns>The similar problems</returns>
    public List<ProblemModel> GetSimilarProblems(string problem, int count)
    {
        var similarProblems = new List<ProblemModel>();

        try
        {
            // Extract concepts from the problem
            var problemConcepts = ExtractConcepts(problem);

            // Calculate similarity scores for each historical problem
            var scoredProblems = _problemHistory.Select(p =>
            {
                var historicalConcepts = ExtractConcepts(p.Description);
                var commonConcepts = problemConcepts.Intersect(historicalConcepts, StringComparer.OrdinalIgnoreCase).Count();
                var similarityScore = commonConcepts / (double)Math.Max(1, Math.Max(problemConcepts.Count, historicalConcepts.Count));
                return (Problem: p, Similarity: similarityScore);
            })
            .OrderByDescending(p => p.Similarity)
            .Take(count)
            .ToList();

            similarProblems = scoredProblems.Select(p => p.Problem).ToList();

            _logger.LogInformation("Found {Count} similar problems", similarProblems.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting similar problems");
        }

        return similarProblems;
    }

    /// <summary>
    /// Extracts concepts from text
    /// </summary>
    /// <param name="text">The text</param>
    /// <returns>The extracted concepts</returns>
    private List<string> ExtractConcepts(string text)
    {
        var concepts = new List<string>();

        // Simple concept extraction by splitting and filtering
        var words = text.Split([' ', ',', '.', ':', ';', '(', ')', '[', ']', '{', '}', '\n', '\r', '\t'],
            StringSplitOptions.RemoveEmptyEntries);

        foreach (var word in words)
        {
            // Only consider words of reasonable length
            if (word.Length >= 4 && word.Length <= 20)
            {
                // Convert to lowercase
                var concept = word.ToLowerInvariant();

                // Add if not already in list
                if (!concepts.Contains(concept))
                {
                    concepts.Add(concept);
                }
            }
        }

        // If no concepts found, add some default ones
        if (concepts.Count == 0)
        {
            concepts.Add("problem");
            concepts.Add("solution");
        }

        return concepts;
    }
}

/// <summary>
/// Represents a problem model
/// </summary>
public class ProblemModel
{
    /// <summary>
    /// Gets or sets the problem ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the problem description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the problem constraints
    /// </summary>
    public List<string> Constraints { get; set; } = [];

    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Gets or sets the solution ID
    /// </summary>
    public string SolutionId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the strategy name
    /// </summary>
    public string StrategyName { get; set; } = string.Empty;
}

/// <summary>
/// Represents a solution strategy
/// </summary>
public class SolutionStrategy
{
    /// <summary>
    /// Gets or sets the strategy name
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the strategy description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the process type
    /// </summary>
    public CreativeProcessType ProcessType { get; set; }

    /// <summary>
    /// Gets or sets the generate function
    /// </summary>
    public Func<string, List<string>, Task<CreativeIdea>> GenerateFunction { get; set; } = (_, __) => AsyncMonad.Return<CreativeIdea>(null);
}

