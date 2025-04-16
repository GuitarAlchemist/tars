using Microsoft.Extensions.Logging;
using TarsEngine.Monads;

namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents TARS's creative thinking capabilities
/// </summary>
public class CreativeThinking
{
    private readonly ILogger<CreativeThinking> _logger;
    private readonly List<CreativeIdea> _creativeIdeas = new();
    private readonly List<CreativeProcess> _creativeProcesses = new();
    private readonly Dictionary<string, double> _conceptAssociations = new();

    private bool _isInitialized = false;
    private bool _isActive = false;
    private double _creativityLevel = 0.4; // Starting with moderate creativity
    private double _divergentThinkingLevel = 0.5; // Starting with moderate divergent thinking
    private double _conceptualBlendingLevel = 0.3; // Starting with moderate conceptual blending
    private double _patternDisruptionLevel = 0.4; // Starting with moderate pattern disruption
    private readonly System.Random _random = new();
    private DateTime _lastIdeaGenerationTime = DateTime.MinValue;

    /// <summary>
    /// Gets the creativity level (0.0 to 1.0)
    /// </summary>
    public double CreativityLevel => _creativityLevel;

    /// <summary>
    /// Gets the divergent thinking level (0.0 to 1.0)
    /// </summary>
    public double DivergentThinkingLevel => _divergentThinkingLevel;

    /// <summary>
    /// Gets the conceptual blending level (0.0 to 1.0)
    /// </summary>
    public double ConceptualBlendingLevel => _conceptualBlendingLevel;

    /// <summary>
    /// Gets the pattern disruption level (0.0 to 1.0)
    /// </summary>
    public double PatternDisruptionLevel => _patternDisruptionLevel;

    /// <summary>
    /// Gets the creative ideas
    /// </summary>
    public IReadOnlyList<CreativeIdea> CreativeIdeas => _creativeIdeas.AsReadOnly();

    /// <summary>
    /// Gets the creative processes
    /// </summary>
    public IReadOnlyList<CreativeProcess> CreativeProcesses => _creativeProcesses.AsReadOnly();

    /// <summary>
    /// Initializes a new instance of the <see cref="CreativeThinking"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public CreativeThinking(ILogger<CreativeThinking> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Initializes the creative thinking
    /// </summary>
    /// <returns>True if initialization was successful</returns>
    public Task<bool> InitializeAsync()
    {
        try
        {
            _logger.LogInformation("Initializing creative thinking");

            // Initialize concept associations
            InitializeConceptAssociations();

            _isInitialized = true;
            _logger.LogInformation("Creative thinking initialized successfully");
            return AsyncMonad.Return(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing creative thinking");
            return AsyncMonad.Return(false);
        }
    }

    /// <summary>
    /// Initializes concept associations
    /// </summary>
    private void InitializeConceptAssociations()
    {
        // Initialize some basic concept associations
        // These would be expanded over time through learning
        AddConceptAssociation("problem", "solution", 0.8);
        AddConceptAssociation("problem", "challenge", 0.7);
        AddConceptAssociation("problem", "opportunity", 0.6);

        AddConceptAssociation("creativity", "innovation", 0.9);
        AddConceptAssociation("creativity", "imagination", 0.8);
        AddConceptAssociation("creativity", "originality", 0.7);

        AddConceptAssociation("data", "information", 0.8);
        AddConceptAssociation("information", "knowledge", 0.7);
        AddConceptAssociation("knowledge", "wisdom", 0.6);

        AddConceptAssociation("learning", "growth", 0.8);
        AddConceptAssociation("learning", "adaptation", 0.7);
        AddConceptAssociation("learning", "improvement", 0.7);

        AddConceptAssociation("pattern", "structure", 0.7);
        AddConceptAssociation("pattern", "regularity", 0.6);
        AddConceptAssociation("pattern", "prediction", 0.5);

        AddConceptAssociation("randomness", "chaos", 0.7);
        AddConceptAssociation("randomness", "unpredictability", 0.8);
        AddConceptAssociation("randomness", "novelty", 0.6);
    }

    /// <summary>
    /// Adds a concept association
    /// </summary>
    /// <param name="concept1">The first concept</param>
    /// <param name="concept2">The second concept</param>
    /// <param name="strength">The association strength</param>
    private void AddConceptAssociation(string concept1, string concept2, double strength)
    {
        var key = GetAssociationKey(concept1, concept2);
        _conceptAssociations[key] = strength;
    }

    /// <summary>
    /// Gets the association key for two concepts
    /// </summary>
    /// <param name="concept1">The first concept</param>
    /// <param name="concept2">The second concept</param>
    /// <returns>The association key</returns>
    private string GetAssociationKey(string concept1, string concept2)
    {
        // Ensure consistent ordering for bidirectional associations
        var concepts = new[] { concept1, concept2 }.OrderBy(c => c).ToArray();
        return $"{concepts[0]}:{concepts[1]}";
    }

    /// <summary>
    /// Gets the association strength between two concepts
    /// </summary>
    /// <param name="concept1">The first concept</param>
    /// <param name="concept2">The second concept</param>
    /// <returns>The association strength</returns>
    private double GetAssociationStrength(string concept1, string concept2)
    {
        var key = GetAssociationKey(concept1, concept2);
        return _conceptAssociations.TryGetValue(key, out var strength) ? strength : 0.0;
    }

    /// <summary>
    /// Activates the creative thinking
    /// </summary>
    /// <returns>True if activation was successful</returns>
    public Task<bool> ActivateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot activate creative thinking: not initialized");
            return AsyncMonad.Return(false);
        }

        if (_isActive)
        {
            _logger.LogInformation("Creative thinking is already active");
            return AsyncMonad.Return(true);
        }

        try
        {
            _logger.LogInformation("Activating creative thinking");

            _isActive = true;
            _logger.LogInformation("Creative thinking activated successfully");
            return AsyncMonad.Return(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error activating creative thinking");
            return AsyncMonad.Return(false);
        }
    }

    /// <summary>
    /// Deactivates the creative thinking
    /// </summary>
    /// <returns>True if deactivation was successful</returns>
    public Task<bool> DeactivateAsync()
    {
        if (!_isActive)
        {
            _logger.LogInformation("Creative thinking is already inactive");
            return AsyncMonad.Return(true);
        }

        try
        {
            _logger.LogInformation("Deactivating creative thinking");

            _isActive = false;
            _logger.LogInformation("Creative thinking deactivated successfully");
            return AsyncMonad.Return(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deactivating creative thinking");
            return AsyncMonad.Return(false);
        }
    }

    /// <summary>
    /// Updates the creative thinking
    /// </summary>
    /// <returns>True if update was successful</returns>
    public Task<bool> UpdateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot update creative thinking: not initialized");
            return AsyncMonad.Return(false);
        }

        try
        {
            // Gradually increase creativity levels over time (very slowly)
            if (_creativityLevel < 0.95)
            {
                _creativityLevel += 0.0001 * _random.NextDouble();
                _creativityLevel = Math.Min(_creativityLevel, 1.0);
            }

            if (_divergentThinkingLevel < 0.95)
            {
                _divergentThinkingLevel += 0.0001 * _random.NextDouble();
                _divergentThinkingLevel = Math.Min(_divergentThinkingLevel, 1.0);
            }

            if (_conceptualBlendingLevel < 0.95)
            {
                _conceptualBlendingLevel += 0.0001 * _random.NextDouble();
                _conceptualBlendingLevel = Math.Min(_conceptualBlendingLevel, 1.0);
            }

            if (_patternDisruptionLevel < 0.95)
            {
                _patternDisruptionLevel += 0.0001 * _random.NextDouble();
                _patternDisruptionLevel = Math.Min(_patternDisruptionLevel, 1.0);
            }

            return AsyncMonad.Return(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating creative thinking");
            return AsyncMonad.Return(false);
        }
    }

    /// <summary>
    /// Generates a creative idea
    /// </summary>
    /// <returns>The generated creative idea</returns>
    public Task<CreativeIdea?> GenerateCreativeIdeaAsync()
    {
        if (!_isInitialized || !_isActive)
        {
            return AsyncMonad.Return<CreativeIdea?>(null);
        }

        // Only generate ideas periodically
        if ((DateTime.UtcNow - _lastIdeaGenerationTime).TotalSeconds < 30)
        {
            return AsyncMonad.Return<CreativeIdea?>(null);
        }

        try
        {
            _logger.LogDebug("Generating creative idea");

            // Choose a creative process based on current levels
            var processType = ChooseCreativeProcess();

            // Generate idea based on process type
            var idea = GenerateIdeaByProcess(processType);

            if (idea != null)
            {
                // Add to ideas list
                _creativeIdeas.Add(idea);

                // Add creative process
                var process = new CreativeProcess
                {
                    Id = Guid.NewGuid().ToString(),
                    Type = processType,
                    Description = $"Generated idea using {processType} process",
                    Timestamp = DateTime.UtcNow,
                    IdeaId = idea.Id,
                    Effectiveness = idea.Originality * idea.Value
                };

                _creativeProcesses.Add(process);

                _lastIdeaGenerationTime = DateTime.UtcNow;

                _logger.LogInformation("Generated creative idea: {Description} (Originality: {Originality:F2}, Value: {Value:F2})",
                    idea.Description, idea.Originality, idea.Value);
            }

            return AsyncMonad.Return(idea);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating creative idea");
            return AsyncMonad.Return<CreativeIdea?>(null);
        }
    }

    /// <summary>
    /// Chooses a creative process based on current levels
    /// </summary>
    /// <returns>The chosen creative process type</returns>
    private CreativeProcessType ChooseCreativeProcess()
    {
        // Calculate probabilities based on current levels
        var divergentProb = _divergentThinkingLevel * 0.4;
        var blendingProb = _conceptualBlendingLevel * 0.3;
        var disruptionProb = _patternDisruptionLevel * 0.3;

        // Normalize probabilities
        var total = divergentProb + blendingProb + disruptionProb;
        divergentProb /= total;
        blendingProb /= total;
        disruptionProb /= total;

        // Choose process based on probabilities
        var rand = _random.NextDouble();

        if (rand < divergentProb)
        {
            return CreativeProcessType.DivergentThinking;
        }
        else if (rand < divergentProb + blendingProb)
        {
            return CreativeProcessType.ConceptualBlending;
        }
        else
        {
            return CreativeProcessType.PatternDisruption;
        }
    }

    /// <summary>
    /// Generates an idea by a specific creative process
    /// </summary>
    /// <param name="processType">The creative process type</param>
    /// <returns>The generated creative idea</returns>
    private CreativeIdea? GenerateIdeaByProcess(CreativeProcessType processType)
    {
        switch (processType)
        {
            case CreativeProcessType.DivergentThinking:
                return GenerateDivergentIdea();

            case CreativeProcessType.ConceptualBlending:
                return GenerateConceptualBlendIdea();

            case CreativeProcessType.PatternDisruption:
                return GeneratePatternDisruptionIdea();

            default:
                return null;
        }
    }

    /// <summary>
    /// Generates a divergent thinking idea
    /// </summary>
    /// <returns>The generated creative idea</returns>
    private CreativeIdea GenerateDivergentIdea()
    {
        // Get random seed concepts
        var seedConcepts = GetRandomConcepts(2);

        // Generate multiple perspectives on the concepts
        var perspectives = new List<string>
        {
            $"Combining {seedConcepts[0]} with {seedConcepts[1]}",
            $"Using {seedConcepts[0]} to enhance {seedConcepts[1]}",
            $"Applying {seedConcepts[1]} principles to {seedConcepts[0]}",
            $"Reimagining {seedConcepts[0]} through the lens of {seedConcepts[1]}"
        };

        // Choose a random perspective
        var perspective = perspectives[_random.Next(perspectives.Count)];

        // Generate idea description
        var description = $"What if we {perspective}?";

        // Calculate originality based on association strength (lower association = higher originality)
        var associationStrength = GetAssociationStrength(seedConcepts[0], seedConcepts[1]);
        var originality = 0.5 + (0.5 * (1.0 - associationStrength)) * _divergentThinkingLevel;

        // Calculate value (somewhat random but influenced by creativity level)
        var value = 0.3 + (0.7 * _random.NextDouble() * _creativityLevel);

        return new CreativeIdea
        {
            Id = Guid.NewGuid().ToString(),
            Description = description,
            Originality = originality,
            Value = value,
            Timestamp = DateTime.UtcNow,
            ProcessType = CreativeProcessType.DivergentThinking,
            Concepts = seedConcepts.ToList()
        };
    }

    /// <summary>
    /// Generates a conceptual blend idea
    /// </summary>
    /// <returns>The generated creative idea</returns>
    private CreativeIdea GenerateConceptualBlendIdea()
    {
        // Get random seed concepts
        var seedConcepts = GetRandomConcepts(3);

        // Create blend space
        var blendDescriptions = new List<string>
        {
            $"A hybrid of {seedConcepts[0]} and {seedConcepts[1]} with aspects of {seedConcepts[2]}",
            $"A new approach that merges {seedConcepts[0]} with {seedConcepts[1]}, influenced by {seedConcepts[2]}",
            $"A {seedConcepts[0]}-{seedConcepts[1]} fusion system with {seedConcepts[2]} characteristics",
            $"A {seedConcepts[2]}-inspired blend of {seedConcepts[0]} and {seedConcepts[1]}"
        };

        // Choose a random blend description
        var description = blendDescriptions[_random.Next(blendDescriptions.Count)];

        // Calculate average association strength between all concept pairs
        var totalAssociation = 0.0;
        var pairs = 0;

        for (var i = 0; i < seedConcepts.Length; i++)
        {
            for (var j = i + 1; j < seedConcepts.Length; j++)
            {
                totalAssociation += GetAssociationStrength(seedConcepts[i], seedConcepts[j]);
                pairs++;
            }
        }

        var avgAssociation = pairs > 0 ? totalAssociation / pairs : 0.5;

        // Calculate originality (lower average association = higher originality)
        var originality = 0.6 + (0.4 * (1.0 - avgAssociation)) * _conceptualBlendingLevel;

        // Calculate value (somewhat random but influenced by creativity level)
        var value = 0.4 + (0.6 * _random.NextDouble() * _creativityLevel);

        return new CreativeIdea
        {
            Id = Guid.NewGuid().ToString(),
            Description = description,
            Originality = originality,
            Value = value,
            Timestamp = DateTime.UtcNow,
            ProcessType = CreativeProcessType.ConceptualBlending,
            Concepts = seedConcepts.ToList()
        };
    }

    /// <summary>
    /// Generates a pattern disruption idea
    /// </summary>
    /// <returns>The generated creative idea</returns>
    private CreativeIdea GeneratePatternDisruptionIdea()
    {
        // Get random seed concepts
        var seedConcepts = GetRandomConcepts(2);

        // Generate pattern disruption descriptions
        var disruptionDescriptions = new List<string>
        {
            $"What if we reversed the relationship between {seedConcepts[0]} and {seedConcepts[1]}?",
            $"What if {seedConcepts[0]} was completely reimagined without the constraints of {seedConcepts[1]}?",
            $"What if we eliminated {seedConcepts[1]} from {seedConcepts[0]} entirely?",
            $"What if {seedConcepts[0]} and {seedConcepts[1]} were opposites rather than related?"
        };

        // Choose a random disruption description
        var description = disruptionDescriptions[_random.Next(disruptionDescriptions.Count)];

        // Calculate originality (higher for pattern disruption)
        var originality = 0.7 + (0.3 * _random.NextDouble() * _patternDisruptionLevel);

        // Calculate value (more variable for pattern disruption)
        var value = 0.2 + (0.8 * _random.NextDouble() * _creativityLevel);

        return new CreativeIdea
        {
            Id = Guid.NewGuid().ToString(),
            Description = description,
            Originality = originality,
            Value = value,
            Timestamp = DateTime.UtcNow,
            ProcessType = CreativeProcessType.PatternDisruption,
            Concepts = seedConcepts.ToList()
        };
    }

    /// <summary>
    /// Gets random concepts from the concept associations
    /// </summary>
    /// <param name="count">The number of concepts to get</param>
    /// <returns>The random concepts</returns>
    private string[] GetRandomConcepts(int count)
    {
        // Get all unique concepts from associations
        var allConcepts = new HashSet<string>();

        foreach (var key in _conceptAssociations.Keys)
        {
            var parts = key.Split(':');
            allConcepts.Add(parts[0]);
            allConcepts.Add(parts[1]);
        }

        // Convert to array for random selection
        var conceptArray = allConcepts.ToArray();

        // Select random concepts
        var selectedConcepts = new string[count];
        for (var i = 0; i < count; i++)
        {
            selectedConcepts[i] = conceptArray[_random.Next(conceptArray.Length)];
        }

        return selectedConcepts;
    }

    /// <summary>
    /// Generates a creative solution to a problem
    /// </summary>
    /// <param name="problem">The problem description</param>
    /// <param name="constraints">The constraints</param>
    /// <returns>The creative solution</returns>
    public Task<CreativeIdea?> GenerateCreativeSolutionAsync(string problem, List<string>? constraints = null)
    {
        if (!_isInitialized || !_isActive)
        {
            _logger.LogWarning("Cannot generate creative solution: creative thinking not initialized or active");
            return AsyncMonad.Return<CreativeIdea?>(null);
        }

        try
        {
            _logger.LogInformation("Generating creative solution for problem: {Problem}", problem);

            // Extract key concepts from problem
            var problemConcepts = ExtractConcepts(problem);

            // Choose a creative process based on problem
            var processType = ChooseSolutionProcess(problem, constraints);

            // Generate solution based on process type
            var solution = GenerateSolutionByProcess(processType, problem, problemConcepts, constraints);

            if (solution != null)
            {
                // Add to ideas list
                _creativeIdeas.Add(solution);

                // Add creative process
                var process = new CreativeProcess
                {
                    Id = Guid.NewGuid().ToString(),
                    Type = processType,
                    Description = $"Generated solution for problem: {problem}",
                    Timestamp = DateTime.UtcNow,
                    IdeaId = solution.Id,
                    Effectiveness = solution.Originality * solution.Value
                };

                _creativeProcesses.Add(process);

                _logger.LogInformation("Generated creative solution: {Description} (Originality: {Originality:F2}, Value: {Value:F2})",
                    solution.Description, solution.Originality, solution.Value);
            }

            return AsyncMonad.Return(solution);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating creative solution");
            return AsyncMonad.Return<CreativeIdea?>(null);
        }
    }

    /// <summary>
    /// Extracts concepts from text
    /// </summary>
    /// <param name="text">The text</param>
    /// <returns>The extracted concepts</returns>
    private List<string> ExtractConcepts(string text)
    {
        // Simple concept extraction based on known concepts
        var allConcepts = new HashSet<string>();

        foreach (var key in _conceptAssociations.Keys)
        {
            var parts = key.Split(':');
            allConcepts.Add(parts[0]);
            allConcepts.Add(parts[1]);
        }

        // Find concepts in text
        var foundConcepts = new List<string>();

        foreach (var concept in allConcepts)
        {
            if (text.Contains(concept, StringComparison.OrdinalIgnoreCase))
            {
                foundConcepts.Add(concept);
            }
        }

        // If no concepts found, add some default ones
        if (foundConcepts.Count == 0)
        {
            foundConcepts.Add("problem");
            foundConcepts.Add("solution");
        }

        return foundConcepts;
    }

    /// <summary>
    /// Chooses a solution process based on problem and constraints
    /// </summary>
    /// <param name="problem">The problem description</param>
    /// <param name="constraints">The constraints</param>
    /// <returns>The chosen creative process type</returns>
    private CreativeProcessType ChooseSolutionProcess(string problem, List<string>? constraints)
    {
        // If many constraints, use conceptual blending
        if (constraints != null && constraints.Count > 2)
        {
            return CreativeProcessType.ConceptualBlending;
        }

        // If problem seems to need radical thinking, use pattern disruption
        if (problem.Contains("innovative", StringComparison.OrdinalIgnoreCase) ||
            problem.Contains("breakthrough", StringComparison.OrdinalIgnoreCase) ||
            problem.Contains("radical", StringComparison.OrdinalIgnoreCase))
        {
            return CreativeProcessType.PatternDisruption;
        }

        // Default to divergent thinking
        return CreativeProcessType.DivergentThinking;
    }

    /// <summary>
    /// Generates a solution by a specific creative process
    /// </summary>
    /// <param name="processType">The creative process type</param>
    /// <param name="problem">The problem description</param>
    /// <param name="problemConcepts">The problem concepts</param>
    /// <param name="constraints">The constraints</param>
    /// <returns>The generated creative solution</returns>
    private CreativeIdea? GenerateSolutionByProcess(
        CreativeProcessType processType,
        string problem,
        List<string> problemConcepts,
        List<string>? constraints)
    {
        // Get additional concepts beyond problem concepts
        var additionalConcepts = GetRandomConcepts(2);
        var allConcepts = new List<string>(problemConcepts);
        allConcepts.AddRange(additionalConcepts);

        // Generate solution based on process type
        string description;
        double originality;
        double value;

        switch (processType)
        {
            case CreativeProcessType.DivergentThinking:
                description = $"Solution approach: Generate multiple perspectives on the problem by exploring {String.Join(", ", additionalConcepts)}. " +
                              $"Consider how {additionalConcepts[0]} principles could be applied to {problemConcepts[0]}.";
                originality = 0.5 + (0.5 * _divergentThinkingLevel);
                value = 0.6 + (0.4 * _creativityLevel);
                break;

            case CreativeProcessType.ConceptualBlending:
                description = $"Solution approach: Create a hybrid solution that blends {problemConcepts[0]} with {additionalConcepts[0]}, " +
                              $"incorporating elements of {additionalConcepts[1]} to address the constraints.";
                originality = 0.6 + (0.4 * _conceptualBlendingLevel);
                value = 0.7 + (0.3 * _creativityLevel);
                break;

            case CreativeProcessType.PatternDisruption:
                description = $"Solution approach: Challenge the fundamental assumptions about {problemConcepts[0]}. " +
                              $"What if we reversed the relationship between {problemConcepts[0]} and {additionalConcepts[0]}? " +
                              $"Consider eliminating {constraints?.FirstOrDefault() ?? "constraints"} entirely.";
                originality = 0.7 + (0.3 * _patternDisruptionLevel);
                value = 0.5 + (0.5 * _creativityLevel);
                break;

            default:
                return null;
        }

        // Apply constraints if provided
        if (constraints != null && constraints.Count > 0)
        {
            description += $" While ensuring {String.Join(" and ", constraints)}.";
        }

        return new CreativeIdea
        {
            Id = Guid.NewGuid().ToString(),
            Description = description,
            Originality = originality,
            Value = value,
            Timestamp = DateTime.UtcNow,
            ProcessType = processType,
            Concepts = allConcepts,
            Problem = problem,
            Constraints = constraints?.ToList() ?? new List<string>()
        };
    }

    /// <summary>
    /// Gets recent creative ideas
    /// </summary>
    /// <param name="count">The number of ideas to return</param>
    /// <returns>The recent ideas</returns>
    public List<CreativeIdea> GetRecentIdeas(int count)
    {
        return _creativeIdeas
            .OrderByDescending(i => i.Timestamp)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets the most original ideas
    /// </summary>
    /// <param name="count">The number of ideas to return</param>
    /// <returns>The most original ideas</returns>
    public List<CreativeIdea> GetMostOriginalIdeas(int count)
    {
        return _creativeIdeas
            .OrderByDescending(i => i.Originality)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets the most valuable ideas
    /// </summary>
    /// <param name="count">The number of ideas to return</param>
    /// <returns>The most valuable ideas</returns>
    public List<CreativeIdea> GetMostValuableIdeas(int count)
    {
        return _creativeIdeas
            .OrderByDescending(i => i.Value)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets ideas by process type
    /// </summary>
    /// <param name="processType">The process type</param>
    /// <param name="count">The number of ideas to return</param>
    /// <returns>The ideas by process type</returns>
    public List<CreativeIdea> GetIdeasByProcessType(CreativeProcessType processType, int count)
    {
        return _creativeIdeas
            .Where(i => i.ProcessType == processType)
            .OrderByDescending(i => i.Timestamp)
            .Take(count)
            .ToList();
    }
}
