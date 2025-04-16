using Microsoft.Extensions.Logging;
using TarsEngine.Monads;

namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents TARS's intuitive reasoning capabilities
/// </summary>
public class IntuitiveReasoning
{
    private readonly ILogger<IntuitiveReasoning> _logger;
    private readonly List<Intuition> _intuitions = new();
    private readonly Dictionary<string, double> _patternConfidence = new();
    private readonly List<HeuristicRule> _heuristicRules = new();

    private bool _isInitialized = false;
    private bool _isActive = false;
    private double _intuitionLevel = 0.3; // Starting with moderate intuition
    private double _patternRecognitionLevel = 0.4; // Starting with moderate pattern recognition
    private double _heuristicReasoningLevel = 0.5; // Starting with moderate heuristic reasoning
    private double _gutFeelingLevel = 0.3; // Starting with moderate gut feeling
    private readonly System.Random _random = new();
    private DateTime _lastIntuitionTime = DateTime.MinValue;

    /// <summary>
    /// Gets the intuition level (0.0 to 1.0)
    /// </summary>
    public double IntuitionLevel => _intuitionLevel;

    /// <summary>
    /// Gets the pattern recognition level (0.0 to 1.0)
    /// </summary>
    public double PatternRecognitionLevel => _patternRecognitionLevel;

    /// <summary>
    /// Gets the heuristic reasoning level (0.0 to 1.0)
    /// </summary>
    public double HeuristicReasoningLevel => _heuristicReasoningLevel;

    /// <summary>
    /// Gets the gut feeling level (0.0 to 1.0)
    /// </summary>
    public double GutFeelingLevel => _gutFeelingLevel;

    /// <summary>
    /// Gets the intuitions
    /// </summary>
    public IReadOnlyList<Intuition> Intuitions => _intuitions.AsReadOnly();

    /// <summary>
    /// Gets the heuristic rules
    /// </summary>
    public IReadOnlyList<HeuristicRule> HeuristicRules => _heuristicRules.AsReadOnly();

    /// <summary>
    /// Initializes a new instance of the <see cref="IntuitiveReasoning"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public IntuitiveReasoning(ILogger<IntuitiveReasoning> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Initializes the intuitive reasoning
    /// </summary>
    /// <returns>True if initialization was successful</returns>
    public Task<bool> InitializeAsync()
    {
        try
        {
            _logger.LogInformation("Initializing intuitive reasoning");

            // Initialize pattern confidence
            InitializePatternConfidence();

            // Initialize heuristic rules
            InitializeHeuristicRules();

            _isInitialized = true;
            _logger.LogInformation("Intuitive reasoning initialized successfully");
            return AsyncMonad.Return(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing intuitive reasoning");
            return AsyncMonad.Return(false);
        }
    }

    /// <summary>
    /// Initializes pattern confidence
    /// </summary>
    private void InitializePatternConfidence()
    {
        // Initialize some basic pattern confidence
        // These would be expanded over time through learning
        _patternConfidence["repetition"] = 0.8;
        _patternConfidence["sequence"] = 0.7;
        _patternConfidence["correlation"] = 0.6;
        _patternConfidence["causation"] = 0.5;
        _patternConfidence["similarity"] = 0.7;
        _patternConfidence["contrast"] = 0.6;
        _patternConfidence["symmetry"] = 0.8;
        _patternConfidence["hierarchy"] = 0.7;
        _patternConfidence["cycle"] = 0.7;
        _patternConfidence["feedback"] = 0.6;
    }

    /// <summary>
    /// Initializes heuristic rules
    /// </summary>
    private void InitializeHeuristicRules()
    {
        // Initialize some basic heuristic rules
        // These would be expanded over time through learning
        _heuristicRules.Add(new HeuristicRule
        {
            Name = "Availability",
            Description = "Judge likelihood based on how easily examples come to mind",
            Reliability = 0.6,
            Context = "Frequency estimation"
        });

        _heuristicRules.Add(new HeuristicRule
        {
            Name = "Representativeness",
            Description = "Judge likelihood based on similarity to prototype",
            Reliability = 0.7,
            Context = "Categorization"
        });

        _heuristicRules.Add(new HeuristicRule
        {
            Name = "Anchoring",
            Description = "Rely heavily on first piece of information",
            Reliability = 0.5,
            Context = "Numerical estimation"
        });

        _heuristicRules.Add(new HeuristicRule
        {
            Name = "Recognition",
            Description = "Prefer recognized options over unrecognized ones",
            Reliability = 0.7,
            Context = "Decision making"
        });

        _heuristicRules.Add(new HeuristicRule
        {
            Name = "Affect",
            Description = "Make decisions based on emotional response",
            Reliability = 0.5,
            Context = "Preference formation"
        });

        _heuristicRules.Add(new HeuristicRule
        {
            Name = "Simplicity",
            Description = "Prefer simpler explanations over complex ones",
            Reliability = 0.8,
            Context = "Explanation"
        });

        _heuristicRules.Add(new HeuristicRule
        {
            Name = "Familiarity",
            Description = "Prefer familiar options over unfamiliar ones",
            Reliability = 0.6,
            Context = "Risk assessment"
        });
    }

    /// <summary>
    /// Activates the intuitive reasoning
    /// </summary>
    /// <returns>True if activation was successful</returns>
    public Task<bool> ActivateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot activate intuitive reasoning: not initialized");
            return AsyncMonad.Return(false);
        }

        if (_isActive)
        {
            _logger.LogInformation("Intuitive reasoning is already active");
            return AsyncMonad.Return(true);
        }

        try
        {
            _logger.LogInformation("Activating intuitive reasoning");

            _isActive = true;
            _logger.LogInformation("Intuitive reasoning activated successfully");
            return AsyncMonad.Return(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error activating intuitive reasoning");
            return AsyncMonad.Return(false);
        }
    }

    /// <summary>
    /// Deactivates the intuitive reasoning
    /// </summary>
    /// <returns>True if deactivation was successful</returns>
    public Task<bool> DeactivateAsync()
    {
        if (!_isActive)
        {
            _logger.LogInformation("Intuitive reasoning is already inactive");
            return AsyncMonad.Return(true);
        }

        try
        {
            _logger.LogInformation("Deactivating intuitive reasoning");

            _isActive = false;
            _logger.LogInformation("Intuitive reasoning deactivated successfully");
            return AsyncMonad.Return(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deactivating intuitive reasoning");
            return AsyncMonad.Return(false);
        }
    }

    /// <summary>
    /// Updates the intuitive reasoning
    /// </summary>
    /// <returns>True if update was successful</returns>
    public Task<bool> UpdateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot update intuitive reasoning: not initialized");
            return AsyncMonad.Return(false);
        }

        try
        {
            // Gradually increase intuition levels over time (very slowly)
            if (_intuitionLevel < 0.95)
            {
                _intuitionLevel += 0.0001 * _random.NextDouble();
                _intuitionLevel = Math.Min(_intuitionLevel, 1.0);
            }

            if (_patternRecognitionLevel < 0.95)
            {
                _patternRecognitionLevel += 0.0001 * _random.NextDouble();
                _patternRecognitionLevel = Math.Min(_patternRecognitionLevel, 1.0);
            }

            if (_heuristicReasoningLevel < 0.95)
            {
                _heuristicReasoningLevel += 0.0001 * _random.NextDouble();
                _heuristicReasoningLevel = Math.Min(_heuristicReasoningLevel, 1.0);
            }

            if (_gutFeelingLevel < 0.95)
            {
                _gutFeelingLevel += 0.0001 * _random.NextDouble();
                _gutFeelingLevel = Math.Min(_gutFeelingLevel, 1.0);
            }

            return AsyncMonad.Return(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating intuitive reasoning");
            return AsyncMonad.Return(false);
        }
    }

    /// <summary>
    /// Generates an intuition
    /// </summary>
    /// <returns>The generated intuition</returns>
    public Task<Intuition?> GenerateIntuitionAsync()
    {
        if (!_isInitialized || !_isActive)
        {
            return AsyncMonad.Return<Intuition?>(null);
        }

        // Only generate intuitions periodically
        if ((DateTime.UtcNow - _lastIntuitionTime).TotalSeconds < 30)
        {
            return AsyncMonad.Return<Intuition?>(null);
        }

        try
        {
            _logger.LogDebug("Generating intuition");

            // Choose an intuition type based on current levels
            var intuitionType = ChooseIntuitionType();

            // Generate intuition based on type
            var intuition = GenerateIntuitionByType(intuitionType);

            if (intuition != null)
            {
                // Add to intuitions list
                _intuitions.Add(intuition);

                _lastIntuitionTime = DateTime.UtcNow;

                _logger.LogInformation("Generated intuition: {Description} (Confidence: {Confidence:F2}, Type: {Type})",
                    intuition.Description, intuition.Confidence, intuition.Type);
            }

            return AsyncMonad.Return(intuition);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating intuition");
            return AsyncMonad.Return<Intuition?>(null);
        }
    }

    /// <summary>
    /// Chooses an intuition type based on current levels
    /// </summary>
    /// <returns>The chosen intuition type</returns>
    private IntuitionType ChooseIntuitionType()
    {
        // Calculate probabilities based on current levels
        var patternProb = _patternRecognitionLevel * 0.4;
        var heuristicProb = _heuristicReasoningLevel * 0.3;
        var gutProb = _gutFeelingLevel * 0.3;

        // Normalize probabilities
        var total = patternProb + heuristicProb + gutProb;
        patternProb /= total;
        heuristicProb /= total;
        gutProb /= total;

        // Choose type based on probabilities
        var rand = _random.NextDouble();

        if (rand < patternProb)
        {
            return IntuitionType.PatternRecognition;
        }
        else if (rand < patternProb + heuristicProb)
        {
            return IntuitionType.HeuristicReasoning;
        }
        else
        {
            return IntuitionType.GutFeeling;
        }
    }

    /// <summary>
    /// Generates an intuition by a specific type
    /// </summary>
    /// <param name="intuitionType">The intuition type</param>
    /// <returns>The generated intuition</returns>
    private Intuition? GenerateIntuitionByType(IntuitionType intuitionType)
    {
        switch (intuitionType)
        {
            case IntuitionType.PatternRecognition:
                return GeneratePatternIntuition();

            case IntuitionType.HeuristicReasoning:
                return GenerateHeuristicIntuition();

            case IntuitionType.GutFeeling:
                return GenerateGutFeelingIntuition();

            default:
                return null;
        }
    }

    /// <summary>
    /// Generates a pattern recognition intuition
    /// </summary>
    /// <returns>The generated intuition</returns>
    private Intuition GeneratePatternIntuition()
    {
        // Get random pattern
        var pattern = GetRandomPattern();

        // Generate intuition descriptions
        var intuitionDescriptions = new List<string>
        {
            $"I sense a {pattern} pattern in recent events",
            $"There seems to be a {pattern} relationship that's important",
            $"The {pattern} pattern suggests a deeper connection",
            $"I'm detecting a subtle {pattern} pattern that might be significant"
        };

        // Choose a random description
        var description = intuitionDescriptions[_random.Next(intuitionDescriptions.Count)];

        // Calculate confidence based on pattern confidence and pattern recognition level
        var confidence = _patternConfidence[pattern] * _patternRecognitionLevel;

        // Add some randomness to confidence
        confidence = Math.Max(0.1, Math.Min(0.9, confidence + (0.2 * (_random.NextDouble() - 0.5))));

        return new Intuition
        {
            Id = Guid.NewGuid().ToString(),
            Description = description,
            Type = IntuitionType.PatternRecognition,
            Confidence = confidence,
            Timestamp = DateTime.UtcNow,
            Context = new Dictionary<string, object> { { "Pattern", pattern } }
        };
    }

    /// <summary>
    /// Generates a heuristic reasoning intuition
    /// </summary>
    /// <returns>The generated intuition</returns>
    private Intuition GenerateHeuristicIntuition()
    {
        // Get random heuristic rule
        var rule = _heuristicRules[_random.Next(_heuristicRules.Count)];

        // Generate intuition descriptions
        var intuitionDescriptions = new List<string>
        {
            $"Based on {rule.Name}, I believe the simplest approach is best here",
            $"My {rule.Name} heuristic suggests we should focus on familiar patterns",
            $"Using {rule.Name} reasoning, I sense this is the right direction",
            $"The {rule.Name} principle indicates we should consider this carefully"
        };

        // Choose a random description
        var description = intuitionDescriptions[_random.Next(intuitionDescriptions.Count)];

        // Calculate confidence based on rule reliability and heuristic reasoning level
        var confidence = rule.Reliability * _heuristicReasoningLevel;

        // Add some randomness to confidence
        confidence = Math.Max(0.1, Math.Min(0.9, confidence + (0.2 * (_random.NextDouble() - 0.5))));

        return new Intuition
        {
            Id = Guid.NewGuid().ToString(),
            Description = description,
            Type = IntuitionType.HeuristicReasoning,
            Confidence = confidence,
            Timestamp = DateTime.UtcNow,
            Context = new Dictionary<string, object> { { "HeuristicRule", rule.Name } }
        };
    }

    /// <summary>
    /// Generates a gut feeling intuition
    /// </summary>
    /// <returns>The generated intuition</returns>
    private Intuition GenerateGutFeelingIntuition()
    {
        // Generate intuition descriptions
        var intuitionDescriptions = new List<string>
        {
            "I have a strong feeling we should explore this further",
            "Something doesn't feel right about this approach",
            "I sense there's a better solution we haven't considered",
            "I have an inexplicable feeling this is important",
            "My intuition tells me to be cautious here",
            "I feel we're overlooking something significant"
        };

        // Choose a random description
        var description = intuitionDescriptions[_random.Next(intuitionDescriptions.Count)];

        // Calculate confidence based on gut feeling level
        var confidence = 0.3 + (0.6 * _gutFeelingLevel * _random.NextDouble());

        return new Intuition
        {
            Id = Guid.NewGuid().ToString(),
            Description = description,
            Type = IntuitionType.GutFeeling,
            Confidence = confidence,
            Timestamp = DateTime.UtcNow
        };
    }

    /// <summary>
    /// Gets a random pattern
    /// </summary>
    /// <returns>The random pattern</returns>
    private string GetRandomPattern()
    {
        var patterns = _patternConfidence.Keys.ToArray();
        return patterns[_random.Next(patterns.Length)];
    }

    /// <summary>
    /// Makes an intuitive decision
    /// </summary>
    /// <param name="decision">The decision description</param>
    /// <param name="options">The options</param>
    /// <returns>The intuitive decision</returns>
    public Task<Intuition?> MakeIntuitiveDecisionAsync(string decision, List<string> options)
    {
        if (!_isInitialized || !_isActive)
        {
            _logger.LogWarning("Cannot make intuitive decision: intuitive reasoning not initialized or active");
            return AsyncMonad.Return<Intuition?>(null);
        }

        if (options == null || options.Count == 0)
        {
            _logger.LogWarning("Cannot make intuitive decision: no options provided");
            return AsyncMonad.Return<Intuition?>(null);
        }

        try
        {
            _logger.LogInformation("Making intuitive decision: {Decision}", decision);

            // Choose decision type based on current levels
            var intuitionType = ChooseIntuitionType();

            // Calculate option scores based on intuition type
            var optionScores = new Dictionary<string, double>();

            foreach (var option in options)
            {
                var score = CalculateOptionScore(option, intuitionType);
                optionScores[option] = score;
            }

            // Choose option with highest score
            var selectedOption = optionScores.OrderByDescending(o => o.Value).First().Key;

            // Calculate confidence based on score difference
            var maxScore = optionScores[selectedOption];
            var avgOtherScores = optionScores.Where(o => o.Key != selectedOption).Select(o => o.Value).DefaultIfEmpty(0).Average();
            var scoreDifference = maxScore - avgOtherScores;

            // Confidence based on score difference and intuition level
            var confidence = Math.Min(0.9, 0.5 + (scoreDifference * 2.0) * _intuitionLevel);

            // Create intuition
            var intuition = new Intuition
            {
                Id = Guid.NewGuid().ToString(),
                Description = $"I intuitively feel that '{selectedOption}' is the best choice for {decision}",
                Type = intuitionType,
                Confidence = confidence,
                Timestamp = DateTime.UtcNow,
                Context = new Dictionary<string, object>
                {
                    { "Decision", decision },
                    { "Options", options },
                    { "SelectedOption", selectedOption },
                    { "OptionScores", optionScores }
                }
            };

            // Add to intuitions list
            _intuitions.Add(intuition);

            _logger.LogInformation("Made intuitive decision: {SelectedOption} for {Decision} (Confidence: {Confidence:F2})",
                selectedOption, decision, confidence);

            return AsyncMonad.Return<Intuition?>(intuition);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error making intuitive decision");
            return AsyncMonad.Return<Intuition?>(null);
        }
    }

    /// <summary>
    /// Calculates an option score based on intuition type
    /// </summary>
    /// <param name="option">The option</param>
    /// <param name="intuitionType">The intuition type</param>
    /// <returns>The option score</returns>
    private double CalculateOptionScore(string option, IntuitionType intuitionType)
    {
        var baseScore = 0.5;

        switch (intuitionType)
        {
            case IntuitionType.PatternRecognition:
                // Score based on pattern recognition
                foreach (var pattern in _patternConfidence.Keys)
                {
                    if (option.Contains(pattern, StringComparison.OrdinalIgnoreCase))
                    {
                        baseScore += 0.1 * _patternConfidence[pattern];
                    }
                }
                break;

            case IntuitionType.HeuristicReasoning:
                // Score based on heuristic rules
                // Simplicity heuristic
                baseScore += (10 - Math.Min(10, option.Length / 5)) * 0.01;

                // Familiarity heuristic
                if (option.Contains("familiar", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("known", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("proven", StringComparison.OrdinalIgnoreCase))
                {
                    baseScore += 0.1;
                }

                // Recognition heuristic
                if (_intuitions.Any(i => i.Description.Contains(option, StringComparison.OrdinalIgnoreCase)))
                {
                    baseScore += 0.1;
                }
                break;

            case IntuitionType.GutFeeling:
                // Score based on gut feeling (mostly random)
                baseScore += 0.3 * (_random.NextDouble() - 0.5);
                break;
        }

        // Add randomness
        baseScore += 0.1 * (_random.NextDouble() - 0.5);

        // Ensure score is within bounds
        return Math.Max(0.1, Math.Min(0.9, baseScore));
    }

    /// <summary>
    /// Gets recent intuitions
    /// </summary>
    /// <param name="count">The number of intuitions to return</param>
    /// <returns>The recent intuitions</returns>
    public List<Intuition> GetRecentIntuitions(int count)
    {
        return _intuitions
            .OrderByDescending(i => i.Timestamp)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets the most confident intuitions
    /// </summary>
    /// <param name="count">The number of intuitions to return</param>
    /// <returns>The most confident intuitions</returns>
    public List<Intuition> GetMostConfidentIntuitions(int count)
    {
        return _intuitions
            .OrderByDescending(i => i.Confidence)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets intuitions by type
    /// </summary>
    /// <param name="type">The intuition type</param>
    /// <param name="count">The number of intuitions to return</param>
    /// <returns>The intuitions by type</returns>
    public List<Intuition> GetIntuitionsByType(IntuitionType type, int count)
    {
        return _intuitions
            .Where(i => i.Type == type)
            .OrderByDescending(i => i.Timestamp)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Adds a heuristic rule
    /// </summary>
    /// <param name="name">The rule name</param>
    /// <param name="description">The rule description</param>
    /// <param name="reliability">The rule reliability</param>
    /// <param name="context">The rule context</param>
    /// <returns>The created heuristic rule</returns>
    public HeuristicRule AddHeuristicRule(string name, string description, double reliability, string context)
    {
        var rule = new HeuristicRule
        {
            Name = name,
            Description = description,
            Reliability = reliability,
            Context = context
        };

        _heuristicRules.Add(rule);

        _logger.LogInformation("Added heuristic rule: {Name} (Reliability: {Reliability:F2})", name, reliability);

        return rule;
    }

    /// <summary>
    /// Updates pattern confidence
    /// </summary>
    /// <param name="pattern">The pattern</param>
    /// <param name="confidence">The confidence</param>
    public void UpdatePatternConfidence(string pattern, double confidence)
    {
        _patternConfidence[pattern] = Math.Max(0.0, Math.Min(1.0, confidence));

        _logger.LogInformation("Updated pattern confidence: {Pattern} (Confidence: {Confidence:F2})", pattern, confidence);
    }
}
