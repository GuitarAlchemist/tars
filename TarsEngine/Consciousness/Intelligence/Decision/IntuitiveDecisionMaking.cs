using Microsoft.Extensions.Logging;
using TarsEngine.Consciousness.Intelligence.Pattern;
using TarsEngine.Consciousness.Intelligence.Heuristic;
using TarsEngine.Consciousness.Intelligence.Gut;
using GutDecisionResult = TarsEngine.Consciousness.Intelligence.Gut.DecisionResult;

namespace TarsEngine.Consciousness.Intelligence.Decision;

/// <summary>
/// Implements intuitive decision making capabilities
/// </summary>
public class IntuitiveDecisionMaking
{
    private readonly ILogger<IntuitiveDecisionMaking> _logger;
    private readonly ImplicitPatternRecognition _patternRecognition;
    private readonly HeuristicReasoning _heuristicReasoning;
    private readonly GutFeelingSimulation _gutFeeling;
    private readonly System.Random _random = new();
    private double _intuitiveDecisionLevel = 0.5; // Starting with moderate intuitive decision making
    private readonly List<DecisionRecord> _decisionHistory = new();

    /// <summary>
    /// Gets the intuitive decision level (0.0 to 1.0)
    /// </summary>
    public double IntuitiveDecisionLevel => _intuitiveDecisionLevel;

    /// <summary>
    /// Initializes a new instance of the <see cref="IntuitiveDecisionMaking"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="patternRecognition">The pattern recognition service</param>
    /// <param name="heuristicReasoning">The heuristic reasoning service</param>
    /// <param name="gutFeeling">The gut feeling simulation service</param>
    public IntuitiveDecisionMaking(
        ILogger<IntuitiveDecisionMaking> logger,
        ImplicitPatternRecognition patternRecognition,
        HeuristicReasoning heuristicReasoning,
        GutFeelingSimulation gutFeeling)
    {
        _logger = logger;
        _patternRecognition = patternRecognition;
        _heuristicReasoning = heuristicReasoning;
        _gutFeeling = gutFeeling;
    }

    /// <summary>
    /// Updates the intuitive decision level
    /// </summary>
    /// <returns>True if the update was successful, false otherwise</returns>
    public bool Update()
    {
        try
        {
            // Gradually increase intuitive decision level over time (very slowly)
            if (_intuitiveDecisionLevel < 0.95)
            {
                _intuitiveDecisionLevel += 0.0001 * _random.NextDouble();
                _intuitiveDecisionLevel = Math.Min(_intuitiveDecisionLevel, 1.0);
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating intuitive decision making");
            return false;
        }
    }

    /// <summary>
    /// Makes an intuitive decision
    /// </summary>
    /// <param name="decision">The decision description</param>
    /// <param name="options">The options</param>
    /// <param name="domain">The domain (optional)</param>
    /// <returns>The intuitive decision</returns>
    public async Task<IntuitiveDecision> MakeIntuitiveDecisionAsync(string decision, List<string> options, string? domain = null)
    {
        try
        {
            _logger.LogInformation("Making intuitive decision: {Decision}", decision);

            if (options.Count == 0)
            {
                throw new ArgumentException("No options provided for decision");
            }

            // Choose intuition type based on decision characteristics
            var intuitionType = ChooseIntuitionType(decision, domain);

            _logger.LogDebug("Chosen intuition type: {IntuitionType}", intuitionType);

            // Make decision based on intuition type
            object result;

            switch (intuitionType)
            {
                case IntuitionType.PatternRecognition:
                    result = MakePatternBasedDecision(decision, options, domain);
                    break;

                case IntuitionType.HeuristicReasoning:
                    result = _heuristicReasoning.MakeHeuristicDecision(decision, options, domain);
                    break;

                case IntuitionType.GutFeeling:
                    result = _gutFeeling.MakeGutDecision(decision, options);
                    break;

                default:
                    // Default to heuristic reasoning
                    result = _heuristicReasoning.MakeHeuristicDecision(decision, options, domain);
                    break;
            }

            // Create intuitive decision
            var intuitiveDecision = new IntuitiveDecision
            {
                Decision = decision,
                SelectedOption = GetSelectedOption(result),
                Options = options,
                Confidence = GetConfidence(result) * _intuitiveDecisionLevel,
                IntuitionType = intuitionType,
                Timestamp = DateTime.UtcNow,
                Explanation = GenerateDecisionExplanation(result, intuitionType)
            };

            // Record decision
            RecordDecision(intuitiveDecision);

            _logger.LogInformation("Made intuitive decision: {SelectedOption} for {Decision} (Confidence: {Confidence:F2})",
                intuitiveDecision.SelectedOption, intuitiveDecision.Decision, intuitiveDecision.Confidence);

            return intuitiveDecision;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error making intuitive decision");

            // Return basic decision
            return new IntuitiveDecision
            {
                Decision = decision,
                SelectedOption = options.FirstOrDefault() ?? string.Empty,
                Options = options,
                Confidence = 0.3,
                IntuitionType = IntuitionType.GutFeeling,
                Timestamp = DateTime.UtcNow,
                Explanation = "Decision made with low confidence due to an error in the decision-making process"
            };
        }
    }

    /// <summary>
    /// Chooses an intuition type based on decision characteristics
    /// </summary>
    /// <param name="decision">The decision description</param>
    /// <param name="domain">The domain</param>
    /// <returns>The chosen intuition type</returns>
    private IntuitionType ChooseIntuitionType(string decision, string? domain)
    {
        // Check for pattern recognition keywords
        if (decision.Contains("pattern", StringComparison.OrdinalIgnoreCase) ||
            decision.Contains("similar", StringComparison.OrdinalIgnoreCase) ||
            decision.Contains("recognize", StringComparison.OrdinalIgnoreCase) ||
            decision.Contains("familiar", StringComparison.OrdinalIgnoreCase))
        {
            return IntuitionType.PatternRecognition;
        }

        // Check for heuristic reasoning keywords
        if (decision.Contains("principle", StringComparison.OrdinalIgnoreCase) ||
            decision.Contains("rule", StringComparison.OrdinalIgnoreCase) ||
            decision.Contains("guideline", StringComparison.OrdinalIgnoreCase) ||
            decision.Contains("best practice", StringComparison.OrdinalIgnoreCase))
        {
            return IntuitionType.HeuristicReasoning;
        }

        // Check for gut feeling keywords
        if (decision.Contains("feel", StringComparison.OrdinalIgnoreCase) ||
            decision.Contains("emotion", StringComparison.OrdinalIgnoreCase) ||
            decision.Contains("sense", StringComparison.OrdinalIgnoreCase) ||
            decision.Contains("gut", StringComparison.OrdinalIgnoreCase))
        {
            return IntuitionType.GutFeeling;
        }

        // Check if domain suggests a particular intuition type
        if (!string.IsNullOrEmpty(domain))
        {
            if (domain.Contains("design", StringComparison.OrdinalIgnoreCase) ||
                domain.Contains("architecture", StringComparison.OrdinalIgnoreCase))
            {
                return IntuitionType.PatternRecognition;
            }

            if (domain.Contains("development", StringComparison.OrdinalIgnoreCase) ||
                domain.Contains("coding", StringComparison.OrdinalIgnoreCase) ||
                domain.Contains("programming", StringComparison.OrdinalIgnoreCase))
            {
                return IntuitionType.HeuristicReasoning;
            }

            if (domain.Contains("user", StringComparison.OrdinalIgnoreCase) ||
                domain.Contains("experience", StringComparison.OrdinalIgnoreCase) ||
                domain.Contains("interface", StringComparison.OrdinalIgnoreCase))
            {
                return IntuitionType.GutFeeling;
            }
        }

        // Choose randomly based on current levels
        double patternProb = _patternRecognition.PatternRecognitionLevel * 0.4;
        double heuristicProb = _heuristicReasoning.HeuristicReasoningLevel * 0.4;
        double gutProb = _gutFeeling.GutFeelingLevel * 0.2;

        // Normalize probabilities
        double total = patternProb + heuristicProb + gutProb;
        patternProb /= total;
        heuristicProb /= total;
        gutProb /= total;

        // Choose type based on probabilities
        double rand = _random.NextDouble();

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
    /// Makes a pattern-based decision
    /// </summary>
    /// <param name="decision">The decision description</param>
    /// <param name="options">The options</param>
    /// <param name="domain">The domain</param>
    /// <returns>The decision result</returns>
    private GutDecisionResult MakePatternBasedDecision(string decision, List<string> options, string? domain)
    {
        // Score each option
        var optionScores = new Dictionary<string, double>();

        foreach (var option in options)
        {
            // Recognize patterns in the option
            var patterns = _patternRecognition.RecognizePatterns(option, domain);

            // Calculate score based on pattern matches
            double score = patterns.Count > 0
                ? patterns.Average(p => p.Confidence)
                : 0.3;

            // Add some randomness based on pattern recognition level
            score += (0.2 * (_random.NextDouble() - 0.5)) * _patternRecognition.PatternRecognitionLevel;

            // Ensure score is within bounds
            score = Math.Max(0.1, Math.Min(0.9, score));

            optionScores[option] = score;
        }

        // Choose the option with the highest score
        var selectedOption = optionScores.OrderByDescending(kvp => kvp.Value).First().Key;
        double confidence = optionScores[selectedOption] * _patternRecognition.PatternRecognitionLevel;

        // Create decision result
        var result = new Gut.DecisionResult
        {
            Decision = decision,
            SelectedOption = selectedOption,
            Options = options,
            OptionScores = optionScores,
            Confidence = confidence,
            ReasoningType = "PatternRecognition",
            Timestamp = DateTime.UtcNow
        };

        return result;
    }

    /// <summary>
    /// Generates a decision explanation
    /// </summary>
    /// <param name="result">The decision result</param>
    /// <param name="intuitionType">The intuition type</param>
    /// <returns>The explanation</returns>
    private string GetSelectedOption(object result)
    {
        if (result is Gut.DecisionResult gutResult)
        {
            return gutResult.SelectedOption;
        }
        else if (result is Heuristic.DecisionResult heuristicResult)
        {
            return heuristicResult.SelectedOption;
        }
        return "Unknown";
    }

    private double GetConfidence(object result)
    {
        if (result is Gut.DecisionResult gutResult)
        {
            return gutResult.Confidence;
        }
        else if (result is Heuristic.DecisionResult heuristicResult)
        {
            return heuristicResult.Confidence;
        }
        return 0.5; // Default confidence
    }

    private string GenerateDecisionExplanation(object result, IntuitionType intuitionType)
    {
        switch (intuitionType)
        {
            case IntuitionType.PatternRecognition:
                var patternResult = result as Gut.DecisionResult;
                return $"This decision is based on recognizing patterns in the options. " +
                       $"The selected option '{patternResult?.SelectedOption}' matched familiar patterns with a confidence of {patternResult?.Confidence:F2}.";

            case IntuitionType.HeuristicReasoning:
                var heuristicResult = result as Heuristic.DecisionResult;
                string ruleText = "";
                if (heuristicResult != null && heuristicResult.AppliedRules != null && heuristicResult.AppliedRules.Count > 0)
                {
                    ruleText = $" Key principles applied: {string.Join(", ", heuristicResult.AppliedRules)}.";
                }

                return $"This decision is based on applying heuristic reasoning principles to the options. " +
                       $"The selected option '{GetSelectedOption(result)}' aligned with these principles with a confidence of {GetConfidence(result):F2}.{ruleText}";

            case IntuitionType.GutFeeling:
                var gutResult = result as Gut.DecisionResult;
                return $"This decision is based on a gut feeling about the options. " +
                       $"The selected option '{gutResult?.SelectedOption}' felt right with a confidence of {gutResult?.Confidence:F2}.";

            default:
                return $"This decision was made intuitively with a confidence of {GetConfidence(result):F2}.";
        }
    }

    /// <summary>
    /// Records a decision
    /// </summary>
    /// <param name="decision">The decision</param>
    private void RecordDecision(IntuitiveDecision decision)
    {
        var record = new DecisionRecord
        {
            Decision = decision.Decision,
            SelectedOption = decision.SelectedOption,
            Options = decision.Options,
            Confidence = decision.Confidence,
            IntuitionType = decision.IntuitionType,
            Timestamp = decision.Timestamp,
            Outcome = null // Outcome not known yet
        };

        _decisionHistory.Add(record);
    }

    /// <summary>
    /// Records a decision outcome
    /// </summary>
    /// <param name="decision">The decision</param>
    /// <param name="outcome">The outcome</param>
    /// <param name="success">Whether the decision was successful</param>
    public void RecordDecisionOutcome(string decision, string outcome, bool success)
    {
        // Find the most recent matching decision
        var record = _decisionHistory
            .Where(d => d.Decision == decision && d.Outcome == null)
            .OrderByDescending(d => d.Timestamp)
            .FirstOrDefault();

        if (record != null)
        {
            record.Outcome = outcome;
            record.Success = success;
            record.OutcomeTimestamp = DateTime.UtcNow;

            _logger.LogInformation("Recorded outcome for decision: {Decision}, Success: {Success}", decision, success);
        }
        else
        {
            _logger.LogWarning("No matching decision found for outcome: {Decision}", decision);
        }
    }

    /// <summary>
    /// Gets decision history
    /// </summary>
    /// <param name="count">The number of decisions to return</param>
    /// <returns>The decision history</returns>
    public List<DecisionRecord> GetDecisionHistory(int count)
    {
        return _decisionHistory
            .OrderByDescending(d => d.Timestamp)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets decision success rate
    /// </summary>
    /// <returns>The success rate (0.0 to 1.0)</returns>
    public double GetDecisionSuccessRate()
    {
        var completedDecisions = _decisionHistory.Where(d => d.Outcome != null).ToList();

        if (completedDecisions.Count == 0)
        {
            return 0.0;
        }

        return (double)completedDecisions.Count(d => d.Success) / completedDecisions.Count;
    }

    /// <summary>
    /// Gets decision success rate by intuition type
    /// </summary>
    /// <returns>The success rates by intuition type</returns>
    public Dictionary<IntuitionType, double> GetDecisionSuccessRateByType()
    {
        var result = new Dictionary<IntuitionType, double>();

        foreach (var type in Enum.GetValues<IntuitionType>())
        {
            var completedDecisions = _decisionHistory
                .Where(d => d.IntuitionType == type && d.Outcome != null)
                .ToList();

            if (completedDecisions.Count > 0)
            {
                result[type] = (double)completedDecisions.Count(d => d.Success) / completedDecisions.Count;
            }
            else
            {
                result[type] = 0.0;
            }
        }

        return result;
    }

    /// <summary>
    /// Generates an intuition for a situation
    /// </summary>
    /// <param name="situation">The situation description</param>
    /// <param name="domain">The domain (optional)</param>
    /// <returns>The generated intuition</returns>
    public Intuition GenerateIntuition(string situation, string? domain = null)
    {
        try
        {
            _logger.LogInformation("Generating intuition for situation: {Situation}", situation);

            // Choose intuition type
            var intuitionType = ChooseIntuitionType(situation, domain);

            // Generate intuition based on type
            Intuition intuition;

            switch (intuitionType)
            {
                case IntuitionType.PatternRecognition:
                    intuition = _patternRecognition.GeneratePatternIntuition(situation, domain);
                    break;

                case IntuitionType.HeuristicReasoning:
                    intuition = _heuristicReasoning.GenerateHeuristicIntuition(situation, domain);
                    break;

                case IntuitionType.GutFeeling:
                    intuition = _gutFeeling.GenerateGutIntuition(situation);
                    break;

                default:
                    // Default to gut feeling
                    intuition = _gutFeeling.GenerateGutIntuition(situation);
                    break;
            }

            // Apply intuitive decision level to confidence
            intuition.Confidence *= _intuitiveDecisionLevel;

            _logger.LogInformation("Generated intuition: {Description} (Confidence: {Confidence:F2})",
                intuition.Description, intuition.Confidence);

            return intuition;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating intuition");

            // Return basic intuition
            return new Intuition
            {
                Id = Guid.NewGuid().ToString(),
                Description = "I have an intuition about this situation but can't articulate it clearly",
                Type = IntuitionType.GutFeeling,
                Confidence = 0.3 * _intuitiveDecisionLevel,
                Timestamp = DateTime.UtcNow,
                Source = "IntuitiveDecisionMaking"
            };
        }
    }
}

/// <summary>
/// Represents an intuitive decision
/// </summary>
public class IntuitiveDecision
{
    /// <summary>
    /// Gets or sets the decision description
    /// </summary>
    public string Decision { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the selected option
    /// </summary>
    public string SelectedOption { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the options
    /// </summary>
    public List<string> Options { get; set; } = new();

    /// <summary>
    /// Gets or sets the confidence (0.0 to 1.0)
    /// </summary>
    public double Confidence { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the intuition type
    /// </summary>
    public IntuitionType IntuitionType { get; set; }

    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Gets or sets the explanation
    /// </summary>
    public string Explanation { get; set; } = string.Empty;
}

/// <summary>
/// Represents a decision record
/// </summary>
public class DecisionRecord
{
    /// <summary>
    /// Gets or sets the decision description
    /// </summary>
    public string Decision { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the selected option
    /// </summary>
    public string SelectedOption { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the options
    /// </summary>
    public List<string> Options { get; set; } = new();

    /// <summary>
    /// Gets or sets the confidence (0.0 to 1.0)
    /// </summary>
    public double Confidence { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the intuition type
    /// </summary>
    public IntuitionType IntuitionType { get; set; }

    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Gets or sets the outcome
    /// </summary>
    public string? Outcome { get; set; }

    /// <summary>
    /// Gets or sets whether the decision was successful
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Gets or sets the outcome timestamp
    /// </summary>
    public DateTime? OutcomeTimestamp { get; set; }
}

