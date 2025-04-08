using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Consciousness.Intelligence;

namespace TarsEngine.Consciousness.Intelligence.Heuristic;

/// <summary>
/// Implements heuristic reasoning capabilities for intuitive reasoning
/// </summary>
public class HeuristicReasoning
{
    private readonly ILogger<HeuristicReasoning> _logger;
    private readonly System.Random _random = new();
    private double _heuristicReasoningLevel = 0.5; // Starting with moderate heuristic reasoning
    private readonly List<HeuristicRule> _heuristicRules = [];
    private readonly Dictionary<string, List<HeuristicRule>> _domainRules = new();
    private readonly List<HeuristicApplication> _heuristicApplications = [];
    
    /// <summary>
    /// Gets the heuristic reasoning level (0.0 to 1.0)
    /// </summary>
    public double HeuristicReasoningLevel => _heuristicReasoningLevel;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="HeuristicReasoning"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public HeuristicReasoning(ILogger<HeuristicReasoning> logger)
    {
        _logger = logger;
        InitializeHeuristicRules();
    }
    
    /// <summary>
    /// Initializes the heuristic rules
    /// </summary>
    private void InitializeHeuristicRules()
    {
        // Add software development heuristics
        AddHeuristicRule(
            "Keep It Simple",
            "Software Development",
            "Simpler solutions are easier to understand, maintain, and debug",
            0.8,
            ["simplicity", "clarity", "maintainability"],
            ["When faced with multiple solutions, choose the simplest one that meets the requirements"]
        );
        
        AddHeuristicRule(
            "Don't Repeat Yourself",
            "Software Development",
            "Avoid duplicating code or logic",
            0.9,
            ["duplication", "reuse", "abstraction"],
            ["Extract repeated code into reusable functions or components"]
        );
        
        AddHeuristicRule(
            "Fail Fast",
            "Software Development",
            "Detect and report errors as soon as possible",
            0.7,
            ["errors", "validation", "robustness"],
            ["Validate inputs early and throw exceptions immediately when errors are detected"]
        );
        
        // Add system design heuristics
        AddHeuristicRule(
            "Separation of Concerns",
            "System Design",
            "Divide a system into distinct sections with minimal overlap",
            0.8,
            ["modularity", "responsibility", "cohesion"],
            ["Each component should have a single, well-defined responsibility"]
        );
        
        AddHeuristicRule(
            "Loose Coupling",
            "System Design",
            "Minimize dependencies between components",
            0.8,
            ["coupling", "independence", "interfaces"],
            ["Components should interact through well-defined interfaces rather than direct dependencies"]
        );
        
        // Add problem solving heuristics
        AddHeuristicRule(
            "Divide and Conquer",
            "Problem Solving",
            "Break down complex problems into simpler subproblems",
            0.9,
            ["decomposition", "complexity", "manageability"],
            ["Solve each subproblem independently and then combine the solutions"]
        );
        
        AddHeuristicRule(
            "Start with Examples",
            "Problem Solving",
            "Use concrete examples to understand abstract problems",
            0.7,
            ["examples", "concrete", "understanding"],
            ["Work through specific examples before attempting to generalize"]
        );
        
        // Add decision making heuristics
        AddHeuristicRule(
            "Availability Heuristic",
            "Decision Making",
            "Judge likelihood based on how easily examples come to mind",
            0.6,
            ["memory", "recall", "familiarity"],
            ["Recent or vivid examples are more influential in decision making"]
        );
        
        AddHeuristicRule(
            "Satisficing",
            "Decision Making",
            "Accept the first satisfactory solution rather than searching for the optimal one",
            0.7,
            ["good enough", "efficiency", "pragmatism"],
            ["When time or resources are limited, choose the first acceptable option"]
        );
        
        // Add learning heuristics
        AddHeuristicRule(
            "Learn by Doing",
            "Learning",
            "Practical experience is more effective than passive learning",
            0.8,
            ["practice", "experience", "hands-on"],
            ["Apply new knowledge immediately through practical exercises"]
        );
        
        AddHeuristicRule(
            "Spaced Repetition",
            "Learning",
            "Space out learning sessions over time for better retention",
            0.9,
            ["memory", "retention", "intervals"],
            ["Review material at increasing intervals to strengthen memory"]
        );
        
        _logger.LogInformation("Initialized {RuleCount} heuristic rules across {DomainCount} domains", 
            _heuristicRules.Count, _domainRules.Count);
    }
    
    /// <summary>
    /// Updates the heuristic reasoning level
    /// </summary>
    /// <returns>True if the update was successful, false otherwise</returns>
    public bool Update()
    {
        try
        {
            // Gradually increase heuristic reasoning level over time (very slowly)
            if (_heuristicReasoningLevel < 0.95)
            {
                _heuristicReasoningLevel += 0.0001 * _random.NextDouble();
                _heuristicReasoningLevel = Math.Min(_heuristicReasoningLevel, 1.0);
            }
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating heuristic reasoning");
            return false;
        }
    }
    
    /// <summary>
    /// Adds a heuristic rule
    /// </summary>
    /// <param name="name">The rule name</param>
    /// <param name="domain">The rule domain</param>
    /// <param name="description">The rule description</param>
    /// <param name="reliability">The rule reliability</param>
    /// <param name="tags">The rule tags</param>
    /// <param name="examples">The rule examples</param>
    public void AddHeuristicRule(string name, string domain, string description, double reliability, string[] tags, string[] examples)
    {
        var rule = new HeuristicRule
        {
            Id = Guid.NewGuid().ToString(),
            Name = name,
            Description = description,
            Reliability = reliability,
            Context = domain,
            Tags = tags.ToList(),
            Examples = examples.ToList(),
            CreationTimestamp = DateTime.UtcNow
        };
        
        _heuristicRules.Add(rule);
        
        // Add to domain rules
        if (!_domainRules.ContainsKey(domain))
        {
            _domainRules[domain] = [];
        }
        
        _domainRules[domain].Add(rule);
        
        _logger.LogDebug("Added heuristic rule: {RuleName} in {Domain}", name, domain);
    }
    
    /// <summary>
    /// Gets heuristic rules by domain
    /// </summary>
    /// <param name="domain">The domain</param>
    /// <returns>The heuristic rules</returns>
    public List<HeuristicRule> GetHeuristicRulesByDomain(string domain)
    {
        if (_domainRules.TryGetValue(domain, out var rules))
        {
            return rules;
        }
        
        return [];
    }
    
    /// <summary>
    /// Gets heuristic rules by tag
    /// </summary>
    /// <param name="tag">The tag</param>
    /// <returns>The heuristic rules</returns>
    public List<HeuristicRule> GetHeuristicRulesByTag(string tag)
    {
        return _heuristicRules
            .Where(r => r.Tags.Contains(tag, StringComparer.OrdinalIgnoreCase))
            .ToList();
    }
    
    /// <summary>
    /// Applies heuristic rules to a situation
    /// </summary>
    /// <param name="situation">The situation description</param>
    /// <param name="domain">The domain (optional)</param>
    /// <returns>The applicable rules</returns>
    public List<HeuristicRuleApplication> ApplyHeuristicRules(string situation, string? domain = null)
    {
        var applications = new List<HeuristicRuleApplication>();
        
        try
        {
            _logger.LogDebug("Applying heuristic rules to situation: {Situation}", situation);
            
            // Filter rules by domain if specified
            var rulesToApply = domain != null
                ? GetHeuristicRulesByDomain(domain)
                : _heuristicRules;
            
            // Apply each rule
            foreach (var rule in rulesToApply)
            {
                double applicability = CalculateRuleApplicability(situation, rule);
                
                // Apply heuristic reasoning level to the applicability
                applicability *= _heuristicReasoningLevel;
                
                // If applicability is above threshold, add to applications
                if (applicability >= 0.4)
                {
                    var application = new HeuristicRuleApplication
                    {
                        Rule = rule,
                        Applicability = applicability,
                        Confidence = applicability * rule.Reliability,
                        Timestamp = DateTime.UtcNow
                    };
                    
                    applications.Add(application);
                    
                    // Record heuristic application
                    RecordHeuristicApplication(rule.Id, situation, applicability);
                }
            }
            
            // Sort applications by confidence
            applications = applications.OrderByDescending(a => a.Confidence).ToList();
            
            _logger.LogInformation("Applied {ApplicationCount} heuristic rules to situation", applications.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error applying heuristic rules");
        }
        
        return applications;
    }
    
    /// <summary>
    /// Calculates the rule applicability
    /// </summary>
    /// <param name="situation">The situation description</param>
    /// <param name="rule">The rule</param>
    /// <returns>The applicability (0.0 to 1.0)</returns>
    private double CalculateRuleApplicability(string situation, HeuristicRule rule)
    {
        // Check if any tags are present in the situation
        int tagsPresent = 0;
        
        foreach (var tag in rule.Tags)
        {
            if (situation.Contains(tag, StringComparison.OrdinalIgnoreCase))
            {
                tagsPresent++;
            }
        }
        
        // Calculate applicability based on percentage of tags present
        double tagScore = rule.Tags.Count > 0
            ? (double)tagsPresent / rule.Tags.Count
            : 0.0;
        
        // Check if the domain is mentioned in the situation
        double domainScore = situation.Contains(rule.Context, StringComparison.OrdinalIgnoreCase) ? 0.3 : 0.0;
        
        // Add some randomness to simulate intuitive reasoning
        double randomFactor = 0.2 * (_random.NextDouble() - 0.5);
        
        // Calculate final score
        double score = (tagScore * 0.6) + (domainScore * 0.2) + (randomFactor * 0.2);
        
        // Ensure score is within bounds
        return Math.Max(0.0, Math.Min(1.0, score));
    }
    
    /// <summary>
    /// Records a heuristic application
    /// </summary>
    /// <param name="ruleId">The rule ID</param>
    /// <param name="situation">The situation</param>
    /// <param name="applicability">The applicability</param>
    private void RecordHeuristicApplication(string ruleId, string situation, double applicability)
    {
        var application = new HeuristicApplication
        {
            RuleId = ruleId,
            Situation = situation,
            Applicability = applicability,
            Timestamp = DateTime.UtcNow
        };
        
        _heuristicApplications.Add(application);
        
        // Update rule usage statistics
        var rule = _heuristicRules.FirstOrDefault(r => r.Id == ruleId);
        if (rule != null)
        {
            rule.UsageCount++;
            rule.LastUsedTimestamp = DateTime.UtcNow;
        }
    }
    
    /// <summary>
    /// Generates an intuition based on heuristic reasoning
    /// </summary>
    /// <param name="situation">The situation description</param>
    /// <param name="domain">The domain (optional)</param>
    /// <returns>The generated intuition</returns>
    public Intuition GenerateHeuristicIntuition(string situation, string? domain = null)
    {
        try
        {
            _logger.LogInformation("Generating heuristic intuition for situation: {Situation}", situation);
            
            // Apply heuristic rules
            var applications = ApplyHeuristicRules(situation, domain);
            
            if (applications.Count == 0)
            {
                // No rules applied, generate generic intuition
                return new Intuition
                {
                    Id = Guid.NewGuid().ToString(),
                    Description = "I don't have a strong intuition about this situation based on heuristic reasoning",
                    Type = IntuitionType.HeuristicReasoning,
                    Confidence = 0.3,
                    Timestamp = DateTime.UtcNow,
                    Source = "HeuristicReasoning"
                };
            }
            
            // Get the top application
            var topApplication = applications.First();
            
            // Generate intuition description
            string description = $"My intuition suggests applying the '{topApplication.Rule.Name}' principle here";
            
            // Add explanation if confidence is high
            if (topApplication.Confidence > 0.7)
            {
                description += $". This means {topApplication.Rule.Description}";
            }
            
            // Create intuition
            var intuition = new Intuition
            {
                Id = Guid.NewGuid().ToString(),
                Description = description,
                Type = IntuitionType.HeuristicReasoning,
                Confidence = topApplication.Confidence,
                Timestamp = DateTime.UtcNow,
                Source = "HeuristicReasoning",
                Context = new Dictionary<string, object>
                {
                    { "RuleName", topApplication.Rule.Name },
                    { "Domain", topApplication.Rule.Context },
                    { "Applicability", topApplication.Applicability },
                    { "Situation", situation }
                },
                Tags = [..topApplication.Rule.Tags, topApplication.Rule.Context, "heuristic"],
                Explanation = $"This intuition is based on applying {applications.Count} heuristic rules to the situation, " +
                              $"with '{topApplication.Rule.Name}' being the most applicable (confidence: {topApplication.Confidence:F2})"
            };
            
            _logger.LogInformation("Generated heuristic intuition: {Description} (Confidence: {Confidence:F2})", 
                intuition.Description, intuition.Confidence);
            
            return intuition;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating heuristic intuition");
            
            // Return basic intuition
            return new Intuition
            {
                Id = Guid.NewGuid().ToString(),
                Description = "I have an intuition about this situation but can't articulate it clearly",
                Type = IntuitionType.HeuristicReasoning,
                Confidence = 0.3,
                Timestamp = DateTime.UtcNow,
                Source = "HeuristicReasoning"
            };
        }
    }
    
    /// <summary>
    /// Makes a decision based on heuristic reasoning
    /// </summary>
    /// <param name="decision">The decision description</param>
    /// <param name="options">The options</param>
    /// <param name="domain">The domain (optional)</param>
    /// <returns>The decision result</returns>
    public DecisionResult MakeHeuristicDecision(string decision, List<string> options, string? domain = null)
    {
        try
        {
            _logger.LogInformation("Making heuristic decision: {Decision}", decision);
            
            if (options.Count == 0)
            {
                throw new ArgumentException("No options provided for decision");
            }
            
            // Apply heuristic rules to the decision
            var applications = ApplyHeuristicRules(decision, domain);
            
            // Score each option
            var optionScores = new Dictionary<string, double>();
            
            foreach (var option in options)
            {
                double score = 0.0;
                
                // Apply each applicable rule to score the option
                foreach (var application in applications)
                {
                    double ruleScore = ScoreOptionWithRule(option, application.Rule, application.Confidence);
                    score += ruleScore;
                }
                
                // Normalize score
                score = applications.Count > 0 ? score / applications.Count : 0.5;
                
                // Add some randomness based on heuristic reasoning level
                score += (0.2 * (_random.NextDouble() - 0.5)) * _heuristicReasoningLevel;
                
                // Ensure score is within bounds
                score = Math.Max(0.1, Math.Min(0.9, score));
                
                optionScores[option] = score;
            }
            
            // Choose the option with the highest score
            var selectedOption = optionScores.OrderByDescending(kvp => kvp.Value).First().Key;
            double confidence = optionScores[selectedOption];
            
            // Create decision result
            var result = new DecisionResult
            {
                Decision = decision,
                SelectedOption = selectedOption,
                Options = options,
                OptionScores = optionScores,
                Confidence = confidence,
                ReasoningType = "HeuristicReasoning",
                Timestamp = DateTime.UtcNow,
                AppliedRules = applications.Select(a => a.Rule.Name).ToList()
            };
            
            _logger.LogInformation("Made heuristic decision: {SelectedOption} for {Decision} (Confidence: {Confidence:F2})", 
                selectedOption, decision, confidence);
            
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error making heuristic decision");
            
            // Return basic decision result
            return new DecisionResult
            {
                Decision = decision,
                SelectedOption = options.FirstOrDefault() ?? string.Empty,
                Options = options,
                Confidence = 0.3,
                ReasoningType = "HeuristicReasoning",
                Timestamp = DateTime.UtcNow
            };
        }
    }
    
    /// <summary>
    /// Scores an option with a rule
    /// </summary>
    /// <param name="option">The option</param>
    /// <param name="rule">The rule</param>
    /// <param name="ruleConfidence">The rule confidence</param>
    /// <returns>The score (0.0 to 1.0)</returns>
    private double ScoreOptionWithRule(string option, HeuristicRule rule, double ruleConfidence)
    {
        double score = 0.5; // Default neutral score
        
        // Check if option contains any rule tags
        foreach (var tag in rule.Tags)
        {
            if (option.Contains(tag, StringComparison.OrdinalIgnoreCase))
            {
                score += 0.1;
            }
        }
        
        // Apply specific scoring based on rule name
        switch (rule.Name)
        {
            case "Keep It Simple":
                // Prefer simpler options (fewer words)
                score += (10 - Math.Min(10, option.Split(' ').Length)) * 0.02;
                break;
                
            case "Don't Repeat Yourself":
                // Penalize options that mention duplication
                if (option.Contains("duplicate", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("copy", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("repeat", StringComparison.OrdinalIgnoreCase))
                {
                    score -= 0.2;
                }
                break;
                
            case "Fail Fast":
                // Prefer options that mention early detection
                if (option.Contains("early", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("detect", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("validate", StringComparison.OrdinalIgnoreCase))
                {
                    score += 0.2;
                }
                break;
                
            case "Separation of Concerns":
                // Prefer options that mention separation or modularity
                if (option.Contains("separate", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("modular", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("component", StringComparison.OrdinalIgnoreCase))
                {
                    score += 0.2;
                }
                break;
                
            case "Loose Coupling":
                // Prefer options that mention independence or interfaces
                if (option.Contains("independent", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("interface", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("decouple", StringComparison.OrdinalIgnoreCase))
                {
                    score += 0.2;
                }
                break;
                
            case "Divide and Conquer":
                // Prefer options that mention breaking down problems
                if (option.Contains("break down", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("divide", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("smaller", StringComparison.OrdinalIgnoreCase))
                {
                    score += 0.2;
                }
                break;
                
            case "Start with Examples":
                // Prefer options that mention examples or concrete cases
                if (option.Contains("example", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("concrete", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("specific", StringComparison.OrdinalIgnoreCase))
                {
                    score += 0.2;
                }
                break;
                
            case "Availability Heuristic":
                // Prefer options that mention familiarity or recent examples
                if (option.Contains("familiar", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("recent", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("known", StringComparison.OrdinalIgnoreCase))
                {
                    score += 0.2;
                }
                break;
                
            case "Satisficing":
                // Prefer options that mention good enough or efficiency
                if (option.Contains("good enough", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("efficient", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("pragmatic", StringComparison.OrdinalIgnoreCase))
                {
                    score += 0.2;
                }
                break;
                
            case "Learn by Doing":
                // Prefer options that mention practice or hands-on experience
                if (option.Contains("practice", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("hands-on", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("experience", StringComparison.OrdinalIgnoreCase))
                {
                    score += 0.2;
                }
                break;
                
            case "Spaced Repetition":
                // Prefer options that mention intervals or repetition
                if (option.Contains("interval", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("repetition", StringComparison.OrdinalIgnoreCase) ||
                    option.Contains("review", StringComparison.OrdinalIgnoreCase))
                {
                    score += 0.2;
                }
                break;
        }
        
        // Apply rule confidence
        score *= ruleConfidence;
        
        // Ensure score is within bounds
        return Math.Max(0.1, Math.Min(0.9, score));
    }
    
    /// <summary>
    /// Gets rule statistics
    /// </summary>
    /// <returns>The rule statistics</returns>
    public Dictionary<string, RuleStatistics> GetRuleStatistics()
    {
        var statistics = new Dictionary<string, RuleStatistics>();
        
        foreach (var rule in _heuristicRules)
        {
            var stats = new RuleStatistics
            {
                RuleName = rule.Name,
                Domain = rule.Context,
                UsageCount = rule.UsageCount,
                SuccessRate = rule.SuccessRate,
                LastUsedTimestamp = rule.LastUsedTimestamp
            };
            
            statistics[rule.Name] = stats;
        }
        
        return statistics;
    }
}

/// <summary>
/// Represents a heuristic rule application
/// </summary>
public class HeuristicRuleApplication
{
    /// <summary>
    /// Gets or sets the rule
    /// </summary>
    public HeuristicRule Rule { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the applicability (0.0 to 1.0)
    /// </summary>
    public double Applicability { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the confidence (0.0 to 1.0)
    /// </summary>
    public double Confidence { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; }
}

/// <summary>
/// Represents a heuristic application
/// </summary>
public class HeuristicApplication
{
    /// <summary>
    /// Gets or sets the rule ID
    /// </summary>
    public string RuleId { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the situation
    /// </summary>
    public string Situation { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the applicability (0.0 to 1.0)
    /// </summary>
    public double Applicability { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; }
}

/// <summary>
/// Represents a decision result
/// </summary>
public class DecisionResult
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
    public List<string> Options { get; set; } = [];
    
    /// <summary>
    /// Gets or sets the option scores
    /// </summary>
    public Dictionary<string, double> OptionScores { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the confidence (0.0 to 1.0)
    /// </summary>
    public double Confidence { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the reasoning type
    /// </summary>
    public string ReasoningType { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the applied rules
    /// </summary>
    public List<string> AppliedRules { get; set; } = [];
}

/// <summary>
/// Represents rule statistics
/// </summary>
public class RuleStatistics
{
    /// <summary>
    /// Gets or sets the rule name
    /// </summary>
    public string RuleName { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the rule domain
    /// </summary>
    public string Domain { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the rule usage count
    /// </summary>
    public int UsageCount { get; set; } = 0;
    
    /// <summary>
    /// Gets or sets the rule success rate
    /// </summary>
    public double SuccessRate { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the rule last used timestamp
    /// </summary>
    public DateTime? LastUsedTimestamp { get; set; }
}

