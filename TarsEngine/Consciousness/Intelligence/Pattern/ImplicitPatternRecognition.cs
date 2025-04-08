using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Consciousness.Intelligence;

namespace TarsEngine.Consciousness.Intelligence.Pattern;

/// <summary>
/// Implements implicit pattern recognition capabilities for intuitive reasoning
/// </summary>
public class ImplicitPatternRecognition
{
    private readonly ILogger<ImplicitPatternRecognition> _logger;
    private readonly System.Random _random = new();
    private double _patternRecognitionLevel = 0.5; // Starting with moderate pattern recognition
    private readonly Dictionary<string, PatternData> _patternDatabase = new();
    private readonly List<PatternInstance> _patternInstances = [];
    private readonly List<string> _domains = [];
    
    /// <summary>
    /// Gets the pattern recognition level (0.0 to 1.0)
    /// </summary>
    public double PatternRecognitionLevel => _patternRecognitionLevel;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="ImplicitPatternRecognition"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public ImplicitPatternRecognition(ILogger<ImplicitPatternRecognition> logger)
    {
        _logger = logger;
        InitializePatternDatabase();
    }
    
    /// <summary>
    /// Initializes the pattern database
    /// </summary>
    private void InitializePatternDatabase()
    {
        // Initialize domains
        _domains.AddRange([
            "Software Development", 
            "System Architecture", 
            "Problem Solving", 
            "Decision Making", 
            "Learning",
            "Communication",
            "Collaboration"
        ]);
        
        // Add software development patterns
        AddPattern(
            "Repeated Code",
            "Software Development",
            "Code that appears multiple times with minor variations",
            0.8,
            ["duplication", "repetition", "copy-paste", "similar blocks"]
        );
        
        AddPattern(
            "Increasing Complexity",
            "Software Development",
            "Code that becomes increasingly complex over time",
            0.7,
            ["complexity", "nested", "convoluted", "hard to understand"]
        );
        
        AddPattern(
            "Inconsistent Naming",
            "Software Development",
            "Inconsistent naming conventions across the codebase",
            0.8,
            ["naming", "inconsistent", "conventions", "style"]
        );
        
        // Add system architecture patterns
        AddPattern(
            "Tight Coupling",
            "System Architecture",
            "Components that are highly dependent on each other",
            0.7,
            ["coupling", "dependency", "interconnected", "tightly bound"]
        );
        
        AddPattern(
            "Single Point of Failure",
            "System Architecture",
            "A component that can cause the entire system to fail",
            0.9,
            ["failure", "bottleneck", "critical", "vulnerable"]
        );
        
        // Add problem solving patterns
        AddPattern(
            "Solution Fixation",
            "Problem Solving",
            "Becoming fixated on a particular solution approach",
            0.6,
            ["fixation", "tunnel vision", "narrow focus", "stuck"]
        );
        
        AddPattern(
            "Premature Optimization",
            "Problem Solving",
            "Optimizing before understanding the actual performance bottlenecks",
            0.7,
            ["optimization", "premature", "performance", "efficiency"]
        );
        
        // Add decision making patterns
        AddPattern(
            "Confirmation Bias",
            "Decision Making",
            "Seeking information that confirms existing beliefs",
            0.8,
            ["bias", "confirmation", "selective", "reinforcing"]
        );
        
        AddPattern(
            "Analysis Paralysis",
            "Decision Making",
            "Overthinking a decision to the point of inaction",
            0.7,
            ["paralysis", "overthinking", "indecision", "hesitation"]
        );
        
        _logger.LogInformation("Initialized pattern database with {PatternCount} patterns across {DomainCount} domains", 
            _patternDatabase.Count, _domains.Count);
    }
    
    /// <summary>
    /// Updates the pattern recognition level
    /// </summary>
    /// <returns>True if the update was successful, false otherwise</returns>
    public bool Update()
    {
        try
        {
            // Gradually increase pattern recognition level over time (very slowly)
            if (_patternRecognitionLevel < 0.95)
            {
                _patternRecognitionLevel += 0.0001 * _random.NextDouble();
                _patternRecognitionLevel = Math.Min(_patternRecognitionLevel, 1.0);
            }
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating pattern recognition");
            return false;
        }
    }
    
    /// <summary>
    /// Adds a pattern to the database
    /// </summary>
    /// <param name="name">The pattern name</param>
    /// <param name="domain">The pattern domain</param>
    /// <param name="description">The pattern description</param>
    /// <param name="reliability">The pattern reliability</param>
    /// <param name="indicators">The pattern indicators</param>
    public void AddPattern(string name, string domain, string description, double reliability, string[] indicators)
    {
        var pattern = new PatternData
        {
            Name = name,
            Domain = domain,
            Description = description,
            Reliability = reliability,
            Indicators = indicators.ToList(),
            CreationTimestamp = DateTime.UtcNow
        };
        
        _patternDatabase[name] = pattern;
        
        _logger.LogDebug("Added pattern to database: {PatternName} in {Domain}", name, domain);
    }
    
    /// <summary>
    /// Recognizes patterns in a situation
    /// </summary>
    /// <param name="situation">The situation description</param>
    /// <param name="domain">The domain (optional)</param>
    /// <returns>The recognized patterns</returns>
    public List<PatternMatch> RecognizePatterns(string situation, string? domain = null)
    {
        var matches = new List<PatternMatch>();
        
        try
        {
            _logger.LogDebug("Recognizing patterns in situation: {Situation}", situation);
            
            // Filter patterns by domain if specified
            var patternsToCheck = domain != null
                ? _patternDatabase.Values.Where(p => p.Domain == domain).ToList()
                : _patternDatabase.Values.ToList();
            
            // Check each pattern for matches
            foreach (var pattern in patternsToCheck)
            {
                double matchScore = CalculatePatternMatchScore(situation, pattern);
                
                // Apply pattern recognition level to the match score
                matchScore *= _patternRecognitionLevel;
                
                // If match score is above threshold, add to matches
                if (matchScore >= 0.4)
                {
                    var match = new PatternMatch
                    {
                        PatternName = pattern.Name,
                        Domain = pattern.Domain,
                        Description = pattern.Description,
                        MatchScore = matchScore,
                        Confidence = matchScore * pattern.Reliability,
                        Timestamp = DateTime.UtcNow
                    };
                    
                    matches.Add(match);
                    
                    // Record pattern instance
                    RecordPatternInstance(pattern.Name, situation, matchScore);
                }
            }
            
            // Sort matches by confidence
            matches = matches.OrderByDescending(m => m.Confidence).ToList();
            
            _logger.LogInformation("Recognized {MatchCount} patterns in situation", matches.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error recognizing patterns");
        }
        
        return matches;
    }
    
    /// <summary>
    /// Calculates the pattern match score
    /// </summary>
    /// <param name="situation">The situation description</param>
    /// <param name="pattern">The pattern</param>
    /// <returns>The match score (0.0 to 1.0)</returns>
    private double CalculatePatternMatchScore(string situation, PatternData pattern)
    {
        // Count how many indicators are present in the situation
        int indicatorsPresent = 0;
        
        foreach (var indicator in pattern.Indicators)
        {
            if (situation.Contains(indicator, StringComparison.OrdinalIgnoreCase))
            {
                indicatorsPresent++;
            }
        }
        
        // Calculate match score based on percentage of indicators present
        double indicatorScore = pattern.Indicators.Count > 0
            ? (double)indicatorsPresent / pattern.Indicators.Count
            : 0.0;
        
        // Add some randomness to simulate intuitive recognition
        double randomFactor = 0.2 * (_random.NextDouble() - 0.5);
        
        // Calculate final score
        double score = indicatorScore + randomFactor;
        
        // Ensure score is within bounds
        return Math.Max(0.0, Math.Min(1.0, score));
    }
    
    /// <summary>
    /// Records a pattern instance
    /// </summary>
    /// <param name="patternName">The pattern name</param>
    /// <param name="situation">The situation</param>
    /// <param name="matchScore">The match score</param>
    private void RecordPatternInstance(string patternName, string situation, double matchScore)
    {
        var instance = new PatternInstance
        {
            PatternName = patternName,
            Situation = situation,
            MatchScore = matchScore,
            Timestamp = DateTime.UtcNow
        };
        
        _patternInstances.Add(instance);
        
        // Update pattern usage statistics
        if (_patternDatabase.TryGetValue(patternName, out var pattern))
        {
            pattern.UsageCount++;
            pattern.LastUsedTimestamp = DateTime.UtcNow;
            pattern.AverageMatchScore = ((pattern.AverageMatchScore * (pattern.UsageCount - 1)) + matchScore) / pattern.UsageCount;
        }
    }
    
    /// <summary>
    /// Generates an intuition based on pattern recognition
    /// </summary>
    /// <param name="situation">The situation description</param>
    /// <param name="domain">The domain (optional)</param>
    /// <returns>The generated intuition</returns>
    public Intuition GeneratePatternIntuition(string situation, string? domain = null)
    {
        try
        {
            _logger.LogInformation("Generating pattern intuition for situation: {Situation}", situation);
            
            // Recognize patterns
            var matches = RecognizePatterns(situation, domain);
            
            if (matches.Count == 0)
            {
                // No patterns recognized, generate generic intuition
                return new Intuition
                {
                    Id = Guid.NewGuid().ToString(),
                    Description = "I don't recognize any specific patterns in this situation",
                    Type = IntuitionType.PatternRecognition,
                    Confidence = 0.3,
                    Timestamp = DateTime.UtcNow,
                    Source = "ImplicitPatternRecognition"
                };
            }
            
            // Get the top match
            var topMatch = matches.First();
            
            // Generate intuition description
            string description = $"I intuitively recognize a {topMatch.PatternName} pattern in this situation";
            
            // Add explanation if confidence is high
            if (topMatch.Confidence > 0.7)
            {
                description += $". This typically indicates {topMatch.Description}";
            }
            
            // Create intuition
            var intuition = new Intuition
            {
                Id = Guid.NewGuid().ToString(),
                Description = description,
                Type = IntuitionType.PatternRecognition,
                Confidence = topMatch.Confidence,
                Timestamp = DateTime.UtcNow,
                Source = "ImplicitPatternRecognition",
                Context = new Dictionary<string, object>
                {
                    { "PatternName", topMatch.PatternName },
                    { "Domain", topMatch.Domain },
                    { "MatchScore", topMatch.MatchScore },
                    { "Situation", situation }
                },
                Tags = [topMatch.Domain, topMatch.PatternName, "pattern"],
                Explanation = $"This intuition is based on recognizing {matches.Count} patterns in the situation, " +
                              $"with {topMatch.PatternName} being the strongest match (confidence: {topMatch.Confidence:F2})"
            };
            
            _logger.LogInformation("Generated pattern intuition: {Description} (Confidence: {Confidence:F2})", 
                intuition.Description, intuition.Confidence);
            
            return intuition;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating pattern intuition");
            
            // Return basic intuition
            return new Intuition
            {
                Id = Guid.NewGuid().ToString(),
                Description = "I have an intuition about this situation but can't articulate it clearly",
                Type = IntuitionType.PatternRecognition,
                Confidence = 0.3,
                Timestamp = DateTime.UtcNow,
                Source = "ImplicitPatternRecognition"
            };
        }
    }
    
    /// <summary>
    /// Predicts outcomes based on pattern recognition
    /// </summary>
    /// <param name="situation">The situation description</param>
    /// <param name="domain">The domain (optional)</param>
    /// <returns>The predicted outcomes</returns>
    public List<PredictedOutcome> PredictOutcomes(string situation, string? domain = null)
    {
        var outcomes = new List<PredictedOutcome>();
        
        try
        {
            _logger.LogInformation("Predicting outcomes for situation: {Situation}", situation);
            
            // Recognize patterns
            var matches = RecognizePatterns(situation, domain);
            
            if (matches.Count == 0)
            {
                // No patterns recognized, return empty list
                return outcomes;
            }
            
            // Generate outcomes for each pattern
            foreach (var match in matches.Take(3)) // Limit to top 3 matches
            {
                // Get pattern data
                if (_patternDatabase.TryGetValue(match.PatternName, out var pattern))
                {
                    // Generate outcome descriptions based on pattern
                    var outcomeDescriptions = GenerateOutcomeDescriptions(pattern);
                    
                    foreach (var description in outcomeDescriptions)
                    {
                        // Calculate probability based on match confidence and pattern reliability
                        double probability = match.Confidence * pattern.Reliability;
                        
                        // Add some randomness to probability
                        probability = Math.Max(0.1, Math.Min(0.9, probability + (0.1 * (_random.NextDouble() - 0.5))));
                        
                        // Create outcome
                        var outcome = new PredictedOutcome
                        {
                            Description = description,
                            Probability = probability,
                            PatternName = pattern.Name,
                            Domain = pattern.Domain,
                            Timestamp = DateTime.UtcNow
                        };
                        
                        outcomes.Add(outcome);
                    }
                }
            }
            
            // Sort outcomes by probability
            outcomes = outcomes.OrderByDescending(o => o.Probability).ToList();
            
            _logger.LogInformation("Predicted {OutcomeCount} outcomes for situation", outcomes.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error predicting outcomes");
        }
        
        return outcomes;
    }
    
    /// <summary>
    /// Generates outcome descriptions for a pattern
    /// </summary>
    /// <param name="pattern">The pattern</param>
    /// <returns>The outcome descriptions</returns>
    private List<string> GenerateOutcomeDescriptions(PatternData pattern)
    {
        var descriptions = new List<string>();
        
        switch (pattern.Name)
        {
            case "Repeated Code":
                descriptions.Add("Maintenance will become increasingly difficult as changes need to be made in multiple places");
                descriptions.Add("Bugs are likely to appear when one instance of the repeated code is updated but others are missed");
                descriptions.Add("The codebase will grow unnecessarily large, making it harder to understand and navigate");
                break;
                
            case "Increasing Complexity":
                descriptions.Add("The code will become harder to understand and maintain over time");
                descriptions.Add("New developers will take longer to onboard and become productive");
                descriptions.Add("The risk of introducing bugs during changes will increase");
                break;
                
            case "Inconsistent Naming":
                descriptions.Add("Developers will struggle to understand the purpose and behavior of components");
                descriptions.Add("Cognitive load will increase as developers need to remember multiple naming conventions");
                descriptions.Add("Code reviews will be less effective as reviewers miss inconsistencies");
                break;
                
            case "Tight Coupling":
                descriptions.Add("Changes to one component will require changes to many other components");
                descriptions.Add("Testing will be more difficult as components cannot be isolated");
                descriptions.Add("Reuse of components in other contexts will be challenging");
                break;
                
            case "Single Point of Failure":
                descriptions.Add("The system will experience downtime when this component fails");
                descriptions.Add("Performance bottlenecks will emerge as load increases");
                descriptions.Add("Scaling the system will be limited by this component");
                break;
                
            case "Solution Fixation":
                descriptions.Add("Alternative, potentially better solutions will be overlooked");
                descriptions.Add("Time and resources will be wasted pursuing a suboptimal approach");
                descriptions.Add("The team will struggle to adapt when the chosen solution proves inadequate");
                break;
                
            case "Premature Optimization":
                descriptions.Add("Development time will be wasted optimizing non-critical parts of the system");
                descriptions.Add("The code will become more complex and harder to maintain");
                descriptions.Add("The actual performance bottlenecks will remain unaddressed");
                break;
                
            case "Confirmation Bias":
                descriptions.Add("Important contradictory information will be overlooked or dismissed");
                descriptions.Add("Decisions will be made based on incomplete or biased information");
                descriptions.Add("The team will be surprised when outcomes don't match expectations");
                break;
                
            case "Analysis Paralysis":
                descriptions.Add("Decision-making will be delayed, potentially missing opportunities");
                descriptions.Add("Team morale will decrease due to lack of progress");
                descriptions.Add("Resources will be wasted on excessive analysis");
                break;
                
            default:
                descriptions.Add($"The {pattern.Name} pattern typically leads to challenges if not addressed");
                descriptions.Add($"Teams that recognize the {pattern.Name} pattern can take proactive steps to mitigate issues");
                break;
        }
        
        return descriptions;
    }
    
    /// <summary>
    /// Gets pattern statistics
    /// </summary>
    /// <returns>The pattern statistics</returns>
    public Dictionary<string, PatternStatistics> GetPatternStatistics()
    {
        var statistics = new Dictionary<string, PatternStatistics>();
        
        foreach (var pattern in _patternDatabase.Values)
        {
            var stats = new PatternStatistics
            {
                PatternName = pattern.Name,
                Domain = pattern.Domain,
                UsageCount = pattern.UsageCount,
                AverageMatchScore = pattern.AverageMatchScore,
                LastUsedTimestamp = pattern.LastUsedTimestamp
            };
            
            statistics[pattern.Name] = stats;
        }
        
        return statistics;
    }
}

/// <summary>
/// Represents pattern data
/// </summary>
public class PatternData
{
    /// <summary>
    /// Gets or sets the pattern name
    /// </summary>
    public string Name { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the pattern domain
    /// </summary>
    public string Domain { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the pattern description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the pattern reliability (0.0 to 1.0)
    /// </summary>
    public double Reliability { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the pattern indicators
    /// </summary>
    public List<string> Indicators { get; set; } = [];
    
    /// <summary>
    /// Gets or sets the pattern creation timestamp
    /// </summary>
    public DateTime CreationTimestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the pattern last used timestamp
    /// </summary>
    public DateTime? LastUsedTimestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the pattern usage count
    /// </summary>
    public int UsageCount { get; set; } = 0;
    
    /// <summary>
    /// Gets or sets the pattern average match score
    /// </summary>
    public double AverageMatchScore { get; set; } = 0.0;
}

/// <summary>
/// Represents a pattern match
/// </summary>
public class PatternMatch
{
    /// <summary>
    /// Gets or sets the pattern name
    /// </summary>
    public string PatternName { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the pattern domain
    /// </summary>
    public string Domain { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the pattern description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the match score (0.0 to 1.0)
    /// </summary>
    public double MatchScore { get; set; } = 0.0;
    
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
/// Represents a pattern instance
/// </summary>
public class PatternInstance
{
    /// <summary>
    /// Gets or sets the pattern name
    /// </summary>
    public string PatternName { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the situation
    /// </summary>
    public string Situation { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the match score (0.0 to 1.0)
    /// </summary>
    public double MatchScore { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; }
}

/// <summary>
/// Represents a predicted outcome
/// </summary>
public class PredictedOutcome
{
    /// <summary>
    /// Gets or sets the outcome description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the outcome probability (0.0 to 1.0)
    /// </summary>
    public double Probability { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the pattern name
    /// </summary>
    public string PatternName { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the pattern domain
    /// </summary>
    public string Domain { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; }
}

/// <summary>
/// Represents pattern statistics
/// </summary>
public class PatternStatistics
{
    /// <summary>
    /// Gets or sets the pattern name
    /// </summary>
    public string PatternName { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the pattern domain
    /// </summary>
    public string Domain { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the pattern usage count
    /// </summary>
    public int UsageCount { get; set; } = 0;
    
    /// <summary>
    /// Gets or sets the pattern average match score
    /// </summary>
    public double AverageMatchScore { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the pattern last used timestamp
    /// </summary>
    public DateTime? LastUsedTimestamp { get; set; }
}

