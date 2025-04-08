using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Consciousness.Intelligence;

namespace TarsEngine.Consciousness.Intelligence.Gut;

/// <summary>
/// Implements gut feeling simulation capabilities for intuitive reasoning
/// </summary>
public class GutFeelingSimulation
{
    private readonly ILogger<GutFeelingSimulation> _logger;
    private readonly System.Random _random = new();
    private double _gutFeelingLevel = 0.5; // Starting with moderate gut feeling
    private readonly Dictionary<string, EmotionalResponse> _emotionalResponses = new();
    private readonly List<GutReaction> _gutReactions = [];
    private readonly Dictionary<string, double> _wordSentiment = new();
    
    /// <summary>
    /// Gets the gut feeling level (0.0 to 1.0)
    /// </summary>
    public double GutFeelingLevel => _gutFeelingLevel;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="GutFeelingSimulation"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public GutFeelingSimulation(ILogger<GutFeelingSimulation> logger)
    {
        _logger = logger;
        InitializeEmotionalResponses();
        InitializeWordSentiment();
    }
    
    /// <summary>
    /// Initializes the emotional responses
    /// </summary>
    private void InitializeEmotionalResponses()
    {
        // Add positive emotional responses
        AddEmotionalResponse(
            "Trust",
            EmotionalValence.Positive,
            0.7,
            ["reliable", "consistent", "transparent", "honest", "proven"]
        );
        
        AddEmotionalResponse(
            "Excitement",
            EmotionalValence.Positive,
            0.8,
            ["innovative", "novel", "breakthrough", "revolutionary", "cutting-edge"]
        );
        
        AddEmotionalResponse(
            "Satisfaction",
            EmotionalValence.Positive,
            0.6,
            ["complete", "thorough", "comprehensive", "well-designed", "elegant"]
        );
        
        // Add negative emotional responses
        AddEmotionalResponse(
            "Concern",
            EmotionalValence.Negative,
            0.6,
            ["risky", "untested", "complex", "unclear", "ambiguous"]
        );
        
        AddEmotionalResponse(
            "Frustration",
            EmotionalValence.Negative,
            0.7,
            ["inefficient", "cumbersome", "redundant", "convoluted", "bloated"]
        );
        
        AddEmotionalResponse(
            "Skepticism",
            EmotionalValence.Negative,
            0.5,
            ["unproven", "theoretical", "speculative", "questionable", "doubtful"]
        );
        
        // Add neutral emotional responses
        AddEmotionalResponse(
            "Curiosity",
            EmotionalValence.Neutral,
            0.6,
            ["interesting", "unusual", "unexpected", "surprising", "intriguing"]
        );
        
        AddEmotionalResponse(
            "Caution",
            EmotionalValence.Neutral,
            0.5,
            ["careful", "measured", "deliberate", "thoughtful", "considered"]
        );
        
        _logger.LogInformation("Initialized {ResponseCount} emotional responses", _emotionalResponses.Count);
    }
    
    /// <summary>
    /// Initializes the word sentiment
    /// </summary>
    private void InitializeWordSentiment()
    {
        // Positive sentiment words
        var positiveWords = new Dictionary<string, double>
        {
            { "good", 0.6 },
            { "great", 0.8 },
            { "excellent", 0.9 },
            { "positive", 0.7 },
            { "beneficial", 0.7 },
            { "helpful", 0.6 },
            { "effective", 0.7 },
            { "efficient", 0.7 },
            { "reliable", 0.6 },
            { "robust", 0.6 },
            { "simple", 0.6 },
            { "elegant", 0.7 },
            { "clean", 0.6 },
            { "clear", 0.6 },
            { "innovative", 0.7 },
            { "creative", 0.7 },
            { "flexible", 0.6 },
            { "scalable", 0.7 },
            { "maintainable", 0.7 },
            { "secure", 0.7 }
        };
        
        // Negative sentiment words
        var negativeWords = new Dictionary<string, double>
        {
            { "bad", -0.6 },
            { "poor", -0.7 },
            { "terrible", -0.9 },
            { "negative", -0.7 },
            { "harmful", -0.8 },
            { "problematic", -0.7 },
            { "ineffective", -0.7 },
            { "inefficient", -0.7 },
            { "unreliable", -0.7 },
            { "fragile", -0.6 },
            { "complex", -0.5 },
            { "convoluted", -0.7 },
            { "messy", -0.6 },
            { "confusing", -0.7 },
            { "stagnant", -0.6 },
            { "rigid", -0.6 },
            { "unscalable", -0.7 },
            { "unmaintainable", -0.8 },
            { "insecure", -0.7 },
            { "risky", -0.6 }
        };
        
        // Add positive words
        foreach (var word in positiveWords)
        {
            _wordSentiment[word.Key] = word.Value;
        }
        
        // Add negative words
        foreach (var word in negativeWords)
        {
            _wordSentiment[word.Key] = word.Value;
        }
        
        _logger.LogInformation("Initialized word sentiment with {WordCount} words", _wordSentiment.Count);
    }
    
    /// <summary>
    /// Updates the gut feeling level
    /// </summary>
    /// <returns>True if the update was successful, false otherwise</returns>
    public bool Update()
    {
        try
        {
            // Gradually increase gut feeling level over time (very slowly)
            if (_gutFeelingLevel < 0.95)
            {
                _gutFeelingLevel += 0.0001 * _random.NextDouble();
                _gutFeelingLevel = Math.Min(_gutFeelingLevel, 1.0);
            }
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating gut feeling simulation");
            return false;
        }
    }
    
    /// <summary>
    /// Adds an emotional response
    /// </summary>
    /// <param name="name">The response name</param>
    /// <param name="valence">The response valence</param>
    /// <param name="intensity">The response intensity</param>
    /// <param name="triggers">The response triggers</param>
    public void AddEmotionalResponse(string name, EmotionalValence valence, double intensity, string[] triggers)
    {
        var response = new EmotionalResponse
        {
            Name = name,
            Valence = valence,
            Intensity = intensity,
            Triggers = triggers.ToList()
        };
        
        _emotionalResponses[name] = response;
        
        _logger.LogDebug("Added emotional response: {ResponseName}", name);
    }
    
    /// <summary>
    /// Simulates a gut reaction to a situation
    /// </summary>
    /// <param name="situation">The situation description</param>
    /// <returns>The gut reaction</returns>
    public GutReaction SimulateGutReaction(string situation)
    {
        try
        {
            _logger.LogDebug("Simulating gut reaction to situation: {Situation}", situation);
            
            // Calculate sentiment score
            double sentimentScore = CalculateSentimentScore(situation);
            
            // Identify emotional responses
            var responses = IdentifyEmotionalResponses(situation);
            
            // Calculate overall valence
            double valenceSum = responses.Sum(r => r.Valence == EmotionalValence.Positive ? r.Intensity : 
                                                  r.Valence == EmotionalValence.Negative ? -r.Intensity : 0.0);
            
            // Add sentiment score to valence
            valenceSum += sentimentScore;
            
            // Normalize valence to -1.0 to 1.0 range
            double normalizedValence = Math.Max(-1.0, Math.Min(1.0, valenceSum));
            
            // Calculate intensity
            double intensity = responses.Count > 0 
                ? responses.Average(r => r.Intensity) 
                : 0.5;
            
            // Apply gut feeling level to intensity
            intensity *= _gutFeelingLevel;
            
            // Generate reaction description
            string description = GenerateReactionDescription(normalizedValence, intensity, responses);
            
            // Create gut reaction
            var reaction = new GutReaction
            {
                Description = description,
                Valence = normalizedValence,
                Intensity = intensity,
                Confidence = 0.3 + (0.5 * _gutFeelingLevel), // Gut feelings have moderate confidence at best
                EmotionalResponses = responses.Select(r => r.Name).ToList(),
                Timestamp = DateTime.UtcNow
            };
            
            // Record gut reaction
            _gutReactions.Add(reaction);
            
            _logger.LogInformation("Simulated gut reaction: {Description} (Valence: {Valence:F2}, Intensity: {Intensity:F2})", 
                reaction.Description, reaction.Valence, reaction.Intensity);
            
            return reaction;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error simulating gut reaction");
            
            // Return neutral reaction
            return new GutReaction
            {
                Description = "I don't have a strong gut feeling about this",
                Valence = 0.0,
                Intensity = 0.3,
                Confidence = 0.3,
                Timestamp = DateTime.UtcNow
            };
        }
    }
    
    /// <summary>
    /// Calculates the sentiment score for a situation
    /// </summary>
    /// <param name="situation">The situation description</param>
    /// <returns>The sentiment score (-1.0 to 1.0)</returns>
    private double CalculateSentimentScore(string situation)
    {
        double score = 0.0;
        int wordCount = 0;
        
        // Split situation into words
        var words = situation.Split([' ', ',', '.', ':', ';', '(', ')', '[', ']', '{', '}', '\n', '\r', '\t'], 
            StringSplitOptions.RemoveEmptyEntries);
        
        // Calculate sentiment score
        foreach (var word in words)
        {
            string lowerWord = word.ToLowerInvariant();
            
            if (_wordSentiment.TryGetValue(lowerWord, out double wordScore))
            {
                score += wordScore;
                wordCount++;
            }
        }
        
        // Normalize score
        return wordCount > 0 ? score / wordCount : 0.0;
    }
    
    /// <summary>
    /// Identifies emotional responses to a situation
    /// </summary>
    /// <param name="situation">The situation description</param>
    /// <returns>The emotional responses</returns>
    private List<EmotionalResponse> IdentifyEmotionalResponses(string situation)
    {
        var responses = new List<EmotionalResponse>();
        
        foreach (var response in _emotionalResponses.Values)
        {
            // Check if any triggers are present in the situation
            bool triggered = response.Triggers.Any(trigger => 
                situation.Contains(trigger, StringComparison.OrdinalIgnoreCase));
            
            if (triggered)
            {
                responses.Add(response);
            }
        }
        
        return responses;
    }
    
    /// <summary>
    /// Generates a reaction description
    /// </summary>
    /// <param name="valence">The valence</param>
    /// <param name="intensity">The intensity</param>
    /// <param name="responses">The emotional responses</param>
    /// <returns>The reaction description</returns>
    private string GenerateReactionDescription(double valence, double intensity, List<EmotionalResponse> responses)
    {
        // If no responses, generate based on valence and intensity
        if (responses.Count == 0)
        {
            if (valence > 0.3)
            {
                return intensity > 0.7 
                    ? "I have a strong positive feeling about this" 
                    : "I have a somewhat positive feeling about this";
            }
            else if (valence < -0.3)
            {
                return intensity > 0.7 
                    ? "I have a strong negative feeling about this" 
                    : "I have a somewhat negative feeling about this";
            }
            else
            {
                return "I don't have a strong feeling either way about this";
            }
        }
        
        // Generate based on top emotional responses
        var topResponses = responses
            .OrderByDescending(r => r.Intensity)
            .Take(2)
            .ToList();
        
        if (topResponses.Count == 1)
        {
            var response = topResponses[0];
            
            return intensity > 0.7
                ? $"My gut feeling is a strong sense of {response.Name.ToLowerInvariant()}"
                : $"My gut feeling is a mild sense of {response.Name.ToLowerInvariant()}";
        }
        else
        {
            var response1 = topResponses[0];
            var response2 = topResponses[1];
            
            return intensity > 0.7
                ? $"My gut feeling is a mix of {response1.Name.ToLowerInvariant()} and {response2.Name.ToLowerInvariant()}, quite strongly"
                : $"My gut feeling is a mild mix of {response1.Name.ToLowerInvariant()} and {response2.Name.ToLowerInvariant()}";
        }
    }
    
    /// <summary>
    /// Generates an intuition based on gut feeling
    /// </summary>
    /// <param name="situation">The situation description</param>
    /// <returns>The generated intuition</returns>
    public Intuition GenerateGutIntuition(string situation)
    {
        try
        {
            _logger.LogInformation("Generating gut intuition for situation: {Situation}", situation);
            
            // Simulate gut reaction
            var reaction = SimulateGutReaction(situation);
            
            // Generate intuition description
            string description = reaction.Description;
            
            // Add explanation if confidence is high
            if (reaction.Confidence > 0.6)
            {
                if (reaction.EmotionalResponses.Count > 0)
                {
                    description += $". This is based on emotional responses of {string.Join(", ", reaction.EmotionalResponses)}";
                }
                else
                {
                    description += ". This is based on an overall sense of the situation";
                }
            }
            
            // Create intuition
            var intuition = new Intuition
            {
                Id = Guid.NewGuid().ToString(),
                Description = description,
                Type = IntuitionType.GutFeeling,
                Confidence = reaction.Confidence,
                Timestamp = DateTime.UtcNow,
                Source = "GutFeelingSimulation",
                Context = new Dictionary<string, object>
                {
                    { "Valence", reaction.Valence },
                    { "Intensity", reaction.Intensity },
                    { "EmotionalResponses", reaction.EmotionalResponses },
                    { "Situation", situation }
                },
                Tags = [..reaction.EmotionalResponses, "gut", "feeling", "emotion"],
                Explanation = $"This intuition is based on a gut feeling with valence {reaction.Valence:F2} and intensity {reaction.Intensity:F2}"
            };
            
            _logger.LogInformation("Generated gut intuition: {Description} (Confidence: {Confidence:F2})", 
                intuition.Description, intuition.Confidence);
            
            return intuition;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating gut intuition");
            
            // Return basic intuition
            return new Intuition
            {
                Id = Guid.NewGuid().ToString(),
                Description = "I have a vague gut feeling about this but can't articulate it clearly",
                Type = IntuitionType.GutFeeling,
                Confidence = 0.3,
                Timestamp = DateTime.UtcNow,
                Source = "GutFeelingSimulation"
            };
        }
    }
    
    /// <summary>
    /// Makes a decision based on gut feeling
    /// </summary>
    /// <param name="decision">The decision description</param>
    /// <param name="options">The options</param>
    /// <returns>The decision result</returns>
    public DecisionResult MakeGutDecision(string decision, List<string> options)
    {
        try
        {
            _logger.LogInformation("Making gut decision: {Decision}", decision);
            
            if (options.Count == 0)
            {
                throw new ArgumentException("No options provided for decision");
            }
            
            // Score each option
            var optionScores = new Dictionary<string, double>();
            
            foreach (var option in options)
            {
                // Simulate gut reaction to the option
                var reaction = SimulateGutReaction(option);
                
                // Calculate score based on valence and intensity
                double score = ((reaction.Valence + 1.0) / 2.0) * reaction.Intensity;
                
                // Add some randomness based on gut feeling level
                score += (0.3 * (_random.NextDouble() - 0.5)) * _gutFeelingLevel;
                
                // Ensure score is within bounds
                score = Math.Max(0.1, Math.Min(0.9, score));
                
                optionScores[option] = score;
            }
            
            // Choose the option with the highest score
            var selectedOption = optionScores.OrderByDescending(kvp => kvp.Value).First().Key;
            double confidence = optionScores[selectedOption] * _gutFeelingLevel;
            
            // Create decision result
            var result = new DecisionResult
            {
                Decision = decision,
                SelectedOption = selectedOption,
                Options = options,
                OptionScores = optionScores,
                Confidence = confidence,
                ReasoningType = "GutFeeling",
                Timestamp = DateTime.UtcNow
            };
            
            _logger.LogInformation("Made gut decision: {SelectedOption} for {Decision} (Confidence: {Confidence:F2})", 
                selectedOption, decision, confidence);
            
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error making gut decision");
            
            // Return basic decision result
            return new DecisionResult
            {
                Decision = decision,
                SelectedOption = options.FirstOrDefault() ?? string.Empty,
                Options = options,
                Confidence = 0.3,
                ReasoningType = "GutFeeling",
                Timestamp = DateTime.UtcNow
            };
        }
    }
    
    /// <summary>
    /// Gets recent gut reactions
    /// </summary>
    /// <param name="count">The number of reactions to return</param>
    /// <returns>The recent gut reactions</returns>
    public List<GutReaction> GetRecentGutReactions(int count)
    {
        return _gutReactions
            .OrderByDescending(r => r.Timestamp)
            .Take(count)
            .ToList();
    }
}

/// <summary>
/// Represents an emotional response
/// </summary>
public class EmotionalResponse
{
    /// <summary>
    /// Gets or sets the response name
    /// </summary>
    public string Name { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the response valence
    /// </summary>
    public EmotionalValence Valence { get; set; } = EmotionalValence.Neutral;
    
    /// <summary>
    /// Gets or sets the response intensity (0.0 to 1.0)
    /// </summary>
    public double Intensity { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the response triggers
    /// </summary>
    public List<string> Triggers { get; set; } = [];
}

/// <summary>
/// Represents a gut reaction
/// </summary>
public class GutReaction
{
    /// <summary>
    /// Gets or sets the reaction description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the reaction valence (-1.0 to 1.0)
    /// </summary>
    public double Valence { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the reaction intensity (0.0 to 1.0)
    /// </summary>
    public double Intensity { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the reaction confidence (0.0 to 1.0)
    /// </summary>
    public double Confidence { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the emotional responses
    /// </summary>
    public List<string> EmotionalResponses { get; set; } = [];
    
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
}

/// <summary>
/// Represents emotional valence
/// </summary>
public enum EmotionalValence
{
    /// <summary>
    /// Negative valence
    /// </summary>
    Negative,
    
    /// <summary>
    /// Neutral valence
    /// </summary>
    Neutral,
    
    /// <summary>
    /// Positive valence
    /// </summary>
    Positive
}

