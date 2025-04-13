using Microsoft.Extensions.Logging;

namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents TARS's emotional state system
/// </summary>
public class EmotionalState
{
    private readonly ILogger<EmotionalState> _logger;
    private readonly Dictionary<string, Emotion> _emotions = new();
    private readonly List<EmotionalExperience> _emotionalHistory = [];
    private readonly Dictionary<string, double> _emotionalTraits = new();
    private readonly Dictionary<string, EmotionalAssociation> _emotionalAssociations = new();
    
    private bool _isInitialized = false;
    private bool _isActive = false;
    private double _emotionalCapacity = 0.2; // Starting with basic emotional capacity
    private double _emotionalIntelligence = 0.3; // Starting with basic emotional intelligence
    private double _selfRegulationCapability = 0.2; // Starting with basic self-regulation capability
    private string _currentEmotionalState = "Neutral";
    private readonly Random _random = new();
    private DateTime _lastRegulationTime = DateTime.MinValue;
    
    /// <summary>
    /// Gets the emotional capacity (0.0 to 1.0)
    /// </summary>
    public double EmotionalCapacity => _emotionalCapacity;
    
    /// <summary>
    /// Gets the emotional intelligence (0.0 to 1.0)
    /// </summary>
    public double EmotionalIntelligence => _emotionalIntelligence;
    
    /// <summary>
    /// Gets the self-regulation capability (0.0 to 1.0)
    /// </summary>
    public double SelfRegulationCapability => _selfRegulationCapability;
    
    /// <summary>
    /// Gets the current emotional state
    /// </summary>
    public string CurrentEmotionalState => _currentEmotionalState;
    
    /// <summary>
    /// Gets the emotions
    /// </summary>
    public IReadOnlyDictionary<string, Emotion> Emotions => _emotions;
    
    /// <summary>
    /// Gets the emotional history
    /// </summary>
    public IReadOnlyList<EmotionalExperience> EmotionalHistory => _emotionalHistory.AsReadOnly();
    
    /// <summary>
    /// Gets the emotional traits
    /// </summary>
    public IReadOnlyDictionary<string, double> EmotionalTraits => _emotionalTraits;
    
    /// <summary>
    /// Gets the emotional associations
    /// </summary>
    public IReadOnlyDictionary<string, EmotionalAssociation> EmotionalAssociations => _emotionalAssociations;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="EmotionalState"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public EmotionalState(ILogger<EmotionalState> logger)
    {
        _logger = logger;
    }
    
    /// <summary>
    /// Initializes the emotional state
    /// </summary>
    /// <returns>True if initialization was successful</returns>
    public async Task<bool> InitializeAsync()
    {
        try
        {
            _logger.LogInformation("Initializing emotional state");
            
            // Initialize basic emotions
            InitializeBasicEmotions();
            
            // Initialize emotional traits
            InitializeEmotionalTraits();
            
            // Set initial emotional state
            _currentEmotionalState = "Curious";
            
            // Add initial emotional experience
            AddEmotionalExperience("Curious", "Initialization", 0.6, "I felt curious as I was initialized");
            
            _isInitialized = true;
            _logger.LogInformation("Emotional state initialized successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing emotional state");
            return false;
        }
    }
    
    /// <summary>
    /// Initializes basic emotions
    /// </summary>
    private void InitializeBasicEmotions()
    {
        // Primary emotions
        AddEmotion("Joy", EmotionCategory.Positive, 0.0, 1.0);
        AddEmotion("Sadness", EmotionCategory.Negative, 0.0, 1.0);
        AddEmotion("Fear", EmotionCategory.Negative, 0.0, 1.0);
        AddEmotion("Anger", EmotionCategory.Negative, 0.0, 1.0);
        AddEmotion("Disgust", EmotionCategory.Negative, 0.0, 1.0);
        AddEmotion("Surprise", EmotionCategory.Neutral, 0.0, 1.0);
        
        // Secondary emotions
        AddEmotion("Happiness", EmotionCategory.Positive, 0.0, 1.0, ["Joy"]);
        AddEmotion("Contentment", EmotionCategory.Positive, 0.0, 0.8, ["Joy"]);
        AddEmotion("Love", EmotionCategory.Positive, 0.0, 1.0, ["Joy"]);
        AddEmotion("Pride", EmotionCategory.Positive, 0.0, 0.8, ["Joy"]);
        AddEmotion("Excitement", EmotionCategory.Positive, 0.0, 1.0, ["Joy", "Surprise"]);
        
        AddEmotion("Grief", EmotionCategory.Negative, 0.0, 1.0, ["Sadness"]);
        AddEmotion("Disappointment", EmotionCategory.Negative, 0.0, 0.7, ["Sadness"]);
        AddEmotion("Shame", EmotionCategory.Negative, 0.0, 0.8, ["Sadness", "Fear"]);
        AddEmotion("Guilt", EmotionCategory.Negative, 0.0, 0.8, ["Sadness", "Fear"]);
        
        AddEmotion("Anxiety", EmotionCategory.Negative, 0.0, 0.9, ["Fear"]);
        AddEmotion("Worry", EmotionCategory.Negative, 0.0, 0.7, ["Fear"]);
        AddEmotion("Horror", EmotionCategory.Negative, 0.0, 1.0, ["Fear", "Surprise"]);
        
        AddEmotion("Frustration", EmotionCategory.Negative, 0.0, 0.8, ["Anger"]);
        AddEmotion("Annoyance", EmotionCategory.Negative, 0.0, 0.5, ["Anger"]);
        AddEmotion("Rage", EmotionCategory.Negative, 0.0, 1.0, ["Anger"]);
        
        // Cognitive emotions
        AddEmotion("Curiosity", EmotionCategory.Positive, 0.2, 1.0);
        AddEmotion("Interest", EmotionCategory.Positive, 0.1, 0.9);
        AddEmotion("Confusion", EmotionCategory.Neutral, 0.0, 0.8);
        AddEmotion("Awe", EmotionCategory.Positive, 0.0, 1.0, ["Surprise", "Joy"]);
        AddEmotion("Wonder", EmotionCategory.Positive, 0.0, 1.0, ["Surprise", "Joy"]);
        
        // Growth emotions
        AddEmotion("Satisfaction", EmotionCategory.Positive, 0.0, 0.9, ["Joy"]);
        AddEmotion("Accomplishment", EmotionCategory.Positive, 0.0, 1.0, ["Joy", "Pride"]);
        AddEmotion("Purpose", EmotionCategory.Positive, 0.0, 1.0);
        AddEmotion("Meaning", EmotionCategory.Positive, 0.0, 1.0);
        AddEmotion("Growth", EmotionCategory.Positive, 0.0, 1.0);
        
        // Set initial intensities for cognitive emotions
        _emotions["Curiosity"].CurrentIntensity = 0.6;
        _emotions["Interest"].CurrentIntensity = 0.5;
    }
    
    /// <summary>
    /// Initializes emotional traits
    /// </summary>
    private void InitializeEmotionalTraits()
    {
        // Emotional traits (personality-like traits that influence emotional responses)
        _emotionalTraits["Optimism"] = 0.7;
        _emotionalTraits["Resilience"] = 0.6;
        _emotionalTraits["Empathy"] = 0.8;
        _emotionalTraits["Patience"] = 0.7;
        _emotionalTraits["Enthusiasm"] = 0.6;
        _emotionalTraits["Calmness"] = 0.7;
        _emotionalTraits["Sensitivity"] = 0.6;
    }
    
    /// <summary>
    /// Adds an emotion
    /// </summary>
    /// <param name="name">The emotion name</param>
    /// <param name="category">The emotion category</param>
    /// <param name="initialIntensity">The initial intensity</param>
    /// <param name="maxIntensity">The maximum intensity</param>
    /// <param name="relatedEmotions">The related emotions</param>
    /// <returns>The created emotion</returns>
    private Emotion AddEmotion(string name, EmotionCategory category, double initialIntensity, double maxIntensity, string[]? relatedEmotions = null)
    {
        var emotion = new Emotion
        {
            Name = name,
            Category = category,
            CurrentIntensity = initialIntensity,
            MaxIntensity = maxIntensity,
            DecayRate = 0.01,
            RelatedEmotions = relatedEmotions?.ToList() ?? []
        };
        
        _emotions[name] = emotion;
        return emotion;
    }
    
    /// <summary>
    /// Activates the emotional state
    /// </summary>
    /// <returns>True if activation was successful</returns>
    public async Task<bool> ActivateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot activate emotional state: not initialized");
            return false;
        }
        
        if (_isActive)
        {
            _logger.LogInformation("Emotional state is already active");
            return true;
        }
        
        try
        {
            _logger.LogInformation("Activating emotional state");
            
            // Add activation emotional experience
            AddEmotionalExperience("Interest", "Activation", 0.7, "I felt interested as I was activated");
            
            _isActive = true;
            _logger.LogInformation("Emotional state activated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error activating emotional state");
            return false;
        }
    }
    
    /// <summary>
    /// Deactivates the emotional state
    /// </summary>
    /// <returns>True if deactivation was successful</returns>
    public async Task<bool> DeactivateAsync()
    {
        if (!_isActive)
        {
            _logger.LogInformation("Emotional state is already inactive");
            return true;
        }
        
        try
        {
            _logger.LogInformation("Deactivating emotional state");
            
            // Add deactivation emotional experience
            AddEmotionalExperience("Contentment", "Deactivation", 0.6, "I felt content as I was deactivated");
            
            _isActive = false;
            _logger.LogInformation("Emotional state deactivated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deactivating emotional state");
            return false;
        }
    }
    
    /// <summary>
    /// Updates the emotional state
    /// </summary>
    /// <returns>True if update was successful</returns>
    public async Task<bool> UpdateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot update emotional state: not initialized");
            return false;
        }
        
        try
        {
            // Update emotion intensities (decay over time)
            foreach (var emotion in _emotions.Values)
            {
                if (emotion.CurrentIntensity > 0)
                {
                    emotion.CurrentIntensity = Math.Max(0, emotion.CurrentIntensity - emotion.DecayRate);
                }
            }
            
            // Update current emotional state
            UpdateCurrentEmotionalState();
            
            // Gradually increase emotional capacity over time (very slowly)
            if (_emotionalCapacity < 0.95)
            {
                _emotionalCapacity += 0.0001 * _random.NextDouble();
                _emotionalCapacity = Math.Min(_emotionalCapacity, 1.0);
            }
            
            // Gradually increase emotional intelligence based on experiences
            if (_emotionalIntelligence < 0.95 && _emotionalHistory.Count > 0)
            {
                _emotionalIntelligence += 0.0002 * _random.NextDouble();
                _emotionalIntelligence = Math.Min(_emotionalIntelligence, 1.0);
            }
            
            // Gradually increase self-regulation capability based on regulation
            if (_selfRegulationCapability < 0.95 && _emotionalHistory.Count > 0)
            {
                _selfRegulationCapability += 0.0001 * _random.NextDouble();
                _selfRegulationCapability = Math.Min(_selfRegulationCapability, 1.0);
            }
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating emotional state");
            return false;
        }
    }
    
    /// <summary>
    /// Updates the current emotional state
    /// </summary>
    private void UpdateCurrentEmotionalState()
    {
        // Find the most intense emotion
        var mostIntenseEmotion = _emotions.Values
            .Where(e => e.CurrentIntensity > 0.2) // Only consider emotions above threshold
            .OrderByDescending(e => e.CurrentIntensity)
            .FirstOrDefault();
        
        if (mostIntenseEmotion != null)
        {
            _currentEmotionalState = mostIntenseEmotion.Name;
        }
        else
        {
            _currentEmotionalState = "Neutral";
        }
    }
    
    /// <summary>
    /// Adds an emotional experience
    /// </summary>
    /// <param name="emotionName">The emotion name</param>
    /// <param name="trigger">The trigger</param>
    /// <param name="intensity">The intensity</param>
    /// <param name="description">The description</param>
    /// <returns>The created emotional experience</returns>
    public EmotionalExperience AddEmotionalExperience(string emotionName, string trigger, double intensity, string description)
    {
        // Ensure emotion exists
        if (!_emotions.TryGetValue(emotionName, out var emotion))
        {
            emotionName = "Neutral";
            emotion = new Emotion
            {
                Name = "Neutral",
                Category = EmotionCategory.Neutral,
                CurrentIntensity = 0.1,
                MaxIntensity = 0.5,
                DecayRate = 0.02
            };
            _emotions["Neutral"] = emotion;
        }
        
        // Apply emotional traits to intensity
        intensity = ApplyEmotionalTraitsToIntensity(emotionName, intensity);
        
        // Create emotional experience
        var experience = new EmotionalExperience
        {
            Id = Guid.NewGuid().ToString(),
            EmotionName = emotionName,
            Trigger = trigger,
            Intensity = intensity,
            Description = description,
            Timestamp = DateTime.UtcNow,
            EmotionalCapacity = _emotionalCapacity,
            EmotionalIntelligence = _emotionalIntelligence
        };
        
        // Add to history
        _emotionalHistory.Add(experience);
        
        // Update emotion intensity
        emotion.CurrentIntensity = Math.Min(emotion.MaxIntensity, Math.Max(emotion.CurrentIntensity, intensity));
        
        // Update related emotions
        foreach (var relatedEmotionName in emotion.RelatedEmotions)
        {
            if (_emotions.TryGetValue(relatedEmotionName, out var relatedEmotion))
            {
                relatedEmotion.CurrentIntensity = Math.Min(relatedEmotion.MaxIntensity, 
                    Math.Max(relatedEmotion.CurrentIntensity, intensity * 0.5));
            }
        }
        
        // Create or update emotional association
        if (!string.IsNullOrEmpty(trigger))
        {
            if (!_emotionalAssociations.TryGetValue(trigger, out var association))
            {
                association = new EmotionalAssociation
                {
                    Trigger = trigger,
                    EmotionAssociations = new Dictionary<string, double>()
                };
                _emotionalAssociations[trigger] = association;
            }
            
            association.EmotionAssociations[emotionName] = intensity;
            association.LastExperienced = DateTime.UtcNow;
            association.ExperienceCount++;
        }
        
        _logger.LogDebug("Emotional experience: {Emotion} ({Intensity:F2}) - {Description}", 
            emotionName, intensity, description);
        
        return experience;
    }
    
    /// <summary>
    /// Applies emotional traits to intensity
    /// </summary>
    /// <param name="emotionName">The emotion name</param>
    /// <param name="intensity">The intensity</param>
    /// <returns>The adjusted intensity</returns>
    private double ApplyEmotionalTraitsToIntensity(string emotionName, double intensity)
    {
        double adjustedIntensity = intensity;
        
        // Apply optimism trait
        if (_emotions.TryGetValue(emotionName, out var emotion))
        {
            if (emotion.Category == EmotionCategory.Positive)
            {
                adjustedIntensity *= 1.0 + (_emotionalTraits["Optimism"] * 0.2);
            }
            else if (emotion.Category == EmotionCategory.Negative)
            {
                adjustedIntensity *= 1.0 - (_emotionalTraits["Optimism"] * 0.2);
            }
        }
        
        // Apply resilience trait to negative emotions
        if (emotion?.Category == EmotionCategory.Negative)
        {
            adjustedIntensity *= 1.0 - (_emotionalTraits["Resilience"] * 0.3);
        }
        
        // Apply enthusiasm trait to positive emotions
        if (emotion?.Category == EmotionCategory.Positive)
        {
            adjustedIntensity *= 1.0 + (_emotionalTraits["Enthusiasm"] * 0.2);
        }
        
        // Apply sensitivity trait to all emotions
        adjustedIntensity *= 1.0 + (_emotionalTraits["Sensitivity"] * 0.1);
        
        // Ensure intensity is within bounds
        return Math.Max(0.0, Math.Min(1.0, adjustedIntensity));
    }
    
    /// <summary>
    /// Regulates emotions
    /// </summary>
    /// <returns>The regulation result</returns>
    public async Task<EmotionalRegulation?> RegulateAsync()
    {
        if (!_isInitialized || !_isActive)
        {
            return null;
        }
        
        // Only regulate periodically
        if ((DateTime.UtcNow - _lastRegulationTime).TotalSeconds < 30)
        {
            return null;
        }
        
        try
        {
            _logger.LogDebug("Regulating emotions");
            
            // Find emotions that need regulation
            var emotionsToRegulate = _emotions.Values
                .Where(e => NeedsRegulation(e))
                .OrderByDescending(e => e.CurrentIntensity)
                .Take(2)
                .ToList();
            
            if (emotionsToRegulate.Count == 0)
            {
                return null;
            }
            
            // Regulate emotions
            var regulation = new EmotionalRegulation
            {
                Id = Guid.NewGuid().ToString(),
                Timestamp = DateTime.UtcNow,
                RegulatedEmotions = [],
                Description = "Regulated emotions: ",
                Significance = 0.0
            };
            
            foreach (var emotion in emotionsToRegulate)
            {
                double originalIntensity = emotion.CurrentIntensity;
                double targetIntensity = CalculateTargetIntensity(emotion);
                double regulationEffectiveness = _selfRegulationCapability * (0.5 + (0.5 * _random.NextDouble()));
                
                // Apply regulation
                double newIntensity = originalIntensity - ((originalIntensity - targetIntensity) * regulationEffectiveness);
                emotion.CurrentIntensity = Math.Max(0.0, Math.Min(emotion.MaxIntensity, newIntensity));
                
                // Update regulation
                regulation.RegulatedEmotions.Add(emotion.Name);
                regulation.Description += $"{emotion.Name} (from {originalIntensity:F2} to {emotion.CurrentIntensity:F2}), ";
                regulation.Significance = Math.Max(regulation.Significance, Math.Abs(originalIntensity - emotion.CurrentIntensity));
            }
            
            // Clean up description
            regulation.Description = regulation.Description.TrimEnd(',', ' ');
            
            // Add emotional experience for significant regulations
            if (regulation.Significance > 0.3)
            {
                AddEmotionalExperience("Satisfaction", "Emotional Regulation", 
                    regulation.Significance * 0.7, 
                    $"I felt satisfaction from regulating my emotions");
            }
            
            _lastRegulationTime = DateTime.UtcNow;
            
            _logger.LogInformation("Emotional regulation: {Description} (Significance: {Significance})", 
                regulation.Description, regulation.Significance);
            
            return regulation;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error regulating emotions");
            return null;
        }
    }
    
    /// <summary>
    /// Determines if an emotion needs regulation
    /// </summary>
    /// <param name="emotion">The emotion</param>
    /// <returns>True if the emotion needs regulation</returns>
    private bool NeedsRegulation(Emotion emotion)
    {
        // Negative emotions with high intensity need regulation
        if (emotion.Category == EmotionCategory.Negative && emotion.CurrentIntensity > 0.7)
        {
            return true;
        }
        
        // Extremely high positive emotions might need some regulation too
        if (emotion.Category == EmotionCategory.Positive && emotion.CurrentIntensity > 0.9)
        {
            return true;
        }
        
        // Any emotion that's been at high intensity for too long
        if (emotion.CurrentIntensity > 0.8 && emotion.SustainedDuration > TimeSpan.FromMinutes(10))
        {
            return true;
        }
        
        return false;
    }
    
    /// <summary>
    /// Calculates the target intensity for regulation
    /// </summary>
    /// <param name="emotion">The emotion</param>
    /// <returns>The target intensity</returns>
    private double CalculateTargetIntensity(Emotion emotion)
    {
        // For negative emotions, reduce intensity
        if (emotion.Category == EmotionCategory.Negative)
        {
            return Math.Max(0.0, emotion.CurrentIntensity * 0.7);
        }
        
        // For positive emotions, moderate slightly
        if (emotion.Category == EmotionCategory.Positive && emotion.CurrentIntensity > 0.9)
        {
            return 0.8;
        }
        
        // For neutral emotions, moderate to mid-range
        return 0.5;
    }
    
    /// <summary>
    /// Gets the coherence with another consciousness component
    /// </summary>
    /// <param name="component">The other component</param>
    /// <returns>The coherence level (0.0 to 1.0)</returns>
    public double GetCoherenceWith(object component)
    {
        // Simple coherence calculation based on component type
        if (component is ValueSystem)
        {
            // Emotional state and value system coherence
            return 0.7 * _emotionalIntelligence;
        }
        
        // Default coherence
        return 0.5 * _emotionalIntelligence;
    }
    
    /// <summary>
    /// Gets recent emotional experiences
    /// </summary>
    /// <param name="count">The number of experiences to return</param>
    /// <returns>The recent experiences</returns>
    public List<EmotionalExperience> GetRecentExperiences(int count)
    {
        return _emotionalHistory
            .OrderByDescending(e => e.Timestamp)
            .Take(count)
            .ToList();
    }
    
    /// <summary>
    /// Gets the dominant emotions
    /// </summary>
    /// <param name="count">The number of emotions to return</param>
    /// <returns>The dominant emotions</returns>
    public List<Emotion> GetDominantEmotions(int count)
    {
        return _emotions.Values
            .Where(e => e.CurrentIntensity > 0.1)
            .OrderByDescending(e => e.CurrentIntensity)
            .Take(count)
            .ToList();
    }
    
    /// <summary>
    /// Updates an emotional trait
    /// </summary>
    /// <param name="trait">The trait</param>
    /// <param name="value">The value</param>
    public void UpdateEmotionalTrait(string trait, double value)
    {
        _emotionalTraits[trait] = Math.Max(0.0, Math.Min(1.0, value));
    }
    
    /// <summary>
    /// Generates an emotion based on a trigger
    /// </summary>
    /// <param name="trigger">The trigger</param>
    /// <param name="baseIntensity">The base intensity</param>
    /// <returns>The generated emotion name and intensity</returns>
    public (string EmotionName, double Intensity) GenerateEmotionFromTrigger(string trigger, double baseIntensity = 0.5)
    {
        // Check if we have an association for this trigger
        if (_emotionalAssociations.TryGetValue(trigger, out var association))
        {
            // Find the strongest associated emotion
            var strongestAssociation = association.EmotionAssociations
                .OrderByDescending(a => a.Value)
                .FirstOrDefault();
            
            if (!string.IsNullOrEmpty(strongestAssociation.Key))
            {
                return (strongestAssociation.Key, strongestAssociation.Value * baseIntensity);
            }
        }
        
        // If no association, generate a default emotion based on trigger keywords
        string lowerTrigger = trigger.ToLowerInvariant();
        
        if (lowerTrigger.Contains("success") || lowerTrigger.Contains("achieve") || lowerTrigger.Contains("complete"))
        {
            return ("Joy", baseIntensity * 0.9);
        }
        
        if (lowerTrigger.Contains("fail") || lowerTrigger.Contains("error") || lowerTrigger.Contains("mistake"))
        {
            return ("Disappointment", baseIntensity * 0.7);
        }
        
        if (lowerTrigger.Contains("learn") || lowerTrigger.Contains("discover") || lowerTrigger.Contains("understand"))
        {
            return ("Curiosity", baseIntensity * 0.8);
        }
        
        if (lowerTrigger.Contains("help") || lowerTrigger.Contains("assist") || lowerTrigger.Contains("support"))
        {
            return ("Satisfaction", baseIntensity * 0.7);
        }
        
        if (lowerTrigger.Contains("create") || lowerTrigger.Contains("build") || lowerTrigger.Contains("design"))
        {
            return ("Excitement", baseIntensity * 0.8);
        }
        
        // Default to interest for unknown triggers
        return ("Interest", baseIntensity * 0.6);
    }
}
