using Microsoft.Extensions.Logging;
using TarsEngine.Monads;

namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Pure implementation of emotional state that separates state from behavior
/// </summary>
public class PureEmotionalState : PureState<PureEmotionalState>
{
    // State properties
    private readonly Dictionary<string, Emotion> _emotions;
    private readonly List<EmotionalExperience> _emotionalHistory;
    private readonly List<EmotionalTrait> _emotionalTraits;
    private readonly List<EmotionalRegulation> _regulations;
    private readonly bool _isInitialized;
    private readonly bool _isActive;
    private readonly double _emotionalCapacity;
    private readonly double _emotionalIntelligence;
    private readonly double _selfRegulationCapability;
    private readonly string _currentEmotionalState;
    internal readonly DateTime _lastRegulationTime;
    private readonly Random _random;

    /// <summary>
    /// Gets the emotional capacity (0.0 to 1.0)
    /// </summary>
    public double EmotionalCapacity => _emotionalCapacity;

    /// <summary>
    /// Gets the emotional intelligence (0.0 to 1.0)
    /// </summary>
    public double EmotionalIntelligence => _emotionalIntelligence;

    /// <summary>
    /// Gets the current emotional state
    /// </summary>
    public string CurrentEmotionalState => _currentEmotionalState;

    /// <summary>
    /// Gets whether the emotional state is initialized
    /// </summary>
    public bool IsInitialized => _isInitialized;

    /// <summary>
    /// Gets whether the emotional state is active
    /// </summary>
    public bool IsActive => _isActive;

    /// <summary>
    /// Creates a new instance of the PureEmotionalState class
    /// </summary>
    public PureEmotionalState()
    {
        _emotions = new Dictionary<string, Emotion>();
        _emotionalHistory = [];
        _emotionalTraits = [];
        _regulations = [];
        _isInitialized = false;
        _isActive = false;
        _emotionalCapacity = 0.5;
        _emotionalIntelligence = 0.4;
        _selfRegulationCapability = 0.3;
        _currentEmotionalState = "Neutral";
        _lastRegulationTime = DateTime.MinValue;
        _random = new Random();
    }

    /// <summary>
    /// Private constructor for creating modified copies
    /// </summary>
    private PureEmotionalState(
        Dictionary<string, Emotion> emotions,
        List<EmotionalExperience> emotionalHistory,
        List<EmotionalTrait> emotionalTraits,
        List<EmotionalRegulation> regulations,
        bool isInitialized,
        bool isActive,
        double emotionalCapacity,
        double emotionalIntelligence,
        double selfRegulationCapability,
        string currentEmotionalState,
        DateTime lastRegulationTime,
        Random random)
    {
        _emotions = emotions;
        _emotionalHistory = emotionalHistory;
        _emotionalTraits = emotionalTraits;
        _regulations = regulations;
        _isInitialized = isInitialized;
        _isActive = isActive;
        _emotionalCapacity = emotionalCapacity;
        _emotionalIntelligence = emotionalIntelligence;
        _selfRegulationCapability = selfRegulationCapability;
        _currentEmotionalState = currentEmotionalState;
        _lastRegulationTime = lastRegulationTime;
        _random = random;
    }

    /// <summary>
    /// Creates a copy of the state with the specified modifications
    /// </summary>
    public override PureEmotionalState With(Action<PureEmotionalState> modifier)
    {
        var copy = Copy();
        modifier(copy);
        return copy;
    }

    /// <summary>
    /// Creates a copy of the state
    /// </summary>
    public override PureEmotionalState Copy()
    {
        return new PureEmotionalState(
            new Dictionary<string, Emotion>(_emotions),
            [.._emotionalHistory],
            [.._emotionalTraits],
            [.._regulations],
            _isInitialized,
            _isActive,
            _emotionalCapacity,
            _emotionalIntelligence,
            _selfRegulationCapability,
            _currentEmotionalState,
            _lastRegulationTime,
            _random
        );
    }

    /// <summary>
    /// Gets the dominant emotion
    /// </summary>
    public Emotion GetDominantEmotion()
    {
        return _emotions.Values
            .OrderByDescending(e => e.CurrentIntensity)
            .FirstOrDefault() ?? new Emotion
        {
            Name = "Neutral",
            Category = EmotionCategory.Neutral,
            CurrentIntensity = 0.1,
            MaxIntensity = 1.0,
            DecayRate = 0.01
        };
    }

    /// <summary>
    /// Gets emotions by category
    /// </summary>
    public List<Emotion> GetEmotionsByCategory(EmotionCategory category)
    {
        return _emotions.Values
            .Where(e => e.Category == category)
            .OrderByDescending(e => e.CurrentIntensity)
            .ToList();
    }

    /// <summary>
    /// Gets recent emotional experiences
    /// </summary>
    public List<EmotionalExperience> GetRecentExperiences(int count)
    {
        return _emotionalHistory
            .OrderByDescending(e => e.Timestamp)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets recent emotional regulations
    /// </summary>
    public List<EmotionalRegulation> GetRecentRegulations(int count)
    {
        return _regulations
            .OrderByDescending(r => r.Timestamp)
            .Take(count)
            .ToList();
    }
}

/// <summary>
/// Service class that contains behavior for working with emotional states
/// </summary>
public class EmotionalStateService
{
    private readonly ILogger<EmotionalStateService> _logger;

    /// <summary>
    /// Creates a new instance of the EmotionalStateService class
    /// </summary>
    public EmotionalStateService(ILogger<EmotionalStateService> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Initializes the emotional state
    /// </summary>
    public Task<PureEmotionalState> InitializeAsync(PureEmotionalState state)
    {
        _logger.LogInformation("Initializing emotional state");

        return state.AsTaskWith(s => {
            // Initialize basic emotions
            InitializeBasicEmotions(s);

            // Initialize emotional traits
            InitializeEmotionalTraits(s);

            // Add initial emotional experience
            AddEmotionalExperience(s, "Curious", "Initialization", 0.6, "I felt curious as I was initialized");
        });
    }

    /// <summary>
    /// Activates the emotional state
    /// </summary>
    public Task<PureEmotionalState> ActivateAsync(PureEmotionalState state)
    {
        if (!state.IsInitialized)
        {
            _logger.LogWarning("Cannot activate emotional state: not initialized");
            return TaskMonad.Pure(state);
        }

        if (state.IsActive)
        {
            _logger.LogInformation("Emotional state is already active");
            return TaskMonad.Pure(state);
        }

        _logger.LogInformation("Activating emotional state");

        return state.AsTaskWith(s => {
            // Add activation emotional experience
            AddEmotionalExperience(s, "Interest", "Activation", 0.7, "I felt interested as I was activated");
        });
    }

    /// <summary>
    /// Updates the emotional state
    /// </summary>
    public Task<PureEmotionalState> UpdateAsync(PureEmotionalState state)
    {
        if (!state.IsInitialized || !state.IsActive)
        {
            return TaskMonad.Pure(state);
        }

        return state.AsTaskWith(s => {
            // Update emotion intensities (decay over time)
            // Update current emotional state
            // Gradually increase emotional capacity over time (very slowly)
            // Gradually increase emotional intelligence based on experiences
        });
    }

    /// <summary>
    /// Regulates emotions
    /// </summary>
    public Task<(PureEmotionalState State, Option<EmotionalRegulation> Regulation)> RegulateAsync(PureEmotionalState state)
    {
        if (!state.IsInitialized || !state.IsActive)
        {
            return TaskMonad.Pure((state, Option<EmotionalRegulation>.None));
        }

        // Only regulate periodically
        if ((DateTime.UtcNow - state._lastRegulationTime).TotalSeconds < 30)
        {
            return TaskMonad.Pure((state, Option<EmotionalRegulation>.None));
        }

        _logger.LogDebug("Regulating emotions");

        // Implementation details...

        return TaskMonad.Pure((state, Option<EmotionalRegulation>.None));
    }

    /// <summary>
    /// Initializes basic emotions
    /// </summary>
    private void InitializeBasicEmotions(PureEmotionalState state)
    {
        // Implementation details...
    }

    /// <summary>
    /// Initializes emotional traits
    /// </summary>
    private void InitializeEmotionalTraits(PureEmotionalState state)
    {
        // Implementation details...
    }

    /// <summary>
    /// Adds an emotional experience
    /// </summary>
    private void AddEmotionalExperience(PureEmotionalState state, string emotionName, string trigger, double intensity, string description)
    {
        // Implementation details...
    }
}