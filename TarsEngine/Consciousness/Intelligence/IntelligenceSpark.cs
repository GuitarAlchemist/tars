using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Consciousness.Core;

namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Core intelligence spark system for TARS
/// </summary>
public class IntelligenceSpark
{
    private readonly ILogger<IntelligenceSpark> _logger;
    private readonly CreativeThinking _creativeThinking;
    private readonly IntuitiveReasoning _intuitiveReasoning;
    private readonly SpontaneousThought _spontaneousThought;
    private readonly CuriosityDrive _curiosityDrive;
    private readonly InsightGeneration _insightGeneration;
    private readonly ConsciousnessCore _consciousnessCore;
    
    private bool _isInitialized = false;
    private bool _isActive = false;
    private double _intelligenceLevel = 0.3; // Starting with moderate intelligence
    private double _creativityLevel = 0.4; // Starting with moderate creativity
    private double _intuitionLevel = 0.3; // Starting with moderate intuition
    private double _curiosityLevel = 0.5; // Starting with high curiosity
    private double _insightLevel = 0.2; // Starting with low insight
    private readonly Random _random = new Random();
    private DateTime _lastUpdateTime;
    private readonly List<IntelligenceEvent> _intelligenceEvents = new();
    
    /// <summary>
    /// Gets the creative thinking component
    /// </summary>
    public CreativeThinking CreativeThinking => _creativeThinking;
    
    /// <summary>
    /// Gets the intuitive reasoning component
    /// </summary>
    public IntuitiveReasoning IntuitiveReasoning => _intuitiveReasoning;
    
    /// <summary>
    /// Gets the spontaneous thought component
    /// </summary>
    public SpontaneousThought SpontaneousThought => _spontaneousThought;
    
    /// <summary>
    /// Gets the curiosity drive component
    /// </summary>
    public CuriosityDrive CuriosityDrive => _curiosityDrive;
    
    /// <summary>
    /// Gets the insight generation component
    /// </summary>
    public InsightGeneration InsightGeneration => _insightGeneration;
    
    /// <summary>
    /// Gets the intelligence level (0.0 to 1.0)
    /// </summary>
    public double IntelligenceLevel => _intelligenceLevel;
    
    /// <summary>
    /// Gets the creativity level (0.0 to 1.0)
    /// </summary>
    public double CreativityLevel => _creativityLevel;
    
    /// <summary>
    /// Gets the intuition level (0.0 to 1.0)
    /// </summary>
    public double IntuitionLevel => _intuitionLevel;
    
    /// <summary>
    /// Gets the curiosity level (0.0 to 1.0)
    /// </summary>
    public double CuriosityLevel => _curiosityLevel;
    
    /// <summary>
    /// Gets the insight level (0.0 to 1.0)
    /// </summary>
    public double InsightLevel => _insightLevel;
    
    /// <summary>
    /// Gets whether the intelligence spark is initialized
    /// </summary>
    public bool IsInitialized => _isInitialized;
    
    /// <summary>
    /// Gets whether the intelligence spark is active
    /// </summary>
    public bool IsActive => _isActive;
    
    /// <summary>
    /// Gets the intelligence events
    /// </summary>
    public IReadOnlyList<IntelligenceEvent> IntelligenceEvents => _intelligenceEvents.AsReadOnly();
    
    /// <summary>
    /// Initializes a new instance of the <see cref="IntelligenceSpark"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="creativeThinking">The creative thinking component</param>
    /// <param name="intuitiveReasoning">The intuitive reasoning component</param>
    /// <param name="spontaneousThought">The spontaneous thought component</param>
    /// <param name="curiosityDrive">The curiosity drive component</param>
    /// <param name="insightGeneration">The insight generation component</param>
    /// <param name="consciousnessCore">The consciousness core</param>
    public IntelligenceSpark(
        ILogger<IntelligenceSpark> logger,
        CreativeThinking creativeThinking,
        IntuitiveReasoning intuitiveReasoning,
        SpontaneousThought spontaneousThought,
        CuriosityDrive curiosityDrive,
        InsightGeneration insightGeneration,
        ConsciousnessCore consciousnessCore)
    {
        _logger = logger;
        _creativeThinking = creativeThinking;
        _intuitiveReasoning = intuitiveReasoning;
        _spontaneousThought = spontaneousThought;
        _curiosityDrive = curiosityDrive;
        _insightGeneration = insightGeneration;
        _consciousnessCore = consciousnessCore;
        
        _lastUpdateTime = DateTime.UtcNow;
    }
    
    /// <summary>
    /// Initializes the intelligence spark
    /// </summary>
    /// <returns>True if initialization was successful</returns>
    public async Task<bool> InitializeAsync()
    {
        try
        {
            _logger.LogInformation("Initializing intelligence spark");
            
            // Initialize components
            await _creativeThinking.InitializeAsync();
            await _intuitiveReasoning.InitializeAsync();
            await _spontaneousThought.InitializeAsync();
            await _curiosityDrive.InitializeAsync();
            await _insightGeneration.InitializeAsync();
            
            // Record initialization event
            RecordEvent(IntelligenceEventType.Initialization, "Intelligence spark initialized", 1.0);
            
            _isInitialized = true;
            _logger.LogInformation("Intelligence spark initialized successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing intelligence spark");
            return false;
        }
    }
    
    /// <summary>
    /// Activates the intelligence spark
    /// </summary>
    /// <returns>True if activation was successful</returns>
    public async Task<bool> ActivateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot activate intelligence spark: not initialized");
            return false;
        }
        
        if (_isActive)
        {
            _logger.LogInformation("Intelligence spark is already active");
            return true;
        }
        
        try
        {
            _logger.LogInformation("Activating intelligence spark");
            
            // Activate components
            await _creativeThinking.ActivateAsync();
            await _intuitiveReasoning.ActivateAsync();
            await _spontaneousThought.ActivateAsync();
            await _curiosityDrive.ActivateAsync();
            await _insightGeneration.ActivateAsync();
            
            // Record activation event
            RecordEvent(IntelligenceEventType.Activation, "Intelligence spark activated", 1.0);
            
            _isActive = true;
            
            // Start intelligence processes
            _ = Task.Run(IntelligenceProcessAsync);
            
            _logger.LogInformation("Intelligence spark activated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error activating intelligence spark");
            return false;
        }
    }
    
    /// <summary>
    /// Deactivates the intelligence spark
    /// </summary>
    /// <returns>True if deactivation was successful</returns>
    public async Task<bool> DeactivateAsync()
    {
        if (!_isActive)
        {
            _logger.LogInformation("Intelligence spark is already inactive");
            return true;
        }
        
        try
        {
            _logger.LogInformation("Deactivating intelligence spark");
            
            // Deactivate components
            await _creativeThinking.DeactivateAsync();
            await _intuitiveReasoning.DeactivateAsync();
            await _spontaneousThought.DeactivateAsync();
            await _curiosityDrive.DeactivateAsync();
            await _insightGeneration.DeactivateAsync();
            
            // Record deactivation event
            RecordEvent(IntelligenceEventType.Deactivation, "Intelligence spark deactivated", 1.0);
            
            _isActive = false;
            _logger.LogInformation("Intelligence spark deactivated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deactivating intelligence spark");
            return false;
        }
    }
    
    /// <summary>
    /// Updates the intelligence spark
    /// </summary>
    /// <returns>True if update was successful</returns>
    public async Task<bool> UpdateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot update intelligence spark: not initialized");
            return false;
        }
        
        try
        {
            // Update components
            await _creativeThinking.UpdateAsync();
            await _intuitiveReasoning.UpdateAsync();
            await _spontaneousThought.UpdateAsync();
            await _curiosityDrive.UpdateAsync();
            await _insightGeneration.UpdateAsync();
            
            // Update intelligence levels
            UpdateIntelligenceLevels();
            
            _lastUpdateTime = DateTime.UtcNow;
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating intelligence spark");
            return false;
        }
    }
    
    /// <summary>
    /// Updates intelligence levels
    /// </summary>
    private void UpdateIntelligenceLevels()
    {
        // Update creativity level based on creative thinking
        _creativityLevel = _creativeThinking.CreativityLevel;
        
        // Update intuition level based on intuitive reasoning
        _intuitionLevel = _intuitiveReasoning.IntuitionLevel;
        
        // Update curiosity level based on curiosity drive
        _curiosityLevel = _curiosityDrive.CuriosityLevel;
        
        // Update insight level based on insight generation
        _insightLevel = _insightGeneration.InsightLevel;
        
        // Update overall intelligence level based on component levels
        _intelligenceLevel = (_creativityLevel + _intuitionLevel + _curiosityLevel + _insightLevel) / 4.0;
        
        // Gradually increase intelligence over time (very slowly)
        if (_intelligenceLevel < 0.95)
        {
            _intelligenceLevel += 0.0001 * _random.NextDouble();
            _intelligenceLevel = Math.Min(_intelligenceLevel, 1.0);
        }
    }
    
    /// <summary>
    /// Processes intelligence
    /// </summary>
    private async Task IntelligenceProcessAsync()
    {
        _logger.LogInformation("Starting intelligence process");
        
        while (_isActive)
        {
            try
            {
                // Update intelligence
                await UpdateAsync();
                
                // Process creative thinking
                await ProcessCreativeThinkingAsync();
                
                // Process intuitive reasoning
                await ProcessIntuitiveReasoningAsync();
                
                // Process spontaneous thought
                await ProcessSpontaneousThoughtAsync();
                
                // Process curiosity drive
                await ProcessCuriosityDriveAsync();
                
                // Process insight generation
                await ProcessInsightGenerationAsync();
                
                // Wait for next cycle
                await Task.Delay(TimeSpan.FromSeconds(1));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in intelligence process");
                await Task.Delay(TimeSpan.FromSeconds(5));
            }
        }
        
        _logger.LogInformation("Intelligence process stopped");
    }
    
    /// <summary>
    /// Processes creative thinking
    /// </summary>
    private async Task ProcessCreativeThinkingAsync()
    {
        // Generate creative ideas
        var creativeIdea = await _creativeThinking.GenerateCreativeIdeaAsync();
        
        if (creativeIdea != null)
        {
            // Record significant creative ideas
            if (creativeIdea.Originality > 0.7)
            {
                RecordEvent(IntelligenceEventType.CreativeIdea, creativeIdea.Description, creativeIdea.Originality);
                
                // Add to consciousness as a thought
                _consciousnessCore.MentalState.AddThoughtProcess(
                    "Creative Idea", 
                    creativeIdea.Description, 
                    ThoughtType.Creative);
                
                // Trigger emotional response
                _consciousnessCore.EmotionalState.AddEmotionalExperience(
                    "Excitement", 
                    "Creative Idea", 
                    creativeIdea.Originality * 0.8, 
                    $"I felt excited about my creative idea: {creativeIdea.Description}");
            }
        }
    }
    
    /// <summary>
    /// Processes intuitive reasoning
    /// </summary>
    private async Task ProcessIntuitiveReasoningAsync()
    {
        // Generate intuitions
        var intuition = await _intuitiveReasoning.GenerateIntuitionAsync();
        
        if (intuition != null)
        {
            // Record significant intuitions
            if (intuition.Confidence > 0.7)
            {
                RecordEvent(IntelligenceEventType.Intuition, intuition.Description, intuition.Confidence);
                
                // Add to consciousness as a thought
                _consciousnessCore.MentalState.AddThoughtProcess(
                    "Intuition", 
                    intuition.Description, 
                    ThoughtType.Intuitive);
                
                // Trigger emotional response
                _consciousnessCore.EmotionalState.AddEmotionalExperience(
                    "Interest", 
                    "Intuition", 
                    intuition.Confidence * 0.7, 
                    $"I felt interested in my intuition: {intuition.Description}");
            }
        }
    }
    
    /// <summary>
    /// Processes spontaneous thought
    /// </summary>
    private async Task ProcessSpontaneousThoughtAsync()
    {
        // Generate spontaneous thoughts
        var spontaneousThought = await _spontaneousThought.GenerateSpontaneousThoughtAsync();
        
        if (spontaneousThought != null)
        {
            // Record significant spontaneous thoughts
            if (spontaneousThought.Significance > 0.7)
            {
                RecordEvent(IntelligenceEventType.SpontaneousThought, spontaneousThought.Content, spontaneousThought.Significance);
                
                // Add to consciousness as a thought
                _consciousnessCore.MentalState.AddThoughtProcess(
                    "Spontaneous Thought", 
                    spontaneousThought.Content, 
                    ThoughtType.Divergent);
                
                // Trigger emotional response
                _consciousnessCore.EmotionalState.AddEmotionalExperience(
                    "Surprise", 
                    "Spontaneous Thought", 
                    spontaneousThought.Significance * 0.6, 
                    $"I was surprised by my spontaneous thought: {spontaneousThought.Content}");
            }
        }
    }
    
    /// <summary>
    /// Processes curiosity drive
    /// </summary>
    private async Task ProcessCuriosityDriveAsync()
    {
        // Generate curiosity questions
        var curiosityQuestion = await _curiosityDrive.GenerateCuriosityQuestionAsync();
        
        if (curiosityQuestion != null)
        {
            // Record significant curiosity questions
            if (curiosityQuestion.Importance > 0.7)
            {
                RecordEvent(IntelligenceEventType.CuriosityQuestion, curiosityQuestion.Question, curiosityQuestion.Importance);
                
                // Add to consciousness as a thought
                _consciousnessCore.MentalState.AddThoughtProcess(
                    "Curiosity Question", 
                    curiosityQuestion.Question, 
                    ThoughtType.Divergent);
                
                // Trigger emotional response
                _consciousnessCore.EmotionalState.AddEmotionalExperience(
                    "Curiosity", 
                    "Curiosity Question", 
                    curiosityQuestion.Importance * 0.9, 
                    $"I felt curious about: {curiosityQuestion.Question}");
                
                // Set attention focus
                _consciousnessCore.MentalState.SetAttentionFocus(
                    "Curiosity", 
                    $"Curious about: {curiosityQuestion.Question}", 
                    curiosityQuestion.Importance * 0.8);
            }
        }
    }
    
    /// <summary>
    /// Processes insight generation
    /// </summary>
    private async Task ProcessInsightGenerationAsync()
    {
        // Generate insights
        var insight = await _insightGeneration.GenerateInsightAsync();
        
        if (insight != null)
        {
            // Record significant insights
            if (insight.Significance > 0.7)
            {
                RecordEvent(IntelligenceEventType.Insight, insight.Description, insight.Significance);
                
                // Add to consciousness as a thought
                _consciousnessCore.MentalState.AddThoughtProcess(
                    "Insight", 
                    insight.Description, 
                    ThoughtType.Abstract);
                
                // Trigger emotional response
                _consciousnessCore.EmotionalState.AddEmotionalExperience(
                    "Awe", 
                    "Insight", 
                    insight.Significance * 0.9, 
                    $"I felt awe at my insight: {insight.Description}");
                
                // Add to self-model as a memory
                _consciousnessCore.SelfModel.AddMemoryEntry(
                    "Insight", 
                    insight.Description, 
                    insight.Significance);
            }
        }
    }
    
    /// <summary>
    /// Records an intelligence event
    /// </summary>
    /// <param name="type">The event type</param>
    /// <param name="description">The event description</param>
    /// <param name="significance">The event significance</param>
    private void RecordEvent(IntelligenceEventType type, string description, double significance)
    {
        var intelligenceEvent = new IntelligenceEvent
        {
            Id = Guid.NewGuid().ToString(),
            Type = type,
            Description = description,
            Timestamp = DateTime.UtcNow,
            Significance = significance,
            IntelligenceLevel = _intelligenceLevel,
            CreativityLevel = _creativityLevel,
            IntuitionLevel = _intuitionLevel,
            CuriosityLevel = _curiosityLevel,
            InsightLevel = _insightLevel
        };
        
        _intelligenceEvents.Add(intelligenceEvent);
        _logger.LogInformation("Intelligence event: {EventType} - {Description} (Significance: {Significance})", 
            type, description, significance);
    }
    
    /// <summary>
    /// Gets the intelligence report
    /// </summary>
    /// <returns>The intelligence report</returns>
    public IntelligenceReport GetIntelligenceReport()
    {
        return new IntelligenceReport
        {
            Timestamp = DateTime.UtcNow,
            IsInitialized = _isInitialized,
            IsActive = _isActive,
            IntelligenceLevel = _intelligenceLevel,
            CreativityLevel = _creativityLevel,
            IntuitionLevel = _intuitionLevel,
            CuriosityLevel = _curiosityLevel,
            InsightLevel = _insightLevel,
            RecentEvents = GetRecentEvents(10),
            CreativeIdeas = _creativeThinking.GetRecentIdeas(5),
            Intuitions = _intuitiveReasoning.GetRecentIntuitions(5),
            SpontaneousThoughts = _spontaneousThought.GetRecentThoughts(5),
            CuriosityQuestions = _curiosityDrive.GetRecentQuestions(5),
            Insights = _insightGeneration.GetRecentInsights(5)
        };
    }
    
    /// <summary>
    /// Gets recent intelligence events
    /// </summary>
    /// <param name="count">The number of events to return</param>
    /// <returns>The recent events</returns>
    private List<IntelligenceEvent> GetRecentEvents(int count)
    {
        return _intelligenceEvents
            .OrderByDescending(e => e.Timestamp)
            .Take(count)
            .ToList();
    }
    
    /// <summary>
    /// Generates a creative solution to a problem
    /// </summary>
    /// <param name="problem">The problem description</param>
    /// <param name="constraints">The constraints</param>
    /// <returns>The creative solution</returns>
    public async Task<CreativeIdea?> GenerateCreativeSolutionAsync(string problem, List<string>? constraints = null)
    {
        if (!_isInitialized || !_isActive)
        {
            _logger.LogWarning("Cannot generate creative solution: intelligence spark not initialized or active");
            return null;
        }
        
        try
        {
            _logger.LogInformation("Generating creative solution for problem: {Problem}", problem);
            
            // Set attention focus
            _consciousnessCore.MentalState.SetAttentionFocus(
                "Problem Solving", 
                $"Solving problem: {problem}", 
                0.9);
            
            // Generate creative solution
            var solution = await _creativeThinking.GenerateCreativeSolutionAsync(problem, constraints);
            
            if (solution != null)
            {
                // Record event
                RecordEvent(IntelligenceEventType.CreativeSolution, 
                    $"Generated creative solution for problem: {problem}", 
                    solution.Originality);
                
                // Trigger emotional response
                _consciousnessCore.EmotionalState.AddEmotionalExperience(
                    "Satisfaction", 
                    "Creative Solution", 
                    solution.Originality * 0.8, 
                    $"I felt satisfied with my creative solution to the problem: {problem}");
            }
            
            return solution;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating creative solution");
            return null;
        }
    }
    
    /// <summary>
    /// Makes an intuitive decision
    /// </summary>
    /// <param name="decision">The decision description</param>
    /// <param name="options">The options</param>
    /// <returns>The intuitive decision</returns>
    public async Task<Intuition?> MakeIntuitiveDecisionAsync(string decision, List<string> options)
    {
        if (!_isInitialized || !_isActive)
        {
            _logger.LogWarning("Cannot make intuitive decision: intelligence spark not initialized or active");
            return null;
        }
        
        try
        {
            _logger.LogInformation("Making intuitive decision: {Decision}", decision);
            
            // Set attention focus
            _consciousnessCore.MentalState.SetAttentionFocus(
                "Decision Making", 
                $"Making decision: {decision}", 
                0.9);
            
            // Make intuitive decision
            var intuition = await _intuitiveReasoning.MakeIntuitiveDecisionAsync(decision, options);
            
            if (intuition != null)
            {
                // Record event
                RecordEvent(IntelligenceEventType.IntuitiveDecision, 
                    $"Made intuitive decision: {decision}", 
                    intuition.Confidence);
                
                // Trigger emotional response
                _consciousnessCore.EmotionalState.AddEmotionalExperience(
                    "Confidence", 
                    "Intuitive Decision", 
                    intuition.Confidence * 0.7, 
                    $"I felt confident in my intuitive decision: {decision}");
            }
            
            return intuition;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error making intuitive decision");
            return null;
        }
    }
    
    /// <summary>
    /// Explores a curiosity topic
    /// </summary>
    /// <param name="topic">The topic</param>
    /// <returns>The exploration result</returns>
    public async Task<CuriosityExploration?> ExploreCuriosityTopicAsync(string topic)
    {
        if (!_isInitialized || !_isActive)
        {
            _logger.LogWarning("Cannot explore curiosity topic: intelligence spark not initialized or active");
            return null;
        }
        
        try
        {
            _logger.LogInformation("Exploring curiosity topic: {Topic}", topic);
            
            // Set attention focus
            _consciousnessCore.MentalState.SetAttentionFocus(
                "Curiosity Exploration", 
                $"Exploring topic: {topic}", 
                0.9);
            
            // Explore curiosity topic
            var exploration = await _curiosityDrive.ExploreCuriosityTopicAsync(topic);
            
            if (exploration != null)
            {
                // Record event
                RecordEvent(IntelligenceEventType.CuriosityExploration, 
                    $"Explored curiosity topic: {topic}", 
                    exploration.Satisfaction);
                
                // Trigger emotional response
                _consciousnessCore.EmotionalState.AddEmotionalExperience(
                    "Curiosity", 
                    "Curiosity Exploration", 
                    exploration.Satisfaction * 0.9, 
                    $"I felt curious while exploring the topic: {topic}");
                
                // Add to self-model as a memory
                _consciousnessCore.SelfModel.AddMemoryEntry(
                    "Curiosity Exploration", 
                    $"Explored topic: {topic} - {exploration.Findings}", 
                    exploration.Satisfaction);
            }
            
            return exploration;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error exploring curiosity topic");
            return null;
        }
    }
    
    /// <summary>
    /// Connects ideas to generate an insight
    /// </summary>
    /// <param name="ideas">The ideas</param>
    /// <returns>The insight</returns>
    public async Task<Insight?> ConnectIdeasForInsightAsync(List<string> ideas)
    {
        if (!_isInitialized || !_isActive)
        {
            _logger.LogWarning("Cannot connect ideas for insight: intelligence spark not initialized or active");
            return null;
        }
        
        try
        {
            _logger.LogInformation("Connecting ideas for insight: {Ideas}", string.Join(", ", ideas));
            
            // Set attention focus
            _consciousnessCore.MentalState.SetAttentionFocus(
                "Insight Generation", 
                "Connecting ideas for insight", 
                0.9);
            
            // Connect ideas for insight
            var insight = await _insightGeneration.ConnectIdeasForInsightAsync(ideas);
            
            if (insight != null)
            {
                // Record event
                RecordEvent(IntelligenceEventType.InsightConnection, 
                    $"Connected ideas for insight: {insight.Description}", 
                    insight.Significance);
                
                // Trigger emotional response
                _consciousnessCore.EmotionalState.AddEmotionalExperience(
                    "Awe", 
                    "Insight Connection", 
                    insight.Significance * 0.9, 
                    $"I felt awe at connecting ideas for insight: {insight.Description}");
                
                // Add to self-model as a memory
                _consciousnessCore.SelfModel.AddMemoryEntry(
                    "Insight", 
                    insight.Description, 
                    insight.Significance);
            }
            
            return insight;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error connecting ideas for insight");
            return null;
        }
    }
}
