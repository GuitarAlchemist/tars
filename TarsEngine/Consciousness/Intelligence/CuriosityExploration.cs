using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents a curiosity exploration
/// </summary>
public class CuriosityExploration
{
    /// <summary>
    /// Gets or sets the exploration ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the topic
    /// </summary>
    public string Topic { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the exploration strategy
    /// </summary>
    public ExplorationStrategy Strategy { get; set; } = ExplorationStrategy.DeepDive;
    
    /// <summary>
    /// Gets or sets the approach
    /// </summary>
    public string Approach { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the findings
    /// </summary>
    public string Findings { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the insights
    /// </summary>
    public List<string> Insights { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the follow-up questions
    /// </summary>
    public List<string> FollowUpQuestions { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the satisfaction (0.0 to 1.0)
    /// </summary>
    public double Satisfaction { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the exploration context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new Dictionary<string, object>();
    
    /// <summary>
    /// Gets or sets the exploration tags
    /// </summary>
    public List<string> Tags { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the question ID
    /// </summary>
    public string? QuestionId { get; set; }
    
    /// <summary>
    /// Gets or sets the related exploration IDs
    /// </summary>
    public List<string> RelatedExplorationIds { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the exploration duration in seconds
    /// </summary>
    public double DurationSeconds { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the exploration resources
    /// </summary>
    public List<string> Resources { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the exploration challenges
    /// </summary>
    public List<string> Challenges { get; set; } = new List<string>();
    
    /// <summary>
    /// Gets or sets the exploration learning
    /// </summary>
    public string Learning { get; set; } = string.Empty;
    
    /// <summary>
    /// Adds an insight
    /// </summary>
    /// <param name="insight">The insight</param>
    public void AddInsight(string insight)
    {
        Insights.Add(insight);
    }
    
    /// <summary>
    /// Adds a follow-up question
    /// </summary>
    /// <param name="followUpQuestion">The follow-up question</param>
    public void AddFollowUpQuestion(string followUpQuestion)
    {
        FollowUpQuestions.Add(followUpQuestion);
    }
    
    /// <summary>
    /// Adds a related exploration
    /// </summary>
    /// <param name="explorationId">The exploration ID</param>
    public void AddRelatedExploration(string explorationId)
    {
        if (!RelatedExplorationIds.Contains(explorationId))
        {
            RelatedExplorationIds.Add(explorationId);
        }
    }
    
    /// <summary>
    /// Adds a resource
    /// </summary>
    /// <param name="resource">The resource</param>
    public void AddResource(string resource)
    {
        Resources.Add(resource);
    }
    
    /// <summary>
    /// Adds a challenge
    /// </summary>
    /// <param name="challenge">The challenge</param>
    public void AddChallenge(string challenge)
    {
        Challenges.Add(challenge);
    }
    
    /// <summary>
    /// Sets the question
    /// </summary>
    /// <param name="questionId">The question ID</param>
    public void SetQuestion(string questionId)
    {
        QuestionId = questionId;
    }
}
