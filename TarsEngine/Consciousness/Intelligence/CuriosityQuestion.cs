namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents a curiosity question
/// </summary>
public class CuriosityQuestion
{
    /// <summary>
    /// Gets or sets the question ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the question
    /// </summary>
    public string Question { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the domain
    /// </summary>
    public string Domain { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the question generation method
    /// </summary>
    public QuestionGenerationMethod Method { get; set; } = QuestionGenerationMethod.InformationGap;
    
    /// <summary>
    /// Gets or sets the question importance (0.0 to 1.0)
    /// </summary>
    public double Importance { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the question timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the question context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the question tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the question status
    /// </summary>
    public QuestionStatus Status { get; set; } = QuestionStatus.Open;
    
    /// <summary>
    /// Gets or sets the answer
    /// </summary>
    public string Answer { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the answer timestamp
    /// </summary>
    public DateTime? AnswerTimestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the answer satisfaction (0.0 to 1.0)
    /// </summary>
    public double AnswerSatisfaction { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the exploration ID
    /// </summary>
    public string? ExplorationId { get; set; }
    
    /// <summary>
    /// Gets or sets the follow-up questions
    /// </summary>
    public List<string> FollowUpQuestions { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the related question IDs
    /// </summary>
    public List<string> RelatedQuestionIds { get; set; } = new();
    
    /// <summary>
    /// Records an answer to the question
    /// </summary>
    /// <param name="answer">The answer</param>
    /// <param name="satisfaction">The satisfaction</param>
    public void RecordAnswer(string answer, double satisfaction)
    {
        Answer = answer;
        AnswerSatisfaction = satisfaction;
        AnswerTimestamp = DateTime.UtcNow;
        Status = QuestionStatus.Answered;
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
    /// Adds a related question
    /// </summary>
    /// <param name="questionId">The question ID</param>
    public void AddRelatedQuestion(string questionId)
    {
        if (!RelatedQuestionIds.Contains(questionId))
        {
            RelatedQuestionIds.Add(questionId);
        }
    }
    
    /// <summary>
    /// Sets the exploration
    /// </summary>
    /// <param name="explorationId">The exploration ID</param>
    public void SetExploration(string explorationId)
    {
        ExplorationId = explorationId;
        Status = QuestionStatus.Explored;
    }
}
