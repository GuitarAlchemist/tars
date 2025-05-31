namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents an intuition
/// </summary>
public class Intuition
{
    /// <summary>
    /// Gets or sets the intuition ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the intuition description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the intuition type
    /// </summary>
    public IntuitionType Type { get; set; } = IntuitionType.GutFeeling;
    
    /// <summary>
    /// Gets or sets the intuition confidence (0.0 to 1.0)
    /// </summary>
    public double Confidence { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the intuition timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the intuition context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the intuition tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the intuition source
    /// </summary>
    public string Source { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the intuition verification status
    /// </summary>
    public VerificationStatus VerificationStatus { get; set; } = VerificationStatus.Unverified;
    
    /// <summary>
    /// Gets or sets the intuition verification timestamp
    /// </summary>
    public DateTime? VerificationTimestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the intuition verification notes
    /// </summary>
    public string VerificationNotes { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the intuition accuracy (0.0 to 1.0)
    /// </summary>
    public double? Accuracy { get; set; }
    
    /// <summary>
    /// Gets or sets the intuition impact (0.0 to 1.0)
    /// </summary>
    public double Impact { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the intuition explanation
    /// </summary>
    public string Explanation { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the intuition decision
    /// </summary>
    public string Decision { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the intuition selected option
    /// </summary>
    public string SelectedOption { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the intuition options
    /// </summary>
    public List<string> Options { get; set; } = new();
    
    /// <summary>
    /// Verifies the intuition
    /// </summary>
    /// <param name="isCorrect">Whether the intuition is correct</param>
    /// <param name="accuracy">The accuracy</param>
    /// <param name="notes">The verification notes</param>
    public void Verify(bool isCorrect, double accuracy, string notes)
    {
        VerificationStatus = isCorrect ? VerificationStatus.Verified : VerificationStatus.Falsified;
        Accuracy = accuracy;
        VerificationNotes = notes;
        VerificationTimestamp = DateTime.UtcNow;
    }
    
    /// <summary>
    /// Adds an explanation for the intuition
    /// </summary>
    /// <param name="explanation">The explanation</param>
    public void AddExplanation(string explanation)
    {
        Explanation = explanation;
    }
    
    /// <summary>
    /// Sets the decision context
    /// </summary>
    /// <param name="decision">The decision</param>
    /// <param name="selectedOption">The selected option</param>
    /// <param name="options">The options</param>
    public void SetDecisionContext(string decision, string selectedOption, List<string> options)
    {
        Decision = decision;
        SelectedOption = selectedOption;
        Options = options;
    }
}
