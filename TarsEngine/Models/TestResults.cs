namespace TarsEngine.Models;

/// <summary>
/// Represents the results of running tests
/// </summary>
public class TestResults
{
    /// <summary>
    /// The list of test results
    /// </summary>
    public List<TestResult> Tests { get; set; } = new List<TestResult>();
    
    /// <summary>
    /// Whether the tests were successful
    /// </summary>
    public bool Success { get; set; } = true;
    
    /// <summary>
    /// The error message if the tests failed
    /// </summary>
    public string ErrorMessage { get; set; } = string.Empty;
    
    /// <summary>
    /// The start time of the tests
    /// </summary>
    public DateTime StartTime { get; set; }
    
    /// <summary>
    /// The end time of the tests
    /// </summary>
    public DateTime EndTime { get; set; }
    
    /// <summary>
    /// The duration of the tests
    /// </summary>
    public TimeSpan Duration { get; set; }
}
