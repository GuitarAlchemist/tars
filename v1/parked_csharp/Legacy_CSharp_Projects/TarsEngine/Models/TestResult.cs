namespace TarsEngine.Models;

/// <summary>
/// Represents the result of running a test
/// </summary>
public class TestResult
{
    /// <summary>
    /// The name of the component being tested
    /// </summary>
    public string ComponentName { get; set; } = string.Empty;
    
    /// <summary>
    /// The path to the test file
    /// </summary>
    public string TestFilePath { get; set; } = string.Empty;
    
    /// <summary>
    /// Whether the test was successful
    /// </summary>
    public bool Success { get; set; } = true;
    
    /// <summary>
    /// The error message if the test failed
    /// </summary>
    public string ErrorMessage { get; set; } = string.Empty;
}
