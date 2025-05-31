namespace TarsEngine.Models;

/// <summary>
/// Represents the result of a test generation operation
/// </summary>
public class TestGenerationResult
{
    /// <summary>
    /// Whether the test generation was successful
    /// </summary>
    public bool Success { get; set; } = true;
    
    /// <summary>
    /// The generated tests
    /// </summary>
    public string GeneratedTests { get; set; } = string.Empty;
    
    /// <summary>
    /// The error message if the test generation failed
    /// </summary>
    public string ErrorMessage { get; set; } = string.Empty;
    
    /// <summary>
    /// The file path where the tests were saved
    /// </summary>
    public string TestFilePath { get; set; } = string.Empty;
}
