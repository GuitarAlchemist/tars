namespace TarsCli.Services.Testing;

/// <summary>
/// Interface for a repository of test patterns
/// </summary>
public interface ITestPatternRepository
{
    /// <summary>
    /// Gets a test pattern for a type and method name
    /// </summary>
    /// <param name="type">Type</param>
    /// <param name="methodName">Method name</param>
    /// <returns>Test pattern or null if not found</returns>
    TestPattern GetPattern(string type, string methodName);

    /// <summary>
    /// Saves a test pattern
    /// </summary>
    /// <param name="pattern">Test pattern</param>
    /// <returns>Task representing the asynchronous operation</returns>
    Task SavePatternAsync(TestPattern pattern);

    /// <summary>
    /// Learns from a successful test
    /// </summary>
    /// <param name="test">Test result</param>
    /// <returns>Task representing the asynchronous operation</returns>
    Task LearnFromSuccessfulTestAsync(TestResult test);
}
