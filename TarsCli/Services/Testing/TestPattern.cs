namespace TarsCli.Services.Testing;

/// <summary>
/// Represents a test pattern
/// </summary>
public class TestPattern
{
    /// <summary>
    /// Gets or sets the type
    /// </summary>
    public string Type { get; set; }

    /// <summary>
    /// Gets or sets the method name
    /// </summary>
    public string MethodName { get; set; }

    /// <summary>
    /// Gets or sets the test data template
    /// </summary>
    public string TestDataTemplate { get; set; }

    /// <summary>
    /// Gets or sets the assertion template
    /// </summary>
    public string AssertionTemplate { get; set; }

    /// <summary>
    /// Gets or sets the success count
    /// </summary>
    public int SuccessCount { get; set; }

    /// <summary>
    /// Gets or sets the failure count
    /// </summary>
    public int FailureCount { get; set; }

    /// <summary>
    /// Generates test data for a parameter
    /// </summary>
    /// <param name="parameterName">Parameter name</param>
    /// <returns>Test data code</returns>
    public string GenerateTestData(string parameterName)
    {
        return TestDataTemplate.Replace("{name}", parameterName);
    }

    /// <summary>
    /// Generates assertions
    /// </summary>
    /// <returns>Assertion code</returns>
    public string GenerateAssertions()
    {
        return AssertionTemplate;
    }
}
