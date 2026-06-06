namespace TarsEngine.Services.Adapters;

/// <summary>
/// Adapter for converting between TarsEngine.Services.TestExecutionResult and TarsEngine.Models.TestExecutionResult
/// </summary>
public static class TestExecutionResultAdapter
{
    /// <summary>
    /// Converts from TarsEngine.Services.TestExecutionResult to TarsEngine.Models.TestExecutionResult
    /// </summary>
    /// <param name="serviceResult">The service result to convert</param>
    /// <returns>The model result</returns>
    public static TarsEngine.Models.TestExecutionResult ToModel(TestExecutionResult serviceResult)
    {
        if (serviceResult == null)
            throw new ArgumentNullException(nameof(serviceResult));

        return new TarsEngine.Models.TestExecutionResult
        {
            IsSuccessful = serviceResult.IsSuccessful,
            Output = serviceResult.Output ?? string.Empty,
            ErrorMessage = serviceResult.ErrorMessage ?? string.Empty,
            TotalTests = serviceResult.TotalTests,
            PassedTests = serviceResult.PassedTests,
            FailedTests = serviceResult.FailedTests,
            SkippedTests = serviceResult.SkippedTests,
            StartedAt = DateTime.UtcNow,
            CompletedAt = DateTime.UtcNow,
            DurationMs = serviceResult.DurationMs,
            TestFailures = serviceResult.TestFailures
        };
    }

    /// <summary>
    /// Converts from TarsEngine.Models.TestExecutionResult to TarsEngine.Services.TestExecutionResult
    /// </summary>
    /// <param name="modelResult">The model result to convert</param>
    /// <returns>The service result</returns>
    public static TestExecutionResult ToService(TarsEngine.Models.TestExecutionResult modelResult)
    {
        if (modelResult == null)
            throw new ArgumentNullException(nameof(modelResult));

        return new TestExecutionResult
        {
            IsSuccessful = modelResult.IsSuccessful,
            Output = modelResult.Output ?? string.Empty,
            ErrorMessage = modelResult.ErrorMessage ?? string.Empty,
            TotalTests = modelResult.TotalTests,
            PassedTests = modelResult.PassedTests,
            FailedTests = modelResult.FailedTests,
            SkippedTests = modelResult.SkippedTests,
            StartedAt = modelResult.StartedAt,
            CompletedAt = modelResult.CompletedAt,
            Duration = TimeSpan.FromMilliseconds(modelResult.DurationMs).ToString(),
            TestFailures = modelResult.TestFailures
        };
    }


}