using TarsCli.Services.Testing;

namespace TarsCli.Services.Adapters
{
    /// <summary>
    /// Adapter for converting between different TestResult types
    /// </summary>
    public static class TestResultAdapter
    {
        /// <summary>
        /// Converts from TarsCli.Services.Testing.TestResult to TarsCli.Services.TestResult
        /// </summary>
        /// <param name="testResult">The Testing test result to convert</param>
        /// <returns>The converted test result</returns>
        public static TestResult ToServiceTestResult(Testing.TestResult testResult)
        {
            return new TestResult
            {
                ErrorMessage = testResult.ErrorMessage
            };
        }

        /// <summary>
        /// Converts from TarsCli.Services.TestResult to TarsCli.Services.Testing.TestResult
        /// </summary>
        /// <param name="testResult">The service test result to convert</param>
        /// <returns>The converted test result</returns>
        public static Testing.TestResult ToTestingTestResult(TestResult testResult)
        {
            return new Testing.TestResult
            {
                TestName = string.Empty, // Default value
                Status = Testing.TestStatus.Failed, // Default value
                Duration = "0ms", // Default value
                ErrorMessage = testResult.ErrorMessage
            };
        }
    }
}
