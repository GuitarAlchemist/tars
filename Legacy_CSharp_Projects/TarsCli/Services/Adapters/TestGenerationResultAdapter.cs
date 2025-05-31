using TarsCli.Services.Testing;

namespace TarsCli.Services.Adapters
{
    /// <summary>
    /// Adapter for converting between different TestGenerationResult types
    /// </summary>
    public static class TestGenerationResultAdapter
    {
        /// <summary>
        /// Converts from TarsCli.Services.Testing.TestGenerationResult to TarsCli.Services.TestGenerationResult
        /// </summary>
        /// <param name="result">The Testing result to convert</param>
        /// <returns>The converted result</returns>
        public static TestGenerationResult ToServiceTestGenerationResult(Testing.TestGenerationResult result)
        {
            return new TestGenerationResult
            {
                Success = result.Success,
                ErrorMessage = result.ErrorMessage,
                TestFilePath = result.TestFilePath
            };
        }

        /// <summary>
        /// Converts from TarsCli.Services.TestGenerationResult to TarsCli.Services.Testing.TestGenerationResult
        /// </summary>
        /// <param name="result">The service result to convert</param>
        /// <returns>The converted result</returns>
        public static Testing.TestGenerationResult ToTestingTestGenerationResult(TestGenerationResult result)
        {
            return new Testing.TestGenerationResult
            {
                Success = result.Success,
                ErrorMessage = result.ErrorMessage,
                SourceFilePath = string.Empty, // Default value
                TestFilePath = string.Empty, // Default value
                TestFileContent = string.Empty, // Default value
                Tests = new List<Testing.TestCase>() // Default value
            };
        }
    }
}
