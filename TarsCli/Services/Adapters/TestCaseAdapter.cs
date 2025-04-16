using TarsCli.Services.Testing;

namespace TarsCli.Services.Adapters
{
    /// <summary>
    /// Adapter for converting between different TestCase types
    /// </summary>
    public static class TestCaseAdapter
    {
        /// <summary>
        /// Converts from TarsCli.Services.Testing.TestCase to TarsCli.Services.TestCase
        /// </summary>
        /// <param name="testCase">The Testing test case to convert</param>
        /// <returns>The converted test case</returns>
        public static TestCase ToServiceTestCase(Testing.TestCase testCase)
        {
            return new TestCase
            {
                Name = testCase.Name,
                Description = testCase.Description
            };
        }

        /// <summary>
        /// Converts from TarsCli.Services.TestCase to TarsCli.Services.Testing.TestCase
        /// </summary>
        /// <param name="testCase">The service test case to convert</param>
        /// <returns>The converted test case</returns>
        public static Testing.TestCase ToTestingTestCase(TestCase testCase)
        {
            return new Testing.TestCase
            {
                Name = testCase.Name,
                Description = testCase.Description,
                Type = TestType.Unit, // Default value
                TargetMethod = string.Empty, // Default value
                TestCode = string.Empty // Default value
            };
        }
    }
}
