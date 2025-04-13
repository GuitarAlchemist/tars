using System.Collections.Generic;
using System.Threading.Tasks;
using TarsCli.Services.CodeGeneration;

namespace TarsCli.Services.Testing
{
    /// <summary>
    /// Interface for test generators
    /// </summary>
    public interface ITestGenerator
    {
        /// <summary>
        /// Generates tests for a file
        /// </summary>
        /// <param name="filePath">Path to the file to test</param>
        /// <param name="fileContent">Content of the file to test</param>
        /// <returns>Test generation result</returns>
        Task<TestGenerationResult> GenerateTestsAsync(string filePath, string fileContent);

        /// <summary>
        /// Gets the supported file extensions for this generator
        /// </summary>
        /// <returns>List of supported file extensions</returns>
        IEnumerable<string> GetSupportedFileExtensions();
    }

    /// <summary>
    /// Result of test generation
    /// </summary>
    public class TestGenerationResult
    {
        /// <summary>
        /// Path to the file being tested
        /// </summary>
        public string SourceFilePath { get; set; }

        /// <summary>
        /// Path to the generated test file
        /// </summary>
        public string TestFilePath { get; set; }

        /// <summary>
        /// Content of the generated test file
        /// </summary>
        public string TestFileContent { get; set; }

        /// <summary>
        /// Whether the generation was successful
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// Error message if the generation failed
        /// </summary>
        public string ErrorMessage { get; set; }

        /// <summary>
        /// List of tests generated
        /// </summary>
        public List<TestCase> Tests { get; set; } = new List<TestCase>();

        /// <summary>
        /// Additional information about the generation
        /// </summary>
        public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();
    }

    /// <summary>
    /// Represents a test case
    /// </summary>
    public class TestCase
    {
        /// <summary>
        /// Name of the test
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Description of the test
        /// </summary>
        public string Description { get; set; }

        /// <summary>
        /// Type of the test
        /// </summary>
        public TestType Type { get; set; }

        /// <summary>
        /// Name of the method or function being tested
        /// </summary>
        public string TargetMethod { get; set; }

        /// <summary>
        /// Test code
        /// </summary>
        public string TestCode { get; set; }
    }

    /// <summary>
    /// Type of test
    /// </summary>
    public enum TestType
    {
        /// <summary>
        /// Unit test
        /// </summary>
        Unit,

        /// <summary>
        /// Integration test
        /// </summary>
        Integration,

        /// <summary>
        /// Functional test
        /// </summary>
        Functional,

        /// <summary>
        /// Performance test
        /// </summary>
        Performance,

        /// <summary>
        /// Security test
        /// </summary>
        Security
    }
}
