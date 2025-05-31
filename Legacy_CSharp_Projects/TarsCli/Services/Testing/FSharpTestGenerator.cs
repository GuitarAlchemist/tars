using System.Text;
using System.Text.RegularExpressions;

namespace TarsCli.Services.Testing;

/// <summary>
/// Generator for F# tests
/// </summary>
public class FSharpTestGenerator : ITestGenerator
{
    private readonly ILogger<FSharpTestGenerator> _logger;

    /// <summary>
    /// Initializes a new instance of the FSharpTestGenerator class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    public FSharpTestGenerator(ILogger<FSharpTestGenerator> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <inheritdoc/>
    public async Task<TestGenerationResult> GenerateTestsAsync(string filePath, string fileContent)
    {
        _logger.LogInformation($"Generating tests for F# file: {filePath}");

        var result = new TestGenerationResult
        {
            SourceFilePath = filePath,
            Success = false
        };

        try
        {
            // Parse the file to extract modules and functions
            var (moduleName, functions) = ParseFile(fileContent);

            // Generate test file path
            var directory = Path.GetDirectoryName(filePath);
            var fileName = Path.GetFileNameWithoutExtension(filePath);
            var testDirectory = Path.Combine(directory, "Tests");
            Directory.CreateDirectory(testDirectory);
            result.TestFilePath = Path.Combine(testDirectory, $"{fileName}Tests.fs");

            // Generate test file content
            var testFileContent = GenerateTestFileContent(moduleName, functions);
            result.TestFileContent = testFileContent;

            // Generate test cases
            foreach (var function in functions)
            {
                var testCase = GenerateTestCase(moduleName, function);
                result.Tests.Add(testCase);
            }

            result.Success = true;
            _logger.LogInformation($"Generated {result.Tests.Count} tests for {filePath}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating tests for {filePath}");
            result.ErrorMessage = ex.Message;
        }

        return await Task.FromResult(result);
    }

    /// <inheritdoc/>
    public IEnumerable<string> GetSupportedFileExtensions()
    {
        return [".fs", ".fsx"];
    }

    /// <summary>
    /// Parses an F# file to extract module name and functions
    /// </summary>
    /// <param name="fileContent">Content of the file</param>
    /// <returns>Tuple containing module name and functions</returns>
    private (string ModuleName, List<FunctionInfo> Functions) ParseFile(string fileContent)
    {
        var moduleName = string.Empty;
        var functions = new List<FunctionInfo>();

        // Extract module name
        var moduleMatch = Regex.Match(fileContent, @"module\s+([a-zA-Z0-9_\.]+)");
        if (moduleMatch.Success)
        {
            moduleName = moduleMatch.Groups[1].Value;
        }

        // Extract functions
        var functionMatches = Regex.Matches(fileContent, @"let\s+(?:rec\s+)?([a-zA-Z0-9_]+)(?:\s+[a-zA-Z0-9_]+)*(?:\s*:\s*([^=]+))?\s*=");
        foreach (Match functionMatch in functionMatches)
        {
            var functionName = functionMatch.Groups[1].Value;
            var functionType = functionMatch.Groups[2].Success ? functionMatch.Groups[2].Value.Trim() : string.Empty;

            // Skip internal functions (starting with lowercase)
            if (char.IsLower(functionName[0]) && !functionName.EndsWith("Async"))
            {
                continue;
            }

            // Parse parameters from the type signature
            var parameters = new List<(string Type, string Name)>();
            if (!string.IsNullOrWhiteSpace(functionType))
            {
                var paramMatches = Regex.Matches(functionType, @"([a-zA-Z0-9_']+)\s*:\s*([^->]+)");
                foreach (Match paramMatch in paramMatches)
                {
                    parameters.Add((paramMatch.Groups[2].Value.Trim(), paramMatch.Groups[1].Value));
                }
            }

            functions.Add(new FunctionInfo
            {
                Name = functionName,
                Type = functionType,
                Parameters = parameters,
                IsAsync = functionName.EndsWith("Async") || (functionType != null && functionType.Contains("Async"))
            });
        }

        return (moduleName, functions);
    }

    /// <summary>
    /// Generates test file content
    /// </summary>
    /// <param name="moduleName">Module name</param>
    /// <param name="functions">List of functions</param>
    /// <returns>Test file content</returns>
    private string GenerateTestFileContent(string moduleName, List<FunctionInfo> functions)
    {
        var sb = new StringBuilder();

        // Add module declaration
        sb.AppendLine($"module {moduleName}Tests");
        sb.AppendLine();

        // Add open statements
        sb.AppendLine("open System");
        sb.AppendLine("open Microsoft.VisualStudio.TestTools.UnitTesting");
        sb.AppendLine($"open {moduleName}");
        sb.AppendLine();

        // Add test functions
        foreach (var function in functions)
        {
            var testScenario = GetTestScenario(function);
                
            sb.AppendLine("[<TestClass>]");
            sb.AppendLine($"type {function.Name}Tests() =");
            sb.AppendLine();
                
            sb.AppendLine("    [<TestMethod>]");
            sb.AppendLine($"    member _.``{function.Name} should {testScenario.ToLowerInvariant().Replace('_', ' ')}``() =");
            sb.AppendLine("        // Arrange");
            sb.AppendLine($"        // TODO: Set up test data for {function.Name}");
            sb.AppendLine();
            sb.AppendLine("        // Act");
            sb.AppendLine($"        // TODO: Call {function.Name} with test data");
            sb.AppendLine();
            sb.AppendLine("        // Assert");
            sb.AppendLine("        // TODO: Verify the results");
            sb.AppendLine("        Assert.Fail(\"Test not implemented\")");
            sb.AppendLine();
        }

        return sb.ToString();
    }

    /// <summary>
    /// Generates a test case for a function
    /// </summary>
    /// <param name="moduleName">Module name</param>
    /// <param name="function">Function information</param>
    /// <returns>Test case</returns>
    private TestCase GenerateTestCase(string moduleName, FunctionInfo function)
    {
        var testScenario = GetTestScenario(function);
        var testName = $"{function.Name}_Should_{testScenario}";

        var sb = new StringBuilder();
        sb.AppendLine("// Arrange");
        sb.AppendLine($"// TODO: Set up test data for {function.Name}");
        sb.AppendLine();
        sb.AppendLine("// Act");
        sb.AppendLine($"// TODO: Call {function.Name} with test data");
        sb.AppendLine();
        sb.AppendLine("// Assert");
        sb.AppendLine("// TODO: Verify the results");
        sb.AppendLine("Assert.Fail(\"Test not implemented\")");

        return new TestCase
        {
            Name = testName,
            Description = $"Tests that {function.Name} {testScenario.ToLowerInvariant().Replace('_', ' ')}",
            Type = TestType.Unit,
            TargetMethod = function.Name,
            TestCode = sb.ToString()
        };
    }

    /// <summary>
    /// Gets a test scenario description based on the function
    /// </summary>
    /// <param name="function">Function information</param>
    /// <returns>Test scenario description</returns>
    private string GetTestScenario(FunctionInfo function)
    {
        // Generate a test scenario based on the function name and type
        if (function.Name.StartsWith("Get"))
        {
            return "Return_Expected_Results";
        }
        else if (function.Name.StartsWith("Set"))
        {
            return "Set_Value_Correctly";
        }
        else if (function.Name.StartsWith("Is") || function.Name.StartsWith("Has") || function.Name.StartsWith("Can"))
        {
            return "Return_True_When_Condition_Met";
        }
        else if (function.Name.StartsWith("Add"))
        {
            return "Add_Item_Successfully";
        }
        else if (function.Name.StartsWith("Remove"))
        {
            return "Remove_Item_Successfully";
        }
        else if (function.Name.StartsWith("Update"))
        {
            return "Update_Item_Successfully";
        }
        else if (function.Name.StartsWith("Delete"))
        {
            return "Delete_Item_Successfully";
        }
        else if (function.Name.StartsWith("Create"))
        {
            return "Create_Item_Successfully";
        }
        else if (function.Name.StartsWith("Find"))
        {
            return "Find_Item_When_Exists";
        }
        else if (function.Name.StartsWith("Validate"))
        {
            return "Return_True_For_Valid_Input";
        }
        else if (function.Name.StartsWith("Process"))
        {
            return "Process_Input_Correctly";
        }
        else if (function.Name.StartsWith("Convert"))
        {
            return "Convert_Input_To_Expected_Output";
        }
        else if (function.Name.StartsWith("Parse"))
        {
            return "Parse_Input_Correctly";
        }
        else if (function.Name.StartsWith("Format"))
        {
            return "Format_Input_Correctly";
        }
        else if (function.Name.StartsWith("Calculate"))
        {
            return "Calculate_Correct_Result";
        }
        else if (function.Name.StartsWith("Generate"))
        {
            return "Generate_Expected_Output";
        }
        else if (function.Type != null && function.Type.Contains("unit"))
        {
            return "Complete_Successfully";
        }
        else if (function.Type != null && function.Type.Contains("bool"))
        {
            return "Return_True_For_Valid_Input";
        }
        else if (function.Type != null && function.Type.Contains("string"))
        {
            return "Return_Expected_String";
        }
        else if (function.Type != null && (function.Type.Contains("int") || function.Type.Contains("float") || function.Type.Contains("decimal")))
        {
            return "Return_Expected_Value";
        }
        else if (function.Type != null && (function.Type.Contains("list") || function.Type.Contains("seq") || function.Type.Contains("array")))
        {
            return "Return_Expected_Collection";
        }
        else
        {
            return "Work_As_Expected";
        }
    }

    /// <summary>
    /// Information about a function
    /// </summary>
    private class FunctionInfo
    {
        /// <summary>
        /// Name of the function
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Type signature of the function
        /// </summary>
        public string Type { get; set; }

        /// <summary>
        /// Parameters of the function
        /// </summary>
        public List<(string Type, string Name)> Parameters { get; set; } = [];

        /// <summary>
        /// Whether the function is asynchronous
        /// </summary>
        public bool IsAsync { get; set; }
    }
}