using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services.Testing
{
    /// <summary>
    /// Generator for C# tests
    /// </summary>
    public class CSharpTestGenerator : ITestGenerator
    {
        private readonly ILogger<CSharpTestGenerator> _logger;

        /// <summary>
        /// Initializes a new instance of the CSharpTestGenerator class
        /// </summary>
        /// <param name="logger">Logger instance</param>
        public CSharpTestGenerator(ILogger<CSharpTestGenerator> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <inheritdoc/>
        public async Task<TestGenerationResult> GenerateTestsAsync(string filePath, string fileContent)
        {
            _logger.LogInformation($"Generating tests for C# file: {filePath}");

            var result = new TestGenerationResult
            {
                SourceFilePath = filePath,
                Success = false
            };

            try
            {
                // Parse the file to extract classes and methods
                var (namespaceName, className, methods) = ParseFile(fileContent);

                // Generate test file path
                var directory = Path.GetDirectoryName(filePath);
                var fileName = Path.GetFileNameWithoutExtension(filePath);
                var testDirectory = Path.Combine(directory, "Tests");
                Directory.CreateDirectory(testDirectory);
                result.TestFilePath = Path.Combine(testDirectory, $"{fileName}Tests.cs");

                // Generate test file content
                var testFileContent = GenerateTestFileContent(namespaceName, className, methods);
                result.TestFileContent = testFileContent;

                // Generate test cases
                foreach (var method in methods)
                {
                    var testCase = GenerateTestCase(className, method);
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
            return new[] { ".cs" };
        }

        /// <summary>
        /// Parses a C# file to extract namespace, class name, and methods
        /// </summary>
        /// <param name="fileContent">Content of the file</param>
        /// <returns>Tuple containing namespace, class name, and methods</returns>
        private (string NamespaceName, string ClassName, List<MethodInfo> Methods) ParseFile(string fileContent)
        {
            var namespaceName = string.Empty;
            var className = string.Empty;
            var methods = new List<MethodInfo>();

            // Extract namespace
            var namespaceMatch = Regex.Match(fileContent, @"namespace\s+([a-zA-Z0-9_\.]+)");
            if (namespaceMatch.Success)
            {
                namespaceName = namespaceMatch.Groups[1].Value;
            }

            // Extract class name
            var classMatch = Regex.Match(fileContent, @"(?:public|internal)\s+(?:static\s+)?(?:abstract\s+)?class\s+([a-zA-Z0-9_]+)");
            if (classMatch.Success)
            {
                className = classMatch.Groups[1].Value;
            }

            // Extract methods
            var methodMatches = Regex.Matches(fileContent, @"(?:public|internal|protected)\s+(?:static\s+)?(?:virtual\s+)?(?:override\s+)?(?:async\s+)?([a-zA-Z0-9_<>]+)\s+([a-zA-Z0-9_]+)\s*\((.*?)\)(?:\s*=>\s*.*?;|\s*\{)");
            foreach (Match methodMatch in methodMatches)
            {
                var returnType = methodMatch.Groups[1].Value;
                var methodName = methodMatch.Groups[2].Value;
                var parameters = methodMatch.Groups[3].Value;

                // Skip property getters and setters
                if (methodName == "get" || methodName == "set")
                {
                    continue;
                }

                // Parse parameters
                var paramList = new List<(string Type, string Name)>();
                if (!string.IsNullOrWhiteSpace(parameters))
                {
                    var paramParts = SplitParameters(parameters);
                    foreach (var part in paramParts)
                    {
                        var paramMatch = Regex.Match(part.Trim(), @"([a-zA-Z0-9_<>\.]+)\s+([a-zA-Z0-9_]+)(?:\s*=.*)?$");
                        if (paramMatch.Success)
                        {
                            paramList.Add((paramMatch.Groups[1].Value, paramMatch.Groups[2].Value));
                        }
                    }
                }

                methods.Add(new MethodInfo
                {
                    Name = methodName,
                    ReturnType = returnType,
                    Parameters = paramList,
                    IsAsync = returnType.Contains("Task") || returnType.Contains("Async") || methodName.EndsWith("Async")
                });
            }

            return (namespaceName, className, methods);
        }

        /// <summary>
        /// Splits parameters string into individual parameters, handling generic types correctly
        /// </summary>
        /// <param name="parameters">Parameters string</param>
        /// <returns>List of parameter strings</returns>
        private List<string> SplitParameters(string parameters)
        {
            var result = new List<string>();
            var currentParam = new StringBuilder();
            var angleBracketCount = 0;
            var parenCount = 0;

            foreach (var c in parameters)
            {
                if (c == '<')
                {
                    angleBracketCount++;
                    currentParam.Append(c);
                }
                else if (c == '>')
                {
                    angleBracketCount--;
                    currentParam.Append(c);
                }
                else if (c == '(')
                {
                    parenCount++;
                    currentParam.Append(c);
                }
                else if (c == ')')
                {
                    parenCount--;
                    currentParam.Append(c);
                }
                else if (c == ',' && angleBracketCount == 0 && parenCount == 0)
                {
                    result.Add(currentParam.ToString());
                    currentParam.Clear();
                }
                else
                {
                    currentParam.Append(c);
                }
            }

            if (currentParam.Length > 0)
            {
                result.Add(currentParam.ToString());
            }

            return result;
        }

        /// <summary>
        /// Generates test file content
        /// </summary>
        /// <param name="namespaceName">Namespace name</param>
        /// <param name="className">Class name</param>
        /// <param name="methods">List of methods</param>
        /// <returns>Test file content</returns>
        private string GenerateTestFileContent(string namespaceName, string className, List<MethodInfo> methods)
        {
            var sb = new StringBuilder();

            // Add using statements
            sb.AppendLine("using System;");
            sb.AppendLine("using System.Collections.Generic;");
            sb.AppendLine("using System.Linq;");
            sb.AppendLine("using System.Threading.Tasks;");
            sb.AppendLine("using Microsoft.VisualStudio.TestTools.UnitTesting;");
            sb.AppendLine("using Moq;");
            sb.AppendLine($"using {namespaceName};");
            sb.AppendLine();

            // Add namespace
            sb.AppendLine($"namespace {namespaceName}.Tests");
            sb.AppendLine("{");
            sb.AppendLine("    /// <summary>");
            sb.AppendLine($"    /// Tests for the {className} class");
            sb.AppendLine("    /// </summary>");
            sb.AppendLine("    [TestClass]");
            sb.AppendLine($"    public class {className}Tests");
            sb.AppendLine("    {");

            // Add test methods
            foreach (var method in methods)
            {
                sb.AppendLine("        /// <summary>");
                sb.AppendLine($"        /// Tests the {method.Name} method");
                sb.AppendLine("        /// </summary>");
                sb.AppendLine("        [TestMethod]");
                sb.AppendLine($"        public {(method.IsAsync ? "async Task" : "void")} {method.Name}_Should{GetTestScenario(method)}()");
                sb.AppendLine("        {");
                sb.AppendLine("            // Arrange");
                sb.AppendLine($"            // TODO: Set up test data and dependencies for {method.Name}");
                sb.AppendLine();
                sb.AppendLine("            // Act");
                sb.AppendLine($"            // TODO: Call {method.Name} with test data");
                sb.AppendLine();
                sb.AppendLine("            // Assert");
                sb.AppendLine("            // TODO: Verify the results");
                sb.AppendLine("            Assert.Fail(\"Test not implemented\");");
                sb.AppendLine("        }");
                sb.AppendLine();
            }

            // Close class and namespace
            sb.AppendLine("    }");
            sb.AppendLine("}");

            return sb.ToString();
        }

        /// <summary>
        /// Generates a test case for a method
        /// </summary>
        /// <param name="className">Class name</param>
        /// <param name="method">Method information</param>
        /// <returns>Test case</returns>
        private TestCase GenerateTestCase(string className, MethodInfo method)
        {
            var testScenario = GetTestScenario(method);
            var testName = $"{method.Name}_Should{testScenario}";

            var sb = new StringBuilder();
            sb.AppendLine("// Arrange");
            sb.AppendLine($"// TODO: Set up test data and dependencies for {method.Name}");
            sb.AppendLine();
            sb.AppendLine("// Act");
            sb.AppendLine($"// TODO: Call {method.Name} with test data");
            sb.AppendLine();
            sb.AppendLine("// Assert");
            sb.AppendLine("// TODO: Verify the results");
            sb.AppendLine("Assert.Fail(\"Test not implemented\");");

            return new TestCase
            {
                Name = testName,
                Description = $"Tests that {method.Name} {testScenario.ToLowerInvariant().Replace('_', ' ')}",
                Type = TestType.Unit,
                TargetMethod = method.Name,
                TestCode = sb.ToString()
            };
        }

        /// <summary>
        /// Gets a test scenario description based on the method
        /// </summary>
        /// <param name="method">Method information</param>
        /// <returns>Test scenario description</returns>
        private string GetTestScenario(MethodInfo method)
        {
            // Generate a test scenario based on the method name and return type
            if (method.Name.StartsWith("Get"))
            {
                return "Return_Expected_Results";
            }
            else if (method.Name.StartsWith("Set"))
            {
                return "Set_Value_Correctly";
            }
            else if (method.Name.StartsWith("Is") || method.Name.StartsWith("Has") || method.Name.StartsWith("Can"))
            {
                return "Return_True_When_Condition_Met";
            }
            else if (method.Name.StartsWith("Add"))
            {
                return "Add_Item_Successfully";
            }
            else if (method.Name.StartsWith("Remove"))
            {
                return "Remove_Item_Successfully";
            }
            else if (method.Name.StartsWith("Update"))
            {
                return "Update_Item_Successfully";
            }
            else if (method.Name.StartsWith("Delete"))
            {
                return "Delete_Item_Successfully";
            }
            else if (method.Name.StartsWith("Create"))
            {
                return "Create_Item_Successfully";
            }
            else if (method.Name.StartsWith("Find"))
            {
                return "Find_Item_When_Exists";
            }
            else if (method.Name.StartsWith("Validate"))
            {
                return "Return_True_For_Valid_Input";
            }
            else if (method.Name.StartsWith("Process"))
            {
                return "Process_Input_Correctly";
            }
            else if (method.Name.StartsWith("Convert"))
            {
                return "Convert_Input_To_Expected_Output";
            }
            else if (method.Name.StartsWith("Parse"))
            {
                return "Parse_Input_Correctly";
            }
            else if (method.Name.StartsWith("Format"))
            {
                return "Format_Input_Correctly";
            }
            else if (method.Name.StartsWith("Calculate"))
            {
                return "Calculate_Correct_Result";
            }
            else if (method.Name.StartsWith("Generate"))
            {
                return "Generate_Expected_Output";
            }
            else if (method.ReturnType == "void" || method.ReturnType == "Task")
            {
                return "Complete_Successfully";
            }
            else if (method.ReturnType.Contains("bool"))
            {
                return "Return_True_For_Valid_Input";
            }
            else if (method.ReturnType.Contains("string"))
            {
                return "Return_Expected_String";
            }
            else if (method.ReturnType.Contains("int") || method.ReturnType.Contains("double") || method.ReturnType.Contains("float"))
            {
                return "Return_Expected_Value";
            }
            else if (method.ReturnType.Contains("List") || method.ReturnType.Contains("IEnumerable") || method.ReturnType.Contains("Array"))
            {
                return "Return_Expected_Collection";
            }
            else
            {
                return "Work_As_Expected";
            }
        }

        /// <summary>
        /// Information about a method
        /// </summary>
        private class MethodInfo
        {
            /// <summary>
            /// Name of the method
            /// </summary>
            public string Name { get; set; }

            /// <summary>
            /// Return type of the method
            /// </summary>
            public string ReturnType { get; set; }

            /// <summary>
            /// Parameters of the method
            /// </summary>
            public List<(string Type, string Name)> Parameters { get; set; } = new List<(string Type, string Name)>();

            /// <summary>
            /// Whether the method is asynchronous
            /// </summary>
            public bool IsAsync { get; set; }
        }
    }
}
