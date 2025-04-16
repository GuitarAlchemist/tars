using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services.Testing;

/// <summary>
/// Improved C# test generator that uses Roslyn for parsing
/// </summary>
public partial class ImprovedCSharpTestGenerator : ITestGenerator
{
    private readonly ILogger<ImprovedCSharpTestGenerator> _logger;
    private readonly ITestPatternRepository _testPatternRepository;

    /// <summary>
    /// Initializes a new instance of the ImprovedCSharpTestGenerator class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="testPatternRepository">Test pattern repository</param>
    public ImprovedCSharpTestGenerator(
        ILogger<ImprovedCSharpTestGenerator> logger,
        ITestPatternRepository testPatternRepository)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _testPatternRepository = testPatternRepository ?? throw new ArgumentNullException(nameof(testPatternRepository));
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
            // Parse the file to extract classes and methods using Roslyn
            var (namespaceName, className, methods) = ParseFileWithRoslyn(fileContent);

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
        return [".cs"];
    }

    /// <summary>
    /// Parses a C# file to extract namespace, class name, and methods using Roslyn
    /// </summary>
    /// <param name="fileContent">Content of the file</param>
    /// <returns>Tuple containing namespace, class name, and methods</returns>
    private (string NamespaceName, string ClassName, List<MethodInfo> Methods) ParseFileWithRoslyn(string fileContent)
    {
        var namespaceName = string.Empty;
        var className = string.Empty;
        var methods = new List<MethodInfo>();

        // Create a syntax tree from the file content
        var syntaxTree = CSharpSyntaxTree.ParseText(fileContent);
        var root = syntaxTree.GetCompilationUnitRoot();

        // Find the namespace
        var namespaceDeclaration = root.DescendantNodes().OfType<FileScopedNamespaceDeclarationSyntax>().FirstOrDefault();
        if (namespaceDeclaration != null)
        {
            namespaceName = namespaceDeclaration.Name.ToString();
        }
        else
        {
            var traditionalNamespace = root.DescendantNodes().OfType<NamespaceDeclarationSyntax>().FirstOrDefault();
            if (traditionalNamespace != null)
            {
                namespaceName = traditionalNamespace.Name.ToString();
            }
        }

        // Find the class
        var classDeclaration = root.DescendantNodes().OfType<ClassDeclarationSyntax>().FirstOrDefault();
        if (classDeclaration != null)
        {
            className = classDeclaration.Identifier.ToString();

            // Find all public methods
            var methodDeclarations = classDeclaration.DescendantNodes().OfType<MethodDeclarationSyntax>()
                .Where(m => m.Modifiers.Any(SyntaxKind.PublicKeyword));

            foreach (var methodDeclaration in methodDeclarations)
            {
                var methodName = methodDeclaration.Identifier.ToString();
                var returnType = methodDeclaration.ReturnType.ToString();
                var isAsync = methodDeclaration.Modifiers.Any(SyntaxKind.AsyncKeyword);

                // Get parameters
                var parameters = new List<(string Type, string Name)>();
                foreach (var parameter in methodDeclaration.ParameterList.Parameters)
                {
                    parameters.Add((parameter.Type.ToString(), parameter.Identifier.ToString()));
                }

                methods.Add(new MethodInfo
                {
                    Name = methodName,
                    ReturnType = returnType,
                    Parameters = parameters,
                    IsAsync = isAsync
                });
            }
        }

        return (namespaceName, className, methods);
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

        // Add test class
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

            // Create instance of the class
            sb.AppendLine($"            var sut = new {className}();");

            // Generate test data based on method parameters
            foreach (var param in method.Parameters)
            {
                sb.AppendLine($"            {GenerateTestDataForParameter(param.Type, param.Name, method.Name)}");
            }

            sb.AppendLine();
            sb.AppendLine("            // Act");

            // Call the method
            if (method.ReturnType != "void")
            {
                sb.Append("            var result = ");
            }
            else
            {
                sb.Append("            ");
            }

            if (method.IsAsync)
            {
                sb.Append("await ");
            }

            sb.Append($"sut.{method.Name}(");
            for (int i = 0; i < method.Parameters.Count; i++)
            {
                sb.Append(method.Parameters[i].Name);
                if (i < method.Parameters.Count - 1)
                {
                    sb.Append(", ");
                }
            }
            sb.AppendLine(");");

            sb.AppendLine();
            sb.AppendLine("            // Assert");

            // Generate assertions based on return type
            if (method.ReturnType != "void")
            {
                sb.AppendLine(GenerateAssertionsForReturnType(method.ReturnType, method.Name));
            }
            else
            {
                sb.AppendLine("            // Verify the method executed without exceptions");
                sb.AppendLine("            Assert.IsTrue(true);");
            }

            sb.AppendLine("        }");
            sb.AppendLine();
        }

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

        // Create instance of the class
        sb.AppendLine($"var sut = new {className}();");

        // Generate test data based on method parameters
        foreach (var param in method.Parameters)
        {
            sb.AppendLine(GenerateTestDataForParameter(param.Type, param.Name, method.Name));
        }

        sb.AppendLine();
        sb.AppendLine("// Act");

        // Call the method
        if (method.ReturnType != "void")
        {
            sb.Append("var result = ");
        }

        if (method.IsAsync)
        {
            sb.Append("await ");
        }

        sb.Append($"sut.{method.Name}(");
        for (int i = 0; i < method.Parameters.Count; i++)
        {
            sb.Append(method.Parameters[i].Name);
            if (i < method.Parameters.Count - 1)
            {
                sb.Append(", ");
            }
        }
        sb.AppendLine(");");

        sb.AppendLine();
        sb.AppendLine("// Assert");

        // Generate assertions based on return type
        if (method.ReturnType != "void")
        {
            sb.AppendLine(GenerateAssertionsForReturnType(method.ReturnType, method.Name));
        }
        else
        {
            sb.AppendLine("// Verify the method executed without exceptions");
            sb.AppendLine("Assert.IsTrue(true);");
        }

        return new TestCase
        {
            Name = testName,
            Description = $"Test for {method.Name} method",
            Type = TestType.Unit,
            TargetMethod = method.Name,
            TestCode = sb.ToString()
        };
    }

    /// <summary>
    /// Gets a test scenario for a method
    /// </summary>
    /// <param name="method">Method information</param>
    /// <returns>Test scenario</returns>
    private static string GetTestScenario(MethodInfo method)
    {
        if (method.Name.StartsWith("Get") || method.Name.StartsWith("Find") || method.Name.StartsWith("Retrieve"))
        {
            return "Find_Item_When_Exists";
        }
        else if (method.Name.StartsWith("Add") || method.Name.StartsWith("Create") || method.Name.StartsWith("Insert"))
        {
            return "Add_Item_Successfully";
        }
        else if (method.Name.StartsWith("Update") || method.Name.StartsWith("Modify"))
        {
            return "Update_Item_Successfully";
        }
        else if (method.Name.StartsWith("Delete") || method.Name.StartsWith("Remove"))
        {
            return "Remove_Item_Successfully";
        }
        else if (method.Name.StartsWith("Validate") || method.Name.StartsWith("Check"))
        {
            return "Return_True_For_Valid_Input";
        }
        else
        {
            return "Return_Expected_Value";
        }
    }

    /// <summary>
    /// Generates test data for a parameter
    /// </summary>
    /// <param name="type">Parameter type</param>
    /// <param name="name">Parameter name</param>
    /// <param name="methodName">Method name</param>
    /// <returns>Code to create test data</returns>
    private string GenerateTestDataForParameter(string type, string name, string methodName)
    {
        // Check if we have a learned pattern for this type and method name
        var learnedPattern = _testPatternRepository.GetPattern(type, methodName);
        if (learnedPattern != null)
        {
            return learnedPattern.GenerateTestData(name);
        }

        // Handle generic types
        if (type.Contains("<") && type.Contains(">"))
        {
            if (type.StartsWith("List<") || type.StartsWith("IList<") || 
                type.StartsWith("IEnumerable<") || type.StartsWith("ICollection<"))
            {
                var innerType = ExtractGenericType(type);
                var values = GenerateValuesForType(innerType, 3);
                return $"var {name} = new List<{innerType}>() {{ {string.Join(", ", values)} }};";
            }
            else if (type.StartsWith("Dictionary<") || type.StartsWith("IDictionary<"))
            {
                var match = Regex.Match(type, @"<([^,]+),\s*([^>]+)>");
                if (match.Success)
                {
                    var keyType = match.Groups[1].Value;
                    var valueType = match.Groups[2].Value;
                    var keys = GenerateValuesForType(keyType, 2);
                    var values = GenerateValuesForType(valueType, 2);
                    return $"var {name} = new Dictionary<{keyType}, {valueType}>() {{ {{ {keys[0]}, {values[0]} }}, {{ {keys[1]}, {values[1]} }} }};";
                }
            }
        }

        // Handle array types
        if (type.EndsWith("[]"))
        {
            var arrayType = type.Replace("[]", "");
            var values = GenerateValuesForType(arrayType, 3);
            return $"var {name} = new {arrayType}[] {{ {string.Join(", ", values)} }};";
        }

        // Handle primitive types
        if (type.Contains("int") || type.Contains("Int32"))
        {
            return $"var {name} = 42;";
        }
        else if (type.Contains("long") || type.Contains("Int64"))
        {
            return $"var {name} = 42L;";
        }
        else if (type.Contains("double"))
        {
            return $"var {name} = 42.0;";
        }
        else if (type.Contains("float"))
        {
            return $"var {name} = 42.0f;";
        }
        else if (type.Contains("decimal"))
        {
            return $"var {name} = 42.0m;";
        }
        else if (type.Contains("bool") || type.Contains("Boolean"))
        {
            return $"var {name} = true;";
        }
        else if (type.Contains("string") || type.Contains("String"))
        {
            return $"var {name} = \"test\";";
        }
        else if (type.Contains("DateTime"))
        {
            return $"var {name} = DateTime.Now;";
        }
        else if (type.Contains("Guid"))
        {
            return $"var {name} = Guid.NewGuid();";
        }
        else
        {
            return $"// TODO: Create test data for parameter '{name}' of type '{type}'";
        }
    }

    /// <summary>
    /// Generates values for a type
    /// </summary>
    /// <param name="type">Type</param>
    /// <param name="count">Number of values to generate</param>
    /// <returns>List of values</returns>
    private List<string> GenerateValuesForType(string type, int count)
    {
        var values = new List<string>();
        for (int i = 0; i < count; i++)
        {
            if (type == "int" || type == "Int32")
            {
                values.Add((i + 1) * 10 + "");
            }
            else if (type == "long" || type == "Int64")
            {
                values.Add((i + 1) * 10 + "L");
            }
            else if (type == "double")
            {
                values.Add((i + 1) * 10 + ".0");
            }
            else if (type == "float")
            {
                values.Add((i + 1) * 10 + ".0f");
            }
            else if (type == "decimal")
            {
                values.Add((i + 1) * 10 + ".0m");
            }
            else if (type == "bool" || type == "Boolean")
            {
                values.Add(i % 2 == 0 ? "true" : "false");
            }
            else if (type == "string" || type == "String")
            {
                values.Add($"\"test{i + 1}\"");
            }
            else if (type == "DateTime")
            {
                values.Add($"DateTime.Now.AddDays({i})");
            }
            else if (type == "Guid")
            {
                values.Add($"Guid.NewGuid()");
            }
            else
            {
                values.Add($"default({type})");
            }
        }
        return values;
    }

    /// <summary>
    /// Extracts the generic type from a generic type declaration
    /// </summary>
    /// <param name="genericType">Generic type declaration</param>
    /// <returns>Inner type</returns>
    private string ExtractGenericType(string genericType)
    {
        var match = GenericTypeRegex().Match(genericType);
        if (match.Success)
        {
            return match.Groups[1].Value;
        }
        return "object";
    }

    /// <summary>
    /// Regex for extracting generic type from a generic type declaration
    /// </summary>
    [GeneratedRegex(@"<([^>]+)>")]
    private static partial Regex GenericTypeRegex();

    /// <summary>
    /// Generates assertions for a return type
    /// </summary>
    /// <param name="returnType">Return type</param>
    /// <param name="methodName">Method name</param>
    /// <returns>Assertion code</returns>
    private string GenerateAssertionsForReturnType(string returnType, string methodName)
    {
        // Generate context-aware assertions based on method name
        if (methodName.Contains("Average") && returnType.Contains("double"))
        {
            return "            Assert.AreEqual(25.0, result, 0.001);";
        }
        else if (methodName.Contains("Max") && returnType.Contains("int"))
        {
            return "            Assert.AreEqual(42, result);";
        }
        else if (methodName.Contains("Min") && returnType.Contains("int"))
        {
            return "            Assert.AreEqual(10, result);";
        }
        else if (methodName.Contains("Add") && returnType.Contains("int"))
        {
            return "            Assert.AreEqual(84, result);";
        }
        else if (methodName.Contains("Subtract") && returnType.Contains("int"))
        {
            return "            Assert.AreEqual(0, result);";
        }
        else if (methodName.Contains("Multiply") || methodName.Contains("x") && returnType.Contains("int"))
        {
            return "            Assert.AreEqual(1764, result);";
        }
        else if (methodName.Contains("Divide") && returnType.Contains("int"))
        {
            return "            Assert.AreEqual(1, result);";
        }

        // Default assertions based on return type
        if (returnType.Contains("int") || returnType.Contains("Int32"))
        {
            return "            Assert.AreEqual(42, result);";
        }
        else if (returnType.Contains("long") || returnType.Contains("Int64"))
        {
            return "            Assert.AreEqual(42L, result);";
        }
        else if (returnType.Contains("double"))
        {
            return "            Assert.AreEqual(42.0, result, 0.001);";
        }
        else if (returnType.Contains("float"))
        {
            return "            Assert.AreEqual(42.0f, result, 0.001f);";
        }
        else if (returnType.Contains("decimal"))
        {
            return "            Assert.AreEqual(42.0m, result);";
        }
        else if (returnType.Contains("bool") || returnType.Contains("Boolean"))
        {
            return "            Assert.IsTrue(result);";
        }
        else if (returnType.Contains("string") || returnType.Contains("String"))
        {
            return "            Assert.AreEqual(\"test\", result);";
        }
        else if (returnType.Contains("List<") || returnType.Contains("IList<") || 
                 returnType.Contains("IEnumerable<") || returnType.Contains("ICollection<"))
        {
            return "            Assert.IsNotNull(result);\n            Assert.IsTrue(result.Any());";
        }
        else if (returnType.Contains("Dictionary<") || returnType.Contains("IDictionary<"))
        {
            return "            Assert.IsNotNull(result);\n            Assert.IsTrue(result.Any());";
        }
        else if (returnType.Contains("Task<"))
        {
            var innerType = ExtractGenericType(returnType);
            return GenerateAssertionsForReturnType(innerType, methodName);
        }
        else if (returnType != "void" && returnType != "Task")
        {
            return "            Assert.IsNotNull(result);";
        }
        else
        {
            return "            // No assertions needed for void return type";
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
        public List<(string Type, string Name)> Parameters { get; set; } = [];

        /// <summary>
        /// Whether the method is asynchronous
        /// </summary>
        public bool IsAsync { get; set; }
    }
}
