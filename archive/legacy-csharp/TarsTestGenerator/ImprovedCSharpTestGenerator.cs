using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace TarsTestGenerator
{
    /// <summary>
    /// Improved C# test generator that uses Roslyn for parsing and can handle generic types
    /// </summary>
    public class ImprovedCSharpTestGenerator
    {
        private readonly TestPatternRepository _patternRepository;

        public ImprovedCSharpTestGenerator()
        {
            _patternRepository = new TestPatternRepository();
        }

        /// <summary>
        /// Generates test code for a C# source file
        /// </summary>
        public string GenerateTests(string sourceCode, string className = null)
        {
            try
            {
                // Parse the source code
                var syntaxTree = CSharpSyntaxTree.ParseText(sourceCode);
                var root = syntaxTree.GetCompilationUnitRoot();

                // Find all class declarations
                var classDeclarations = root.DescendantNodes().OfType<ClassDeclarationSyntax>();
                
                if (!classDeclarations.Any())
                {
                    Console.WriteLine("No classes found in the source code");
                    return string.Empty;
                }

                var testBuilder = new StringBuilder();
                testBuilder.AppendLine("using Microsoft.VisualStudio.TestTools.UnitTesting;");
                testBuilder.AppendLine("using System;");
                testBuilder.AppendLine("using System.Collections.Generic;");
                testBuilder.AppendLine("using System.Linq;");
                testBuilder.AppendLine();

                // Get the namespace
                var namespaceDeclaration = root.DescendantNodes().OfType<NamespaceDeclarationSyntax>().FirstOrDefault();
                string namespaceName = namespaceDeclaration?.Name.ToString() ?? "DefaultNamespace";

                testBuilder.AppendLine($"namespace {namespaceName}.Tests");
                testBuilder.AppendLine("{");

                foreach (var classDeclaration in classDeclarations)
                {
                    // Skip if we're looking for a specific class and this isn't it
                    if (!string.IsNullOrEmpty(className) && classDeclaration.Identifier.Text != className)
                        continue;

                    testBuilder.AppendLine($"    [TestClass]");
                    testBuilder.AppendLine($"    public class {classDeclaration.Identifier.Text}Tests");
                    testBuilder.AppendLine("    {");

                    // Get all methods in the class
                    var methodDeclarations = classDeclaration.DescendantNodes().OfType<MethodDeclarationSyntax>();
                    
                    foreach (var methodDeclaration in methodDeclarations)
                    {
                        // Skip constructors, private methods, and property accessors
                        if (methodDeclaration.Identifier.Text == classDeclaration.Identifier.Text ||
                            methodDeclaration.Modifiers.Any(m => m.Text == "private") ||
                            methodDeclaration.Parent is AccessorListSyntax)
                            continue;

                        GenerateTestMethod(testBuilder, classDeclaration, methodDeclaration);
                    }

                    testBuilder.AppendLine("    }");
                }

                testBuilder.AppendLine("}");
                return testBuilder.ToString();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error generating tests: {ex.Message}");
                return string.Empty;
            }
        }

        private void GenerateTestMethod(StringBuilder testBuilder, ClassDeclarationSyntax classDeclaration, MethodDeclarationSyntax methodDeclaration)
        {
            string methodName = methodDeclaration.Identifier.Text;
            string testMethodName = $"Test_{methodName}_ShouldWork";

            testBuilder.AppendLine($"        [TestMethod]");
            testBuilder.AppendLine($"        public void {testMethodName}()");
            testBuilder.AppendLine("        {");

            // Create an instance of the class
            testBuilder.AppendLine($"            // Arrange");
            testBuilder.AppendLine($"            var sut = new {classDeclaration.Identifier.Text}();");

            // Generate parameters for the method call
            var parameters = methodDeclaration.ParameterList.Parameters;
            var parameterValues = new List<string>();

            foreach (var parameter in parameters)
            {
                string paramName = parameter.Identifier.Text;
                string paramType = parameter.Type.ToString();
                string paramValue = GenerateValueForType(paramType, paramName);
                
                testBuilder.AppendLine($"            var {paramName} = {paramValue};");
                parameterValues.Add(paramName);
            }

            // Generate the method call
            testBuilder.AppendLine();
            testBuilder.AppendLine($"            // Act");
            
            string returnType = methodDeclaration.ReturnType.ToString();
            bool isVoid = returnType == "void";
            
            if (!isVoid)
            {
                testBuilder.AppendLine($"            var result = sut.{methodName}({string.Join(", ", parameterValues)});");
            }
            else
            {
                testBuilder.AppendLine($"            sut.{methodName}({string.Join(", ", parameterValues)});");
            }

            // Generate assertions
            testBuilder.AppendLine();
            testBuilder.AppendLine($"            // Assert");
            
            if (!isVoid)
            {
                // Generate appropriate assertions based on return type
                GenerateAssertionsForType(testBuilder, returnType, methodName);
            }
            else
            {
                testBuilder.AppendLine($"            // TODO: Add assertions to verify the method's behavior");
            }

            testBuilder.AppendLine("        }");
            testBuilder.AppendLine();
        }

        private string GenerateValueForType(string typeName, string paramName)
        {
            // Check if we have a pattern for this parameter
            var pattern = _patternRepository.GetPatternForParameter(paramName);
            if (pattern != null)
            {
                return pattern;
            }

            // Handle generic types
            if (typeName.Contains("<"))
            {
                return HandleGenericType(typeName);
            }

            // Handle arrays
            if (typeName.EndsWith("[]"))
            {
                string elementType = typeName.Substring(0, typeName.Length - 2);
                return $"new {elementType}[] {{ {GenerateValueForType(elementType, "item")} }}";
            }

            // Handle basic types
            switch (typeName)
            {
                case "int":
                    return "42";
                case "double":
                case "float":
                    return "3.14";
                case "decimal":
                    return "3.14m";
                case "string":
                    return "\"test\"";
                case "bool":
                    return "true";
                case "DateTime":
                    return "DateTime.Now";
                case "Guid":
                    return "Guid.NewGuid()";
                default:
                    return $"new {typeName}()";
            }
        }

        private string HandleGenericType(string typeName)
        {
            // Extract the generic type and its type arguments
            int openBracketIndex = typeName.IndexOf('<');
            int closeBracketIndex = typeName.LastIndexOf('>');
            
            if (openBracketIndex < 0 || closeBracketIndex < 0)
                return $"new {typeName}()";
                
            string baseType = typeName.Substring(0, openBracketIndex);
            string typeArgs = typeName.Substring(openBracketIndex + 1, closeBracketIndex - openBracketIndex - 1);
            
            // Split multiple type arguments
            var typeArgList = typeArgs.Split(',').Select(t => t.Trim()).ToList();
            
            // Handle common generic collections
            switch (baseType)
            {
                case "List":
                    return $"new {typeName}() {{ {GenerateValueForType(typeArgList[0], "item")}, {GenerateValueForType(typeArgList[0], "item2")}, {GenerateValueForType(typeArgList[0], "item3")} }}";
                
                case "Dictionary":
                    if (typeArgList.Count >= 2)
                    {
                        string keyType = typeArgList[0];
                        string valueType = typeArgList[1];
                        return $"new {typeName}() {{ {{ {GenerateValueForType(keyType, "key")}, {GenerateValueForType(valueType, "value")} }}, {{ {GenerateValueForType(keyType, "key2")}, {GenerateValueForType(valueType, "value2")} }} }}";
                    }
                    break;
                    
                case "IEnumerable":
                case "ICollection":
                case "IList":
                    return $"new List<{typeArgList[0]}>() {{ {GenerateValueForType(typeArgList[0], "item")}, {GenerateValueForType(typeArgList[0], "item2")} }}";
                    
                case "IReadOnlyList":
                case "IReadOnlyCollection":
                    return $"new List<{typeArgList[0]}>() {{ {GenerateValueForType(typeArgList[0], "item")}, {GenerateValueForType(typeArgList[0], "item2")} }}.AsReadOnly()";
                    
                case "HashSet":
                    return $"new {typeName}() {{ {GenerateValueForType(typeArgList[0], "item")}, {GenerateValueForType(typeArgList[0], "item2")} }}";
                    
                case "Queue":
                case "Stack":
                    return $"new {typeName}(new[] {{ {GenerateValueForType(typeArgList[0], "item")}, {GenerateValueForType(typeArgList[0], "item2")} }})";
                    
                case "KeyValuePair":
                    if (typeArgList.Count >= 2)
                    {
                        string keyType = typeArgList[0];
                        string valueType = typeArgList[1];
                        return $"new {typeName}({GenerateValueForType(keyType, "key")}, {GenerateValueForType(valueType, "value")})";
                    }
                    break;
            }
            
            // Default for unknown generic types
            return $"new {typeName}()";
        }

        private void GenerateAssertionsForType(StringBuilder testBuilder, string returnType, string methodName)
        {
            // Check if we have a pattern for this method
            var pattern = _patternRepository.GetPatternForMethod(methodName);
            if (pattern != null)
            {
                testBuilder.AppendLine($"            {pattern}");
                return;
            }

            // Generate context-aware assertions based on method name
            if (methodName.StartsWith("Get") || methodName.StartsWith("Find") || methodName.StartsWith("Retrieve"))
            {
                testBuilder.AppendLine($"            Assert.IsNotNull(result, \"Result should not be null\");");
            }
            else if (methodName.StartsWith("Calculate") || methodName.StartsWith("Compute"))
            {
                if (returnType == "int" || returnType == "double" || returnType == "float" || returnType == "decimal")
                {
                    testBuilder.AppendLine($"            Assert.IsTrue(result >= 0, \"Result should be non-negative\");");
                }
            }
            else if (methodName.StartsWith("Is") || methodName.StartsWith("Has") || methodName.StartsWith("Can"))
            {
                if (returnType == "bool")
                {
                    testBuilder.AppendLine($"            Assert.IsTrue(result, \"Expected result to be true\");");
                }
            }
            else if (methodName.StartsWith("Count") || methodName.StartsWith("Sum"))
            {
                if (returnType == "int" || returnType == "long")
                {
                    testBuilder.AppendLine($"            Assert.IsTrue(result >= 0, \"Count should be non-negative\");");
                }
            }
            else if (methodName.StartsWith("Average"))
            {
                if (returnType == "double" || returnType == "float" || returnType == "decimal")
                {
                    testBuilder.AppendLine($"            Assert.IsTrue(!double.IsNaN((double)result), \"Average should not be NaN\");");
                }
            }
            else if (methodName.StartsWith("Convert") || methodName.StartsWith("Transform"))
            {
                testBuilder.AppendLine($"            Assert.IsNotNull(result, \"Converted result should not be null\");");
            }
            else if (methodName.StartsWith("Validate") || methodName.StartsWith("Check"))
            {
                if (returnType == "bool")
                {
                    testBuilder.AppendLine($"            Assert.IsTrue(result, \"Validation should pass\");");
                }
            }
            else if (methodName.StartsWith("Parse") || methodName.StartsWith("Format"))
            {
                testBuilder.AppendLine($"            Assert.IsNotNull(result, \"Parsed result should not be null\");");
            }
            else if (methodName.StartsWith("Max") || methodName.StartsWith("Min"))
            {
                if (returnType == "int" || returnType == "double" || returnType == "float" || returnType == "decimal")
                {
                    if (methodName.StartsWith("Max"))
                    {
                        testBuilder.AppendLine($"            Assert.IsTrue(result >= 0, \"Max value should be non-negative\");");
                    }
                    else
                    {
                        testBuilder.AppendLine($"            Assert.IsTrue(result <= 100, \"Min value should be reasonable\");");
                    }
                }
            }
            else
            {
                // Default assertions based on return type
                switch (returnType)
                {
                    case "int":
                    case "long":
                    case "short":
                    case "byte":
                        testBuilder.AppendLine($"            Assert.IsTrue(result >= 0, \"Result should be non-negative\");");
                        break;
                    case "double":
                    case "float":
                    case "decimal":
                        testBuilder.AppendLine($"            Assert.IsTrue(result >= 0, \"Result should be non-negative\");");
                        break;
                    case "string":
                        testBuilder.AppendLine($"            Assert.IsFalse(string.IsNullOrEmpty(result), \"Result should not be empty\");");
                        break;
                    case "bool":
                        testBuilder.AppendLine($"            Assert.IsTrue(result, \"Expected result to be true\");");
                        break;
                    default:
                        if (returnType.StartsWith("List<") || returnType.StartsWith("IEnumerable<") || 
                            returnType.StartsWith("ICollection<") || returnType.StartsWith("IList<") ||
                            returnType.EndsWith("[]"))
                        {
                            testBuilder.AppendLine($"            Assert.IsNotNull(result, \"Result collection should not be null\");");
                        }
                        else
                        {
                            testBuilder.AppendLine($"            Assert.IsNotNull(result, \"Result should not be null\");");
                        }
                        break;
                }
            }
        }
    }

    /// <summary>
    /// Repository for storing and retrieving test patterns
    /// </summary>
    public class TestPatternRepository
    {
        private readonly Dictionary<string, string> _methodPatterns = new Dictionary<string, string>();
        private readonly Dictionary<string, string> _parameterPatterns = new Dictionary<string, string>();
        private readonly string _patternFilePath;

        public TestPatternRepository()
        {
            _patternFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "test_patterns.json");
            LoadPatterns();
        }

        public string GetPatternForMethod(string methodName)
        {
            if (_methodPatterns.TryGetValue(methodName, out string pattern))
            {
                return pattern;
            }
            return null;
        }

        public string GetPatternForParameter(string paramName)
        {
            if (_parameterPatterns.TryGetValue(paramName, out string pattern))
            {
                return pattern;
            }
            return null;
        }

        public void AddMethodPattern(string methodName, string pattern)
        {
            _methodPatterns[methodName] = pattern;
            SavePatterns();
        }

        public void AddParameterPattern(string paramName, string pattern)
        {
            _parameterPatterns[paramName] = pattern;
            SavePatterns();
        }

        private void LoadPatterns()
        {
            try
            {
                if (File.Exists(_patternFilePath))
                {
                    string json = File.ReadAllText(_patternFilePath);
                    var patterns = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, string>>>(json);
                    
                    if (patterns.TryGetValue("methods", out var methodPatterns))
                    {
                        foreach (var kvp in methodPatterns)
                        {
                            _methodPatterns[kvp.Key] = kvp.Value;
                        }
                    }
                    
                    if (patterns.TryGetValue("parameters", out var paramPatterns))
                    {
                        foreach (var kvp in paramPatterns)
                        {
                            _parameterPatterns[kvp.Key] = kvp.Value;
                        }
                    }
                }
            }
            catch (Exception)
            {
                // Ignore errors when loading patterns
            }
        }

        private void SavePatterns()
        {
            try
            {
                var patterns = new Dictionary<string, Dictionary<string, string>>
                {
                    ["methods"] = _methodPatterns,
                    ["parameters"] = _parameterPatterns
                };
                
                string json = System.Text.Json.JsonSerializer.Serialize(patterns, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(_patternFilePath, json);
            }
            catch (Exception)
            {
                // Ignore errors when saving patterns
            }
        }
    }
}
