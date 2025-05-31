using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Formatting;
using Microsoft.Extensions.Logging;

namespace TarsEngine.Services.CodeAnalysis
{
    /// <summary>
    /// Transforms code by applying targeted fixes.
    /// </summary>
    public class CodeTransformer
    {
        private readonly ILogger<CodeTransformer> _logger;
        private readonly PatternDetector _patternDetector;

        /// <summary>
        /// Initializes a new instance of the <see cref="CodeTransformer"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="patternDetector">The pattern detector.</param>
        public CodeTransformer(
            ILogger<CodeTransformer> logger,
            PatternDetector patternDetector)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _patternDetector = patternDetector ?? throw new ArgumentNullException(nameof(patternDetector));
        }

        /// <summary>
        /// Represents a transformation result.
        /// </summary>
        public class TransformationResult
        {
            /// <summary>
            /// Gets or sets the original code.
            /// </summary>
            public string OriginalCode { get; set; }

            /// <summary>
            /// Gets or sets the transformed code.
            /// </summary>
            public string TransformedCode { get; set; }

            /// <summary>
            /// Gets or sets the applied transformations.
            /// </summary>
            public List<string> AppliedTransformations { get; set; } = new List<string>();

            /// <summary>
            /// Gets or sets the transformation errors.
            /// </summary>
            public List<string> Errors { get; set; } = new List<string>();

            /// <summary>
            /// Gets or sets a value indicating whether the transformation was successful.
            /// </summary>
            public bool Success => Errors.Count == 0;
        }

        /// <summary>
        /// Transforms code by applying targeted fixes.
        /// </summary>
        /// <param name="code">The code to transform.</param>
        /// <param name="transformationTypes">The types of transformations to apply.</param>
        /// <returns>The transformation result.</returns>
        public TransformationResult TransformCode(string code, IEnumerable<string> transformationTypes)
        {
            _logger.LogInformation("Transforming code");

            var result = new TransformationResult
            {
                OriginalCode = code,
                TransformedCode = code
            };

            try
            {
                // Detect patterns
                var patterns = _patternDetector.DetectPatterns(code);

                // Filter patterns by transformation types
                var filteredPatterns = patterns
                    .Where(p => transformationTypes.Contains(p.Category))
                    .ToList();

                if (filteredPatterns.Count == 0)
                {
                    _logger.LogInformation("No patterns found for the specified transformation types");
                    return result;
                }

                // Parse the code
                var syntaxTree = CSharpSyntaxTree.ParseText(code);
                var root = syntaxTree.GetCompilationUnitRoot();

                // Apply transformations
                var transformedRoot = root;
                foreach (var pattern in filteredPatterns)
                {
                    try
                    {
                        var transformer = GetTransformer(pattern);
                        if (transformer != null)
                        {
                            transformedRoot = transformer(transformedRoot, pattern);
                            result.AppliedTransformations.Add($"Applied {pattern.Name} fix at {pattern.Location}");
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error applying transformation for pattern {PatternName}", pattern.Name);
                        result.Errors.Add($"Error applying transformation for pattern {pattern.Name}: {ex.Message}");
                    }
                }

                // Format the transformed code
                var workspace = new AdhocWorkspace();
                var formattedRoot = Formatter.Format(transformedRoot, workspace);

                // Update the transformed code
                result.TransformedCode = formattedRoot.ToFullString();

                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error transforming code");
                result.Errors.Add($"Error transforming code: {ex.Message}");
                return result;
            }
        }

        private Func<CompilationUnitSyntax, PatternDetector.DetectedPattern, CompilationUnitSyntax> GetTransformer(PatternDetector.DetectedPattern pattern)
        {
            switch (pattern.Name)
            {
                case "String Concatenation in Loop":
                    return TransformStringConcatenation;
                case "LINQ in Loop":
                    return TransformLinqInLoop;
                case "Object Creation in Loop":
                    return TransformObjectCreationInLoop;
                case "Inefficient Collection Usage":
                    return TransformInefficientCollectionUsage;
                case "Missing Null Check":
                    return TransformMissingNullCheck;
                case "Potential Division by Zero":
                case "Division by Zero":
                    return TransformDivisionByZero;
                case "Missing Try-Catch":
                    return TransformMissingTryCatch;
                case "Magic Number":
                    return TransformMagicNumber;
                case "Long Method":
                    return TransformLongMethod;
                case "Missing Comments":
                    return TransformMissingComments;
                case "Poor Naming":
                    return TransformPoorNaming;
                case "SQL Injection Vulnerability":
                    return TransformSqlInjection;
                case "Hardcoded Credentials":
                    return TransformHardcodedCredentials;
                case "Insecure Random Number Generation":
                    return TransformInsecureRandom;
                default:
                    _logger.LogWarning("No transformer found for pattern {PatternName}", pattern.Name);
                    return null;
            }
        }

        private CompilationUnitSyntax TransformStringConcatenation(CompilationUnitSyntax root, PatternDetector.DetectedPattern pattern)
        {
            // Find the string concatenation
            var concatenation = root.DescendantNodes()
                .OfType<AssignmentExpressionSyntax>()
                .FirstOrDefault(a => a.ToString() == pattern.Code);

            if (concatenation == null)
            {
                return root;
            }

            // Find the variable declaration
            var variableName = concatenation.Left.ToString();
            var variableDeclaration = root.DescendantNodes()
                .OfType<VariableDeclarationSyntax>()
                .FirstOrDefault(v => v.Variables.Any(var => var.Identifier.ToString() == variableName));

            if (variableDeclaration == null)
            {
                return root;
            }

            // Create a StringBuilder declaration
            var builderName = variableName + "Builder";
            var stringBuilderDeclaration = SyntaxFactory.ParseStatement(
                $"var {builderName} = new StringBuilder();\n");

            // Replace string concatenation with StringBuilder.Append
            var newConcatenation = concatenation.ReplaceNode(
                concatenation,
                SyntaxFactory.ParseExpression(
                    $"{builderName}.Append({concatenation.Right})"));

            // Replace the variable declaration
            var newRoot = root.ReplaceNode(
                variableDeclaration,
                SyntaxFactory.ParseStatement(
                    $"var {builderName} = new StringBuilder();\n"));

            // Replace the concatenation
            newRoot = newRoot.ReplaceNode(
                concatenation,
                newConcatenation);

            // Find any usages of the string variable and replace with StringBuilder.ToString()
            var stringUsages = newRoot.DescendantNodes()
                .OfType<IdentifierNameSyntax>()
                .Where(i => i.Identifier.ToString() == variableName &&
                           i.Parent is not AssignmentExpressionSyntax);

            foreach (var usage in stringUsages)
            {
                newRoot = newRoot.ReplaceNode(
                    usage,
                    SyntaxFactory.ParseExpression($"{builderName}.ToString()"));
            }

            return newRoot;
        }

        private CompilationUnitSyntax TransformLinqInLoop(CompilationUnitSyntax root, PatternDetector.DetectedPattern pattern)
        {
            // Find the LINQ operation in loop
            var linq = root.DescendantNodes()
                .OfType<InvocationExpressionSyntax>()
                .FirstOrDefault(i => i.ToString() == pattern.Code);

            if (linq == null)
            {
                return root;
            }

            // Find the loop
            var loop = linq.Ancestors()
                .FirstOrDefault(a => a is ForStatementSyntax ||
                                    a is ForEachStatementSyntax ||
                                    a is WhileStatementSyntax ||
                                    a is DoStatementSyntax);

            if (loop == null)
            {
                return root;
            }

            // Create a variable to store the LINQ result
            var linqResultName = "filtered" + Guid.NewGuid().ToString().Substring(0, 8);
            var linqResultDeclaration = SyntaxFactory.ParseStatement(
                $"var {linqResultName} = {linq};\n");

            // Insert the declaration before the loop
            var newRoot = root.InsertNodesBefore(
                loop,
                new[] { linqResultDeclaration });

            // Replace the LINQ operation with the variable
            newRoot = newRoot.ReplaceNode(
                linq,
                SyntaxFactory.ParseExpression(linqResultName));

            return newRoot;
        }

        private CompilationUnitSyntax TransformObjectCreationInLoop(CompilationUnitSyntax root, PatternDetector.DetectedPattern pattern)
        {
            // Find the object creation in loop
            var creation = root.DescendantNodes()
                .OfType<ObjectCreationExpressionSyntax>()
                .FirstOrDefault(o => o.ToString() == pattern.Code);

            if (creation == null)
            {
                return root;
            }

            // Find the loop
            var loop = creation.Ancestors()
                .FirstOrDefault(a => a is ForStatementSyntax ||
                                    a is ForEachStatementSyntax ||
                                    a is WhileStatementSyntax ||
                                    a is DoStatementSyntax);

            if (loop == null)
            {
                return root;
            }

            // Create a variable to store the object
            var objectName = "cached" + Guid.NewGuid().ToString().Substring(0, 8);
            var objectDeclaration = SyntaxFactory.ParseStatement(
                $"var {objectName} = {creation};\n");

            // Insert the declaration before the loop
            var newRoot = root.InsertNodesBefore(
                loop,
                new[] { objectDeclaration });

            // Replace the object creation with the variable
            newRoot = newRoot.ReplaceNode(
                creation,
                SyntaxFactory.ParseExpression(objectName));

            return newRoot;
        }

        private CompilationUnitSyntax TransformInefficientCollectionUsage(CompilationUnitSyntax root, PatternDetector.DetectedPattern pattern)
        {
            // Find the inefficient collection usage
            var contains = root.DescendantNodes()
                .OfType<InvocationExpressionSyntax>()
                .FirstOrDefault(i => i.ToString() == pattern.Code);

            if (contains == null)
            {
                return root;
            }

            // Find the collection variable
            var memberAccess = contains.Expression as MemberAccessExpressionSyntax;
            if (memberAccess == null)
            {
                return root;
            }

            var collectionName = memberAccess.Expression.ToString();

            // Find the collection declaration
            var collectionDeclaration = root.DescendantNodes()
                .OfType<VariableDeclarationSyntax>()
                .FirstOrDefault(v => v.Variables.Any(var => var.Identifier.ToString() == collectionName));

            if (collectionDeclaration == null)
            {
                return root;
            }

            // Get the collection type
            var collectionType = collectionDeclaration.Type.ToString();
            if (!collectionType.Contains("List<"))
            {
                return root;
            }

            // Extract the element type
            var elementType = collectionType.Substring(5, collectionType.Length - 6);

            // Create a HashSet declaration
            var hashSetName = collectionName + "Set";
            var hashSetDeclaration = SyntaxFactory.ParseStatement(
                $"var {hashSetName} = new HashSet<{elementType}>({collectionName});\n");

            // Insert the HashSet declaration after the collection declaration
            var newRoot = root.InsertNodesAfter(
                collectionDeclaration,
                new[] { hashSetDeclaration });

            // Replace the Contains call with HashSet.Contains
            newRoot = newRoot.ReplaceNode(
                contains,
                SyntaxFactory.ParseExpression(
                    contains.ToString().Replace(collectionName + ".Contains", hashSetName + ".Contains")));

            return newRoot;
        }

        private CompilationUnitSyntax TransformMissingNullCheck(CompilationUnitSyntax root, PatternDetector.DetectedPattern pattern)
        {
            // Extract the parameter name from the pattern description
            var parameterName = pattern.Description.Split('\'')[3];

            // Find the method
            var method = root.DescendantNodes()
                .OfType<MethodDeclarationSyntax>()
                .FirstOrDefault(m => m.ParameterList.ToString() == pattern.Code);

            if (method == null)
            {
                return root;
            }

            // Create a null check
            var nullCheck = SyntaxFactory.ParseStatement(
                $"if ({parameterName} == null)\n" +
                "{\n" +
                $"    throw new ArgumentNullException(nameof({parameterName}));\n" +
                "}\n");

            // Insert the null check at the beginning of the method body
            if (method.Body != null)
            {
                var newBody = method.Body.WithStatements(
                    SyntaxFactory.List(
                        new[] { nullCheck }.Concat(method.Body.Statements)));

                var newMethod = method.WithBody(newBody);
                return root.ReplaceNode(method, newMethod);
            }

            return root;
        }

        private CompilationUnitSyntax TransformDivisionByZero(CompilationUnitSyntax root, PatternDetector.DetectedPattern pattern)
        {
            // Find the division
            var division = root.DescendantNodes()
                .OfType<BinaryExpressionSyntax>()
                .FirstOrDefault(b => b.ToString() == pattern.Code);

            if (division == null)
            {
                return root;
            }

            // Get the divisor
            var divisor = division.Right.ToString();

            // Create a zero check
            var zeroCheck = SyntaxFactory.ParseStatement(
                $"if ({divisor} == 0)\n" +
                "{\n" +
                "    throw new DivideByZeroException();\n" +
                "}\n");

            // Find the statement containing the division
            var statement = division.Ancestors()
                .OfType<StatementSyntax>()
                .FirstOrDefault();

            if (statement == null)
            {
                return root;
            }

            // Insert the zero check before the statement
            return root.InsertNodesBefore(
                statement,
                new[] { zeroCheck });
        }

        private CompilationUnitSyntax TransformMissingTryCatch(CompilationUnitSyntax root, PatternDetector.DetectedPattern pattern)
        {
            // Find the method
            var method = root.DescendantNodes()
                .OfType<MethodDeclarationSyntax>()
                .FirstOrDefault(m => m.ToString() == pattern.Code);

            if (method == null || method.Body == null)
            {
                return root;
            }

            // Create a try-catch block
            var tryCatch = SyntaxFactory.TryStatement(
                SyntaxFactory.Block(method.Body.Statements),
                SyntaxFactory.List(new[] {
                    SyntaxFactory.CatchClause()
                        .WithDeclaration(
                            SyntaxFactory.CatchDeclaration(
                                SyntaxFactory.ParseTypeName("Exception"),
                                SyntaxFactory.Identifier("ex")))
                        .WithBlock(
                            SyntaxFactory.Block(
                                SyntaxFactory.ParseStatement(
                                    "// Log the exception\n" +
                                    "Console.WriteLine($\"Error: {ex.Message}\");\n" +
                                    "throw;\n")))
                }),
                null);

            // Replace the method body with the try-catch block
            var newMethod = method.WithBody(
                SyntaxFactory.Block(tryCatch));

            return root.ReplaceNode(method, newMethod);
        }

        private CompilationUnitSyntax TransformMagicNumber(CompilationUnitSyntax root, PatternDetector.DetectedPattern pattern)
        {
            // Find the magic number
            var magicNumber = root.DescendantNodes()
                .OfType<LiteralExpressionSyntax>()
                .FirstOrDefault(l => l.ToString() == pattern.Code);

            if (magicNumber == null)
            {
                return root;
            }

            // Find the class containing the magic number
            var containingClass = magicNumber.Ancestors()
                .OfType<ClassDeclarationSyntax>()
                .FirstOrDefault();

            if (containingClass == null)
            {
                return root;
            }

            // Create a constant name
            var constantName = "Constant" + magicNumber.ToString().Replace(".", "_");

            // Determine the constant type
            var constantType = "int";
            if (magicNumber.ToString().Contains("."))
            {
                constantType = "double";
            }
            else if (magicNumber.ToString().EndsWith("m"))
            {
                constantType = "decimal";
            }
            else if (magicNumber.ToString().EndsWith("f"))
            {
                constantType = "float";
            }

            // Create a constant declaration
            var constantDeclaration = SyntaxFactory.ParseMemberDeclaration(
                $"private const {constantType} {constantName} = {magicNumber};\n");

            // Add the constant to the class
            var newClass = containingClass.AddMembers(constantDeclaration);

            // Replace the magic number with the constant
            var newRoot = root.ReplaceNode(containingClass, newClass);
            return newRoot.ReplaceNode(
                magicNumber,
                SyntaxFactory.ParseExpression(constantName));
        }

        private CompilationUnitSyntax TransformLongMethod(CompilationUnitSyntax root, PatternDetector.DetectedPattern pattern)
        {
            // This transformation is complex and requires semantic analysis
            // For now, we'll just add a TODO comment
            var method = root.DescendantNodes()
                .OfType<MethodDeclarationSyntax>()
                .FirstOrDefault(m => m.ToString() == pattern.Code);

            if (method == null || method.Body == null)
            {
                return root;
            }

            // Add a TODO comment
            var todoComment = SyntaxFactory.Comment(
                "// TODO: Break down this long method into smaller, more focused methods\n");

            var newMethod = method.WithLeadingTrivia(
                method.GetLeadingTrivia().Add(todoComment));

            return root.ReplaceNode(method, newMethod);
        }

        private CompilationUnitSyntax TransformMissingComments(CompilationUnitSyntax root, PatternDetector.DetectedPattern pattern)
        {
            // Find the method
            var method = root.DescendantNodes()
                .OfType<MethodDeclarationSyntax>()
                .FirstOrDefault(m => m.ToString() == pattern.Code);

            if (method == null)
            {
                return root;
            }

            // Create XML documentation
            var xmlDoc = new StringBuilder();
            xmlDoc.AppendLine("/// <summary>");
            xmlDoc.AppendLine($"/// {method.Identifier} method.");
            xmlDoc.AppendLine("/// </summary>");

            foreach (var parameter in method.ParameterList.Parameters)
            {
                xmlDoc.AppendLine($"/// <param name=\"{parameter.Identifier}\">{parameter.Identifier} parameter.</param>");
            }

            if (method.ReturnType.ToString() != "void")
            {
                xmlDoc.AppendLine("/// <returns>The result.</returns>");
            }

            // Add the XML documentation
            var xmlDocTrivia = SyntaxFactory.ParseLeadingTrivia(xmlDoc.ToString());
            var newMethod = method.WithLeadingTrivia(
                xmlDocTrivia.AddRange(method.GetLeadingTrivia()));

            return root.ReplaceNode(method, newMethod);
        }

        private CompilationUnitSyntax TransformPoorNaming(CompilationUnitSyntax root, PatternDetector.DetectedPattern pattern)
        {
            // Find the variable declaration
            var variableDeclaration = root.DescendantNodes()
                .OfType<VariableDeclarationSyntax>()
                .FirstOrDefault(v => v.ToString() == pattern.Code);

            if (variableDeclaration == null || variableDeclaration.Variables.Count == 0)
            {
                return root;
            }

            // Get the variable
            var variable = variableDeclaration.Variables[0];
            var variableName = variable.Identifier.ToString();

            // Create a better name
            var betterName = "improved" + char.ToUpper(variableName[0]) + variableName.Substring(1);

            // Replace the variable name
            var newVariable = variable.WithIdentifier(
                SyntaxFactory.Identifier(betterName));

            var newVariableDeclaration = variableDeclaration.ReplaceNode(
                variable,
                newVariable);

            // Replace all usages of the variable
            var newRoot = root.ReplaceNode(
                variableDeclaration,
                newVariableDeclaration);

            var variableUsages = newRoot.DescendantNodes()
                .OfType<IdentifierNameSyntax>()
                .Where(i => i.Identifier.ToString() == variableName);

            foreach (var usage in variableUsages)
            {
                newRoot = newRoot.ReplaceNode(
                    usage,
                    SyntaxFactory.IdentifierName(betterName));
            }

            return newRoot;
        }

        private CompilationUnitSyntax TransformSqlInjection(CompilationUnitSyntax root, PatternDetector.DetectedPattern pattern)
        {
            // Find the SQL injection
            var injection = root.DescendantNodes()
                .OfType<InvocationExpressionSyntax>()
                .FirstOrDefault(i => i.ToString() == pattern.Code);

            if (injection == null || injection.ArgumentList.Arguments.Count == 0)
            {
                return root;
            }

            // Get the SQL argument
            var sqlArgument = injection.ArgumentList.Arguments[0];

            // Add a comment about SQL injection
            var comment = SyntaxFactory.Comment(
                "// TODO: Use parameterized queries instead of string concatenation to prevent SQL injection\n");

            // Find the statement containing the injection
            var statement = injection.Ancestors()
                .OfType<StatementSyntax>()
                .FirstOrDefault();

            if (statement == null)
            {
                return root;
            }

            // Add the comment before the statement
            return root.InsertTriviaBefore(
                statement,
                SyntaxFactory.TriviaList(comment));
        }

        private CompilationUnitSyntax TransformHardcodedCredentials(CompilationUnitSyntax root, PatternDetector.DetectedPattern pattern)
        {
            // Find the hardcoded credentials
            var credentials = root.DescendantNodes()
                .OfType<VariableDeclarationSyntax>()
                .FirstOrDefault(v => v.ToString() == pattern.Code);

            if (credentials == null || credentials.Variables.Count == 0)
            {
                return root;
            }

            // Get the variable
            var variable = credentials.Variables[0];
            var variableName = variable.Identifier.ToString();

            // Add a comment about hardcoded credentials
            var comment = SyntaxFactory.Comment(
                "// TODO: Store credentials in a secure configuration or use a secret management system\n");

            // Find the statement containing the credentials
            var statement = credentials.Ancestors()
                .OfType<StatementSyntax>()
                .FirstOrDefault();

            if (statement == null)
            {
                return root;
            }

            // Add the comment before the statement
            return root.InsertTriviaBefore(
                statement,
                SyntaxFactory.TriviaList(comment));
        }

        private CompilationUnitSyntax TransformInsecureRandom(CompilationUnitSyntax root, PatternDetector.DetectedPattern pattern)
        {
            // Find the insecure random
            var random = root.DescendantNodes()
                .OfType<ObjectCreationExpressionSyntax>()
                .FirstOrDefault(o => o.ToString() == pattern.Code);

            if (random == null)
            {
                return root;
            }

            // Create a secure random
            var secureRandom = SyntaxFactory.ParseExpression(
                "System.Security.Cryptography.RandomNumberGenerator.Create()");

            // Replace the insecure random with the secure random
            return root.ReplaceNode(random, secureRandom);
        }
    }
}
