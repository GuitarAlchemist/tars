using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.Extensions.Logging;

namespace TarsEngine.Services.CodeAnalysis
{
    /// <summary>
    /// Detects common code patterns and anti-patterns.
    /// </summary>
    public class PatternDetector
    {
        private readonly ILogger<PatternDetector> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="PatternDetector"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        public PatternDetector(ILogger<PatternDetector> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Represents a detected pattern.
        /// </summary>
        public class DetectedPattern
        {
            /// <summary>
            /// Gets or sets the pattern name.
            /// </summary>
            public string Name { get; set; }

            /// <summary>
            /// Gets or sets the pattern description.
            /// </summary>
            public string Description { get; set; }

            /// <summary>
            /// Gets or sets the pattern category.
            /// </summary>
            public string Category { get; set; }

            /// <summary>
            /// Gets or sets the pattern severity.
            /// </summary>
            public string Severity { get; set; }

            /// <summary>
            /// Gets or sets the pattern location.
            /// </summary>
            public string Location { get; set; }

            /// <summary>
            /// Gets or sets the pattern code.
            /// </summary>
            public string Code { get; set; }

            /// <summary>
            /// Gets or sets the suggested fix.
            /// </summary>
            public string SuggestedFix { get; set; }
        }

        /// <summary>
        /// Detects patterns in code.
        /// </summary>
        /// <param name="code">The code to analyze.</param>
        /// <returns>The detected patterns.</returns>
        public List<DetectedPattern> DetectPatterns(string code)
        {
            _logger.LogInformation("Detecting patterns in code");

            var patterns = new List<DetectedPattern>();

            try
            {
                var syntaxTree = CSharpSyntaxTree.ParseText(code);
                var root = syntaxTree.GetCompilationUnitRoot();

                // Detect performance patterns
                patterns.AddRange(DetectPerformancePatterns(root));

                // Detect error handling patterns
                patterns.AddRange(DetectErrorHandlingPatterns(root));

                // Detect maintainability patterns
                patterns.AddRange(DetectMaintainabilityPatterns(root));

                // Detect security patterns
                patterns.AddRange(DetectSecurityPatterns(root));

                return patterns;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error detecting patterns");
                return patterns;
            }
        }

        private List<DetectedPattern> DetectPerformancePatterns(CompilationUnitSyntax root)
        {
            var patterns = new List<DetectedPattern>();

            // Detect string concatenation in loops
            var stringConcatenations = root.DescendantNodes()
                .OfType<AssignmentExpressionSyntax>()
                .Where(a => a.Kind() == SyntaxKind.AddAssignmentExpression &&
                            a.Left.ToString().Contains("string") &&
                            IsInLoop(a));

            foreach (var concatenation in stringConcatenations)
            {
                patterns.Add(new DetectedPattern
                {
                    Name = "String Concatenation in Loop",
                    Description = "String concatenation in loops can be inefficient due to the immutable nature of strings.",
                    Category = "Performance",
                    Severity = "Medium",
                    Location = GetLocation(concatenation),
                    Code = concatenation.ToString(),
                    SuggestedFix = "Use StringBuilder instead of string concatenation in loops."
                });
            }

            // Detect LINQ operations in loops
            var linqInLoops = root.DescendantNodes()
                .OfType<InvocationExpressionSyntax>()
                .Where(i => i.ToString().Contains(".Where(") ||
                            i.ToString().Contains(".Select(") ||
                            i.ToString().Contains(".OrderBy(") ||
                            i.ToString().Contains(".GroupBy("))
                .Where(IsInLoop);

            foreach (var linq in linqInLoops)
            {
                patterns.Add(new DetectedPattern
                {
                    Name = "LINQ in Loop",
                    Description = "LINQ operations in loops can be inefficient as they create new collections on each iteration.",
                    Category = "Performance",
                    Severity = "Medium",
                    Location = GetLocation(linq),
                    Code = linq.ToString(),
                    SuggestedFix = "Move LINQ operations outside of loops or cache the results."
                });
            }

            // Detect unnecessary object creation in loops
            var objectCreationInLoops = root.DescendantNodes()
                .OfType<ObjectCreationExpressionSyntax>()
                .Where(IsInLoop);

            foreach (var creation in objectCreationInLoops)
            {
                patterns.Add(new DetectedPattern
                {
                    Name = "Object Creation in Loop",
                    Description = "Creating objects in loops can lead to excessive garbage collection.",
                    Category = "Performance",
                    Severity = "Low",
                    Location = GetLocation(creation),
                    Code = creation.ToString(),
                    SuggestedFix = "Move object creation outside of loops or reuse objects."
                });
            }

            // Detect inefficient collection usage
            var listContains = root.DescendantNodes()
                .OfType<InvocationExpressionSyntax>()
                .Where(i => i.ToString().Contains(".Contains(") &&
                            i.Expression is MemberAccessExpressionSyntax memberAccess &&
                            memberAccess.Expression.ToString().Contains("List<"));

            foreach (var contains in listContains)
            {
                patterns.Add(new DetectedPattern
                {
                    Name = "Inefficient Collection Usage",
                    Description = "Using List.Contains() for frequent lookups is inefficient. HashSet or Dictionary would be more appropriate.",
                    Category = "Performance",
                    Severity = "Medium",
                    Location = GetLocation(contains),
                    Code = contains.ToString(),
                    SuggestedFix = "Use HashSet or Dictionary for frequent lookups."
                });
            }

            return patterns;
        }

        private List<DetectedPattern> DetectErrorHandlingPatterns(CompilationUnitSyntax root)
        {
            var patterns = new List<DetectedPattern>();

            // Detect missing null checks
            var methods = root.DescendantNodes()
                .OfType<MethodDeclarationSyntax>()
                .Where(m => m.ParameterList.Parameters.Any(p => !p.Type.ToString().EndsWith("?") &&
                                                               !IsValueType(p.Type.ToString())));

            foreach (var method in methods)
            {
                var parameters = method.ParameterList.Parameters
                    .Where(p => !p.Type.ToString().EndsWith("?") &&
                               !IsValueType(p.Type.ToString()))
                    .Select(p => p.Identifier.ToString());

                var methodBody = method.Body?.ToString() ?? "";

                foreach (var parameter in parameters)
                {
                    if (!methodBody.Contains($"if ({parameter} == null)") &&
                        !methodBody.Contains($"if (null == {parameter})") &&
                        !methodBody.Contains($"if ({parameter} is null)") &&
                        !methodBody.Contains($"?? throw") &&
                        !methodBody.Contains("ArgumentNullException"))
                    {
                        patterns.Add(new DetectedPattern
                        {
                            Name = "Missing Null Check",
                            Description = $"Method '{method.Identifier}' does not check if parameter '{parameter}' is null.",
                            Category = "Error Handling",
                            Severity = "High",
                            Location = GetLocation(method),
                            Code = method.ParameterList.ToString(),
                            SuggestedFix = $"Add a null check for parameter '{parameter}'."
                        });
                    }
                }
            }

            // Detect potential division by zero
            var divisions = root.DescendantNodes()
                .OfType<BinaryExpressionSyntax>()
                .Where(b => b.Kind() == SyntaxKind.DivideExpression);

            foreach (var division in divisions)
            {
                var right = division.Right.ToString();
                if (right == "0" || right == "0.0" || right == "0f" || right == "0d" || right == "0m")
                {
                    patterns.Add(new DetectedPattern
                    {
                        Name = "Division by Zero",
                        Description = "Division by zero will throw a DivideByZeroException.",
                        Category = "Error Handling",
                        Severity = "Critical",
                        Location = GetLocation(division),
                        Code = division.ToString(),
                        SuggestedFix = "Add a check to prevent division by zero."
                    });
                }
                else if (!IsConstant(right) && !IsCheckedForZero(division))
                {
                    patterns.Add(new DetectedPattern
                    {
                        Name = "Potential Division by Zero",
                        Description = "Potential division by zero if the divisor is zero.",
                        Category = "Error Handling",
                        Severity = "High",
                        Location = GetLocation(division),
                        Code = division.ToString(),
                        SuggestedFix = "Add a check to prevent division by zero."
                    });
                }
            }

            // Detect missing try-catch blocks
            var methodsWithoutTryCatch = root.DescendantNodes()
                .OfType<MethodDeclarationSyntax>()
                .Where(m => m.Body != null &&
                           !m.Body.DescendantNodes().OfType<TryStatementSyntax>().Any() &&
                           m.Body.DescendantNodes().OfType<InvocationExpressionSyntax>().Any());

            foreach (var method in methodsWithoutTryCatch)
            {
                patterns.Add(new DetectedPattern
                {
                    Name = "Missing Try-Catch",
                    Description = $"Method '{method.Identifier}' does not have try-catch blocks for error handling.",
                    Category = "Error Handling",
                    Severity = "Medium",
                    Location = GetLocation(method),
                    Code = method.ToString(),
                    SuggestedFix = "Add try-catch blocks to handle potential exceptions."
                });
            }

            return patterns;
        }

        private List<DetectedPattern> DetectMaintainabilityPatterns(CompilationUnitSyntax root)
        {
            var patterns = new List<DetectedPattern>();

            // Detect magic numbers
            var magicNumbers = root.DescendantNodes()
                .OfType<LiteralExpressionSyntax>()
                .Where(l => l.Kind() == SyntaxKind.NumericLiteralExpression &&
                           !l.ToString().Equals("0") &&
                           !l.ToString().Equals("1") &&
                           !l.ToString().Equals("2") &&
                           !l.ToString().Equals("-1") &&
                           !IsInConstantDeclaration(l));

            foreach (var number in magicNumbers)
            {
                patterns.Add(new DetectedPattern
                {
                    Name = "Magic Number",
                    Description = "Magic numbers make code harder to understand and maintain.",
                    Category = "Maintainability",
                    Severity = "Low",
                    Location = GetLocation(number),
                    Code = number.ToString(),
                    SuggestedFix = "Replace magic numbers with named constants."
                });
            }

            // Detect long methods
            var longMethods = root.DescendantNodes()
                .OfType<MethodDeclarationSyntax>()
                .Where(m => m.Body != null && m.Body.Statements.Count > 30);

            foreach (var method in longMethods)
            {
                patterns.Add(new DetectedPattern
                {
                    Name = "Long Method",
                    Description = $"Method '{method.Identifier}' is too long ({method.Body.Statements.Count} statements).",
                    Category = "Maintainability",
                    Severity = "Medium",
                    Location = GetLocation(method),
                    Code = method.ToString(),
                    SuggestedFix = "Break down long methods into smaller, more focused methods."
                });
            }

            // Detect missing comments
            var methodsWithoutComments = root.DescendantNodes()
                .OfType<MethodDeclarationSyntax>()
                .Where(m => m.HasLeadingTrivia &&
                           !m.GetLeadingTrivia().ToString().Contains("///") &&
                           !m.GetLeadingTrivia().ToString().Contains("/*"));

            foreach (var method in methodsWithoutComments)
            {
                patterns.Add(new DetectedPattern
                {
                    Name = "Missing Comments",
                    Description = $"Method '{method.Identifier}' does not have XML documentation.",
                    Category = "Maintainability",
                    Severity = "Low",
                    Location = GetLocation(method),
                    Code = method.ToString(),
                    SuggestedFix = "Add XML documentation to public members for better code understanding."
                });
            }

            // Detect poor naming
            var poorlyNamedVariables = root.DescendantNodes()
                .OfType<VariableDeclarationSyntax>()
                .Where(v => v.Variables.Any(var => var.Identifier.ToString().Length <= 2 &&
                                                 !var.Identifier.ToString().Equals("i") &&
                                                 !var.Identifier.ToString().Equals("j") &&
                                                 !var.Identifier.ToString().Equals("k")));

            foreach (var variable in poorlyNamedVariables)
            {
                patterns.Add(new DetectedPattern
                {
                    Name = "Poor Naming",
                    Description = "Variable names should be descriptive and meaningful.",
                    Category = "Maintainability",
                    Severity = "Low",
                    Location = GetLocation(variable),
                    Code = variable.ToString(),
                    SuggestedFix = "Use descriptive variable names that convey their purpose."
                });
            }

            return patterns;
        }

        private List<DetectedPattern> DetectSecurityPatterns(CompilationUnitSyntax root)
        {
            var patterns = new List<DetectedPattern>();

            // Detect SQL injection vulnerabilities
            var sqlInjections = root.DescendantNodes()
                .OfType<InvocationExpressionSyntax>()
                .Where(i => i.ToString().Contains("ExecuteNonQuery") ||
                            i.ToString().Contains("ExecuteReader") ||
                            i.ToString().Contains("ExecuteScalar"))
                .Where(i => {
                    var arguments = i.ArgumentList.Arguments;
                    if (arguments.Count == 0)
                        return false;

                    var sqlArgument = arguments[0].ToString();
                    return sqlArgument.Contains("+") || sqlArgument.Contains("$");
                });

            foreach (var injection in sqlInjections)
            {
                patterns.Add(new DetectedPattern
                {
                    Name = "SQL Injection Vulnerability",
                    Description = "Potential SQL injection vulnerability detected.",
                    Category = "Security",
                    Severity = "Critical",
                    Location = GetLocation(injection),
                    Code = injection.ToString(),
                    SuggestedFix = "Use parameterized queries or prepared statements instead of string concatenation."
                });
            }

            // Detect hardcoded credentials
            var hardcodedCredentials = root.DescendantNodes()
                .OfType<VariableDeclarationSyntax>()
                .Where(v => v.Variables.Any(var => 
                    (var.Identifier.ToString().Contains("password") ||
                     var.Identifier.ToString().Contains("secret") ||
                     var.Identifier.ToString().Contains("key") ||
                     var.Identifier.ToString().Contains("token")) &&
                    var.Initializer != null &&
                    var.Initializer.Value.Kind() == SyntaxKind.StringLiteralExpression));

            foreach (var credential in hardcodedCredentials)
            {
                patterns.Add(new DetectedPattern
                {
                    Name = "Hardcoded Credentials",
                    Description = "Hardcoded credentials are a security risk.",
                    Category = "Security",
                    Severity = "Critical",
                    Location = GetLocation(credential),
                    Code = credential.ToString(),
                    SuggestedFix = "Store credentials in a secure configuration or use a secret management system."
                });
            }

            // Detect insecure random number generation
            var insecureRandom = root.DescendantNodes()
                .OfType<ObjectCreationExpressionSyntax>()
                .Where(o => o.Type.ToString() == "Random");

            foreach (var random in insecureRandom)
            {
                patterns.Add(new DetectedPattern
                {
                    Name = "Insecure Random Number Generation",
                    Description = "System.Random is not cryptographically secure.",
                    Category = "Security",
                    Severity = "Medium",
                    Location = GetLocation(random),
                    Code = random.ToString(),
                    SuggestedFix = "Use RNGCryptoServiceProvider or RandomNumberGenerator for security-sensitive operations."
                });
            }

            return patterns;
        }

        private bool IsInLoop(SyntaxNode node)
        {
            var parent = node.Parent;
            while (parent != null)
            {
                if (parent is ForStatementSyntax ||
                    parent is ForEachStatementSyntax ||
                    parent is WhileStatementSyntax ||
                    parent is DoStatementSyntax)
                {
                    return true;
                }
                parent = parent.Parent;
            }
            return false;
        }

        private bool IsValueType(string typeName)
        {
            return typeName == "int" ||
                   typeName == "long" ||
                   typeName == "float" ||
                   typeName == "double" ||
                   typeName == "decimal" ||
                   typeName == "bool" ||
                   typeName == "char" ||
                   typeName == "byte" ||
                   typeName == "sbyte" ||
                   typeName == "short" ||
                   typeName == "ushort" ||
                   typeName == "uint" ||
                   typeName == "ulong";
        }

        private bool IsConstant(string expression)
        {
            return int.TryParse(expression, out _) ||
                   double.TryParse(expression, out _) ||
                   decimal.TryParse(expression, out _);
        }

        private bool IsCheckedForZero(BinaryExpressionSyntax division)
        {
            var parent = division.Parent;
            while (parent != null && !(parent is MethodDeclarationSyntax))
            {
                if (parent is IfStatementSyntax ifStatement)
                {
                    var condition = ifStatement.Condition.ToString();
                    if (condition.Contains(division.Right.ToString()) &&
                        (condition.Contains("== 0") ||
                         condition.Contains("!= 0") ||
                         condition.Contains("> 0") ||
                         condition.Contains("< 0")))
                    {
                        return true;
                    }
                }
                parent = parent.Parent;
            }
            return false;
        }

        private bool IsInConstantDeclaration(LiteralExpressionSyntax literal)
        {
            var parent = literal.Parent;
            while (parent != null && !(parent is MethodDeclarationSyntax))
            {
                if (parent is FieldDeclarationSyntax field &&
                    field.Modifiers.Any(m => m.Kind() == SyntaxKind.ConstKeyword))
                {
                    return true;
                }
                parent = parent.Parent;
            }
            return false;
        }

        private string GetLocation(SyntaxNode node)
        {
            var location = node.GetLocation();
            var lineSpan = location.GetLineSpan();
            return $"Line {lineSpan.StartLinePosition.Line + 1}, Column {lineSpan.StartLinePosition.Character + 1}";
        }
    }
}
