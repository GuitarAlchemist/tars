using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.Extensions.Logging;

namespace TarsEngine.Services.CodeAnalysis
{
    /// <summary>
    /// A basic code analyzer that uses Roslyn to analyze C# code.
    /// </summary>
    public class BasicCodeAnalyzer : ICodeAnalyzer
    {
        private readonly ILogger<BasicCodeAnalyzer> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="BasicCodeAnalyzer"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        public BasicCodeAnalyzer(ILogger<BasicCodeAnalyzer> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Analyzes code.
        /// </summary>
        /// <param name="code">The code to analyze.</param>
        /// <returns>The analysis result.</returns>
        public Task<string> AnalyzeCodeAsync(string code)
        {
            _logger.LogInformation("Analyzing code using Roslyn");

            try
            {
                var syntaxTree = CSharpSyntaxTree.ParseText(code);
                var root = syntaxTree.GetCompilationUnitRoot();

                var issues = new List<string>();

                // Analyze performance issues
                var performanceIssues = AnalyzePerformanceIssues(root);
                if (performanceIssues.Any())
                {
                    issues.Add("## Performance Issues\n\n" + string.Join("\n", performanceIssues.Select(i => $"- {i}")));
                }

                // Analyze error handling issues
                var errorHandlingIssues = AnalyzeErrorHandlingIssues(root);
                if (errorHandlingIssues.Any())
                {
                    issues.Add("## Error Handling Issues\n\n" + string.Join("\n", errorHandlingIssues.Select(i => $"- {i}")));
                }

                // Analyze maintainability issues
                var maintainabilityIssues = AnalyzeMaintainabilityIssues(root);
                if (maintainabilityIssues.Any())
                {
                    issues.Add("## Maintainability Issues\n\n" + string.Join("\n", maintainabilityIssues.Select(i => $"- {i}")));
                }

                if (issues.Any())
                {
                    return Task.FromResult(string.Join("\n\n", issues));
                }
                else
                {
                    return Task.FromResult("No issues found.");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error analyzing code");
                return Task.FromResult($"Analysis failed: {ex.Message}");
            }
        }

        private List<string> AnalyzePerformanceIssues(CompilationUnitSyntax root)
        {
            var issues = new List<string>();

            // Find string concatenation in loops
            var stringConcatenations = root.DescendantNodes()
                .OfType<AssignmentExpressionSyntax>()
                .Where(a => a.Kind() == SyntaxKind.AddAssignmentExpression &&
                            a.Left.ToString().Contains("string") &&
                            IsInLoop(a));

            if (stringConcatenations.Any())
            {
                issues.Add("String concatenation in loops detected. Consider using StringBuilder instead.");
            }

            // Find LINQ operations in loops
            var linqInLoops = root.DescendantNodes()
                .OfType<InvocationExpressionSyntax>()
                .Where(i => i.ToString().Contains(".Where(") ||
                            i.ToString().Contains(".Select(") ||
                            i.ToString().Contains(".OrderBy(") ||
                            i.ToString().Contains(".GroupBy("))
                .Where(IsInLoop);

            if (linqInLoops.Any())
            {
                issues.Add("LINQ operations in loops detected. Consider moving LINQ operations outside of loops.");
            }

            // Find unnecessary object creation in loops
            var objectCreationInLoops = root.DescendantNodes()
                .OfType<ObjectCreationExpressionSyntax>()
                .Where(IsInLoop);

            if (objectCreationInLoops.Any())
            {
                issues.Add("Object creation in loops detected. Consider reusing objects or moving creation outside of loops.");
            }

            return issues;
        }

        private List<string> AnalyzeErrorHandlingIssues(CompilationUnitSyntax root)
        {
            var issues = new List<string>();

            // Find methods without null checks
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
                        issues.Add($"Method '{method.Identifier}' does not check if parameter '{parameter}' is null.");
                    }
                }
            }

            // Find potential division by zero
            var divisions = root.DescendantNodes()
                .OfType<BinaryExpressionSyntax>()
                .Where(b => b.Kind() == SyntaxKind.DivideExpression);

            foreach (var division in divisions)
            {
                var right = division.Right.ToString();
                if (right == "0" || right == "0.0" || right == "0f" || right == "0d" || right == "0m")
                {
                    issues.Add($"Division by zero detected: {division}");
                }
                else if (!IsConstant(right) && !IsCheckedForZero(division))
                {
                    issues.Add($"Potential division by zero detected: {division}");
                }
            }

            return issues;
        }

        private List<string> AnalyzeMaintainabilityIssues(CompilationUnitSyntax root)
        {
            var issues = new List<string>();

            // Find magic numbers
            var magicNumbers = root.DescendantNodes()
                .OfType<LiteralExpressionSyntax>()
                .Where(l => l.Kind() == SyntaxKind.NumericLiteralExpression &&
                           !l.ToString().Equals("0") &&
                           !l.ToString().Equals("1") &&
                           !l.ToString().Equals("2") &&
                           !l.ToString().Equals("-1") &&
                           !IsInConstantDeclaration(l));

            if (magicNumbers.Any())
            {
                issues.Add("Magic numbers detected. Consider using named constants instead.");
            }

            // Find long methods
            var longMethods = root.DescendantNodes()
                .OfType<MethodDeclarationSyntax>()
                .Where(m => m.Body != null && m.Body.Statements.Count > 30);

            foreach (var method in longMethods)
            {
                issues.Add($"Method '{method.Identifier}' is too long ({method.Body.Statements.Count} statements). Consider breaking it down into smaller methods.");
            }

            // Find duplicate code
            var methodBodies = root.DescendantNodes()
                .OfType<MethodDeclarationSyntax>()
                .Where(m => m.Body != null)
                .Select(m => m.Body.ToString())
                .ToList();

            for (int i = 0; i < methodBodies.Count; i++)
            {
                for (int j = i + 1; j < methodBodies.Count; j++)
                {
                    if (IsSimilar(methodBodies[i], methodBodies[j]))
                    {
                        issues.Add("Duplicate or similar code detected. Consider extracting common code into a separate method.");
                        break;
                    }
                }
            }

            return issues;
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

        private bool IsSimilar(string body1, string body2)
        {
            // Simple similarity check: if the bodies are more than 80% similar in length and content
            var minLength = Math.Min(body1.Length, body2.Length);
            var maxLength = Math.Max(body1.Length, body2.Length);

            if ((double)minLength / maxLength < 0.8)
            {
                return false;
            }

            var commonChars = 0;
            for (int i = 0; i < minLength; i++)
            {
                if (body1[i] == body2[i])
                {
                    commonChars++;
                }
            }

            return (double)commonChars / maxLength > 0.8;
        }
    }
}
