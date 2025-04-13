using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Validates semantic correctness of code changes
/// </summary>
public class SemanticValidator
{
    private readonly ILogger<SemanticValidator> _logger;
    private readonly Dictionary<string, Func<string, string, string, Task<List<ValidationResult>>>> _validators;

    /// <summary>
    /// Initializes a new instance of the <see cref="SemanticValidator"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public SemanticValidator(ILogger<SemanticValidator> logger)
    {
        _logger = logger;
        _validators = new Dictionary<string, Func<string, string, string, Task<List<ValidationResult>>>>(StringComparer.OrdinalIgnoreCase)
        {
            { ".cs", ValidateCSharpSemanticsAsync },
            { ".fs", ValidateFSharpSemanticsAsync }
        };
    }

    /// <summary>
    /// Validates semantics of a file
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <param name="content">The file content</param>
    /// <param name="projectPath">The project path</param>
    /// <returns>The list of validation results</returns>
    public async Task<List<ValidationResult>> ValidateFileSemanticsAsync(string filePath, string content, string projectPath)
    {
        try
        {
            _logger.LogInformation("Validating semantics of file: {FilePath}", filePath);

            // Get file extension
            var extension = Path.GetExtension(filePath);

            // Check if we have a validator for this file type
            if (_validators.TryGetValue(extension, out var validator))
            {
                return await validator(filePath, content, projectPath);
            }

            // No validator for this file type, return success
            _logger.LogInformation("No semantic validator available for file type: {Extension}", extension);
            return
            [
                new ValidationResult
                {
                    RuleName = "SemanticValidation",
                    IsPassed = true,
                    Severity = ValidationRuleSeverity.Information,
                    Message = $"No semantic validation available for file type: {extension}",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow
                }
            ];
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating semantics of file: {FilePath}", filePath);
            return
            [
                new ValidationResult
                {
                    RuleName = "SemanticValidation",
                    IsPassed = false,
                    Severity = ValidationRuleSeverity.Error,
                    Message = $"Error validating semantics: {ex.Message}",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow,
                    Details = ex.ToString(),
                    Exception = ex
                }
            ];
        }
    }

    /// <summary>
    /// Validates C# semantics
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <param name="content">The file content</param>
    /// <param name="projectPath">The project path</param>
    /// <returns>The list of validation results</returns>
    private async Task<List<ValidationResult>> ValidateCSharpSemanticsAsync(string filePath, string content, string projectPath)
    {
        try
        {
            _logger.LogInformation("Validating C# semantics of file: {FilePath}", filePath);

            var results = new List<ValidationResult>();

            // For a full semantic analysis, we would need to compile the code with references
            // For simplicity, we'll just do some basic semantic checks

            // Parse the code
            var tree = CSharpSyntaxTree.ParseText(content);
            var root = await tree.GetRootAsync();

            // Check for undefined variables
            var undefinedVariables = await Task.Run(() => FindUndefinedVariables(root));
            foreach (var variable in undefinedVariables)
            {
                results.Add(new ValidationResult
                {
                    RuleName = "CSharpUndefinedVariable",
                    IsPassed = false,
                    Severity = ValidationRuleSeverity.Error,
                    Message = $"Undefined variable: {variable.Name} at line {variable.Line}, column {variable.Column}",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow,
                    Details = $"The variable '{variable.Name}' is used but not defined in the current scope",
                    Metadata = new Dictionary<string, string>
                    {
                        { "Line", variable.Line.ToString() },
                        { "Column", variable.Column.ToString() },
                        { "VariableName", variable.Name }
                    }
                });
            }

            // Check for unused variables
            var unusedVariables = await Task.Run(() => FindUnusedVariables(root));
            foreach (var variable in unusedVariables)
            {
                results.Add(new ValidationResult
                {
                    RuleName = "CSharpUnusedVariable",
                    IsPassed = true, // Unused variables don't fail validation
                    Severity = ValidationRuleSeverity.Warning,
                    Message = $"Unused variable: {variable.Name} at line {variable.Line}, column {variable.Column}",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow,
                    Details = $"The variable '{variable.Name}' is defined but not used in the current scope",
                    Metadata = new Dictionary<string, string>
                    {
                        { "Line", variable.Line.ToString() },
                        { "Column", variable.Column.ToString() },
                        { "VariableName", variable.Name }
                    }
                });
            }

            // Check for null reference issues
            var nullReferenceIssues = await Task.Run(() => FindPotentialNullReferenceIssues(root));
            foreach (var issue in nullReferenceIssues)
            {
                results.Add(new ValidationResult
                {
                    RuleName = "CSharpNullReference",
                    IsPassed = true, // Potential null references don't fail validation
                    Severity = ValidationRuleSeverity.Warning,
                    Message = $"Potential null reference: {issue.Expression} at line {issue.Line}, column {issue.Column}",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow,
                    Details = $"The expression '{issue.Expression}' might result in a null reference exception",
                    Metadata = new Dictionary<string, string>
                    {
                        { "Line", issue.Line.ToString() },
                        { "Column", issue.Column.ToString() },
                        { "Expression", issue.Expression }
                    }
                });
            }

            // If no errors, add a success result
            if (!results.Any(r => r.Severity == ValidationRuleSeverity.Error))
            {
                results.Add(new ValidationResult
                {
                    RuleName = "CSharpSemanticValidation",
                    IsPassed = true,
                    Severity = ValidationRuleSeverity.Information,
                    Message = "C# semantic validation passed",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow
                });
            }

            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating C# semantics of file: {FilePath}", filePath);
            return
            [
                new ValidationResult
                {
                    RuleName = "CSharpSemanticValidation",
                    IsPassed = false,
                    Severity = ValidationRuleSeverity.Error,
                    Message = $"Error validating C# semantics: {ex.Message}",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow,
                    Details = ex.ToString(),
                    Exception = ex
                }
            ];
        }
    }

    /// <summary>
    /// Finds undefined variables in C# code
    /// </summary>
    /// <param name="root">The syntax root</param>
    /// <returns>The list of undefined variables</returns>
    private List<(string Name, int Line, int Column)> FindUndefinedVariables(SyntaxNode root)
    {
        var result = new List<(string Name, int Line, int Column)>();
        var declaredVariables = new HashSet<string>();

        // Find all variable declarations
        var variableDeclarations = root.DescendantNodes().OfType<VariableDeclarationSyntax>();
        foreach (var declaration in variableDeclarations)
        {
            foreach (var variable in declaration.Variables)
            {
                declaredVariables.Add(variable.Identifier.Text);
            }
        }

        // Find all parameter declarations
        var parameters = root.DescendantNodes().OfType<ParameterSyntax>();
        foreach (var parameter in parameters)
        {
            declaredVariables.Add(parameter.Identifier.Text);
        }

        // Find all identifiers that might be variables
        var identifiers = root.DescendantNodes().OfType<IdentifierNameSyntax>();
        foreach (var identifier in identifiers)
        {
            // Skip if it's a type name or a member access
            if (identifier.Parent is TypeSyntax || identifier.Parent is MemberAccessExpressionSyntax)
            {
                continue;
            }

            // Skip if it's a method name
            if (identifier.Parent is InvocationExpressionSyntax invocation && invocation.Expression == identifier)
            {
                continue;
            }

            // Skip if it's a declared variable
            if (declaredVariables.Contains(identifier.Identifier.Text))
            {
                continue;
            }

            // Skip common known identifiers
            if (IsKnownIdentifier(identifier.Identifier.Text))
            {
                continue;
            }

            // Get the line and column
            var lineSpan = identifier.GetLocation().GetLineSpan();
            var line = lineSpan.StartLinePosition.Line + 1;
            var column = lineSpan.StartLinePosition.Character + 1;

            result.Add((identifier.Identifier.Text, line, column));
        }

        return result;
    }

    /// <summary>
    /// Finds unused variables in C# code
    /// </summary>
    /// <param name="root">The syntax root</param>
    /// <returns>The list of unused variables</returns>
    private List<(string Name, int Line, int Column)> FindUnusedVariables(SyntaxNode root)
    {
        var result = new List<(string Name, int Line, int Column)>();
        var variableUses = new Dictionary<string, int>();

        // Find all variable declarations and initialize use count to 0
        var variableDeclarations = root.DescendantNodes().OfType<VariableDeclarationSyntax>();
        foreach (var declaration in variableDeclarations)
        {
            foreach (var variable in declaration.Variables)
            {
                variableUses[variable.Identifier.Text] = 0;
            }
        }

        // Find all parameter declarations and initialize use count to 0
        var parameters = root.DescendantNodes().OfType<ParameterSyntax>();
        foreach (var parameter in parameters)
        {
            variableUses[parameter.Identifier.Text] = 0;
        }

        // Find all identifiers that might be variables
        var identifiers = root.DescendantNodes().OfType<IdentifierNameSyntax>();
        foreach (var identifier in identifiers)
        {
            // Skip if it's a type name or a member access
            if (identifier.Parent is TypeSyntax || identifier.Parent is MemberAccessExpressionSyntax)
            {
                continue;
            }

            // Skip if it's a method name
            if (identifier.Parent is InvocationExpressionSyntax invocation && invocation.Expression == identifier)
            {
                continue;
            }

            // Skip if it's a variable declaration
            if (identifier.Parent is VariableDeclaratorSyntax)
            {
                continue;
            }

            // Increment use count if it's a known variable
            if (variableUses.ContainsKey(identifier.Identifier.Text))
            {
                variableUses[identifier.Identifier.Text]++;
            }
        }

        // Find unused variables
        foreach (var declaration in variableDeclarations)
        {
            foreach (var variable in declaration.Variables)
            {
                if (variableUses.TryGetValue(variable.Identifier.Text, out var useCount) && useCount == 0)
                {
                    // Get the line and column
                    var lineSpan = variable.GetLocation().GetLineSpan();
                    var line = lineSpan.StartLinePosition.Line + 1;
                    var column = lineSpan.StartLinePosition.Character + 1;

                    result.Add((variable.Identifier.Text, line, column));
                }
            }
        }

        // Find unused parameters
        foreach (var parameter in parameters)
        {
            if (variableUses.TryGetValue(parameter.Identifier.Text, out var useCount) && useCount == 0)
            {
                // Get the line and column
                var lineSpan = parameter.GetLocation().GetLineSpan();
                var line = lineSpan.StartLinePosition.Line + 1;
                var column = lineSpan.StartLinePosition.Character + 1;

                result.Add((parameter.Identifier.Text, line, column));
            }
        }

        return result;
    }

    /// <summary>
    /// Finds potential null reference issues in C# code
    /// </summary>
    /// <param name="root">The syntax root</param>
    /// <returns>The list of potential null reference issues</returns>
    private List<(string Expression, int Line, int Column)> FindPotentialNullReferenceIssues(SyntaxNode root)
    {
        var result = new List<(string Expression, int Line, int Column)>();

        // Find all member access expressions
        var memberAccesses = root.DescendantNodes().OfType<MemberAccessExpressionSyntax>();
        foreach (var memberAccess in memberAccesses)
        {
            // Check if the expression is a potential null reference
            if (IsPotentialNullReference(memberAccess.Expression))
            {
                // Get the line and column
                var lineSpan = memberAccess.GetLocation().GetLineSpan();
                var line = lineSpan.StartLinePosition.Line + 1;
                var column = lineSpan.StartLinePosition.Character + 1;

                result.Add((memberAccess.ToString(), line, column));
            }
        }

        // Find all element access expressions
        var elementAccesses = root.DescendantNodes().OfType<ElementAccessExpressionSyntax>();
        foreach (var elementAccess in elementAccesses)
        {
            // Check if the expression is a potential null reference
            if (IsPotentialNullReference(elementAccess.Expression))
            {
                // Get the line and column
                var lineSpan = elementAccess.GetLocation().GetLineSpan();
                var line = lineSpan.StartLinePosition.Line + 1;
                var column = lineSpan.StartLinePosition.Character + 1;

                result.Add((elementAccess.ToString(), line, column));
            }
        }

        return result;
    }

    /// <summary>
    /// Checks if an expression is a potential null reference
    /// </summary>
    /// <param name="expression">The expression</param>
    /// <returns>True if the expression is a potential null reference, false otherwise</returns>
    private bool IsPotentialNullReference(ExpressionSyntax expression)
    {
        // Check if the expression is a method invocation
        if (expression is InvocationExpressionSyntax)
        {
            return true;
        }

        // Check if the expression is a member access
        if (expression is MemberAccessExpressionSyntax)
        {
            return true;
        }

        // Check if the expression is a conditional access
        if (expression is ConditionalAccessExpressionSyntax)
        {
            return false; // Conditional access already handles null
        }

        // Check if the expression is a null literal
        if (expression is LiteralExpressionSyntax literal && literal.Token.ValueText == "null")
        {
            return true;
        }

        // Check if the expression is a default expression
        if (expression is DefaultExpressionSyntax)
        {
            return true;
        }

        return false;
    }

    /// <summary>
    /// Checks if an identifier is a known identifier
    /// </summary>
    /// <param name="identifier">The identifier</param>
    /// <returns>True if the identifier is known, false otherwise</returns>
    private bool IsKnownIdentifier(string identifier)
    {
        // Common C# keywords
        var keywords = new HashSet<string>
        {
            "this", "base", "null", "true", "false", "var", "dynamic", "object", "string", "int", "long", "float", "double", "decimal",
            "bool", "byte", "char", "short", "uint", "ulong", "ushort", "sbyte", "void", "new", "typeof", "sizeof", "nameof", "is", "as",
            "class", "struct", "interface", "enum", "delegate", "event", "namespace", "using", "public", "private", "protected", "internal",
            "static", "readonly", "const", "virtual", "abstract", "override", "sealed", "extern", "unsafe", "volatile", "async", "await",
            "if", "else", "switch", "case", "default", "for", "foreach", "while", "do", "break", "continue", "return", "yield", "goto",
            "try", "catch", "finally", "throw", "lock", "checked", "unchecked", "fixed", "stackalloc", "value", "ref", "out", "in", "params"
        };

        return keywords.Contains(identifier);
    }

    /// <summary>
    /// Validates F# semantics
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <param name="content">The file content</param>
    /// <param name="projectPath">The project path</param>
    /// <returns>The list of validation results</returns>
    private async Task<List<ValidationResult>> ValidateFSharpSemanticsAsync(string filePath, string content, string projectPath)
    {
        try
        {
            _logger.LogInformation("Validating F# semantics of file: {FilePath}", filePath);

            // For F#, we'll use a simpler approach since we don't have direct access to the F# compiler
            // In a real implementation, you would use the F# Compiler Services
            var results = new List<ValidationResult>();

            // Check for basic semantic issues
            var issues = await Task.Run(() => CheckFSharpBasicSemantics(content));

            if (issues.Count > 0)
            {
                foreach (var issue in issues)
                {
                    results.Add(new ValidationResult
                    {
                        RuleName = "FSharpSemanticValidation",
                        IsPassed = false,
                        Severity = ValidationRuleSeverity.Error,
                        Message = issue.Message,
                        Target = filePath,
                        Timestamp = DateTime.UtcNow,
                        Details = issue.Details,
                        Metadata = new Dictionary<string, string>
                        {
                            { "Line", issue.Line.ToString() },
                            { "Column", issue.Column.ToString() }
                        }
                    });
                }
            }
            else
            {
                // No semantic issues found
                results.Add(new ValidationResult
                {
                    RuleName = "FSharpSemanticValidation",
                    IsPassed = true,
                    Severity = ValidationRuleSeverity.Information,
                    Message = "F# semantic validation passed",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow
                });
            }

            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating F# semantics of file: {FilePath}", filePath);
            return
            [
                new ValidationResult
                {
                    RuleName = "FSharpSemanticValidation",
                    IsPassed = false,
                    Severity = ValidationRuleSeverity.Error,
                    Message = $"Error validating F# semantics: {ex.Message}",
                    Target = filePath,
                    Timestamp = DateTime.UtcNow,
                    Details = ex.ToString(),
                    Exception = ex
                }
            ];
        }
    }

    /// <summary>
    /// Checks F# code for basic semantic issues
    /// </summary>
    /// <param name="content">The file content</param>
    /// <returns>The list of semantic issues</returns>
    private List<(int Line, int Column, string Message, string Details)> CheckFSharpBasicSemantics(string content)
    {
        var issues = new List<(int Line, int Column, string Message, string Details)>();
        var lines = content.Split('\n');

        // Check for type mismatches in function calls
        // This is a simplified check and would need to be more sophisticated in a real implementation
        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            
            // Check for potential type mismatches in function calls
            if (line.Contains("(") && line.Contains(")"))
            {
                var functionCallMatch = System.Text.RegularExpressions.Regex.Match(line, @"(\w+)\s*\(([^)]*)\)");
                if (functionCallMatch.Success)
                {
                    var functionName = functionCallMatch.Groups[1].Value;
                    var arguments = functionCallMatch.Groups[2].Value;
                    
                    // Check for common type mismatch patterns
                    if (functionName == "int" && arguments.Contains("."))
                    {
                        issues.Add((i + 1, functionCallMatch.Index + 1, "Potential type mismatch", $"Converting floating-point value to int may lose precision: {functionCallMatch.Value}"));
                    }
                    else if (functionName == "string" && arguments.Contains("null"))
                    {
                        issues.Add((i + 1, functionCallMatch.Index + 1, "Potential null reference", $"Converting null to string may cause issues: {functionCallMatch.Value}"));
                    }
                }
            }
            
            // Check for potential null references
            if (line.Contains("null"))
            {
                var nullMatch = System.Text.RegularExpressions.Regex.Match(line, @"(\w+)\s*\.\s*(\w+)");
                if (nullMatch.Success)
                {
                    var objectName = nullMatch.Groups[1].Value;
                    if (line.Contains($"{objectName} = null") || line.Contains($"{objectName} <- null"))
                    {
                        issues.Add((i + 1, nullMatch.Index + 1, "Potential null reference", $"Object {objectName} is set to null but later accessed: {nullMatch.Value}"));
                    }
                }
            }
            
            // Check for unused let bindings
            if (line.Trim().StartsWith("let ") && !line.Contains("="))
            {
                var letMatch = System.Text.RegularExpressions.Regex.Match(line, @"let\s+(\w+)");
                if (letMatch.Success)
                {
                    var variableName = letMatch.Groups[1].Value;
                    bool isUsed = false;
                    
                    // Check if the variable is used in subsequent lines
                    for (int j = i + 1; j < lines.Length; j++)
                    {
                        if (lines[j].Contains(variableName))
                        {
                            isUsed = true;
                            break;
                        }
                    }
                    
                    if (!isUsed)
                    {
                        issues.Add((i + 1, letMatch.Index + 1, "Unused let binding", $"The let binding '{variableName}' is not used"));
                    }
                }
            }
        }

        return issues;
    }

    /// <summary>
    /// Gets the supported file extensions
    /// </summary>
    /// <returns>The list of supported file extensions</returns>
    public List<string> GetSupportedFileExtensions()
    {
        return _validators.Keys.ToList();
    }
}
