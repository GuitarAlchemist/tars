using System.Text;
using System.Text.RegularExpressions;

namespace TarsCli.Services;

/// <summary>
/// Engine for parsing and executing metascripts
/// </summary>
public class MetascriptEngine
{
    private readonly ILogger<MetascriptEngine> _logger;
    private readonly DynamicFSharpCompilerService _fsharpCompiler;

    public MetascriptEngine(ILogger<MetascriptEngine> logger, DynamicFSharpCompilerService fsharpCompiler)
    {
        _logger = logger;
        _fsharpCompiler = fsharpCompiler;
    }

    /// <summary>
    /// Represents a transformation rule in a metascript
    /// </summary>
    public class TransformationRule
    {
        public string Name { get; set; }
        public string Pattern { get; set; }
        public string Replacement { get; set; }
        public List<string> RequiredNamespaces { get; set; } = [];
        public string Description { get; set; }
    }

    /// <summary>
    /// Parses a metascript file into transformation rules
    /// </summary>
    /// <param name="metascriptContent">The content of the metascript file</param>
    /// <returns>List of transformation rules</returns>
    public List<TransformationRule> ParseMetascript(string metascriptContent)
    {
        try
        {
            _logger.LogInformation("Parsing metascript");
                
            var rules = new List<TransformationRule>();
                
            // Regular expression to match rule definitions
            // Format: rule Name { match: "pattern" replace: "replacement" requires: "namespace1; namespace2" description: "description" }
            var rulePattern = @"rule\s+(\w+)\s*{\s*match:\s*""(.*?)""\s*replace:\s*""(.*?)""\s*(?:requires:\s*""(.*?)""\s*)?(?:description:\s*""(.*?)""\s*)?}";
                
            var matches = Regex.Matches(metascriptContent, rulePattern, RegexOptions.Singleline);
                
            foreach (Match match in matches)
            {
                var rule = new TransformationRule
                {
                    Name = match.Groups[1].Value,
                    Pattern = match.Groups[2].Value,
                    Replacement = match.Groups[3].Value
                };
                    
                // Parse required namespaces
                if (match.Groups[4].Success)
                {
                    rule.RequiredNamespaces = match.Groups[4].Value
                        .Split([';'], StringSplitOptions.RemoveEmptyEntries)
                        .Select(ns => ns.Trim())
                        .ToList();
                }
                    
                // Parse description
                if (match.Groups[5].Success)
                {
                    rule.Description = match.Groups[5].Value;
                }
                else
                {
                    rule.Description = $"Apply {rule.Name} transformation";
                }
                    
                rules.Add(rule);
            }
                
            _logger.LogInformation($"Parsed {rules.Count} rules from metascript");
            return rules;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error parsing metascript");
            throw;
        }
    }

    /// <summary>
    /// Generates F# code that implements the transformation rules
    /// </summary>
    /// <param name="rules">The transformation rules to implement</param>
    /// <returns>F# code as a string</returns>
    public string GenerateFSharpCode(List<TransformationRule> rules)
    {
        try
        {
            _logger.LogInformation("Generating F# code for transformation rules");
                
            var sb = new StringBuilder();
                
            // Add module declaration and imports
            sb.AppendLine("namespace TarsEngineFSharp");
            sb.AppendLine();
            sb.AppendLine("module DynamicTransformations =");
            sb.AppendLine("    open System");
            sb.AppendLine("    open System.Text.RegularExpressions");
                
            // Add any required namespaces
            var allNamespaces = rules
                .SelectMany(r => r.RequiredNamespaces)
                .Distinct()
                .OrderBy(ns => ns);
                
            foreach (var ns in allNamespaces)
            {
                sb.AppendLine($"    open {ns}");
            }
                
            sb.AppendLine();
                
            // Define the transformation function for each rule
            for (int i = 0; i < rules.Count; i++)
            {
                var rule = rules[i];
                    
                sb.AppendLine($"    // {rule.Description}");
                sb.AppendLine($"    let applyRule{i} (code: string) =");
                    
                // Convert the pattern to a regex pattern
                var regexPattern = ConvertPatternToRegex(rule.Pattern);
                var replacementPattern = ConvertReplacementToRegex(rule.Replacement);
                    
                sb.AppendLine($"        let pattern = @\"{regexPattern}\"");
                sb.AppendLine($"        let replacement = \"{replacementPattern}\"");
                sb.AppendLine($"        Regex.Replace(code, pattern, replacement, RegexOptions.Multiline)");
                sb.AppendLine();
            }
                
            // Define the main transformation function that applies all rules
            sb.AppendLine("    // Apply all transformation rules");
            sb.AppendLine("    let applyAllRules (code: string) =");
                
            // Chain all the rule applications
            if (rules.Count > 0)
            {
                sb.Append("        code");
                    
                for (int i = 0; i < rules.Count; i++)
                {
                    sb.AppendLine();
                    sb.Append($"        |> applyRule{i}");
                }
                    
                sb.AppendLine();
            }
            else
            {
                sb.AppendLine("        code // No rules to apply");
            }
                
            return sb.ToString();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating F# code");
            throw;
        }
    }

    /// <summary>
    /// Applies transformation rules to code
    /// </summary>
    /// <param name="code">The code to transform</param>
    /// <param name="rules">The transformation rules to apply</param>
    /// <returns>The transformed code</returns>
    public async Task<string> ApplyTransformationsAsync(string code, List<TransformationRule> rules)
    {
        try
        {
            _logger.LogInformation("Applying transformations to code");
                
            // Generate F# code for the transformations
            var fsharpCode = GenerateFSharpCode(rules);
                
            // Compile the F# code
            var assembly = await _fsharpCompiler.CompileFSharpCodeAsync(fsharpCode, "DynamicTransformations");
                
            // Get the module type
            var moduleType = _fsharpCompiler.GetTypeFromAssembly(assembly, "TarsEngineFSharp.DynamicTransformations");
                
            // Invoke the applyAllRules function
            var transformedCode = _fsharpCompiler.InvokeStaticMethod(moduleType, "applyAllRules", code);
                
            return transformedCode as string;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error applying transformations");
            throw;
        }
    }

    /// <summary>
    /// Converts a metascript pattern to a regex pattern
    /// </summary>
    private string ConvertPatternToRegex(string pattern)
    {
        // This is a simplified conversion - a real implementation would be more sophisticated
        return pattern
            .Replace("\\", "\\\\")  // Escape backslashes
            .Replace("\"", "\\\"")  // Escape quotes
            .Replace("$", @"(\w+)")  // Replace variables with capture groups
            .Replace("{", @"\{")    // Escape braces
            .Replace("}", @"\}")
            .Replace("(", @"\(")    // Escape parentheses
            .Replace(")", @"\)")
            .Replace("[", @"\[")    // Escape brackets
            .Replace("]", @"\]")
            .Replace(".", @"\.");   // Escape dots
    }

    /// <summary>
    /// Converts a metascript replacement to a regex replacement
    /// </summary>
    private string ConvertReplacementToRegex(string replacement)
    {
        // This is a simplified conversion - a real implementation would be more sophisticated
        return replacement
            .Replace("\\", "\\\\")  // Escape backslashes
            .Replace("\"", "\\\"")  // Escape quotes
            .Replace("$1", "$1")    // Keep capture group references
            .Replace("$2", "$2")
            .Replace("$3", "$3")
            .Replace("$4", "$4")
            .Replace("$5", "$5")
            .Replace("$", "$$");    // Escape other dollar signs
    }
}