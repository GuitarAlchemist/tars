using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Extracts code structures from C# code
/// </summary>
public class CSharpStructureExtractor(ILogger<CSharpStructureExtractor> logger) : ICodeStructureExtractor
{
    private readonly ILogger<CSharpStructureExtractor> _logger = logger;

    /// <inheritdoc/>
    public string Language => "csharp";

    /// <inheritdoc/>
    public List<CodeStructure> ExtractStructures(string content)
    {
        var structures = new List<CodeStructure>();

        try
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                return structures;
            }

            // Extract namespaces
            var namespaceRegex = new Regex(@"namespace\s+([a-zA-Z0-9_.]+)\s*{", RegexOptions.Compiled);
            var namespaceMatches = namespaceRegex.Matches(content);
            foreach (Match match in namespaceMatches)
            {
                if (match.Groups.Count > 1)
                {
                    var namespaceName = match.Groups[1].Value;
                    structures.Add(new CodeStructure
                    {
                        Type = StructureType.Namespace,
                        Name = namespaceName,
                        Location = new CodeLocation
                        {
                            StartLine = GetLineNumber(content, match.Index),
                            Namespace = namespaceName
                        }
                    });
                }
            }

            // Extract classes
            var classRegex = new Regex(@"(public|private|protected|internal)?\s*(static|abstract|sealed)?\s*class\s+([a-zA-Z0-9_]+)(?:<[^>]+>)?(?:\s*:\s*[^{]+)?", RegexOptions.Compiled);
            var classMatches = classRegex.Matches(content);
            foreach (Match match in classMatches)
            {
                if (match.Groups.Count > 3)
                {
                    var className = match.Groups[3].Value;
                    var namespaceName = GetNamespaceForPosition(structures, match.Index, content);
                    structures.Add(new CodeStructure
                    {
                        Type = StructureType.Class,
                        Name = className,
                        Location = new CodeLocation
                        {
                            StartLine = GetLineNumber(content, match.Index),
                            Namespace = namespaceName,
                            ClassName = className
                        }
                    });
                }
            }

            // Extract interfaces
            var interfaceRegex = new Regex(@"(public|private|protected|internal)?\s*interface\s+([a-zA-Z0-9_]+)(?:<[^>]+>)?(?:\s*:\s*[^{]+)?", RegexOptions.Compiled);
            var interfaceMatches = interfaceRegex.Matches(content);
            foreach (Match match in interfaceMatches)
            {
                if (match.Groups.Count > 2)
                {
                    var interfaceName = match.Groups[2].Value;
                    var namespaceName = GetNamespaceForPosition(structures, match.Index, content);
                    structures.Add(new CodeStructure
                    {
                        Type = StructureType.Interface,
                        Name = interfaceName,
                        Location = new CodeLocation
                        {
                            StartLine = GetLineNumber(content, match.Index),
                            Namespace = namespaceName,
                            ClassName = interfaceName
                        }
                    });
                }
            }

            // Extract methods
            var methodRegex = new Regex(@"(public|private|protected|internal)?\s*(static|virtual|abstract|override|async)?\s*([a-zA-Z0-9_<>]+)\s+([a-zA-Z0-9_]+)\s*\(", RegexOptions.Compiled);
            var methodMatches = methodRegex.Matches(content);
            foreach (Match match in methodMatches)
            {
                if (match.Groups.Count > 4)
                {
                    var methodName = match.Groups[4].Value;
                    var returnType = match.Groups[3].Value;
                    var className = GetClassForPosition(structures, match.Index, content);
                    var namespaceName = GetNamespaceForPosition(structures, match.Index, content);
                    structures.Add(new CodeStructure
                    {
                        Type = StructureType.Method,
                        Name = methodName,
                        ParentName = className,
                        Location = new CodeLocation
                        {
                            StartLine = GetLineNumber(content, match.Index),
                            Namespace = namespaceName,
                            ClassName = className
                        },
                        Properties = new Dictionary<string, string>
                        {
                            { "ReturnType", returnType }
                        }
                    });
                }
            }

            // Extract properties
            var propertyRegex = new Regex(@"(public|private|protected|internal)?\s*(static|virtual|abstract|override)?\s*([a-zA-Z0-9_<>]+)\s+([a-zA-Z0-9_]+)\s*{\s*(get|set)?", RegexOptions.Compiled);
            var propertyMatches = propertyRegex.Matches(content);
            foreach (Match match in propertyMatches)
            {
                if (match.Groups.Count > 4)
                {
                    var propertyName = match.Groups[4].Value;
                    var propertyType = match.Groups[3].Value;
                    var className = GetClassForPosition(structures, match.Index, content);
                    var namespaceName = GetNamespaceForPosition(structures, match.Index, content);
                    structures.Add(new CodeStructure
                    {
                        Type = StructureType.Property,
                        Name = propertyName,
                        ParentName = className,
                        Location = new CodeLocation
                        {
                            StartLine = GetLineNumber(content, match.Index),
                            Namespace = namespaceName,
                            ClassName = className
                        },
                        Properties = new Dictionary<string, string>
                        {
                            { "PropertyType", propertyType }
                        }
                    });
                }
            }

            // Calculate structure sizes and update end lines
            CalculateStructureSizes(structures, content);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting structures from C# code");
        }

        return structures;
    }

    /// <inheritdoc/>
    public string GetNamespaceForPosition(List<CodeStructure> structures, int position, string content)
    {
        var lineNumber = GetLineNumber(content, position);
        var namespaceStructures = structures
            .Where(s => s.Type == StructureType.Namespace)
            .ToList();

        foreach (var ns in namespaceStructures)
        {
            if (ns.Location.StartLine <= lineNumber && 
                (ns.Location.EndLine == 0 || ns.Location.EndLine >= lineNumber))
            {
                return ns.Name;
            }
        }

        return string.Empty;
    }

    /// <inheritdoc/>
    public string GetClassForPosition(List<CodeStructure> structures, int position, string content)
    {
        var lineNumber = GetLineNumber(content, position);
        var classStructures = structures
            .Where(s => s.Type == StructureType.Class)
            .ToList();

        foreach (var cls in classStructures)
        {
            if (cls.Location.StartLine <= lineNumber && 
                (cls.Location.EndLine == 0 || cls.Location.EndLine >= lineNumber))
            {
                return cls.Name;
            }
        }

        return string.Empty;
    }

    /// <inheritdoc/>
    public void CalculateStructureSizes(List<CodeStructure> structures, string content)
    {
        // Find matching braces to determine end positions
        foreach (var structure in structures)
        {
            if (structure.Type == StructureType.Namespace || 
                structure.Type == StructureType.Class || 
                structure.Type == StructureType.Interface)
            {
                // Find the opening brace position
                var startLine = structure.Location.StartLine;
                var startLinePos = GetPositionForLine(content, startLine);
                var openBracePos = content.IndexOf('{', startLinePos);
                    
                if (openBracePos != -1)
                {
                    // Find the matching closing brace
                    var endPos = FindMatchingBrace(content, openBracePos);
                    if (endPos != -1)
                    {
                        var endLine = GetLineNumber(content, endPos);
                        structure.Location.EndLine = endLine;
                        structure.Size = endLine - startLine + 1;
                    }
                }
            }
            else if (structure.Type == StructureType.Method)
            {
                // Find the opening brace position
                var startLine = structure.Location.StartLine;
                var startLinePos = GetPositionForLine(content, startLine);
                var openParenPos = content.IndexOf('(', startLinePos);
                    
                if (openParenPos != -1)
                {
                    var closeParenPos = FindMatchingParenthesis(content, openParenPos);
                    if (closeParenPos != -1)
                    {
                        var openBracePos = content.IndexOf('{', closeParenPos);
                        if (openBracePos != -1)
                        {
                            // Find the matching closing brace
                            var endPos = FindMatchingBrace(content, openBracePos);
                            if (endPos != -1)
                            {
                                var endLine = GetLineNumber(content, endPos);
                                structure.Location.EndLine = endLine;
                                structure.Size = endLine - startLine + 1;
                            }
                        }
                    }
                }
            }
            else if (structure.Type == StructureType.Property)
            {
                // Find the opening brace position
                var startLine = structure.Location.StartLine;
                var startLinePos = GetPositionForLine(content, startLine);
                var openBracePos = content.IndexOf('{', startLinePos);
                    
                if (openBracePos != -1)
                {
                    // Find the matching closing brace
                    var endPos = FindMatchingBrace(content, openBracePos);
                    if (endPos != -1)
                    {
                        var endLine = GetLineNumber(content, endPos);
                        structure.Location.EndLine = endLine;
                        structure.Size = endLine - startLine + 1;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Gets the line number for a position in the content
    /// </summary>
    public int GetLineNumber(string content, int position)
    {
        if (string.IsNullOrEmpty(content) || position < 0 || position >= content.Length)
        {
            return 0;
        }

        // Count newlines before the position
        return content[..position].Count(c => c == '\n') + 1;
    }

    /// <summary>
    /// Gets the position for a line number in the content
    /// </summary>
    private int GetPositionForLine(string content, int lineNumber)
    {
        if (string.IsNullOrEmpty(content) || lineNumber <= 0)
        {
            return 0;
        }

        int currentLine = 1;
        int position = 0;

        while (currentLine < lineNumber && position < content.Length)
        {
            if (content[position] == '\n')
            {
                currentLine++;
            }
            position++;
        }

        return position;
    }

    /// <summary>
    /// Finds the position of the matching closing brace
    /// </summary>
    private static int FindMatchingBrace(string content, int openBracePos)
    {
        int braceCount = 1;
        for (int i = openBracePos + 1; i < content.Length; i++)
        {
            if (content[i] == '{')
            {
                braceCount++;
            }
            else if (content[i] == '}')
            {
                braceCount--;
                if (braceCount == 0)
                {
                    return i;
                }
            }
        }
        return -1;
    }

    /// <summary>
    /// Finds the position of the matching closing parenthesis
    /// </summary>
    private static int FindMatchingParenthesis(string content, int openParenPos)
    {
        int parenCount = 1;
        for (int i = openParenPos + 1; i < content.Length; i++)
        {
            if (content[i] == '(')
            {
                parenCount++;
            }
            else if (content[i] == ')')
            {
                parenCount--;
                if (parenCount == 0)
                {
                    return i;
                }
            }
        }
        return -1;
    }
}