using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Extracts code structures from F# code
/// </summary>
public class FSharpStructureExtractor(ILogger<FSharpStructureExtractor> logger) : ICodeStructureExtractor
{
    private readonly ILogger<FSharpStructureExtractor> _logger = logger;

    /// <inheritdoc/>
    public string Language => "fsharp";

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

            // Extract modules
            var moduleRegex = new Regex(@"module\s+([a-zA-Z0-9_.]+)(?:\s*=)?", RegexOptions.Compiled);
            var moduleMatches = moduleRegex.Matches(content);
            foreach (Match match in moduleMatches)
            {
                if (match.Groups.Count > 1)
                {
                    var moduleName = match.Groups[1].Value;
                    structures.Add(new CodeStructure
                    {
                        Type = StructureType.Module,
                        Name = moduleName,
                        Location = new CodeLocation
                        {
                            StartLine = GetLineNumber(content, match.Index),
                            Namespace = moduleName
                        }
                    });
                }
            }

            // Extract types (records, discriminated unions, classes)
            var typeRegex = new Regex(@"type\s+([a-zA-Z0-9_]+)(?:<[^>]+>)?(?:\s*=|\s*\(|\s*:)", RegexOptions.Compiled);
            var typeMatches = typeRegex.Matches(content);
            foreach (Match match in typeMatches)
            {
                if (match.Groups.Count > 1)
                {
                    var typeName = match.Groups[1].Value;
                    var moduleName = GetModuleForPosition(structures, match.Index, content);

                    // Determine type kind (record, union, class)
                    var typeKind = DetermineTypeKind(content, match.Index);

                    structures.Add(new CodeStructure
                    {
                        Type = typeKind,
                        Name = typeName,
                        Location = new CodeLocation
                        {
                            StartLine = GetLineNumber(content, match.Index),
                            Namespace = moduleName,
                            ClassName = typeName
                        },
                        Properties = new Dictionary<string, string>
                        {
                            { "TypeKind", typeKind.ToString() }
                        }
                    });
                }
            }

            // Extract functions
            var functionRegex = new Regex(@"let\s+(rec\s+)?([a-zA-Z0-9_]+)(?:\s+[a-zA-Z0-9_]+)*\s*=", RegexOptions.Compiled);
            var functionMatches = functionRegex.Matches(content);
            foreach (Match match in functionMatches)
            {
                if (match.Groups.Count > 2)
                {
                    var functionName = match.Groups[2].Value;
                    var moduleName = GetModuleForPosition(structures, match.Index, content);
                    var isRecursive = !string.IsNullOrEmpty(match.Groups[1].Value);

                    structures.Add(new CodeStructure
                    {
                        Type = StructureType.Function,
                        Name = functionName,
                        Location = new CodeLocation
                        {
                            StartLine = GetLineNumber(content, match.Index),
                            Namespace = moduleName
                        },
                        Properties = new Dictionary<string, string>
                        {
                            { "IsRecursive", isRecursive.ToString() }
                        }
                    });
                }
            }

            // Extract members
            var memberRegex = new Regex(@"member\s+(?:this|self|_)\.([a-zA-Z0-9_]+)", RegexOptions.Compiled);
            var memberMatches = memberRegex.Matches(content);
            foreach (Match match in memberMatches)
            {
                if (match.Groups.Count > 1)
                {
                    var memberName = match.Groups[1].Value;
                    var moduleName = GetModuleForPosition(structures, match.Index, content);
                    var className = GetClassForPosition(structures, match.Index, content);

                    structures.Add(new CodeStructure
                    {
                        Type = StructureType.Method,
                        Name = memberName,
                        ParentName = className,
                        Location = new CodeLocation
                        {
                            StartLine = GetLineNumber(content, match.Index),
                            Namespace = moduleName,
                            ClassName = className
                        }
                    });
                }
            }

            // Extract active patterns
            var activePatternRegex = new Regex(@"\(\|([^|]+)\|\)", RegexOptions.Compiled);
            var activePatternMatches = activePatternRegex.Matches(content);
            foreach (Match match in activePatternMatches)
            {
                if (match.Groups.Count > 1)
                {
                    var patternName = match.Groups[1].Value;
                    var moduleName = GetModuleForPosition(structures, match.Index, content);

                    structures.Add(new CodeStructure
                    {
                        Type = StructureType.Function,
                        Name = patternName,
                        Location = new CodeLocation
                        {
                            StartLine = GetLineNumber(content, match.Index),
                            Namespace = moduleName
                        },
                        Properties = new Dictionary<string, string>
                        {
                            { "IsActivePattern", "true" }
                        }
                    });
                }
            }

            // Calculate structure sizes and update end lines
            CalculateStructureSizes(structures, content);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting structures from F# code");
        }

        return structures;
    }

    /// <inheritdoc/>
    public string GetNamespaceForPosition(List<CodeStructure> structures, int position, string content)
    {
        // In F#, modules are equivalent to namespaces
        var lineNumber = GetLineNumber(content, position);
        var moduleStructures = structures
            .Where(s => s.Type == StructureType.Module)
            .ToList();

        foreach (var module in moduleStructures)
        {
            if (module.Location.StartLine <= lineNumber &&
                (module.Location.EndLine == 0 || module.Location.EndLine >= lineNumber))
            {
                return module.Name;
            }
        }

        return string.Empty;
    }

    /// <inheritdoc/>
    public string GetClassForPosition(List<CodeStructure> structures, int position, string content)
    {
        var lineNumber = GetLineNumber(content, position);
        var classStructures = structures
            .Where(s => s.Type == StructureType.Class || s.Type == StructureType.Record || s.Type == StructureType.Union)
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
        // F# doesn't use braces for scope, so we need to use indentation to determine structure boundaries
        var lines = content.Split('\n');

        // Sort structures by start line
        var sortedStructures = structures.OrderBy(s => s.Location.StartLine).ToList();

        for (int i = 0; i < sortedStructures.Count; i++)
        {
            var structure = sortedStructures[i];
            var startLine = structure.Location.StartLine;

            if (startLine <= 0 || startLine > lines.Length)
            {
                continue;
            }

            // Get the indentation of the structure definition
            var structureIndent = GetIndentation(lines[startLine - 1]);

            // Find the end line by looking for the next line with the same or less indentation
            var endLine = startLine;
            for (int j = startLine; j < lines.Length; j++)
            {
                var line = lines[j];

                // Skip empty lines
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                var indent = GetIndentation(line);

                // If we find a line with the same or less indentation, we've reached the end of the structure
                if (indent <= structureIndent && j > startLine)
                {
                    endLine = j - 1;
                    break;
                }

                // If we reach the end of the file, use that as the end line
                if (j == lines.Length - 1)
                {
                    endLine = j;
                }
            }

            // Update the structure
            structure.Location.EndLine = endLine;
            structure.Size = endLine - startLine + 1;

            // Check if there's a next structure that starts before this one ends
            if (i < sortedStructures.Count - 1)
            {
                var nextStructure = sortedStructures[i + 1];
                if (nextStructure.Location.StartLine <= structure.Location.EndLine)
                {
                    // Adjust the end line to be just before the next structure
                    structure.Location.EndLine = nextStructure.Location.StartLine - 1;
                    structure.Size = structure.Location.EndLine - structure.Location.StartLine + 1;
                }
            }

            // Ensure size is at least 1
            if (structure.Size <= 0)
            {
                structure.Size = 1;
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
    /// Gets the module for a position in the content
    /// </summary>
    private string GetModuleForPosition(List<CodeStructure> structures, int position, string content)
    {
        var lineNumber = GetLineNumber(content, position);
        var moduleStructures = structures
            .Where(s => s.Type == StructureType.Module)
            .ToList();

        foreach (var module in moduleStructures)
        {
            if (module.Location.StartLine <= lineNumber &&
                (module.Location.EndLine == 0 || module.Location.EndLine >= lineNumber))
            {
                return module.Name;
            }
        }

        return string.Empty;
    }

    /// <summary>
    /// Determines the type kind (record, union, class) based on the content
    /// </summary>
    private StructureType DetermineTypeKind(string content, int position)
    {
        // Get the content after the type definition
        var afterType = content.Substring(position);

        // Check for record type
        if (afterType.Contains("{") && afterType.Contains(":"))
        {
            return StructureType.Record;
        }

        // Check for discriminated union
        if (afterType.Contains("|"))
        {
            return StructureType.Union;
        }

        // Check for class
        if (afterType.Contains("class") || afterType.Contains("new()"))
        {
            return StructureType.Class;
        }

        // Default to class
        return StructureType.Class;
    }

    /// <summary>
    /// Gets the indentation level of a line
    /// </summary>
    private static int GetIndentation(string line)
    {
        int indent = 0;
        foreach (char c in line)
        {
            if (c == ' ')
            {
                indent++;
            }
            else if (c == '\t')
            {
                indent += 4; // Assuming tabs are 4 spaces
            }
            else
            {
                break;
            }
        }
        return indent;
    }
}