using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Generic implementation of the code structure extractor
/// </summary>
public class GenericStructureExtractor : ICodeStructureExtractor
{
    private readonly ILogger<GenericStructureExtractor> _logger;

    /// <summary>
    /// Initializes a new instance of the <see cref="GenericStructureExtractor"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    public GenericStructureExtractor(ILogger<GenericStructureExtractor> logger)
    {
        _logger = logger;
    }

    /// <inheritdoc/>
    public string Language => "Generic";

    /// <inheritdoc/>
    public List<CodeStructure> ExtractStructures(string content)
    {
        _logger.LogInformation("Extracting structures with generic extractor");
        return new List<CodeStructure>();
    }

    /// <inheritdoc/>
    public string GetClassForPosition(List<CodeStructure> structures, int position, string content)
    {
        return string.Empty;
    }

    /// <inheritdoc/>
    public string GetMethodForPosition(List<CodeStructure> structures, int position, string content)
    {
        return string.Empty;
    }

    /// <inheritdoc/>
    public string GetNamespaceForPosition(List<CodeStructure> structures, int position, string content)
    {
        return string.Empty;
    }

    /// <inheritdoc/>
    public void CalculateStructureSizes(List<CodeStructure> structures, string content)
    {
        // Do nothing
    }
}
