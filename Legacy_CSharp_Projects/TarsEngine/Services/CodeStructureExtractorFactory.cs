using Microsoft.Extensions.Logging;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Implementation of the code structure extractor factory
/// </summary>
public class CodeStructureExtractorFactory : ICodeStructureExtractorFactory
{
    private readonly ILogger<CodeStructureExtractorFactory> _logger;
    private readonly CSharpStructureExtractor _csharpExtractor;
    private readonly FSharpStructureExtractor _fsharpExtractor;
    private readonly GenericStructureExtractor _genericExtractor;

    /// <summary>
    /// Initializes a new instance of the <see cref="CodeStructureExtractorFactory"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    /// <param name="csharpExtractor">C# structure extractor</param>
    /// <param name="fsharpExtractor">F# structure extractor</param>
    /// <param name="genericExtractor">Generic structure extractor</param>
    public CodeStructureExtractorFactory(
        ILogger<CodeStructureExtractorFactory> logger,
        CSharpStructureExtractor csharpExtractor,
        FSharpStructureExtractor fsharpExtractor,
        GenericStructureExtractor genericExtractor)
    {
        _logger = logger;
        _csharpExtractor = csharpExtractor;
        _fsharpExtractor = fsharpExtractor;
        _genericExtractor = genericExtractor;
    }

    /// <inheritdoc/>
    public ICodeStructureExtractor GetExtractor(string language)
    {
        _logger.LogInformation($"Getting code structure extractor for {language}");
        
        return language.ToLowerInvariant() switch
        {
            "c#" or "csharp" => _csharpExtractor,
            "f#" or "fsharp" => _fsharpExtractor,
            _ => _genericExtractor
        };
    }
}
