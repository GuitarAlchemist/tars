namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for code structure extractor factory
/// </summary>
public interface ICodeStructureExtractorFactory
{
    /// <summary>
    /// Gets a code structure extractor for a specific language
    /// </summary>
    /// <param name="language">Programming language</param>
    /// <returns>Code structure extractor</returns>
    ICodeStructureExtractor GetExtractor(string language);
}
