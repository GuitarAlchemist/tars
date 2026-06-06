namespace TarsEngine.Services.Adapters;

/// <summary>
/// Converter for ProgrammingLanguage enum and string values
/// </summary>
public static class ProgrammingLanguageConverter
{
    /// <summary>
    /// Converts a ProgrammingLanguage enum value to a string language name
    /// </summary>
    /// <param name="language">The ProgrammingLanguage enum value</param>
    /// <returns>The corresponding language name as a string</returns>
    public static string ToString(ProgrammingLanguage language)
    {
        return language.ToString();
    }
}