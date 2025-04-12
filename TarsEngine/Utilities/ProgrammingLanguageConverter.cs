using System;
using TarsEngine.Services;

namespace TarsEngine.Utilities
{
    /// <summary>
    /// Provides conversion methods for ProgrammingLanguage enum
    /// </summary>
    public static class ProgrammingLanguageConverter
    {
        /// <summary>
        /// Converts a language string to a ProgrammingLanguage enum value
        /// </summary>
        /// <param name="language">The language string</param>
        /// <returns>The corresponding ProgrammingLanguage enum value</returns>
        public static ProgrammingLanguage FromString(string language)
        {
            return language?.ToLowerInvariant() switch
            {
                "csharp" => ProgrammingLanguage.CSharp,
                "fsharp" => ProgrammingLanguage.FSharp,
                "javascript" => ProgrammingLanguage.JavaScript,
                "typescript" => ProgrammingLanguage.TypeScript,
                "python" => ProgrammingLanguage.Python,
                "java" => ProgrammingLanguage.Java,
                "cpp" => ProgrammingLanguage.Cpp,
                _ => ProgrammingLanguage.Unknown
            };
        }

        /// <summary>
        /// Converts a ProgrammingLanguage enum value to a language string
        /// </summary>
        /// <param name="language">The ProgrammingLanguage enum value</param>
        /// <returns>The corresponding language string</returns>
        public static string ToString(ProgrammingLanguage language)
        {
            return language switch
            {
                ProgrammingLanguage.CSharp => "csharp",
                ProgrammingLanguage.FSharp => "fsharp",
                ProgrammingLanguage.JavaScript => "javascript",
                ProgrammingLanguage.TypeScript => "typescript",
                ProgrammingLanguage.Python => "python",
                ProgrammingLanguage.Java => "java",
                ProgrammingLanguage.Cpp => "cpp",
                _ => "unknown"
            };
        }
    }
}
