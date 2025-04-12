using System.Text.RegularExpressions;

namespace TarsEngine.Services
{
    /// <summary>
    /// Contains generated regex patterns for code analysis
    /// </summary>
    public static partial class RegexPatterns
    {
        /// <summary>
        /// Regex for matching C# namespaces
        /// </summary>
        [GeneratedRegex(@"namespace\s+([a-zA-Z0-9_.]+)\s*{", RegexOptions.Compiled)]
        public static partial Regex CSharpNamespace();

        /// <summary>
        /// Regex for matching C# classes
        /// </summary>
        [GeneratedRegex(@"(public|private|protected|internal)?\s*(static|abstract|sealed)?\s*class\s+([a-zA-Z0-9_]+)(?:<[^>]+>)?(?:\s*:\s*[^{]+)?", RegexOptions.Compiled)]
        public static partial Regex CSharpClass();

        /// <summary>
        /// Regex for matching C# interfaces
        /// </summary>
        [GeneratedRegex(@"(public|private|protected|internal)?\s*interface\s+([a-zA-Z0-9_]+)(?:<[^>]+>)?(?:\s*:\s*[^{]+)?", RegexOptions.Compiled)]
        public static partial Regex CSharpInterface();

        /// <summary>
        /// Regex for matching C# methods
        /// </summary>
        [GeneratedRegex(@"(public|private|protected|internal)?\s*(static|virtual|abstract|override|async)?\s*([a-zA-Z0-9_<>]+)\s+([a-zA-Z0-9_]+)\s*\(", RegexOptions.Compiled)]
        public static partial Regex CSharpMethod();

        /// <summary>
        /// Regex for matching C# properties
        /// </summary>
        [GeneratedRegex(@"(public|private|protected|internal)?\s*(static|virtual|abstract|override)?\s*([a-zA-Z0-9_<>]+)\s+([a-zA-Z0-9_]+)\s*{\s*(get|set)?", RegexOptions.Compiled)]
        public static partial Regex CSharpProperty();

        /// <summary>
        /// Regex for matching SQL injection vulnerabilities
        /// </summary>
        [GeneratedRegex(@"SqlCommand\s*\(\s*[""'].*?\+\s*[^""']+\s*\+", RegexOptions.Compiled)]
        public static partial Regex SqlInjection();

        /// <summary>
        /// Regex for matching XSS vulnerabilities
        /// </summary>
        [GeneratedRegex(@"Response\.Write\s*\(\s*[^""']*\s*\)", RegexOptions.Compiled)]
        public static partial Regex XssVulnerability();

        /// <summary>
        /// Regex for matching hardcoded credentials
        /// </summary>
        [GeneratedRegex(@"(password|pwd|passwd|secret|key|token|apikey)\s*=\s*[""'][^""']+[""']", RegexOptions.IgnoreCase | RegexOptions.Compiled)]
        public static partial Regex HardcodedCredentials();

        /// <summary>
        /// Regex for matching operators in code
        /// </summary>
        [GeneratedRegex(@"[+\-*/=<>!&|^~%]|==|!=|<=|>=|&&|\|\||<<|>>|\+\+|--|->|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=", RegexOptions.Compiled)]
        public static partial Regex Operators();

        /// <summary>
        /// Regex for matching operands in code
        /// </summary>
        [GeneratedRegex(@"\b[a-zA-Z_][a-zA-Z0-9_]*\b|""[^""]*""|'[^']*'|\d+(\.\d+)?", RegexOptions.Compiled)]
        public static partial Regex Operands();

        /// <summary>
        /// Regex for matching F# modules
        /// </summary>
        [GeneratedRegex(@"module\s+([a-zA-Z0-9_.]+)(?:\s*=)?", RegexOptions.Compiled)]
        public static partial Regex FSharpModule();

        /// <summary>
        /// Regex for matching F# types
        /// </summary>
        [GeneratedRegex(@"type\s+([a-zA-Z0-9_]+)(?:<[^>]+>)?(?:\s*=|\s*\(|\s*:)", RegexOptions.Compiled)]
        public static partial Regex FSharpType();

        /// <summary>
        /// Regex for matching F# functions
        /// </summary>
        [GeneratedRegex(@"let\s+(rec\s+)?([a-zA-Z0-9_]+)(?:\s+[a-zA-Z0-9_]+)*\s*=", RegexOptions.Compiled)]
        public static partial Regex FSharpFunction();

        /// <summary>
        /// Regex for matching F# union cases
        /// </summary>
        [GeneratedRegex(@"\|\s*[a-zA-Z0-9_]+", RegexOptions.Compiled)]
        public static partial Regex FSharpUnionCase();

        /// <summary>
        /// Regex for matching F# record fields
        /// </summary>
        [GeneratedRegex(@"[a-zA-Z0-9_]+\s*:\s*[a-zA-Z0-9_<>]+", RegexOptions.Compiled)]
        public static partial Regex FSharpRecordField();

        /// <summary>
        /// Regex for matching generic type parameters
        /// </summary>
        [GeneratedRegex(@"<[^>]+>", RegexOptions.Compiled)]
        public static partial Regex GenericTypeParameter();

        /// <summary>
        /// Regex for matching F# interface implementations
        /// </summary>
        [GeneratedRegex(@"interface\s+[a-zA-Z0-9_<>]+", RegexOptions.Compiled)]
        public static partial Regex FSharpInterface();

        /// <summary>
        /// Regex for matching F# member definitions
        /// </summary>
        [GeneratedRegex(@"member\s+[a-zA-Z0-9_]+\.", RegexOptions.Compiled)]
        public static partial Regex FSharpMember();

        /// <summary>
        /// Regex for matching F# match expressions
        /// </summary>
        [GeneratedRegex(@"\bmatch\b", RegexOptions.Compiled)]
        public static partial Regex FSharpMatch();

        /// <summary>
        /// Regex for matching F# pattern cases
        /// </summary>
        [GeneratedRegex(@"\|\s*[^-]+\s*->", RegexOptions.Compiled)]
        public static partial Regex FSharpPatternCase();

        /// <summary>
        /// Regex for matching F# active patterns
        /// </summary>
        [GeneratedRegex(@"\(\|[^|]+\|\)", RegexOptions.Compiled)]
        public static partial Regex FSharpActivePattern();

        /// <summary>
        /// Regex for matching F# nested patterns
        /// </summary>
        [GeneratedRegex(@"match\s+[^w]+\s+with\s+[^m]+match", RegexOptions.Compiled)]
        public static partial Regex FSharpNestedPattern();

        /// <summary>
        /// Regex for matching F# unsafe code
        /// </summary>
        [GeneratedRegex(@"NativeInterop\.NativePtr|Microsoft\.FSharp\.NativeInterop|fixed\s+[a-zA-Z0-9_]+", RegexOptions.Compiled)]
        public static partial Regex FSharpUnsafeCode();

        /// <summary>
        /// Regex for matching F# external data access
        /// </summary>
        [GeneratedRegex(@"System\.IO\.File\.ReadAllText|System\.Net\.WebClient|HttpClient\.GetStringAsync", RegexOptions.Compiled)]
        public static partial Regex FSharpExternalData();
    }
}
