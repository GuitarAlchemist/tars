namespace TarsEngine.FSharp.Core.Analysis

open System
open System.Text.RegularExpressions
open Microsoft.Extensions.Logging

/// <summary>
/// Implementation of ISecurityIssueDetector for C# and F# languages.
/// </summary>
type SecurityIssueDetector(logger: ILogger<SecurityIssueDetector>) =
    
    /// <summary>
    /// Gets the language supported by this detector.
    /// </summary>
    member _.Language = "csharp"
    
    /// <summary>
    /// Detects issues in the provided content.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    member _.DetectIssues(content: string) =
        try
            logger.LogInformation("Detecting security issues")
            
            // Detect SQL injection vulnerabilities
            let sqlInjectionIssues = this.DetectSqlInjection(content)
            
            // Detect cross-site scripting vulnerabilities
            let xssIssues = this.DetectXss(content)
            
            // Detect insecure cryptography
            let insecureCryptographyIssues = this.DetectInsecureCryptography(content)
            
            // Detect hardcoded credentials
            let hardcodedCredentialsIssues = this.DetectHardcodedCredentials(content)
            
            // Combine all issues
            List.concat [
                sqlInjectionIssues
                xssIssues
                insecureCryptographyIssues
                hardcodedCredentialsIssues
            ]
        with
        | ex ->
            logger.LogError(ex, "Error detecting security issues")
            []
    
    /// <summary>
    /// Detects issues in a file.
    /// </summary>
    /// <param name="filePath">The path to the file.</param>
    /// <returns>A list of detected issues.</returns>
    member this.DetectIssuesInFile(filePath: string) =
        try
            logger.LogInformation("Detecting security issues in file: {FilePath}", filePath)
            
            // Read the file content
            let content = System.IO.File.ReadAllText(filePath)
            
            // Detect issues in the content
            let issues = this.DetectIssues(content)
            
            // Update issues with file path
            issues |> List.map (fun issue -> { issue with FilePath = Some filePath })
        with
        | ex ->
            logger.LogError(ex, "Error detecting security issues in file: {FilePath}", filePath)
            []
    
    /// <summary>
    /// Gets the supported issue types.
    /// </summary>
    /// <returns>A list of supported issue types.</returns>
    member _.GetSupportedIssueTypes() =
        [
            CodeIssueType.Security
        ]
    
    /// <summary>
    /// Detects SQL injection vulnerabilities.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    member _.DetectSqlInjection(content: string) =
        try
            // Define patterns for SQL injection vulnerabilities
            let patterns = [
                @"string\s+sql\s*=\s*""[^""]*\+\s*[^""]*""", "String concatenation in SQL query"
                @"ExecuteQuery\s*\(\s*""[^""]*\+\s*[^""]*""\s*\)", "String concatenation in ExecuteQuery"
                @"ExecuteNonQuery\s*\(\s*""[^""]*\+\s*[^""]*""\s*\)", "String concatenation in ExecuteNonQuery"
                @"ExecuteScalar\s*\(\s*""[^""]*\+\s*[^""]*""\s*\)", "String concatenation in ExecuteScalar"
                @"ExecuteReader\s*\(\s*""[^""]*\+\s*[^""]*""\s*\)", "String concatenation in ExecuteReader"
                @"SqlCommand\s*\(\s*""[^""]*\+\s*[^""]*""\s*\)", "String concatenation in SqlCommand"
                @"OleDbCommand\s*\(\s*""[^""]*\+\s*[^""]*""\s*\)", "String concatenation in OleDbCommand"
                @"OdbcCommand\s*\(\s*""[^""]*\+\s*[^""]*""\s*\)", "String concatenation in OdbcCommand"
                @"MySqlCommand\s*\(\s*""[^""]*\+\s*[^""]*""\s*\)", "String concatenation in MySqlCommand"
                @"NpgsqlCommand\s*\(\s*""[^""]*\+\s*[^""]*""\s*\)", "String concatenation in NpgsqlCommand"
            ]
            
            // Detect issues using patterns
            patterns
            |> List.collect (fun (pattern, message) ->
                Regex.Matches(content, pattern, RegexOptions.IgnoreCase ||| RegexOptions.Multiline)
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let lineNumber = content.Substring(0, m.Index).Split('\n').Length
                    let columnNumber = m.Index - content.Substring(0, m.Index).LastIndexOf('\n') - 1
                    let codeSnippet = m.Value
                    
                    {
                        IssueType = CodeIssueType.Security
                        Severity = IssueSeverity.Critical
                        Message = $"Potential SQL injection vulnerability: {message}"
                        LineNumber = Some lineNumber
                        ColumnNumber = Some columnNumber
                        FilePath = None
                        CodeSnippet = Some codeSnippet
                        SuggestedFix = Some "Use parameterized queries instead of string concatenation"
                        AdditionalInfo = Map.empty
                    }
                )
                |> Seq.toList
            )
        with
        | ex ->
            logger.LogError(ex, "Error detecting SQL injection vulnerabilities")
            []
    
    /// <summary>
    /// Detects cross-site scripting vulnerabilities.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    member _.DetectXss(content: string) =
        try
            // Define patterns for XSS vulnerabilities
            let patterns = [
                @"Response\.Write\s*\(\s*[^""]*\s*\)", "Unencoded output in Response.Write"
                @"<%=\s*[^""]*\s*%>", "Unencoded output in <%= %>"
                @"@Html\.Raw\s*\(\s*[^""]*\s*\)", "Unencoded output in Html.Raw"
                @"innerHTML\s*=\s*[^""]*", "Unencoded output in innerHTML"
                @"document\.write\s*\(\s*[^""]*\s*\)", "Unencoded output in document.write"
            ]
            
            // Detect issues using patterns
            patterns
            |> List.collect (fun (pattern, message) ->
                Regex.Matches(content, pattern, RegexOptions.IgnoreCase ||| RegexOptions.Multiline)
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let lineNumber = content.Substring(0, m.Index).Split('\n').Length
                    let columnNumber = m.Index - content.Substring(0, m.Index).LastIndexOf('\n') - 1
                    let codeSnippet = m.Value
                    
                    {
                        IssueType = CodeIssueType.Security
                        Severity = IssueSeverity.Critical
                        Message = $"Potential cross-site scripting vulnerability: {message}"
                        LineNumber = Some lineNumber
                        ColumnNumber = Some columnNumber
                        FilePath = None
                        CodeSnippet = Some codeSnippet
                        SuggestedFix = Some "Use HTML encoding before outputting user input"
                        AdditionalInfo = Map.empty
                    }
                )
                |> Seq.toList
            )
        with
        | ex ->
            logger.LogError(ex, "Error detecting cross-site scripting vulnerabilities")
            []
    
    /// <summary>
    /// Detects insecure cryptography.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    member _.DetectInsecureCryptography(content: string) =
        try
            // Define patterns for insecure cryptography
            let patterns = [
                @"MD5", "MD5 is a weak hashing algorithm"
                @"SHA1", "SHA1 is a weak hashing algorithm"
                @"DES", "DES is a weak encryption algorithm"
                @"RC2", "RC2 is a weak encryption algorithm"
                @"RC4", "RC4 is a weak encryption algorithm"
                @"ECB", "ECB mode is insecure for encryption"
                @"new RNGCryptoServiceProvider\(\)", "RNGCryptoServiceProvider is deprecated"
                @"Random\(\)", "System.Random is not cryptographically secure"
            ]
            
            // Detect issues using patterns
            patterns
            |> List.collect (fun (pattern, message) ->
                Regex.Matches(content, pattern, RegexOptions.IgnoreCase ||| RegexOptions.Multiline)
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let lineNumber = content.Substring(0, m.Index).Split('\n').Length
                    let columnNumber = m.Index - content.Substring(0, m.Index).LastIndexOf('\n') - 1
                    let codeSnippet = m.Value
                    
                    {
                        IssueType = CodeIssueType.Security
                        Severity = IssueSeverity.Error
                        Message = $"Insecure cryptography: {message}"
                        LineNumber = Some lineNumber
                        ColumnNumber = Some columnNumber
                        FilePath = None
                        CodeSnippet = Some codeSnippet
                        SuggestedFix = Some "Use modern cryptographic algorithms and practices"
                        AdditionalInfo = Map.empty
                    }
                )
                |> Seq.toList
            )
        with
        | ex ->
            logger.LogError(ex, "Error detecting insecure cryptography")
            []
    
    /// <summary>
    /// Detects hardcoded credentials.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    member _.DetectHardcodedCredentials(content: string) =
        try
            // Define patterns for hardcoded credentials
            let patterns = [
                @"password\s*=\s*""[^""]+""", "Hardcoded password"
                @"pwd\s*=\s*""[^""]+""", "Hardcoded password"
                @"apikey\s*=\s*""[^""]+""", "Hardcoded API key"
                @"api_key\s*=\s*""[^""]+""", "Hardcoded API key"
                @"secret\s*=\s*""[^""]+""", "Hardcoded secret"
                @"connectionString\s*=\s*""[^""]*password=[^""]+""", "Hardcoded password in connection string"
                @"connectionString\s*=\s*""[^""]*pwd=[^""]+""", "Hardcoded password in connection string"
                @"connectionString\s*=\s*""[^""]*user id=[^""]+""", "Hardcoded user ID in connection string"
                @"connectionString\s*=\s*""[^""]*uid=[^""]+""", "Hardcoded user ID in connection string"
                @"token\s*=\s*""[^""]+""", "Hardcoded token"
            ]
            
            // Detect issues using patterns
            patterns
            |> List.collect (fun (pattern, message) ->
                Regex.Matches(content, pattern, RegexOptions.IgnoreCase ||| RegexOptions.Multiline)
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let lineNumber = content.Substring(0, m.Index).Split('\n').Length
                    let columnNumber = m.Index - content.Substring(0, m.Index).LastIndexOf('\n') - 1
                    let codeSnippet = m.Value
                    
                    {
                        IssueType = CodeIssueType.Security
                        Severity = IssueSeverity.Critical
                        Message = $"Hardcoded credentials: {message}"
                        LineNumber = Some lineNumber
                        ColumnNumber = Some columnNumber
                        FilePath = None
                        CodeSnippet = Some codeSnippet
                        SuggestedFix = Some "Store credentials in a secure configuration system"
                        AdditionalInfo = Map.empty
                    }
                )
                |> Seq.toList
            )
        with
        | ex ->
            logger.LogError(ex, "Error detecting hardcoded credentials")
            []
    
    interface IIssueDetector with
        member this.Language = this.Language
        member this.DetectIssues(content) = this.DetectIssues(content)
        member this.DetectIssuesInFile(filePath) = this.DetectIssuesInFile(filePath)
        member this.GetSupportedIssueTypes() = this.GetSupportedIssueTypes()
    
    interface ISecurityIssueDetector with
        member this.DetectSqlInjection(content) = this.DetectSqlInjection(content)
        member this.DetectXss(content) = this.DetectXss(content)
        member this.DetectInsecureCryptography(content) = this.DetectInsecureCryptography(content)
        member this.DetectHardcodedCredentials(content) = this.DetectHardcodedCredentials(content)
