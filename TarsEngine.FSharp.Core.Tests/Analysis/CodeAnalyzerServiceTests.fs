namespace TarsEngine.FSharp.Core.Tests.Analysis

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Xunit
open TarsEngine.FSharp.Core.Analysis

/// <summary>
/// Tests for the CodeAnalyzerService class.
/// </summary>
module CodeAnalyzerServiceTests =
    
    /// <summary>
    /// Mock logger for testing.
    /// </summary>
    type MockLogger<'T>() =
        interface ILogger<'T> with
            member _.Log<'TState>(logLevel, eventId, state, ex, formatter) =
                // Do nothing
                ()
            
            member _.IsEnabled(logLevel) = true
            
            member _.BeginScope<'TState>(state) =
                { new IDisposable with
                    member _.Dispose() = ()
                }
    
    /// <summary>
    /// Test that the analyzer can analyze C# code.
    /// </summary>
    [<Fact>]
    let ``AnalyzeContentAsync with C# code should succeed``() = task {
        // Arrange
        let csharpStructureExtractorLogger = MockLogger<CSharpStructureExtractor>() :> ILogger<CSharpStructureExtractor>
        let csharpStructureExtractor = CSharpStructureExtractor(csharpStructureExtractorLogger)
        
        let securityIssueDetectorLogger = MockLogger<SecurityIssueDetector>() :> ILogger<SecurityIssueDetector>
        let securityIssueDetector = SecurityIssueDetector(securityIssueDetectorLogger)
        
        let csharpAnalyzerLogger = MockLogger<CSharpAnalyzer>() :> ILogger<CSharpAnalyzer>
        let csharpAnalyzer = CSharpAnalyzer(
            csharpAnalyzerLogger,
            csharpStructureExtractor,
            securityIssueDetector,
            securityIssueDetector :> IPerformanceIssueDetector,
            securityIssueDetector :> IComplexityIssueDetector,
            securityIssueDetector :> IStyleIssueDetector
        )
        
        let fsharpStructureExtractorLogger = MockLogger<FSharpStructureExtractor>() :> ILogger<FSharpStructureExtractor>
        let fsharpStructureExtractor = FSharpStructureExtractor(fsharpStructureExtractorLogger)
        
        let fsharpAnalyzerLogger = MockLogger<FSharpAnalyzer>() :> ILogger<FSharpAnalyzer>
        let fsharpAnalyzer = FSharpAnalyzer(
            fsharpAnalyzerLogger,
            fsharpStructureExtractor,
            securityIssueDetector,
            securityIssueDetector :> IPerformanceIssueDetector,
            securityIssueDetector :> IComplexityIssueDetector,
            securityIssueDetector :> IStyleIssueDetector
        )
        
        let codeAnalyzerServiceLogger = MockLogger<CodeAnalyzerService>() :> ILogger<CodeAnalyzerService>
        let codeAnalyzerService = CodeAnalyzerService(
            codeAnalyzerServiceLogger,
            [csharpAnalyzer :> ILanguageAnalyzer; fsharpAnalyzer :> ILanguageAnalyzer]
        )
        
        let code = """
            using System;
            using System.Data.SqlClient;

            namespace MyApp
            {
                public class Program
                {
                    public static void Main(string[] args)
                    {
                        string username = args[0];
                        string password = "hardcoded_password";
                        
                        // SQL injection vulnerability
                        string sql = "SELECT * FROM Users WHERE Username = '" + username + "'";
                        
                        using (var connection = new SqlConnection("Server=myServerAddress;Database=myDataBase;User Id=myUsername;Password=myPassword;"))
                        {
                            connection.Open();
                            
                            using (var command = new SqlCommand(sql, connection))
                            {
                                using (var reader = command.ExecuteReader())
                                {
                                    while (reader.Read())
                                    {
                                        Console.WriteLine(reader["Name"]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            """
        
        // Act
        let! result = codeAnalyzerService.AnalyzeContentAsync(code, "csharp")
        
        // Assert
        Assert.Equal("csharp", result.Language)
        Assert.NotEmpty(result.Issues)
        Assert.NotEmpty(result.Structures)
        Assert.NotEmpty(result.Metrics)
        
        // Check for SQL injection issue
        let sqlInjectionIssue = result.Issues |> List.tryFind (fun i -> i.Message.Contains("SQL injection"))
        Assert.True(sqlInjectionIssue.IsSome)
        
        // Check for hardcoded credentials issue
        let hardcodedCredentialsIssue = result.Issues |> List.tryFind (fun i -> i.Message.Contains("Hardcoded credentials"))
        Assert.True(hardcodedCredentialsIssue.IsSome)
    }
    
    /// <summary>
    /// Test that the analyzer can analyze F# code.
    /// </summary>
    [<Fact>]
    let ``AnalyzeContentAsync with F# code should succeed``() = task {
        // Arrange
        let csharpStructureExtractorLogger = MockLogger<CSharpStructureExtractor>() :> ILogger<CSharpStructureExtractor>
        let csharpStructureExtractor = CSharpStructureExtractor(csharpStructureExtractorLogger)
        
        let securityIssueDetectorLogger = MockLogger<SecurityIssueDetector>() :> ILogger<SecurityIssueDetector>
        let securityIssueDetector = SecurityIssueDetector(securityIssueDetectorLogger)
        
        let csharpAnalyzerLogger = MockLogger<CSharpAnalyzer>() :> ILogger<CSharpAnalyzer>
        let csharpAnalyzer = CSharpAnalyzer(
            csharpAnalyzerLogger,
            csharpStructureExtractor,
            securityIssueDetector,
            securityIssueDetector :> IPerformanceIssueDetector,
            securityIssueDetector :> IComplexityIssueDetector,
            securityIssueDetector :> IStyleIssueDetector
        )
        
        let fsharpStructureExtractorLogger = MockLogger<FSharpStructureExtractor>() :> ILogger<FSharpStructureExtractor>
        let fsharpStructureExtractor = FSharpStructureExtractor(fsharpStructureExtractorLogger)
        
        let fsharpAnalyzerLogger = MockLogger<FSharpAnalyzer>() :> ILogger<FSharpAnalyzer>
        let fsharpAnalyzer = FSharpAnalyzer(
            fsharpAnalyzerLogger,
            fsharpStructureExtractor,
            securityIssueDetector,
            securityIssueDetector :> IPerformanceIssueDetector,
            securityIssueDetector :> IComplexityIssueDetector,
            securityIssueDetector :> IStyleIssueDetector
        )
        
        let codeAnalyzerServiceLogger = MockLogger<CodeAnalyzerService>() :> ILogger<CodeAnalyzerService>
        let codeAnalyzerService = CodeAnalyzerService(
            codeAnalyzerServiceLogger,
            [csharpAnalyzer :> ILanguageAnalyzer; fsharpAnalyzer :> ILanguageAnalyzer]
        )
        
        let code = """
            open System
            open System.Data.SqlClient

            module Program =
                let executeQuery (username: string) =
                    let password = "hardcoded_password"
                    
                    // SQL injection vulnerability
                    let sql = "SELECT * FROM Users WHERE Username = '" + username + "'"
                    
                    use connection = new SqlConnection("Server=myServerAddress;Database=myDataBase;User Id=myUsername;Password=myPassword;")
                    connection.Open()
                    
                    use command = new SqlCommand(sql, connection)
                    use reader = command.ExecuteReader()
                    
                    while reader.Read() do
                        printfn "%s" (reader.["Name"].ToString())
                
                [<EntryPoint>]
                let main args =
                    executeQuery args.[0]
                    0
            """
        
        // Act
        let! result = codeAnalyzerService.AnalyzeContentAsync(code, "fsharp")
        
        // Assert
        Assert.Equal("fsharp", result.Language)
        Assert.NotEmpty(result.Issues)
        Assert.NotEmpty(result.Structures)
        Assert.NotEmpty(result.Metrics)
        
        // Check for SQL injection issue
        let sqlInjectionIssue = result.Issues |> List.tryFind (fun i -> i.Message.Contains("SQL injection"))
        Assert.True(sqlInjectionIssue.IsSome)
        
        // Check for hardcoded credentials issue
        let hardcodedCredentialsIssue = result.Issues |> List.tryFind (fun i -> i.Message.Contains("Hardcoded credentials"))
        Assert.True(hardcodedCredentialsIssue.IsSome)
    }
    
    /// <summary>
    /// Test that the analyzer can get supported languages.
    /// </summary>
    [<Fact>]
    let ``GetSupportedLanguagesAsync should return supported languages``() = task {
        // Arrange
        let csharpStructureExtractorLogger = MockLogger<CSharpStructureExtractor>() :> ILogger<CSharpStructureExtractor>
        let csharpStructureExtractor = CSharpStructureExtractor(csharpStructureExtractorLogger)
        
        let securityIssueDetectorLogger = MockLogger<SecurityIssueDetector>() :> ILogger<SecurityIssueDetector>
        let securityIssueDetector = SecurityIssueDetector(securityIssueDetectorLogger)
        
        let csharpAnalyzerLogger = MockLogger<CSharpAnalyzer>() :> ILogger<CSharpAnalyzer>
        let csharpAnalyzer = CSharpAnalyzer(
            csharpAnalyzerLogger,
            csharpStructureExtractor,
            securityIssueDetector,
            securityIssueDetector :> IPerformanceIssueDetector,
            securityIssueDetector :> IComplexityIssueDetector,
            securityIssueDetector :> IStyleIssueDetector
        )
        
        let fsharpStructureExtractorLogger = MockLogger<FSharpStructureExtractor>() :> ILogger<FSharpStructureExtractor>
        let fsharpStructureExtractor = FSharpStructureExtractor(fsharpStructureExtractorLogger)
        
        let fsharpAnalyzerLogger = MockLogger<FSharpAnalyzer>() :> ILogger<FSharpAnalyzer>
        let fsharpAnalyzer = FSharpAnalyzer(
            fsharpAnalyzerLogger,
            fsharpStructureExtractor,
            securityIssueDetector,
            securityIssueDetector :> IPerformanceIssueDetector,
            securityIssueDetector :> IComplexityIssueDetector,
            securityIssueDetector :> IStyleIssueDetector
        )
        
        let codeAnalyzerServiceLogger = MockLogger<CodeAnalyzerService>() :> ILogger<CodeAnalyzerService>
        let codeAnalyzerService = CodeAnalyzerService(
            codeAnalyzerServiceLogger,
            [csharpAnalyzer :> ILanguageAnalyzer; fsharpAnalyzer :> ILanguageAnalyzer]
        )
        
        // Act
        let! languages = codeAnalyzerService.GetSupportedLanguagesAsync()
        
        // Assert
        Assert.Contains("csharp", languages)
        Assert.Contains("fsharp", languages)
    }
    
    /// <summary>
    /// Test that the analyzer can get issues for a file.
    /// </summary>
    [<Fact>]
    let ``GetIssuesForFileAsync should return issues for a file``() = task {
        // Arrange
        let csharpStructureExtractorLogger = MockLogger<CSharpStructureExtractor>() :> ILogger<CSharpStructureExtractor>
        let csharpStructureExtractor = CSharpStructureExtractor(csharpStructureExtractorLogger)
        
        let securityIssueDetectorLogger = MockLogger<SecurityIssueDetector>() :> ILogger<SecurityIssueDetector>
        let securityIssueDetector = SecurityIssueDetector(securityIssueDetectorLogger)
        
        let csharpAnalyzerLogger = MockLogger<CSharpAnalyzer>() :> ILogger<CSharpAnalyzer>
        let csharpAnalyzer = CSharpAnalyzer(
            csharpAnalyzerLogger,
            csharpStructureExtractor,
            securityIssueDetector,
            securityIssueDetector :> IPerformanceIssueDetector,
            securityIssueDetector :> IComplexityIssueDetector,
            securityIssueDetector :> IStyleIssueDetector
        )
        
        let fsharpStructureExtractorLogger = MockLogger<FSharpStructureExtractor>() :> ILogger<FSharpStructureExtractor>
        let fsharpStructureExtractor = FSharpStructureExtractor(fsharpStructureExtractorLogger)
        
        let fsharpAnalyzerLogger = MockLogger<FSharpAnalyzer>() :> ILogger<FSharpAnalyzer>
        let fsharpAnalyzer = FSharpAnalyzer(
            fsharpAnalyzerLogger,
            fsharpStructureExtractor,
            securityIssueDetector,
            securityIssueDetector :> IPerformanceIssueDetector,
            securityIssueDetector :> IComplexityIssueDetector,
            securityIssueDetector :> IStyleIssueDetector
        )
        
        let codeAnalyzerServiceLogger = MockLogger<CodeAnalyzerService>() :> ILogger<CodeAnalyzerService>
        let codeAnalyzerService = CodeAnalyzerService(
            codeAnalyzerServiceLogger,
            [csharpAnalyzer :> ILanguageAnalyzer; fsharpAnalyzer :> ILanguageAnalyzer]
        )
        
        // Create a temporary file
        let tempFilePath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.cs")
        let code = """
            using System;
            using System.Data.SqlClient;

            namespace MyApp
            {
                public class Program
                {
                    public static void Main(string[] args)
                    {
                        string username = args[0];
                        string password = "hardcoded_password";
                        
                        // SQL injection vulnerability
                        string sql = "SELECT * FROM Users WHERE Username = '" + username + "'";
                        
                        using (var connection = new SqlConnection("Server=myServerAddress;Database=myDataBase;User Id=myUsername;Password=myPassword;"))
                        {
                            connection.Open();
                            
                            using (var command = new SqlCommand(sql, connection))
                            {
                                using (var reader = command.ExecuteReader())
                                {
                                    while (reader.Read())
                                    {
                                        Console.WriteLine(reader["Name"]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            """
        File.WriteAllText(tempFilePath, code)
        
        // Act
        let! issues = codeAnalyzerService.GetIssuesForFileAsync(tempFilePath, [CodeIssueType.Security])
        
        // Assert
        Assert.NotEmpty(issues)
        Assert.All(issues, fun i -> Assert.Equal(CodeIssueType.Security, i.IssueType))
        
        // Check for SQL injection issue
        let sqlInjectionIssue = issues |> List.tryFind (fun i -> i.Message.Contains("SQL injection"))
        Assert.True(sqlInjectionIssue.IsSome)
        
        // Check for hardcoded credentials issue
        let hardcodedCredentialsIssue = issues |> List.tryFind (fun i -> i.Message.Contains("Hardcoded credentials"))
        Assert.True(hardcodedCredentialsIssue.IsSome)
        
        // Cleanup
        File.Delete(tempFilePath)
    }
