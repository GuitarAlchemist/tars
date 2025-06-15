namespace TarsEngine.FSharp.Core.Tests.Analysis

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Xunit
open TarsEngine.FSharp.Core.Analysis

/// <summary>
/// Tests for the issue detector classes.
/// </summary>
module IssueDetectorTests =
    
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
    /// Test that the security issue detector can detect SQL injection vulnerabilities.
    /// </summary>
    [<Fact>]
    let ``SecurityIssueDetector can detect SQL injection vulnerabilities``() =
        // Arrange
        let logger = MockLogger<SecurityIssueDetector>() :> ILogger<SecurityIssueDetector>
        let detector = SecurityIssueDetector(logger)
        
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
        let issues = detector.DetectSqlInjection(code)
        
        // Assert
        Assert.NotEmpty(issues)
        Assert.Contains(issues, fun i -> i.Message.Contains("SQL injection"))
    
    /// <summary>
    /// Test that the security issue detector can detect hardcoded credentials.
    /// </summary>
    [<Fact>]
    let ``SecurityIssueDetector can detect hardcoded credentials``() =
        // Arrange
        let logger = MockLogger<SecurityIssueDetector>() :> ILogger<SecurityIssueDetector>
        let detector = SecurityIssueDetector(logger)
        
        let code = """
            using System;
            using System.Data.SqlClient;

            namespace MyApp
            {
                public class Program
                {
                    public static void Main(string[] args)
                    {
                        string username = "admin";
                        string password = "hardcoded_password";
                        
                        using (var connection = new SqlConnection("Server=myServerAddress;Database=myDataBase;User Id=myUsername;Password=myPassword;"))
                        {
                            connection.Open();
                        }
                    }
                }
            }
            """
        
        // Act
        let issues = detector.DetectHardcodedCredentials(code)
        
        // Assert
        Assert.NotEmpty(issues)
        Assert.Contains(issues, fun i -> i.Message.Contains("Hardcoded credentials"))
    
    /// <summary>
    /// Test that the performance issue detector can detect inefficient loops.
    /// </summary>
    [<Fact>]
    let ``PerformanceIssueDetector can detect inefficient loops``() =
        // Arrange
        let logger = MockLogger<PerformanceIssueDetector>() :> ILogger<PerformanceIssueDetector>
        let detector = PerformanceIssueDetector(logger)
        
        let code = """
            using System;
            using System.Collections.Generic;

            namespace MyApp
            {
                public class Program
                {
                    public static void Main(string[] args)
                    {
                        List<string> items = new List<string>();
                        
                        // Inefficient loop
                        for (int i = 0; i < items.Count; i++)
                        {
                            Console.WriteLine(items[i]);
                        }
                    }
                }
            }
            """
        
        // Act
        let issues = detector.DetectInefficientLoops(code)
        
        // Assert
        Assert.NotEmpty(issues)
        Assert.Contains(issues, fun i -> i.Message.Contains("Inefficient loop"))
    
    /// <summary>
    /// Test that the performance issue detector can detect inefficient string operations.
    /// </summary>
    [<Fact>]
    let ``PerformanceIssueDetector can detect inefficient string operations``() =
        // Arrange
        let logger = MockLogger<PerformanceIssueDetector>() :> ILogger<PerformanceIssueDetector>
        let detector = PerformanceIssueDetector(logger)
        
        let code = """
            using System;
            using System.Collections.Generic;

            namespace MyApp
            {
                public class Program
                {
                    public static void Main(string[] args)
                    {
                        string result = "";
                        
                        // Inefficient string operation
                        for (int i = 0; i < 1000; i++)
                        {
                            result += i.ToString();
                        }
                        
                        Console.WriteLine(result);
                    }
                }
            }
            """
        
        // Act
        let issues = detector.DetectInefficientStringOperations(code)
        
        // Assert
        Assert.NotEmpty(issues)
        Assert.Contains(issues, fun i -> i.Message.Contains("String concatenation in a loop"))
    
    /// <summary>
    /// Test that the complexity issue detector can detect methods with too many parameters.
    /// </summary>
    [<Fact>]
    let ``ComplexityIssueDetector can detect methods with too many parameters``() =
        // Arrange
        let logger = MockLogger<ComplexityIssueDetector>() :> ILogger<ComplexityIssueDetector>
        let detector = ComplexityIssueDetector(logger)
        
        let code = """
            using System;

            namespace MyApp
            {
                public class Program
                {
                    // Method with too many parameters
                    public static void ProcessData(string param1, string param2, string param3, string param4, string param5, string param6, string param7, string param8, string param9)
                    {
                        Console.WriteLine(param1 + param2 + param3 + param4 + param5 + param6 + param7 + param8 + param9);
                    }
                }
            }
            """
        
        // Act
        let issues = detector.DetectTooManyParameters(code)
        
        // Assert
        Assert.NotEmpty(issues)
        Assert.Contains(issues, fun i -> i.Message.Contains("too many parameters"))
    
    /// <summary>
    /// Test that the complexity issue detector can detect long methods.
    /// </summary>
    [<Fact>]
    let ``ComplexityIssueDetector can detect long methods``() =
        // Arrange
        let logger = MockLogger<ComplexityIssueDetector>() :> ILogger<ComplexityIssueDetector>
        let detector = ComplexityIssueDetector(logger)
        
        // Create a long method
        let lines = [| for i in 1..60 -> $"            Console.WriteLine(\"Line {i}\");" |]
        let methodBody = String.Join(Environment.NewLine, lines)
        
        let code = $"""
            using System;

            namespace MyApp
            {{
                public class Program
                {{
                    // Long method
                    public static void LongMethod()
                    {{
{methodBody}
                    }}
                }}
            }}
            """
        
        // Act
        let issues = detector.DetectLongMethods(code, [])
        
        // Assert
        Assert.NotEmpty(issues)
        Assert.Contains(issues, fun i -> i.Message.Contains("too long"))
    
    /// <summary>
    /// Test that the style issue detector can detect naming convention violations.
    /// </summary>
    [<Fact>]
    let ``StyleIssueDetector can detect naming convention violations``() =
        // Arrange
        let logger = MockLogger<StyleIssueDetector>() :> ILogger<StyleIssueDetector>
        let detector = StyleIssueDetector(logger)
        
        let code = """
            using System;

            namespace MyApp
            {
                // Class name should start with uppercase letter
                public class program
                {
                    // Method name should start with lowercase letter
                    public static void Main(string[] args)
                    {
                        // Variable name is fine
                        string message = "Hello, world!";
                        Console.WriteLine(message);
                    }
                    
                    // Property name should start with lowercase letter
                    public string Message { get; set; }
                }
                
                // Interface name should start with 'I'
                public interface myInterface
                {
                    void DoSomething();
                }
            }
            """
        
        // Act
        let issues = detector.DetectNamingConventionViolations(code)
        
        // Assert
        Assert.NotEmpty(issues)
        Assert.Contains(issues, fun i -> i.Message.Contains("Class name should start with an uppercase letter"))
        Assert.Contains(issues, fun i -> i.Message.Contains("Interface name should start with 'I'"))
    
    /// <summary>
    /// Test that the style issue detector can detect formatting issues.
    /// </summary>
    [<Fact>]
    let ``StyleIssueDetector can detect formatting issues``() =
        // Arrange
        let logger = MockLogger<StyleIssueDetector>() :> ILogger<StyleIssueDetector>
        let detector = StyleIssueDetector(logger)
        
        let code = """
            using System;

            namespace MyApp
            {
                public class Program
                {
                    public static void Main(string[] args)
                    {
                        // Missing space after if
                        if(args.Length > 0)Console.WriteLine(args[0]);
                        
                        // Empty if block
                        if (args.Length > 1) {}
                        
                        // Trailing whitespace
                        Console.WriteLine("Hello, world!");    
                    }
                }
            }
            """
        
        // Act
        let issues = detector.DetectFormattingIssues(code)
        
        // Assert
        Assert.NotEmpty(issues)
        Assert.Contains(issues, fun i -> i.Message.Contains("Missing space after if statement") || i.Message.Contains("Empty if block") || i.Message.Contains("Trailing whitespace"))
