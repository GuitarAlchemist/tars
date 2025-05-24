namespace TarsEngine.FSharp.Core.CodeGen.Testing.Runners

open System
open System.Collections.Generic
open Microsoft.Extensions.Logging

/// <summary>
/// Factory for creating test runners.
/// </summary>
type TestRunnerFactory(logger: ILogger<TestRunnerFactory>, testRunners: ITestRunner seq) =
    
    /// <summary>
    /// Gets a test runner for a test framework.
    /// </summary>
    /// <param name="testFramework">The test framework to get a runner for.</param>
    /// <returns>The test runner, if found.</returns>
    member _.GetTestRunner(testFramework: string) =
        testRunners 
        |> Seq.tryFind (fun r -> 
            r.SupportedFrameworks 
            |> List.exists (fun f -> f.Equals(testFramework, StringComparison.OrdinalIgnoreCase)))
    
    /// <summary>
    /// Gets all test runners.
    /// </summary>
    /// <returns>The list of all test runners.</returns>
    member _.GetAllTestRunners() =
        testRunners |> Seq.toList
    
    /// <summary>
    /// Gets the supported test frameworks.
    /// </summary>
    /// <returns>The list of supported test frameworks.</returns>
    member _.GetSupportedTestFrameworks() =
        testRunners 
        |> Seq.collect (fun r -> r.SupportedFrameworks) 
        |> Seq.distinct 
        |> Seq.toList
