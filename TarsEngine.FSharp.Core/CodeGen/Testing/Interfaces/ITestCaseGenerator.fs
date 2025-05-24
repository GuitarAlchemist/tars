namespace TarsEngine.FSharp.Core.CodeGen.Testing.Interfaces

open System.Threading.Tasks
open TarsEngine.FSharp.Core.CodeGen.Testing
open TarsEngine.FSharp.Core.CodeGen.Testing.Models

/// <summary>
/// Interface for generating test cases.
/// </summary>
type ITestCaseGenerator =
    /// <summary>
    /// Generates test cases for a method.
    /// </summary>
    /// <param name="method">The method to generate test cases for.</param>
    /// <param name="testFramework">The test framework to use.</param>
    /// <param name="language">The language to use.</param>
    /// <returns>The list of generated test cases.</returns>
    abstract member GenerateTestCasesForMethod : method:ExtractedMethod * testFramework:string * language:string -> Task<TestCase list>
    
    /// <summary>
    /// Generates test cases for a class.
    /// </summary>
    /// <param name="class">The class to generate test cases for.</param>
    /// <param name="testFramework">The test framework to use.</param>
    /// <param name="language">The language to use.</param>
    /// <returns>The test suite containing the generated test cases.</returns>
    abstract member GenerateTestCasesForClass : class':ExtractedClass * testFramework:string * language:string -> Task<TestSuite>
    
    /// <summary>
    /// Generates test cases for code.
    /// </summary>
    /// <param name="code">The code to generate test cases for.</param>
    /// <param name="testFramework">The test framework to use.</param>
    /// <param name="language">The language to use.</param>
    /// <returns>The list of test suites containing the generated test cases.</returns>
    abstract member GenerateTestCasesForCode : code:string * testFramework:string * language:string -> Task<TestSuite list>
    
    /// <summary>
    /// Gets the supported test frameworks.
    /// </summary>
    /// <returns>The list of supported test frameworks.</returns>
    abstract member GetSupportedTestFrameworks : unit -> string list
    
    /// <summary>
    /// Gets the supported languages.
    /// </summary>
    /// <returns>The list of supported languages.</returns>
    abstract member GetSupportedLanguages : unit -> string list
