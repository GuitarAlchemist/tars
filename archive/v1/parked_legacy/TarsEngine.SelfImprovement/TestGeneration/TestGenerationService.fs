namespace TarsEngine.SelfImprovement.TestGeneration

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.SelfImprovement.Common

/// Configuration options for the test generation service
type TestGenerationOptions = {
    /// Whether to include assertions in the generated tests
    IncludeAssertions: bool
    /// Whether to generate tests for private methods
    IncludePrivateMethods: bool
    /// Whether to generate tests for internal methods
    IncludeInternalMethods: bool
    /// The test framework to use (MSTest, NUnit, xUnit)
    TestFramework: string
    /// The maximum number of tests to generate per class
    MaxTestsPerClass: int option
    /// The maximum number of assertions to generate per test
    MaxAssertionsPerTest: int option
    /// Whether to use test patterns from the repository
    UseTestPatterns: bool
    /// The path to the test pattern repository file
    TestPatternRepositoryPath: string option
}

/// Default test generation options
module TestGenerationDefaults =
    let options = {
        IncludeAssertions = true
        IncludePrivateMethods = false
        IncludeInternalMethods = true
        TestFramework = "MSTest"
        MaxTestsPerClass = None
        MaxAssertionsPerTest = None
        UseTestPatterns = true
        TestPatternRepositoryPath = None
    }

/// Result of a test generation operation
type TestGenerationResult = {
    /// The generated test code
    GeneratedCode: string
    /// The source file path
    SourceFilePath: string
    /// The output file path
    OutputFilePath: string option
    /// The class name for which tests were generated
    ClassName: string option
    /// The number of test methods generated
    TestMethodCount: int
    /// The number of assertions generated
    AssertionCount: int
    /// Any warnings or information messages
    Messages: string list
}

/// Interface for the test generation service
type ITestGenerationService =
    /// Generate tests for a source file
    abstract member GenerateTests: sourceFilePath: string * outputFilePath: string option * className: string option * options: TestGenerationOptions option -> Task<Result<TestGenerationResult, string>>
    
    /// Generate tests for source code
    abstract member GenerateTestsFromCode: sourceCode: string * className: string option * options: TestGenerationOptions option -> Task<Result<string, string>>

/// Implementation of the test generation service
type TestGenerationService(logger: ILogger<TestGenerationService>) =
    
    let mutable testGenerator = Unchecked.defaultof<obj> // Will be initialized with the C# ImprovedCSharpTestGenerator
    
    let initializeTestGenerator() =
        // This will be implemented to create an instance of the C# ImprovedCSharpTestGenerator
        // For now, we'll just log that this is a placeholder
        logger.LogInformation("TestGenerationService: Initializing test generator (placeholder)")
        Task.CompletedTask
    
    let countTestMethods (testCode: string) =
        // Count the number of test methods in the generated code
        // This is a simple implementation that counts occurrences of [TestMethod]
        let testMethodCount = 
            testCode.Split([|"[TestMethod]"|], StringSplitOptions.None).Length - 1
        testMethodCount
    
    let countAssertions (testCode: string) =
        // Count the number of assertions in the generated code
        // This is a simple implementation that counts occurrences of Assert.
        let assertionCount = 
            testCode.Split([|"Assert."|], StringSplitOptions.None).Length - 1
        assertionCount
    
    interface ITestGenerationService with
        member this.GenerateTests(sourceFilePath, outputFilePath, className, options) = 
            task {
                try
                    // Check if the source file exists
                    if not (File.Exists(sourceFilePath)) then
                        return Error $"Source file {sourceFilePath} does not exist"
                    
                    // Read the source code
                    let! sourceCode = File.ReadAllTextAsync(sourceFilePath)
                    
                    // Generate tests from the source code
                    let! testCodeResult = (this :> ITestGenerationService).GenerateTestsFromCode(sourceCode, className, options)
                    
                    match testCodeResult with
                    | Ok testCode ->
                        // Write the test code to the output file if specified
                        match outputFilePath with
                        | Some path ->
                            do! File.WriteAllTextAsync(path, testCode)
                            logger.LogInformation("TestGenerationService: Tests written to {OutputFilePath}", path)
                        | None ->
                            logger.LogInformation("TestGenerationService: Tests generated but not written to file")
                        
                        // Count test methods and assertions
                        let testMethodCount = countTestMethods testCode
                        let assertionCount = countAssertions testCode
                        
                        // Create the result
                        let result = {
                            GeneratedCode = testCode
                            SourceFilePath = sourceFilePath
                            OutputFilePath = outputFilePath
                            ClassName = className
                            TestMethodCount = testMethodCount
                            AssertionCount = assertionCount
                            Messages = []
                        }
                        
                        return Ok result
                    | Error error ->
                        return Error error
                with
                | ex ->
                    logger.LogError(ex, "TestGenerationService: Error generating tests for {SourceFilePath}", sourceFilePath)
                    return Error $"Error generating tests: {ex.Message}"
            }
        
        member this.GenerateTestsFromCode(sourceCode, className, options) =
            task {
                try
                    // Initialize the test generator if needed
                    if obj.ReferenceEquals(testGenerator, null) then
                        do! initializeTestGenerator()
                    
                    // For now, return a placeholder result
                    // This will be replaced with actual test generation using the C# ImprovedCSharpTestGenerator
                    logger.LogInformation("TestGenerationService: Generating tests from code (placeholder)")
                    
                    let testCode = """
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace Tests
{
    [TestClass]
    public class PlaceholderTests
    {
        [TestMethod]
        public void Test_Placeholder_ShouldWork()
        {
            // Arrange
            var sut = new object();
            
            // Act
            var result = sut.ToString();
            
            // Assert
            Assert.IsNotNull(result);
        }
    }
}
"""
                    return Ok testCode
                with
                | ex ->
                    logger.LogError(ex, "TestGenerationService: Error generating tests from code")
                    return Error $"Error generating tests: {ex.Message}"
            }
    
    /// Initialize the service
    member this.Initialize() =
        task {
            try
                do! initializeTestGenerator()
                return Ok()
            with
            | ex ->
                logger.LogError(ex, "TestGenerationService: Error initializing service")
                return Error $"Error initializing service: {ex.Message}"
        }

/// Factory for creating test generation services
type TestGenerationServiceFactory(loggerFactory: ILoggerFactory) =
    
    /// Create a new test generation service
    member this.CreateTestGenerationService() =
        let logger = loggerFactory.CreateLogger<TestGenerationService>()
        let service = new TestGenerationService(logger)
        service
