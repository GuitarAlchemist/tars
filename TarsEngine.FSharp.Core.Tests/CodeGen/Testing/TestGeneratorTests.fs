namespace TarsEngine.FSharp.Core.Tests.CodeGen.Testing

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Xunit
open TarsEngine.FSharp.Core.CodeGen.Testing
open TarsEngine.FSharp.Core.CodeGen.Testing.Assertions
open TarsEngine.FSharp.Core.CodeGen.Testing.Generators
open TarsEngine.FSharp.Core.CodeGen.Testing.Interfaces
open TarsEngine.FSharp.Core.CodeGen.Testing.Models

/// <summary>
/// Tests for the TestGenerator class.
/// </summary>
module TestGeneratorTests =
    
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
    /// Test that the test generator can generate tests for C# code.
    /// </summary>
    [<Fact>]
    let ``TestGenerator can generate tests for C# code``() = task {
        // Arrange
        let logger = MockLogger<TestGenerator>() :> ILogger<TestGenerator>
        
        let valueGeneratorLogger = MockLogger<TestValueGenerator>() :> ILogger<TestValueGenerator>
        let valueGenerator = TestValueGenerator(valueGeneratorLogger)
        
        let assertionFormatterLogger = MockLogger<XUnitAssertionFormatter>() :> ILogger<XUnitAssertionFormatter>
        let assertionFormatter = XUnitAssertionFormatter(assertionFormatterLogger)
        
        let assertionGeneratorLogger = MockLogger<PrimitiveAssertionGenerator>() :> ILogger<PrimitiveAssertionGenerator>
        let assertionGenerator = PrimitiveAssertionGenerator(assertionGeneratorLogger, [assertionFormatter])
        
        let codeAnalyzerLogger = MockLogger<TestCodeAnalyzer>() :> ILogger<TestCodeAnalyzer>
        let codeAnalyzer = TestCodeAnalyzer(codeAnalyzerLogger)
        
        let testCaseGeneratorLogger = MockLogger<BasicTestCaseGenerator>() :> ILogger<BasicTestCaseGenerator>
        let testCaseGenerator = BasicTestCaseGenerator(testCaseGeneratorLogger, valueGenerator, assertionGenerator, codeAnalyzer)
        
        let templateManagerLogger = MockLogger<TestTemplateManager>() :> ILogger<TestTemplateManager>
        let templateManager = TestTemplateManager(templateManagerLogger)
        
        let testGenerator = TestGenerator(logger, testCaseGenerator, templateManager)
        
        let code = """
            using System;

            namespace MyApp
            {
                public class Calculator
                {
                    public int Add(int a, int b)
                    {
                        return a + b;
                    }
                    
                    public int Subtract(int a, int b)
                    {
                        return a - b;
                    }
                    
                    public int Multiply(int a, int b)
                    {
                        return a * b;
                    }
                    
                    public int Divide(int a, int b)
                    {
                        if (b == 0)
                        {
                            throw new DivideByZeroException();
                        }
                        
                        return a / b;
                    }
                }
            }
            """
        
        // Act
        let! result = testGenerator.GenerateTests(code, "xunit")
        
        // Assert
        Assert.NotNull(result)
        Assert.NotEmpty(result.GeneratedTestCode)
        Assert.Equal(code, result.SourceCode)
        Assert.Equal("xunit", result.TestFramework)
        
        // Check that the generated code contains test methods for each method in the Calculator class
        Assert.Contains("Add", result.GeneratedTestCode)
        Assert.Contains("Subtract", result.GeneratedTestCode)
        Assert.Contains("Multiply", result.GeneratedTestCode)
        Assert.Contains("Divide", result.GeneratedTestCode)
    }
    
    /// <summary>
    /// Test that the test generator can generate tests for F# code.
    /// </summary>
    [<Fact>]
    let ``TestGenerator can generate tests for F# code``() = task {
        // Arrange
        let logger = MockLogger<TestGenerator>() :> ILogger<TestGenerator>
        
        let valueGeneratorLogger = MockLogger<TestValueGenerator>() :> ILogger<TestValueGenerator>
        let valueGenerator = TestValueGenerator(valueGeneratorLogger)
        
        let assertionFormatterLogger = MockLogger<XUnitAssertionFormatter>() :> ILogger<XUnitAssertionFormatter>
        let assertionFormatter = XUnitAssertionFormatter(assertionFormatterLogger)
        
        let assertionGeneratorLogger = MockLogger<PrimitiveAssertionGenerator>() :> ILogger<PrimitiveAssertionGenerator>
        let assertionGenerator = PrimitiveAssertionGenerator(assertionGeneratorLogger, [assertionFormatter])
        
        let codeAnalyzerLogger = MockLogger<TestCodeAnalyzer>() :> ILogger<TestCodeAnalyzer>
        let codeAnalyzer = TestCodeAnalyzer(codeAnalyzerLogger)
        
        let testCaseGeneratorLogger = MockLogger<BasicTestCaseGenerator>() :> ILogger<BasicTestCaseGenerator>
        let testCaseGenerator = BasicTestCaseGenerator(testCaseGeneratorLogger, valueGenerator, assertionGenerator, codeAnalyzer)
        
        let templateManagerLogger = MockLogger<TestTemplateManager>() :> ILogger<TestTemplateManager>
        let templateManager = TestTemplateManager(templateManagerLogger)
        
        let testGenerator = TestGenerator(logger, testCaseGenerator, templateManager)
        
        let code = """
            namespace MyApp

            module Calculator =
                let add a b = a + b
                
                let subtract a b = a - b
                
                let multiply a b = a * b
                
                let divide a b =
                    if b = 0 then
                        failwith "Division by zero"
                    else
                        a / b
            """
        
        // Act
        let! result = testGenerator.GenerateTests(code, "xunit")
        
        // Assert
        Assert.NotNull(result)
        Assert.NotEmpty(result.GeneratedTestCode)
        Assert.Equal(code, result.SourceCode)
        Assert.Equal("xunit", result.TestFramework)
        
        // Check that the generated code contains test methods for each function in the Calculator module
        Assert.Contains("add", result.GeneratedTestCode)
        Assert.Contains("subtract", result.GeneratedTestCode)
        Assert.Contains("multiply", result.GeneratedTestCode)
        Assert.Contains("divide", result.GeneratedTestCode)
    }
    
    /// <summary>
    /// Test that the test generator can suggest tests for C# code.
    /// </summary>
    [<Fact>]
    let ``TestGenerator can suggest tests for C# code``() = task {
        // Arrange
        let logger = MockLogger<TestGenerator>() :> ILogger<TestGenerator>
        
        let valueGeneratorLogger = MockLogger<TestValueGenerator>() :> ILogger<TestValueGenerator>
        let valueGenerator = TestValueGenerator(valueGeneratorLogger)
        
        let assertionFormatterLogger = MockLogger<XUnitAssertionFormatter>() :> ILogger<XUnitAssertionFormatter>
        let assertionFormatter = XUnitAssertionFormatter(assertionFormatterLogger)
        
        let assertionGeneratorLogger = MockLogger<PrimitiveAssertionGenerator>() :> ILogger<PrimitiveAssertionGenerator>
        let assertionGenerator = PrimitiveAssertionGenerator(assertionGeneratorLogger, [assertionFormatter])
        
        let codeAnalyzerLogger = MockLogger<TestCodeAnalyzer>() :> ILogger<TestCodeAnalyzer>
        let codeAnalyzer = TestCodeAnalyzer(codeAnalyzerLogger)
        
        let testCaseGeneratorLogger = MockLogger<BasicTestCaseGenerator>() :> ILogger<BasicTestCaseGenerator>
        let testCaseGenerator = BasicTestCaseGenerator(testCaseGeneratorLogger, valueGenerator, assertionGenerator, codeAnalyzer)
        
        let templateManagerLogger = MockLogger<TestTemplateManager>() :> ILogger<TestTemplateManager>
        let templateManager = TestTemplateManager(templateManagerLogger)
        
        let testGenerator = TestGenerator(logger, testCaseGenerator, templateManager)
        
        let code = """
            using System;

            namespace MyApp
            {
                public class Calculator
                {
                    public int Add(int a, int b)
                    {
                        return a + b;
                    }
                    
                    public int Subtract(int a, int b)
                    {
                        return a - b;
                    }
                    
                    public int Multiply(int a, int b)
                    {
                        return a * b;
                    }
                    
                    public int Divide(int a, int b)
                    {
                        if (b == 0)
                        {
                            throw new DivideByZeroException();
                        }
                        
                        return a / b;
                    }
                }
            }
            """
        
        // Act
        let! suggestedTests = testGenerator.SuggestTests(code, "xunit")
        
        // Assert
        Assert.NotEmpty(suggestedTests)
        
        // Check that there are suggested tests for each method in the Calculator class
        Assert.Contains(suggestedTests, fun t -> t.Target.Contains("Add"))
        Assert.Contains(suggestedTests, fun t -> t.Target.Contains("Subtract"))
        Assert.Contains(suggestedTests, fun t -> t.Target.Contains("Multiply"))
        Assert.Contains(suggestedTests, fun t -> t.Target.Contains("Divide"))
    }
