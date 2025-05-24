namespace TarsEngine.FSharp.Core.CodeGen.Testing.Generators

open System
open System.Collections.Generic
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.CodeGen.Testing
open TarsEngine.FSharp.Core.CodeGen.Testing.Assertions
open TarsEngine.FSharp.Core.CodeGen.Testing.Interfaces
open TarsEngine.FSharp.Core.CodeGen.Testing.Models

/// <summary>
/// Basic implementation of ITestCaseGenerator.
/// </summary>
type BasicTestCaseGenerator(logger: ILogger<BasicTestCaseGenerator>, 
                            valueGenerator: TestValueGenerator,
                            assertionGenerator: PrimitiveAssertionGenerator,
                            codeAnalyzer: TestCodeAnalyzer) =
    
    /// <summary>
    /// Gets the supported test frameworks.
    /// </summary>
    /// <returns>The list of supported test frameworks.</returns>
    member _.GetSupportedTestFrameworks() =
        [
            "xunit"
            "nunit"
            "mstest"
        ]
    
    /// <summary>
    /// Gets the supported languages.
    /// </summary>
    /// <returns>The list of supported languages.</returns>
    member _.GetSupportedLanguages() =
        [
            "csharp"
            "fsharp"
        ]
    
    /// <summary>
    /// Generates a test case name for a method.
    /// </summary>
    /// <param name="method">The method to generate a test case name for.</param>
    /// <returns>The generated test case name.</returns>
    member _.GenerateTestCaseName(method: ExtractedMethod) =
        $"{method.Name}_Should"
    
    /// <summary>
    /// Generates a test case description for a method.
    /// </summary>
    /// <param name="method">The method to generate a test case description for.</param>
    /// <returns>The generated test case description.</returns>
    member _.GenerateTestCaseDescription(method: ExtractedMethod) =
        $"Test that {method.Name} works correctly"
    
    /// <summary>
    /// Generates parameter values for a method.
    /// </summary>
    /// <param name="method">The method to generate parameter values for.</param>
    /// <returns>The list of generated parameter values.</returns>
    member _.GenerateParameterValues(method: ExtractedMethod) =
        method.Parameters
        |> List.map (fun (name, typeName) ->
            let paramType = 
                match typeName with
                | "int" | "Int32" -> typeof<int>
                | "double" | "Double" -> typeof<double>
                | "float" | "Single" -> typeof<float32>
                | "decimal" | "Decimal" -> typeof<decimal>
                | "bool" | "Boolean" -> typeof<bool>
                | "string" | "String" -> typeof<string>
                | "DateTime" -> typeof<DateTime>
                | "Guid" -> typeof<Guid>
                | _ -> typeof<obj>
            
            let value = valueGenerator.GenerateValue(paramType)
            let valueCode = valueGenerator.GenerateValueCode(value, paramType, "csharp")
            
            {
                Name = name
                Type = typeName
                Value = valueCode
                SetupCode = None
            }
        )
    
    /// <summary>
    /// Generates a test case for a method.
    /// </summary>
    /// <param name="method">The method to generate a test case for.</param>
    /// <param name="testFramework">The test framework to use.</param>
    /// <param name="language">The language to use.</param>
    /// <returns>The generated test case.</returns>
    member this.GenerateTestCaseForMethod(method: ExtractedMethod, testFramework: string, language: string) =
        try
            logger.LogInformation("Generating test case for method: {MethodName}", method.Name)
            
            // Generate parameter values
            let parameters = this.GenerateParameterValues(method)
            
            // Generate execution code
            let parameterList = parameters |> List.map (fun p -> p.Value) |> String.concat ", "
            let executionCode = 
                if method.ReturnType <> "void" && method.ReturnType <> "unit" && method.ReturnType <> "Task" && method.ReturnType <> "Task<unit>" then
                    $"var result = {method.ClassName}.{method.Name}({parameterList});"
                else
                    $"{method.ClassName}.{method.Name}({parameterList});"
            
            // Generate assertions
            let assertions = 
                if method.ReturnType <> "void" && method.ReturnType <> "unit" && method.ReturnType <> "Task" && method.ReturnType <> "Task<unit>" then
                    // Generate assertions for the result
                    let resultType = 
                        match method.ReturnType with
                        | "int" | "Int32" -> typeof<int>
                        | "double" | "Double" -> typeof<double>
                        | "float" | "Single" -> typeof<float32>
                        | "decimal" | "Decimal" -> typeof<decimal>
                        | "bool" | "Boolean" -> typeof<bool>
                        | "string" | "String" -> typeof<string>
                        | "DateTime" -> typeof<DateTime>
                        | "Guid" -> typeof<Guid>
                        | _ -> typeof<obj>
                    
                    let expectedValue = valueGenerator.GenerateValue(resultType)
                    let expectedValueCode = valueGenerator.GenerateValueCode(expectedValue, resultType, language)
                    
                    assertionGenerator.GenerateAssertionsForPrimitive(expectedValue, "result", testFramework, language)
                else
                    // Generate assertions for void methods
                    [assertionGenerator.GenerateTrueAssertion("true", testFramework = testFramework, language = language)]
            
            // Create the test case
            {
                Name = this.GenerateTestCaseName(method)
                Description = this.GenerateTestCaseDescription(method)
                MethodName = method.Name
                ClassName = method.ClassName
                Namespace = method.Namespace
                Parameters = parameters
                SetupCode = None
                ExecutionCode = executionCode
                Result = 
                    if method.ReturnType <> "void" && method.ReturnType <> "unit" && method.ReturnType <> "Task" && method.ReturnType <> "Task<unit>" then
                        Some {
                            Type = method.ReturnType
                            Value = None
                            VariableName = Some "result"
                        }
                    else
                        None
                Assertions = assertions
                CleanupCode = None
                TestFramework = testFramework
                Language = language
                AdditionalInfo = Map.empty
            }
        with
        | ex ->
            logger.LogError(ex, "Error generating test case for method: {MethodName}", method.Name)
            {
                Name = $"{method.Name}_Test"
                Description = $"Test for {method.Name}"
                MethodName = method.Name
                ClassName = method.ClassName
                Namespace = method.Namespace
                Parameters = []
                SetupCode = None
                ExecutionCode = $"// Error generating test case: {ex.Message}"
                Result = None
                Assertions = []
                CleanupCode = None
                TestFramework = testFramework
                Language = language
                AdditionalInfo = Map.empty
            }
    
    /// <summary>
    /// Generates test cases for a method.
    /// </summary>
    /// <param name="method">The method to generate test cases for.</param>
    /// <param name="testFramework">The test framework to use.</param>
    /// <param name="language">The language to use.</param>
    /// <returns>The list of generated test cases.</returns>
    member this.GenerateTestCasesForMethod(method: ExtractedMethod, testFramework: string, language: string) =
        task {
            try
                logger.LogInformation("Generating test cases for method: {MethodName}", method.Name)
                
                // Generate a basic test case
                let basicTestCase = this.GenerateTestCaseForMethod(method, testFramework, language)
                
                // TODO: Generate additional test cases for edge cases, null inputs, etc.
                
                return [basicTestCase]
            with
            | ex ->
                logger.LogError(ex, "Error generating test cases for method: {MethodName}", method.Name)
                return []
        }
    
    /// <summary>
    /// Generates test cases for a class.
    /// </summary>
    /// <param name="class">The class to generate test cases for.</param>
    /// <param name="testFramework">The test framework to use.</param>
    /// <param name="language">The language to use.</param>
    /// <returns>The test suite containing the generated test cases.</returns>
    member this.GenerateTestCasesForClass(class': ExtractedClass, testFramework: string, language: string) =
        task {
            try
                logger.LogInformation("Generating test cases for class: {ClassName}", class'.Name)
                
                // Generate test cases for each method
                let! testCases = 
                    class'.Methods
                    |> List.filter (fun m -> not (m.Modifiers |> List.exists (fun mod -> mod = "private" || mod = "internal")))
                    |> List.map (fun m -> this.GenerateTestCasesForMethod(m, testFramework, language))
                    |> Task.WhenAll
                
                // Create the test suite
                return {
                    Name = $"{class'.Name}Tests"
                    Description = $"Tests for {class'.Name}"
                    ClassName = class'.Name
                    Namespace = class'.Namespace
                    TestCases = testCases |> Array.toList |> List.concat
                    SetupCode = None
                    CleanupCode = None
                    TestFramework = testFramework
                    Language = language
                    AdditionalInfo = Map.empty
                }
            with
            | ex ->
                logger.LogError(ex, "Error generating test cases for class: {ClassName}", class'.Name)
                return {
                    Name = $"{class'.Name}Tests"
                    Description = $"Tests for {class'.Name}"
                    ClassName = class'.Name
                    Namespace = class'.Namespace
                    TestCases = []
                    SetupCode = None
                    CleanupCode = None
                    TestFramework = testFramework
                    Language = language
                    AdditionalInfo = Map.empty
                }
        }
    
    /// <summary>
    /// Generates test cases for code.
    /// </summary>
    /// <param name="code">The code to generate test cases for.</param>
    /// <param name="testFramework">The test framework to use.</param>
    /// <param name="language">The language to use.</param>
    /// <returns>The list of test suites containing the generated test cases.</returns>
    member this.GenerateTestCasesForCode(code: string, testFramework: string, language: string) =
        task {
            try
                logger.LogInformation("Generating test cases for code")
                
                // Analyze the code
                let methods, classes = 
                    match language.ToLowerInvariant() with
                    | "csharp" -> codeAnalyzer.ExtractMethodsFromCSharp(code), codeAnalyzer.ExtractClassesFromCSharp(code)
                    | "fsharp" -> codeAnalyzer.ExtractMethodsFromFSharp(code), codeAnalyzer.ExtractClassesFromFSharp(code)
                    | _ -> [], []
                
                // Generate test suites for each class
                let! testSuites = 
                    classes
                    |> List.map (fun c -> this.GenerateTestCasesForClass(c, testFramework, language))
                    |> Task.WhenAll
                
                return testSuites |> Array.toList
            with
            | ex ->
                logger.LogError(ex, "Error generating test cases for code")
                return []
        }
    
    interface ITestCaseGenerator with
        member this.GenerateTestCasesForMethod(method, testFramework, language) = this.GenerateTestCasesForMethod(method, testFramework, language)
        member this.GenerateTestCasesForClass(class', testFramework, language) = this.GenerateTestCasesForClass(class', testFramework, language)
        member this.GenerateTestCasesForCode(code, testFramework, language) = this.GenerateTestCasesForCode(code, testFramework, language)
        member this.GetSupportedTestFrameworks() = this.GetSupportedTestFrameworks()
        member this.GetSupportedLanguages() = this.GetSupportedLanguages()
