namespace TarsEngine.FSharp.Core.CodeGen.Testing.Assertions

open System
open System.Collections.Generic
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.CodeGen.Testing.Models

/// <summary>
/// Base class for assertion generators.
/// </summary>
type AssertionGeneratorBase(logger: ILogger<AssertionGeneratorBase>, formatters: IAssertionFormatter seq) =
    
    /// <summary>
    /// Gets a formatter for a test framework and language.
    /// </summary>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The formatter, if found.</returns>
    member _.GetFormatter(testFramework: string, language: string) =
        formatters 
        |> Seq.tryFind (fun f -> 
            f.TestFramework.Equals(testFramework, StringComparison.OrdinalIgnoreCase) && 
            f.Language.Equals(language, StringComparison.OrdinalIgnoreCase))
    
    /// <summary>
    /// Generates an assertion.
    /// </summary>
    /// <param name="assertionType">The type of assertion.</param>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The generated assertion.</returns>
    member this.GenerateAssertion(assertionType: AssertionType, ?expected: string, ?actual: string, ?message: string, ?testFramework: string, ?language: string) =
        try
            // Get the formatter
            let framework = defaultArg testFramework "xunit"
            let lang = defaultArg language "csharp"
            
            match this.GetFormatter(framework, lang) with
            | Some formatter ->
                // Format the assertion
                let code = formatter.Format(assertionType, ?expected = expected, ?actual = actual, ?message = message)
                
                // Create the assertion
                {
                    Type = AssertionTypeExtensions.getName assertionType
                    ExpectedValue = expected
                    ActualValue = defaultArg actual ""
                    Message = message
                    Code = code
                }
            | None ->
                logger.LogError("No formatter found for test framework {TestFramework} and language {Language}", framework, lang)
                {
                    Type = AssertionTypeExtensions.getName assertionType
                    ExpectedValue = expected
                    ActualValue = defaultArg actual ""
                    Message = message
                    Code = $"// No formatter found for test framework {framework} and language {lang}"
                }
        with
        | ex ->
            logger.LogError(ex, "Error generating assertion: {AssertionType}", AssertionTypeExtensions.getName assertionType)
            {
                Type = AssertionTypeExtensions.getName assertionType
                ExpectedValue = expected
                ActualValue = defaultArg actual ""
                Message = message
                Code = $"// Error generating assertion: {ex.Message}"
            }
    
    /// <summary>
    /// Generates an equality assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The generated assertion.</returns>
    member this.GenerateEqualityAssertion(expected: string, actual: string, ?message: string, ?testFramework: string, ?language: string) =
        this.GenerateAssertion(AssertionType.Equal, expected, actual, ?message = message, ?testFramework = testFramework, ?language = language)
    
    /// <summary>
    /// Generates a non-equality assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The generated assertion.</returns>
    member this.GenerateNotEqualAssertion(expected: string, actual: string, ?message: string, ?testFramework: string, ?language: string) =
        this.GenerateAssertion(AssertionType.NotEqual, expected, actual, ?message = message, ?testFramework = testFramework, ?language = language)
    
    /// <summary>
    /// Generates a true assertion.
    /// </summary>
    /// <param name="condition">The condition to check.</param>
    /// <param name="message">The assertion message.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The generated assertion.</returns>
    member this.GenerateTrueAssertion(condition: string, ?message: string, ?testFramework: string, ?language: string) =
        this.GenerateAssertion(AssertionType.True, actual = condition, ?message = message, ?testFramework = testFramework, ?language = language)
    
    /// <summary>
    /// Generates a false assertion.
    /// </summary>
    /// <param name="condition">The condition to check.</param>
    /// <param name="message">The assertion message.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The generated assertion.</returns>
    member this.GenerateFalseAssertion(condition: string, ?message: string, ?testFramework: string, ?language: string) =
        this.GenerateAssertion(AssertionType.False, actual = condition, ?message = message, ?testFramework = testFramework, ?language = language)
    
    /// <summary>
    /// Generates a null assertion.
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <param name="message">The assertion message.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The generated assertion.</returns>
    member this.GenerateNullAssertion(value: string, ?message: string, ?testFramework: string, ?language: string) =
        this.GenerateAssertion(AssertionType.Null, actual = value, ?message = message, ?testFramework = testFramework, ?language = language)
    
    /// <summary>
    /// Generates a not null assertion.
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <param name="message">The assertion message.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The generated assertion.</returns>
    member this.GenerateNotNullAssertion(value: string, ?message: string, ?testFramework: string, ?language: string) =
        this.GenerateAssertion(AssertionType.NotNull, actual = value, ?message = message, ?testFramework = testFramework, ?language = language)
    
    /// <summary>
    /// Generates a throws assertion.
    /// </summary>
    /// <param name="exceptionType">The expected exception type.</param>
    /// <param name="action">The action that should throw.</param>
    /// <param name="message">The assertion message.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The generated assertion.</returns>
    member this.GenerateThrowsAssertion(exceptionType: string, action: string, ?message: string, ?testFramework: string, ?language: string) =
        this.GenerateAssertion(AssertionType.Throws, exceptionType, action, ?message = message, ?testFramework = testFramework, ?language = language)
    
    /// <summary>
    /// Generates a does not throw assertion.
    /// </summary>
    /// <param name="action">The action that should not throw.</param>
    /// <param name="message">The assertion message.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The generated assertion.</returns>
    member this.GenerateDoesNotThrowAssertion(action: string, ?message: string, ?testFramework: string, ?language: string) =
        this.GenerateAssertion(AssertionType.DoesNotThrow, actual = action, ?message = message, ?testFramework = testFramework, ?language = language)
    
    /// <summary>
    /// Generates a contains assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The generated assertion.</returns>
    member this.GenerateContainsAssertion(expected: string, actual: string, ?message: string, ?testFramework: string, ?language: string) =
        this.GenerateAssertion(AssertionType.Contains, expected, actual, ?message = message, ?testFramework = testFramework, ?language = language)
    
    /// <summary>
    /// Generates a does not contain assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The generated assertion.</returns>
    member this.GenerateDoesNotContainAssertion(expected: string, actual: string, ?message: string, ?testFramework: string, ?language: string) =
        this.GenerateAssertion(AssertionType.DoesNotContain, expected, actual, ?message = message, ?testFramework = testFramework, ?language = language)
    
    /// <summary>
    /// Generates an empty assertion.
    /// </summary>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The generated assertion.</returns>
    member this.GenerateEmptyAssertion(actual: string, ?message: string, ?testFramework: string, ?language: string) =
        this.GenerateAssertion(AssertionType.Empty, actual = actual, ?message = message, ?testFramework = testFramework, ?language = language)
    
    /// <summary>
    /// Generates a not empty assertion.
    /// </summary>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The generated assertion.</returns>
    member this.GenerateNotEmptyAssertion(actual: string, ?message: string, ?testFramework: string, ?language: string) =
        this.GenerateAssertion(AssertionType.NotEmpty, actual = actual, ?message = message, ?testFramework = testFramework, ?language = language)
    
    /// <summary>
    /// Generates a custom assertion.
    /// </summary>
    /// <param name="code">The custom assertion code.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The generated assertion.</returns>
    member this.GenerateCustomAssertion(code: string, ?testFramework: string, ?language: string) =
        this.GenerateAssertion(AssertionType.Custom, actual = code, ?testFramework = testFramework, ?language = language)
