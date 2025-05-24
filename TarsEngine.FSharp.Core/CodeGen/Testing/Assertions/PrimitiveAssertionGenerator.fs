namespace TarsEngine.FSharp.Core.CodeGen.Testing.Assertions

open System
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.CodeGen.Testing.Models

/// <summary>
/// Generator for primitive type assertions.
/// </summary>
type PrimitiveAssertionGenerator(logger: ILogger<PrimitiveAssertionGenerator>, formatters: IAssertionFormatter seq) =
    inherit AssertionGeneratorBase(logger :> ILogger<AssertionGeneratorBase>, formatters)
    
    /// <summary>
    /// Generates assertions for a boolean value.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The list of generated assertions.</returns>
    member this.GenerateAssertionsForBoolean(expected: bool, actual: string, ?testFramework: string, ?language: string) =
        if expected then
            [this.GenerateTrueAssertion(actual, ?testFramework = testFramework, ?language = language)]
        else
            [this.GenerateFalseAssertion(actual, ?testFramework = testFramework, ?language = language)]
    
    /// <summary>
    /// Generates assertions for an integer value.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The list of generated assertions.</returns>
    member this.GenerateAssertionsForInteger(expected: int, actual: string, ?testFramework: string, ?language: string) =
        [this.GenerateEqualityAssertion(expected.ToString(), actual, ?testFramework = testFramework, ?language = language)]
    
    /// <summary>
    /// Generates assertions for a double value.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="delta">The allowed delta.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The list of generated assertions.</returns>
    member this.GenerateAssertionsForDouble(expected: double, actual: string, delta: double, ?testFramework: string, ?language: string) =
        let framework = defaultArg testFramework "xunit"
        let lang = defaultArg language "csharp"
        
        match this.GetFormatter(framework, lang) with
        | Some formatter ->
            let code = 
                match framework with
                | "xunit" -> $"Assert.Equal({expected}, {actual}, {delta});"
                | "nunit" -> $"Assert.AreEqual({expected}, {actual}, {delta});"
                | "mstest" -> $"Assert.AreEqual({expected}, {actual}, {delta});"
                | _ -> $"// No formatter found for test framework {framework}"
            
            [{
                Type = AssertionTypeExtensions.getName AssertionType.Equal
                ExpectedValue = Some (expected.ToString())
                ActualValue = actual
                Message = None
                Code = code
            }]
        | None ->
            logger.LogError("No formatter found for test framework {TestFramework} and language {Language}", framework, lang)
            [{
                Type = AssertionTypeExtensions.getName AssertionType.Equal
                ExpectedValue = Some (expected.ToString())
                ActualValue = actual
                Message = None
                Code = $"// No formatter found for test framework {framework} and language {lang}"
            }]
    
    /// <summary>
    /// Generates assertions for a string value.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The list of generated assertions.</returns>
    member this.GenerateAssertionsForString(expected: string, actual: string, ?testFramework: string, ?language: string) =
        if String.IsNullOrEmpty(expected) then
            [this.GenerateEmptyAssertion(actual, ?testFramework = testFramework, ?language = language)]
        else
            [this.GenerateEqualityAssertion($"\"{expected}\"", actual, ?testFramework = testFramework, ?language = language)]
    
    /// <summary>
    /// Generates assertions for a null value.
    /// </summary>
    /// <param name="actual">The actual value.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The list of generated assertions.</returns>
    member this.GenerateAssertionsForNull(actual: string, ?testFramework: string, ?language: string) =
        [this.GenerateNullAssertion(actual, ?testFramework = testFramework, ?language = language)]
    
    /// <summary>
    /// Generates assertions for a non-null value.
    /// </summary>
    /// <param name="actual">The actual value.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The list of generated assertions.</returns>
    member this.GenerateAssertionsForNotNull(actual: string, ?testFramework: string, ?language: string) =
        [this.GenerateNotNullAssertion(actual, ?testFramework = testFramework, ?language = language)]
    
    /// <summary>
    /// Generates assertions for a primitive value.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="testFramework">The test framework.</param>
    /// <param name="language">The language.</param>
    /// <returns>The list of generated assertions.</returns>
    member this.GenerateAssertionsForPrimitive(expected: obj, actual: string, ?testFramework: string, ?language: string) =
        if expected = null then
            this.GenerateAssertionsForNull(actual, ?testFramework = testFramework, ?language = language)
        else
            match expected with
            | :? bool as b -> this.GenerateAssertionsForBoolean(b, actual, ?testFramework = testFramework, ?language = language)
            | :? int as i -> this.GenerateAssertionsForInteger(i, actual, ?testFramework = testFramework, ?language = language)
            | :? double as d -> this.GenerateAssertionsForDouble(d, actual, 0.0001, ?testFramework = testFramework, ?language = language)
            | :? string as s -> this.GenerateAssertionsForString(s, actual, ?testFramework = testFramework, ?language = language)
            | _ -> [this.GenerateEqualityAssertion(expected.ToString(), actual, ?testFramework = testFramework, ?language = language)]
