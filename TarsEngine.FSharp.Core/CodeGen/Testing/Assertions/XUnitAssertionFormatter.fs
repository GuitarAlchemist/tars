namespace TarsEngine.FSharp.Core.CodeGen.Testing.Assertions

open System
open Microsoft.Extensions.Logging

/// <summary>
/// Formatter for xUnit assertions.
/// </summary>
type XUnitAssertionFormatter(logger: ILogger<XUnitAssertionFormatter>) =
    
    /// <summary>
    /// Gets the name of the test framework.
    /// </summary>
    member _.TestFramework = "xunit"
    
    /// <summary>
    /// Gets the language of the formatter.
    /// </summary>
    member _.Language = "csharp"
    
    /// <summary>
    /// Formats an equality assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatEqual(expected: string, actual: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.Equal({expected}, {actual}, \"{msg}\");"
        | None -> $"Assert.Equal({expected}, {actual});"
    
    /// <summary>
    /// Formats a non-equality assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatNotEqual(expected: string, actual: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.NotEqual({expected}, {actual}, \"{msg}\");"
        | None -> $"Assert.NotEqual({expected}, {actual});"
    
    /// <summary>
    /// Formats a same instance assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatSame(expected: string, actual: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.Same({expected}, {actual}, \"{msg}\");"
        | None -> $"Assert.Same({expected}, {actual});"
    
    /// <summary>
    /// Formats a not same instance assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatNotSame(expected: string, actual: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.NotSame({expected}, {actual}, \"{msg}\");"
        | None -> $"Assert.NotSame({expected}, {actual});"
    
    /// <summary>
    /// Formats a greater than assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatGreater(expected: string, actual: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.True({actual} > {expected}, \"{msg}\");"
        | None -> $"Assert.True({actual} > {expected});"
    
    /// <summary>
    /// Formats a less than assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatLess(expected: string, actual: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.True({actual} < {expected}, \"{msg}\");"
        | None -> $"Assert.True({actual} < {expected});"
    
    /// <summary>
    /// Formats a greater than or equal assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatGreaterOrEqual(expected: string, actual: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.True({actual} >= {expected}, \"{msg}\");"
        | None -> $"Assert.True({actual} >= {expected});"
    
    /// <summary>
    /// Formats a less than or equal assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatLessOrEqual(expected: string, actual: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.True({actual} <= {expected}, \"{msg}\");"
        | None -> $"Assert.True({actual} <= {expected});"
    
    /// <summary>
    /// Formats a contains assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatContains(expected: string, actual: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.Contains({expected}, {actual}, \"{msg}\");"
        | None -> $"Assert.Contains({expected}, {actual});"
    
    /// <summary>
    /// Formats a does not contain assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatDoesNotContain(expected: string, actual: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.DoesNotContain({expected}, {actual}, \"{msg}\");"
        | None -> $"Assert.DoesNotContain({expected}, {actual});"
    
    /// <summary>
    /// Formats an empty assertion.
    /// </summary>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatEmpty(actual: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.Empty({actual}, \"{msg}\");"
        | None -> $"Assert.Empty({actual});"
    
    /// <summary>
    /// Formats a not empty assertion.
    /// </summary>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatNotEmpty(actual: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.NotEmpty({actual}, \"{msg}\");"
        | None -> $"Assert.NotEmpty({actual});"
    
    /// <summary>
    /// Formats a count assertion.
    /// </summary>
    /// <param name="expected">The expected count.</param>
    /// <param name="actual">The actual collection.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatCount(expected: string, actual: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.Equal({expected}, {actual}.Count, \"{msg}\");"
        | None -> $"Assert.Equal({expected}, {actual}.Count);"
    
    /// <summary>
    /// Formats a throws assertion.
    /// </summary>
    /// <param name="exceptionType">The expected exception type.</param>
    /// <param name="action">The action that should throw.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatThrows(exceptionType: string, action: string, ?message: string) =
        match message with
        | Some msg -> $"var ex = Assert.Throws<{exceptionType}>(() => {action});\nAssert.NotNull(ex);\nAssert.Equal(\"{msg}\", ex.Message);"
        | None -> $"var ex = Assert.Throws<{exceptionType}>(() => {action});\nAssert.NotNull(ex);"
    
    /// <summary>
    /// Formats a throws any assertion.
    /// </summary>
    /// <param name="action">The action that should throw.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatThrowsAny(action: string, ?message: string) =
        match message with
        | Some msg -> $"var ex = Record.Exception(() => {action});\nAssert.NotNull(ex, \"{msg}\");"
        | None -> $"var ex = Record.Exception(() => {action});\nAssert.NotNull(ex);"
    
    /// <summary>
    /// Formats a does not throw assertion.
    /// </summary>
    /// <param name="action">The action that should not throw.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatDoesNotThrow(action: string, ?message: string) =
        match message with
        | Some msg -> $"var ex = Record.Exception(() => {action});\nAssert.Null(ex, \"{msg}\");"
        | None -> $"var ex = Record.Exception(() => {action});\nAssert.Null(ex);"
    
    /// <summary>
    /// Formats a true assertion.
    /// </summary>
    /// <param name="condition">The condition to check.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatTrue(condition: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.True({condition}, \"{msg}\");"
        | None -> $"Assert.True({condition});"
    
    /// <summary>
    /// Formats a false assertion.
    /// </summary>
    /// <param name="condition">The condition to check.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatFalse(condition: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.False({condition}, \"{msg}\");"
        | None -> $"Assert.False({condition});"
    
    /// <summary>
    /// Formats a null assertion.
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatNull(value: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.Null({value}, \"{msg}\");"
        | None -> $"Assert.Null({value});"
    
    /// <summary>
    /// Formats a not null assertion.
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatNotNull(value: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.NotNull({value}, \"{msg}\");"
        | None -> $"Assert.NotNull({value});"
    
    /// <summary>
    /// Formats a starts with assertion.
    /// </summary>
    /// <param name="expected">The expected prefix.</param>
    /// <param name="actual">The actual string.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatStartsWith(expected: string, actual: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.StartsWith({expected}, {actual}, \"{msg}\");"
        | None -> $"Assert.StartsWith({expected}, {actual});"
    
    /// <summary>
    /// Formats an ends with assertion.
    /// </summary>
    /// <param name="expected">The expected suffix.</param>
    /// <param name="actual">The actual string.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatEndsWith(expected: string, actual: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.EndsWith({expected}, {actual}, \"{msg}\");"
        | None -> $"Assert.EndsWith({expected}, {actual});"
    
    /// <summary>
    /// Formats a matches assertion.
    /// </summary>
    /// <param name="pattern">The expected pattern.</param>
    /// <param name="actual">The actual string.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatMatches(pattern: string, actual: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.Matches({pattern}, {actual}, \"{msg}\");"
        | None -> $"Assert.Matches({pattern}, {actual});"
    
    /// <summary>
    /// Formats a does not match assertion.
    /// </summary>
    /// <param name="pattern">The expected pattern.</param>
    /// <param name="actual">The actual string.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatDoesNotMatch(pattern: string, actual: string, ?message: string) =
        match message with
        | Some msg -> $"Assert.DoesNotMatch({pattern}, {actual}, \"{msg}\");"
        | None -> $"Assert.DoesNotMatch({pattern}, {actual});"
    
    /// <summary>
    /// Formats a custom assertion.
    /// </summary>
    /// <param name="code">The custom assertion code.</param>
    /// <returns>The formatted assertion.</returns>
    member _.FormatCustom(code: string) =
        if code.EndsWith(";") then code else code + ";"
    
    /// <summary>
    /// Formats an assertion based on the assertion type.
    /// </summary>
    /// <param name="assertionType">The type of assertion.</param>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    member this.Format(assertionType: AssertionType, ?expected: string, ?actual: string, ?message: string) =
        try
            match assertionType with
            | AssertionType.Equal -> 
                this.FormatEqual(expected.Value, actual.Value, ?message = message)
            | AssertionType.NotEqual -> 
                this.FormatNotEqual(expected.Value, actual.Value, ?message = message)
            | AssertionType.Same -> 
                this.FormatSame(expected.Value, actual.Value, ?message = message)
            | AssertionType.NotSame -> 
                this.FormatNotSame(expected.Value, actual.Value, ?message = message)
            | AssertionType.Greater -> 
                this.FormatGreater(expected.Value, actual.Value, ?message = message)
            | AssertionType.Less -> 
                this.FormatLess(expected.Value, actual.Value, ?message = message)
            | AssertionType.GreaterOrEqual -> 
                this.FormatGreaterOrEqual(expected.Value, actual.Value, ?message = message)
            | AssertionType.LessOrEqual -> 
                this.FormatLessOrEqual(expected.Value, actual.Value, ?message = message)
            | AssertionType.Contains -> 
                this.FormatContains(expected.Value, actual.Value, ?message = message)
            | AssertionType.DoesNotContain -> 
                this.FormatDoesNotContain(expected.Value, actual.Value, ?message = message)
            | AssertionType.Empty -> 
                this.FormatEmpty(actual.Value, ?message = message)
            | AssertionType.NotEmpty -> 
                this.FormatNotEmpty(actual.Value, ?message = message)
            | AssertionType.Count -> 
                this.FormatCount(expected.Value, actual.Value, ?message = message)
            | AssertionType.Throws -> 
                this.FormatThrows(expected.Value, actual.Value, ?message = message)
            | AssertionType.ThrowsAny -> 
                this.FormatThrowsAny(actual.Value, ?message = message)
            | AssertionType.DoesNotThrow -> 
                this.FormatDoesNotThrow(actual.Value, ?message = message)
            | AssertionType.True -> 
                this.FormatTrue(actual.Value, ?message = message)
            | AssertionType.False -> 
                this.FormatFalse(actual.Value, ?message = message)
            | AssertionType.Null -> 
                this.FormatNull(actual.Value, ?message = message)
            | AssertionType.NotNull -> 
                this.FormatNotNull(actual.Value, ?message = message)
            | AssertionType.StartsWith -> 
                this.FormatStartsWith(expected.Value, actual.Value, ?message = message)
            | AssertionType.EndsWith -> 
                this.FormatEndsWith(expected.Value, actual.Value, ?message = message)
            | AssertionType.Matches -> 
                this.FormatMatches(expected.Value, actual.Value, ?message = message)
            | AssertionType.DoesNotMatch -> 
                this.FormatDoesNotMatch(expected.Value, actual.Value, ?message = message)
            | AssertionType.Custom -> 
                match actual with
                | Some code -> this.FormatCustom(code)
                | None -> "// Custom assertion with no code"
            | AssertionType.All
            | AssertionType.Any -> 
                "// Not supported in xUnit"
        with
        | ex ->
            logger.LogError(ex, "Error formatting assertion: {AssertionType}", AssertionTypeExtensions.getName assertionType)
            "// Error formatting assertion"
    
    interface IAssertionFormatter with
        member this.TestFramework = this.TestFramework
        member this.Language = this.Language
        member this.FormatEqual(expected, actual, ?message) = this.FormatEqual(expected, actual, ?message = message)
        member this.FormatNotEqual(expected, actual, ?message) = this.FormatNotEqual(expected, actual, ?message = message)
        member this.FormatSame(expected, actual, ?message) = this.FormatSame(expected, actual, ?message = message)
        member this.FormatNotSame(expected, actual, ?message) = this.FormatNotSame(expected, actual, ?message = message)
        member this.FormatGreater(expected, actual, ?message) = this.FormatGreater(expected, actual, ?message = message)
        member this.FormatLess(expected, actual, ?message) = this.FormatLess(expected, actual, ?message = message)
        member this.FormatGreaterOrEqual(expected, actual, ?message) = this.FormatGreaterOrEqual(expected, actual, ?message = message)
        member this.FormatLessOrEqual(expected, actual, ?message) = this.FormatLessOrEqual(expected, actual, ?message = message)
        member this.FormatContains(expected, actual, ?message) = this.FormatContains(expected, actual, ?message = message)
        member this.FormatDoesNotContain(expected, actual, ?message) = this.FormatDoesNotContain(expected, actual, ?message = message)
        member this.FormatEmpty(actual, ?message) = this.FormatEmpty(actual, ?message = message)
        member this.FormatNotEmpty(actual, ?message) = this.FormatNotEmpty(actual, ?message = message)
        member this.FormatCount(expected, actual, ?message) = this.FormatCount(expected, actual, ?message = message)
        member this.FormatThrows(exceptionType, action, ?message) = this.FormatThrows(exceptionType, action, ?message = message)
        member this.FormatThrowsAny(action, ?message) = this.FormatThrowsAny(action, ?message = message)
        member this.FormatDoesNotThrow(action, ?message) = this.FormatDoesNotThrow(action, ?message = message)
        member this.FormatTrue(condition, ?message) = this.FormatTrue(condition, ?message = message)
        member this.FormatFalse(condition, ?message) = this.FormatFalse(condition, ?message = message)
        member this.FormatNull(value, ?message) = this.FormatNull(value, ?message = message)
        member this.FormatNotNull(value, ?message) = this.FormatNotNull(value, ?message = message)
        member this.FormatStartsWith(expected, actual, ?message) = this.FormatStartsWith(expected, actual, ?message = message)
        member this.FormatEndsWith(expected, actual, ?message) = this.FormatEndsWith(expected, actual, ?message = message)
        member this.FormatMatches(pattern, actual, ?message) = this.FormatMatches(pattern, actual, ?message = message)
        member this.FormatDoesNotMatch(pattern, actual, ?message) = this.FormatDoesNotMatch(pattern, actual, ?message = message)
        member this.FormatCustom(code) = this.FormatCustom(code)
        member this.Format(assertionType, ?expected, ?actual, ?message) = this.Format(assertionType, ?expected = expected, ?actual = actual, ?message = message)
