namespace TarsEngine.FSharp.Core.CodeGen.Testing.Assertions

/// <summary>
/// Represents the type of an assertion.
/// </summary>
type AssertionType =
    // Equality assertions
    | Equal
    | NotEqual
    | Same
    | NotSame
    
    // Comparison assertions
    | Greater
    | Less
    | GreaterOrEqual
    | LessOrEqual
    
    // Collection assertions
    | Contains
    | DoesNotContain
    | Empty
    | NotEmpty
    | Count
    | All
    | Any
    
    // Exception assertions
    | Throws
    | ThrowsAny
    | DoesNotThrow
    
    // Boolean assertions
    | True
    | False
    | Null
    | NotNull
    
    // String assertions
    | StartsWith
    | EndsWith
    | Contains
    | Matches
    | DoesNotMatch
    
    // Custom assertions
    | Custom

/// <summary>
/// Extensions for AssertionType.
/// </summary>
module AssertionTypeExtensions =
    /// <summary>
    /// Gets the name of an assertion type.
    /// </summary>
    /// <param name="assertionType">The assertion type.</param>
    /// <returns>The name of the assertion type.</returns>
    let getName (assertionType: AssertionType) =
        match assertionType with
        | Equal -> "Equal"
        | NotEqual -> "NotEqual"
        | Same -> "Same"
        | NotSame -> "NotSame"
        | Greater -> "Greater"
        | Less -> "Less"
        | GreaterOrEqual -> "GreaterOrEqual"
        | LessOrEqual -> "LessOrEqual"
        | Contains -> "Contains"
        | DoesNotContain -> "DoesNotContain"
        | Empty -> "Empty"
        | NotEmpty -> "NotEmpty"
        | Count -> "Count"
        | All -> "All"
        | Any -> "Any"
        | Throws -> "Throws"
        | ThrowsAny -> "ThrowsAny"
        | DoesNotThrow -> "DoesNotThrow"
        | True -> "True"
        | False -> "False"
        | Null -> "Null"
        | NotNull -> "NotNull"
        | StartsWith -> "StartsWith"
        | EndsWith -> "EndsWith"
        | Matches -> "Matches"
        | DoesNotMatch -> "DoesNotMatch"
        | Custom -> "Custom"
    
    /// <summary>
    /// Gets the description of an assertion type.
    /// </summary>
    /// <param name="assertionType">The assertion type.</param>
    /// <returns>The description of the assertion type.</returns>
    let getDescription (assertionType: AssertionType) =
        match assertionType with
        | Equal -> "Verifies that two values are equal"
        | NotEqual -> "Verifies that two values are not equal"
        | Same -> "Verifies that two references are the same instance"
        | NotSame -> "Verifies that two references are not the same instance"
        | Greater -> "Verifies that a value is greater than another value"
        | Less -> "Verifies that a value is less than another value"
        | GreaterOrEqual -> "Verifies that a value is greater than or equal to another value"
        | LessOrEqual -> "Verifies that a value is less than or equal to another value"
        | Contains -> "Verifies that a collection contains an element"
        | DoesNotContain -> "Verifies that a collection does not contain an element"
        | Empty -> "Verifies that a collection is empty"
        | NotEmpty -> "Verifies that a collection is not empty"
        | Count -> "Verifies that a collection has a specific count"
        | All -> "Verifies that all elements in a collection satisfy a condition"
        | Any -> "Verifies that at least one element in a collection satisfies a condition"
        | Throws -> "Verifies that a delegate throws a specific exception"
        | ThrowsAny -> "Verifies that a delegate throws any exception"
        | DoesNotThrow -> "Verifies that a delegate does not throw an exception"
        | True -> "Verifies that a condition is true"
        | False -> "Verifies that a condition is false"
        | Null -> "Verifies that a value is null"
        | NotNull -> "Verifies that a value is not null"
        | StartsWith -> "Verifies that a string starts with a specific substring"
        | EndsWith -> "Verifies that a string ends with a specific substring"
        | Matches -> "Verifies that a string matches a regular expression"
        | DoesNotMatch -> "Verifies that a string does not match a regular expression"
        | Custom -> "Custom assertion"
