namespace TarsEngine.FSharp.Core.CodeGen.Testing.Assertions

/// <summary>
/// Interface for formatting assertions.
/// </summary>
type IAssertionFormatter =
    /// <summary>
    /// Gets the name of the test framework.
    /// </summary>
    abstract member TestFramework : string
    
    /// <summary>
    /// Gets the language of the formatter.
    /// </summary>
    abstract member Language : string
    
    /// <summary>
    /// Formats an equality assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatEqual : expected:string * actual:string * ?message:string -> string
    
    /// <summary>
    /// Formats a non-equality assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatNotEqual : expected:string * actual:string * ?message:string -> string
    
    /// <summary>
    /// Formats a same instance assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatSame : expected:string * actual:string * ?message:string -> string
    
    /// <summary>
    /// Formats a not same instance assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatNotSame : expected:string * actual:string * ?message:string -> string
    
    /// <summary>
    /// Formats a greater than assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatGreater : expected:string * actual:string * ?message:string -> string
    
    /// <summary>
    /// Formats a less than assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatLess : expected:string * actual:string * ?message:string -> string
    
    /// <summary>
    /// Formats a greater than or equal assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatGreaterOrEqual : expected:string * actual:string * ?message:string -> string
    
    /// <summary>
    /// Formats a less than or equal assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatLessOrEqual : expected:string * actual:string * ?message:string -> string
    
    /// <summary>
    /// Formats a contains assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatContains : expected:string * actual:string * ?message:string -> string
    
    /// <summary>
    /// Formats a does not contain assertion.
    /// </summary>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatDoesNotContain : expected:string * actual:string * ?message:string -> string
    
    /// <summary>
    /// Formats an empty assertion.
    /// </summary>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatEmpty : actual:string * ?message:string -> string
    
    /// <summary>
    /// Formats a not empty assertion.
    /// </summary>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatNotEmpty : actual:string * ?message:string -> string
    
    /// <summary>
    /// Formats a count assertion.
    /// </summary>
    /// <param name="expected">The expected count.</param>
    /// <param name="actual">The actual collection.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatCount : expected:string * actual:string * ?message:string -> string
    
    /// <summary>
    /// Formats a throws assertion.
    /// </summary>
    /// <param name="exceptionType">The expected exception type.</param>
    /// <param name="action">The action that should throw.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatThrows : exceptionType:string * action:string * ?message:string -> string
    
    /// <summary>
    /// Formats a throws any assertion.
    /// </summary>
    /// <param name="action">The action that should throw.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatThrowsAny : action:string * ?message:string -> string
    
    /// <summary>
    /// Formats a does not throw assertion.
    /// </summary>
    /// <param name="action">The action that should not throw.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatDoesNotThrow : action:string * ?message:string -> string
    
    /// <summary>
    /// Formats a true assertion.
    /// </summary>
    /// <param name="condition">The condition to check.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatTrue : condition:string * ?message:string -> string
    
    /// <summary>
    /// Formats a false assertion.
    /// </summary>
    /// <param name="condition">The condition to check.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatFalse : condition:string * ?message:string -> string
    
    /// <summary>
    /// Formats a null assertion.
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatNull : value:string * ?message:string -> string
    
    /// <summary>
    /// Formats a not null assertion.
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatNotNull : value:string * ?message:string -> string
    
    /// <summary>
    /// Formats a starts with assertion.
    /// </summary>
    /// <param name="expected">The expected prefix.</param>
    /// <param name="actual">The actual string.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatStartsWith : expected:string * actual:string * ?message:string -> string
    
    /// <summary>
    /// Formats an ends with assertion.
    /// </summary>
    /// <param name="expected">The expected suffix.</param>
    /// <param name="actual">The actual string.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatEndsWith : expected:string * actual:string * ?message:string -> string
    
    /// <summary>
    /// Formats a matches assertion.
    /// </summary>
    /// <param name="pattern">The expected pattern.</param>
    /// <param name="actual">The actual string.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatMatches : pattern:string * actual:string * ?message:string -> string
    
    /// <summary>
    /// Formats a does not match assertion.
    /// </summary>
    /// <param name="pattern">The expected pattern.</param>
    /// <param name="actual">The actual string.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatDoesNotMatch : pattern:string * actual:string * ?message:string -> string
    
    /// <summary>
    /// Formats a custom assertion.
    /// </summary>
    /// <param name="code">The custom assertion code.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member FormatCustom : code:string -> string
    
    /// <summary>
    /// Formats an assertion based on the assertion type.
    /// </summary>
    /// <param name="assertionType">The type of assertion.</param>
    /// <param name="expected">The expected value.</param>
    /// <param name="actual">The actual value.</param>
    /// <param name="message">The assertion message.</param>
    /// <returns>The formatted assertion.</returns>
    abstract member Format : assertionType:AssertionType * ?expected:string * ?actual:string * ?message:string -> string
