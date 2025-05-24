namespace TarsEngine.FSharp.Main.Functional

/// <summary>
/// Module for discriminated union types.
/// Provides a foundation for creating type-safe discriminated unions in F#.
/// </summary>
module DiscriminatedUnion =
    /// <summary>
    /// Helper function to throw an exception when a case is not handled in a pattern match
    /// </summary>
    /// <param name="union">The union that wasn't matched</param>
    /// <returns>Never returns, always throws</returns>
    /// <exception cref="System.InvalidOperationException">Always thrown</exception>
    let unhandledCase<'T> (union: obj) : 'T =
        raise (System.InvalidOperationException($"Unhandled case: {union.GetType().Name}"))
