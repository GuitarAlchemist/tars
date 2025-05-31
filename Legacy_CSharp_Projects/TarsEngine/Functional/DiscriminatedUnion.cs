namespace TarsEngine.Functional;

/// <summary>
/// Base class for discriminated union types.
/// Provides a foundation for creating type-safe discriminated unions in C#.
/// </summary>
public abstract record DiscriminatedUnion
{
    /// <summary>
    /// Prevents external instantiation of the base class
    /// </summary>
    private protected DiscriminatedUnion() { }

    /// <summary>
    /// Helper method to throw an exception when a case is not handled in a pattern match
    /// </summary>
    /// <param name="union">The union that wasn't matched</param>
    /// <returns>Never returns, always throws</returns>
    /// <exception cref="InvalidOperationException">Always thrown</exception>
    public static T UnhandledCase<T>(DiscriminatedUnion union)
    {
        throw new InvalidOperationException($"Unhandled case: {union.GetType().Name}");
    }
}

/// <summary>
/// Extension methods for discriminated unions
/// </summary>
public static class DiscriminatedUnionExtensions
{
    /// <summary>
    /// Performs an action based on the type of the discriminated union
    /// </summary>
    /// <typeparam name="TUnion">The type of the discriminated union</typeparam>
    /// <param name="union">The discriminated union</param>
    /// <param name="action">The action to perform</param>
    public static void Match<TUnion>(this TUnion union, Action<TUnion> action)
        where TUnion : DiscriminatedUnion
    {
        action(union);
    }

    /// <summary>
    /// Maps a discriminated union to a value based on its type
    /// </summary>
    /// <typeparam name="TUnion">The type of the discriminated union</typeparam>
    /// <typeparam name="TResult">The type of the result</typeparam>
    /// <param name="union">The discriminated union</param>
    /// <param name="mapper">The mapping function</param>
    /// <returns>The mapped value</returns>
    public static TResult Match<TUnion, TResult>(this TUnion union, Func<TUnion, TResult> mapper)
        where TUnion : DiscriminatedUnion
    {
        return mapper(union);
    }
}