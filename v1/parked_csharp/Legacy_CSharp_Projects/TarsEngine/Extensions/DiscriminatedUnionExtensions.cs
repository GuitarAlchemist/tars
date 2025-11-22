namespace TarsEngine.Functional;

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