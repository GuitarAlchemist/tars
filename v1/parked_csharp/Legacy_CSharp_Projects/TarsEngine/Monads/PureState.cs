namespace TarsEngine.Monads;

/// <summary>
/// Base class for pure state objects that don't contain behavior
/// </summary>
/// <typeparam name="T">The type of the state</typeparam>
public abstract class PureState<T> where T : PureState<T>
{
    /// <summary>
    /// Creates a copy of the state with the specified modifications
    /// </summary>
    public abstract T With(Action<T> modifier);

    /// <summary>
    /// Creates a copy of the state
    /// </summary>
    public abstract T Copy();
}

/// <summary>
/// Extension methods for working with pure state objects
/// </summary>
public static class PureStateExtensions
{
    /// <summary>
    /// Creates a task that returns the state
    /// </summary>
    public static Task<T> AsTask<T>(this T state) where T : PureState<T> =>
        Task.FromResult(state);

    /// <summary>
    /// Creates a task that returns a modified copy of the state
    /// </summary>
    public static Task<T> AsTaskWith<T>(this T state, Action<T> modifier) where T : PureState<T> =>
        Task.FromResult(state.With(modifier));

    /// <summary>
    /// Creates an AsyncResult that returns the state
    /// </summary>
    public static AsyncResult<T> AsAsyncResult<T>(this T state) where T : PureState<T> =>
        AsyncResult<T>.FromResult(state);

    /// <summary>
    /// Creates an AsyncResult that returns a modified copy of the state
    /// </summary>
    public static AsyncResult<T> AsAsyncResultWith<T>(this T state, Action<T> modifier) where T : PureState<T> =>
        AsyncResult<T>.FromResult(state.With(modifier));
}