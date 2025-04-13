namespace TarsEngine.Monads;

/// <summary>
/// Utility class for working with Task monad to avoid async methods without await
/// </summary>
public static class TaskMonad
{
    /// <summary>
    /// Creates a completed task with the specified result
    /// </summary>
    public static Task<T> Pure<T>(T value) => Task.FromResult(value);

    /// <summary>
    /// Creates a completed task with no result
    /// </summary>
    public static Task PureUnit() => Task.CompletedTask;

    /// <summary>
    /// Applies a function to the result of a task
    /// </summary>
    public static Task<TResult> Map<T, TResult>(this Task<T> task, Func<T, TResult> mapper) =>
        task.ContinueWith(t => mapper(t.Result));

    /// <summary>
    /// Applies a function that returns a task to the result of a task
    /// </summary>
    public static Task<TResult> Bind<T, TResult>(this Task<T> task, Func<T, Task<TResult>> binder) =>
        task.ContinueWith(t => binder(t.Result)).Unwrap();

    /// <summary>
    /// Applies an action to the result of a task
    /// </summary>
    public static Task<T> Do<T>(this Task<T> task, Action<T> action) =>
        task.ContinueWith(t => {
            action(t.Result);
            return t.Result;
        });

    /// <summary>
    /// Converts a synchronous function to an asynchronous one
    /// </summary>
    public static Task<T> ToTask<T>(this Func<T> func) => Task.FromResult(func());

    /// <summary>
    /// Converts a synchronous action to an asynchronous one
    /// </summary>
    public static Task ToTask(this Action action) => Task.Run(action);

    /// <summary>
    /// Converts a synchronous function to an asynchronous one with a delay
    /// </summary>
    public static async Task<T> ToTaskWithDelay<T>(this Func<T> func, int delayMs)
    {
        await Task.Delay(delayMs);
        return func();
    }

    /// <summary>
    /// Converts a synchronous action to an asynchronous one with a delay
    /// </summary>
    public static async Task ToTaskWithDelay(this Action action, int delayMs)
    {
        await Task.Delay(delayMs);
        action();
    }
}