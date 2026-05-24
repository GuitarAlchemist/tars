namespace TarsEngine.Monads;

/// <summary>
/// Represents an asynchronous operation that might fail.
/// Used to handle async operations that might fail in a functional way.
/// </summary>
/// <typeparam name="T">The type of the success value</typeparam>
/// <typeparam name="TError">The type of the error value</typeparam>
public class AsyncResultError<T, TError>
{
    private readonly Task<Result<T, TError>> _task;

    private AsyncResultError(Task<Result<T, TError>> task)
    {
        _task = task;
    }

    /// <summary>
    /// Creates a successful AsyncResultError with a value
    /// </summary>
    public static AsyncResultError<T, TError> Success(T value) =>
        new(Task.FromResult(Result<T, TError>.Success(value)));

    /// <summary>
    /// Creates a failed AsyncResultError with an error
    /// </summary>
    public static AsyncResultError<T, TError> Failure(TError error) =>
        new(Task.FromResult(Result<T, TError>.Failure(error)));

    /// <summary>
    /// Creates an AsyncResultError from a task that returns a Result
    /// </summary>
    public static AsyncResultError<T, TError> FromTask(Task<Result<T, TError>> task) =>
        new(task);

    /// <summary>
    /// Creates an AsyncResultError from a task that returns a value
    /// </summary>
    public static AsyncResultError<T, TError> FromValueTask(Task<T> task, Func<Exception, TError> errorMapper) =>
        new(task.ContinueWith(t => t.IsFaulted ? Result<T, TError>.Failure(errorMapper(t.Exception!.InnerException!)) : Result<T, TError>.Success(t.Result)));

    /// <summary>
    /// Runs the async operation and returns the result
    /// </summary>
    public Task<Result<T, TError>> RunAsync() => _task;

    /// <summary>
    /// Applies a function to the value if successful when the operation completes
    /// </summary>
    public AsyncResultError<TResult, TError> Map<TResult>(Func<T, TResult> mapper) =>
        new(_task.ContinueWith(t => t.Result.Map(mapper)));

    /// <summary>
    /// Applies a function that returns a Result to the value if successful when the operation completes
    /// </summary>
    public AsyncResultError<TResult, TError> Bind<TResult>(Func<T, Result<TResult, TError>> binder) =>
        new(_task.ContinueWith(t => t.Result.Bind(binder)));

    /// <summary>
    /// Applies a function that returns an AsyncResultError to the value if successful when the operation completes
    /// </summary>
    public AsyncResultError<TResult, TError> Bind<TResult>(Func<T, AsyncResultError<TResult, TError>> binder) =>
        new(_task.ContinueWith(t =>
            t.Result.Match(
                success: value => binder(value)._task,
                failure: error => Task.FromResult(Result<TResult, TError>.Failure(error))
            )).Unwrap());

    /// <summary>
    /// Performs an action on the value if successful when the operation completes
    /// </summary>
    public AsyncResultError<T, TError> Do(Action<T> action) =>
        new(_task.ContinueWith(t => {
            t.Result.IfSuccess(action);
            return t.Result;
        }));
}

/// <summary>
/// A simpler version of AsyncResultError that uses Exception as the error type
/// </summary>
public static class AsyncResultError
{
    /// <summary>
    /// Creates a successful AsyncResultError with a value
    /// </summary>
    public static AsyncResultError<T, Exception> Success<T>(T value) =>
        AsyncResultError<T, Exception>.Success(value);

    /// <summary>
    /// Creates a failed AsyncResultError with an exception
    /// </summary>
    public static AsyncResultError<T, Exception> Failure<T>(Exception error) =>
        AsyncResultError<T, Exception>.Failure(error);

    /// <summary>
    /// Tries to execute a function asynchronously and returns an AsyncResultError
    /// </summary>
    public static AsyncResultError<T, Exception> TryAsync<T>(Func<Task<T>> func)
    {
        try
        {
            return AsyncResultError<T, Exception>.FromValueTask(func(), ex => ex);
        }
        catch (Exception ex)
        {
            return Failure<T>(ex);
        }
    }
}