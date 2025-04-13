namespace TarsEngine.Monads;

/// <summary>
/// Extension methods for the Result monad to support async operations
/// </summary>
public static class ResultExtensions
{
    /// <summary>
    /// Applies an async function to the value if successful, or returns a failure with the same error if not
    /// </summary>
    public static async Task<Result<TResult, TError>> MapAsync<T, TResult, TError>(
        this Result<T, TError> result, 
        Func<T, Task<TResult>> mapper)
    {
        return result.IsSuccess
            ? Result<TResult, TError>.Success(await mapper(result.Value))
            : Result<TResult, TError>.Failure(result.Error);
    }

    /// <summary>
    /// Applies an async function that returns a Result to the value if successful, 
    /// or returns a failure with the same error if not
    /// </summary>
    public static async Task<Result<TResult, TError>> BindAsync<T, TResult, TError>(
        this Result<T, TError> result, 
        Func<T, Task<Result<TResult, TError>>> binder)
    {
        return result.IsSuccess
            ? await binder(result.Value)
            : Result<TResult, TError>.Failure(result.Error);
    }

    /// <summary>
    /// Tries to execute an async function and returns a Result
    /// </summary>
    public static async Task<Result<T, Exception>> TryAsync<T>(Func<Task<T>> func)
    {
        try
        {
            return Result.Success(await func());
        }
        catch (Exception ex)
        {
            return Result.Failure<T>(ex);
        }
    }

    /// <summary>
    /// Performs an async action if the result is successful
    /// </summary>
    public static async Task<Result<T, TError>> IfSuccessAsync<T, TError>(
        this Result<T, TError> result, 
        Func<T, Task> action)
    {
        if (result.IsSuccess)
        {
            await action(result.Value);
        }
        return result;
    }

    /// <summary>
    /// Performs an async action if the result is a failure
    /// </summary>
    public static async Task<Result<T, TError>> IfFailureAsync<T, TError>(
        this Result<T, TError> result, 
        Func<TError, Task> action)
    {
        if (result.IsFailure)
        {
            await action(result.Error);
        }
        return result;
    }
}