using System;
using System.Threading.Tasks;

namespace DuplicationAnalyzerTests.Monads;

/// <summary>
/// Represents the result of an operation that can succeed or fail
/// </summary>
/// <typeparam name="T">The type of the result value</typeparam>
public class Result<T>
{
    /// <summary>
    /// Gets a value indicating whether the operation was successful
    /// </summary>
    public bool IsSuccess { get; }

    /// <summary>
    /// Gets the value of the result (if successful)
    /// </summary>
    public T Value { get; }

    /// <summary>
    /// Gets the error message (if failed)
    /// </summary>
    public string Error { get; }

    /// <summary>
    /// Gets the exception (if failed)
    /// </summary>
    public Exception? Exception { get; }

    private Result(bool isSuccess, T? value, string error, Exception? exception)
    {
        IsSuccess = isSuccess;
        Value = value!;
        Error = error;
        Exception = exception;
    }

    /// <summary>
    /// Creates a successful result
    /// </summary>
    /// <param name="value">The result value</param>
    /// <returns>A successful result</returns>
    public static Result<T> Success(T value) => new Result<T>(true, value, string.Empty, null);

    /// <summary>
    /// Creates a failed result
    /// </summary>
    /// <param name="error">The error message</param>
    /// <returns>A failed result</returns>
    public static Result<T> Failure(string error) => new Result<T>(false, default!, error, null);

    /// <summary>
    /// Creates a failed result
    /// </summary>
    /// <param name="exception">The exception</param>
    /// <returns>A failed result</returns>
    public static Result<T> Failure(Exception exception) => new Result<T>(false, default!, exception.Message, exception);

    /// <summary>
    /// Binds the result to another operation
    /// </summary>
    /// <typeparam name="TResult">The type of the new result</typeparam>
    /// <param name="func">The function to apply to the value</param>
    /// <returns>The new result</returns>
    public Result<TResult> Bind<TResult>(Func<T, Result<TResult>> func)
    {
        if (!IsSuccess)
        {
            return Result<TResult>.Failure(Error);
        }

        try
        {
            return func(Value);
        }
        catch (Exception ex)
        {
            return Result<TResult>.Failure(ex);
        }
    }

    /// <summary>
    /// Maps the result to a new value
    /// </summary>
    /// <typeparam name="TResult">The type of the new result</typeparam>
    /// <param name="func">The function to apply to the value</param>
    /// <returns>The new result</returns>
    public Result<TResult> Map<TResult>(Func<T, TResult> func)
    {
        if (!IsSuccess)
        {
            return Result<TResult>.Failure(Error);
        }

        try
        {
            return Result<TResult>.Success(func(Value));
        }
        catch (Exception ex)
        {
            return Result<TResult>.Failure(ex);
        }
    }

    /// <summary>
    /// Binds the result to an async operation
    /// </summary>
    /// <typeparam name="TResult">The type of the new result</typeparam>
    /// <param name="func">The async function to apply to the value</param>
    /// <returns>The new result</returns>
    public async Task<Result<TResult>> BindAsync<TResult>(Func<T, Task<Result<TResult>>> func)
    {
        if (!IsSuccess)
        {
            return Result<TResult>.Failure(Error);
        }

        try
        {
            return await func(Value);
        }
        catch (Exception ex)
        {
            return Result<TResult>.Failure(ex);
        }
    }

    /// <summary>
    /// Maps the result to a new async value
    /// </summary>
    /// <typeparam name="TResult">The type of the new result</typeparam>
    /// <param name="func">The async function to apply to the value</param>
    /// <returns>The new result</returns>
    public async Task<Result<TResult>> MapAsync<TResult>(Func<T, Task<TResult>> func)
    {
        if (!IsSuccess)
        {
            return Result<TResult>.Failure(Error);
        }

        try
        {
            return Result<TResult>.Success(await func(Value));
        }
        catch (Exception ex)
        {
            return Result<TResult>.Failure(ex);
        }
    }
}

/// <summary>
/// Static helper methods for Result monad
/// </summary>
public static class Result
{
    /// <summary>
    /// Creates a successful result
    /// </summary>
    /// <typeparam name="T">The type of the result value</typeparam>
    /// <param name="value">The result value</param>
    /// <returns>A successful result</returns>
    public static Result<T> Success<T>(T value) => Result<T>.Success(value);

    /// <summary>
    /// Creates a failed result
    /// </summary>
    /// <typeparam name="T">The type of the result value</typeparam>
    /// <param name="error">The error message</param>
    /// <returns>A failed result</returns>
    public static Result<T> Failure<T>(string error) => Result<T>.Failure(error);

    /// <summary>
    /// Creates a failed result
    /// </summary>
    /// <typeparam name="T">The type of the result value</typeparam>
    /// <param name="exception">The exception</param>
    /// <returns>A failed result</returns>
    public static Result<T> Failure<T>(Exception exception) => Result<T>.Failure(exception);

    /// <summary>
    /// Tries to execute a function and returns a result
    /// </summary>
    /// <typeparam name="T">The type of the result value</typeparam>
    /// <param name="func">The function to execute</param>
    /// <returns>The result of the function</returns>
    public static Result<T> Try<T>(Func<T> func)
    {
        try
        {
            return Success(func());
        }
        catch (Exception ex)
        {
            return Failure<T>(ex);
        }
    }

    /// <summary>
    /// Tries to execute an async function and returns a result
    /// </summary>
    /// <typeparam name="T">The type of the result value</typeparam>
    /// <param name="func">The async function to execute</param>
    /// <returns>The result of the function</returns>
    public static async Task<Result<T>> TryAsync<T>(Func<Task<T>> func)
    {
        try
        {
            return Success(await func());
        }
        catch (Exception ex)
        {
            return Failure<T>(ex);
        }
    }
}
