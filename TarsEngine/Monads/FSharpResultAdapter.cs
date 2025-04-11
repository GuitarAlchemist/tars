using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.FSharp.Core;
using TarsEngine.Unified;

namespace TarsEngine.Monads
{
    /// <summary>
    /// Adapter for working with F# Result types in C#.
    /// </summary>
    /// <typeparam name="TSuccess">The success type</typeparam>
    /// <typeparam name="TFailure">The failure type (must be an Exception)</typeparam>
    public readonly struct FSharpResult<TSuccess, TFailure> where TFailure : Exception
    {
        private readonly TarsEngine.Unified.FSharpResult<TSuccess, TFailure> _adapter;

        /// <summary>
        /// Creates a new FSharpResult from an F# Result.
        /// </summary>
        /// <param name="adapter">The unified adapter</param>
        public FSharpResult(TarsEngine.Unified.FSharpResult<TSuccess, TFailure> adapter)
        {
            _adapter = adapter;
        }

        /// <summary>
        /// Creates a new FSharpResult from an F# Result.
        /// </summary>
        /// <param name="result">The F# Result</param>
        public FSharpResult(TarsEngineFSharp.Core.Monads.Result<TSuccess, TFailure> result)
        {
            // Convert F# Result to Unified FSharpResult
            _adapter = result switch
            {
                TarsEngineFSharp.Core.Monads.Result<TSuccess, TFailure>.Success success =>
                    TarsEngine.Unified.FSharpResult<TSuccess, TFailure>.Success(success.Item),
                TarsEngineFSharp.Core.Monads.Result<TSuccess, TFailure>.Failure failure =>
                    TarsEngine.Unified.FSharpResult<TSuccess, TFailure>.Failure(failure.Item),
                _ => throw new ArgumentException("Invalid F# Result type", nameof(result))
            };
        }

        /// <summary>
        /// Creates a success result.
        /// </summary>
        /// <param name="value">The success value</param>
        /// <returns>A success result</returns>
        public static FSharpResult<TSuccess, TFailure> Success(TSuccess value) =>
            new(TarsEngine.Unified.FSharpResult<TSuccess, TFailure>.Success(value));

        /// <summary>
        /// Creates a failure result.
        /// </summary>
        /// <param name="error">The error</param>
        /// <returns>A failure result</returns>
        public static FSharpResult<TSuccess, TFailure> Failure(TFailure error) =>
            new(TarsEngine.Unified.FSharpResult<TSuccess, TFailure>.Failure(error));

        /// <summary>
        /// Returns true if the result is a success.
        /// </summary>
        public bool IsSuccess => _adapter.IsSuccess;

        /// <summary>
        /// Returns true if the result is a failure.
        /// </summary>
        public bool IsFailure => _adapter.IsFailure;

        /// <summary>
        /// Gets the success value or throws an exception if the result is a failure.
        /// </summary>
        public TSuccess Value => _adapter.Value;

        /// <summary>
        /// Gets the error or throws an exception if the result is a success.
        /// </summary>
        public TFailure Error => _adapter.Error;

        /// <summary>
        /// Gets the success value or a default value if the result is a failure.
        /// </summary>
        /// <param name="defaultValue">The default value</param>
        /// <returns>The success value or the default value</returns>
        public TSuccess GetValueOrDefault(TSuccess defaultValue) =>
            _adapter.GetValueOrDefault(defaultValue);

        /// <summary>
        /// Gets the success value or computes a default value if the result is a failure.
        /// </summary>
        /// <param name="defaultFn">The function to compute the default value</param>
        /// <returns>The success value or the computed default value</returns>
        public TSuccess GetValueOrElse(Func<TFailure, TSuccess> defaultFn) =>
            _adapter.GetValueOrElse(defaultFn);

        /// <summary>
        /// Maps the success value using the provided function.
        /// </summary>
        /// <typeparam name="TResult">The result type</typeparam>
        /// <param name="mapper">The mapping function</param>
        /// <returns>A new result with the mapped value</returns>
        public FSharpResult<TResult, TFailure> Map<TResult>(Func<TSuccess, TResult> mapper) =>
            new(_adapter.Map(mapper));

        /// <summary>
        /// Maps the error using the provided function.
        /// </summary>
        /// <typeparam name="TNewFailure">The new error type</typeparam>
        /// <param name="mapper">The mapping function</param>
        /// <returns>A new result with the mapped error</returns>
        public FSharpResult<TSuccess, TNewFailure> MapError<TNewFailure>(Func<TFailure, TNewFailure> mapper) where TNewFailure : Exception =>
            new(_adapter.MapError(mapper));

        /// <summary>
        /// Binds the success value using the provided function.
        /// </summary>
        /// <typeparam name="TResult">The result type</typeparam>
        /// <param name="binder">The binding function</param>
        /// <returns>A new result</returns>
        public FSharpResult<TResult, TFailure> Bind<TResult>(Func<TSuccess, FSharpResult<TResult, TFailure>> binder) =>
            new(_adapter.Bind(value =>
            {
                var result = binder(value);
                return result._adapter; // Access the internal adapter directly
            }));

        /// <summary>
        /// Applies a function to the success value and a function to the error.
        /// </summary>
        /// <typeparam name="TResult">The result type</typeparam>
        /// <param name="onSuccess">The function to apply to the success value</param>
        /// <param name="onFailure">The function to apply to the error</param>
        /// <returns>The result of applying the appropriate function</returns>
        public TResult Match<TResult>(Func<TSuccess, TResult> onSuccess, Func<TFailure, TResult> onFailure) =>
            _adapter.Match(onSuccess, onFailure);

        /// <summary>
        /// Executes an action based on whether the result is a success or failure.
        /// </summary>
        /// <param name="onSuccess">The action to execute if the result is a success</param>
        /// <param name="onFailure">The action to execute if the result is a failure</param>
        public void Match(Action<TSuccess> onSuccess, Action<TFailure> onFailure) =>
            _adapter.Match(onSuccess, onFailure);

        /// <summary>
        /// Converts the result to a Task.
        /// </summary>
        /// <returns>A task that completes with the success value or fails with the error</returns>
        public Task<TSuccess> AsTask() =>
            _adapter.AsTask();

        /// <summary>
        /// Converts the F# Result to a C# Result.
        /// </summary>
        /// <returns>A C# Result</returns>
        public Result<TSuccess, TFailure> ToCSharpResult()
        {
            if (IsSuccess)
                return Result<TSuccess, TFailure>.Success(Value);
            else
                return Result<TSuccess, TFailure>.Failure(Error);
        }

        /// <summary>
        /// Gets the underlying F# Result.
        /// </summary>
        public object AsFSharpResult()
        {
            // Return the adapter itself as the F# Result
            return _adapter;
        }
    }

    /// <summary>
    /// Extension methods for working with F# Results.
    /// </summary>
    public static class FSharpResultExtensions
    {
        /// <summary>
        /// Converts a C# Result to an F# Result.
        /// </summary>
        /// <typeparam name="TSuccess">The success type</typeparam>
        /// <typeparam name="TFailure">The failure type</typeparam>
        /// <param name="result">The C# Result</param>
        /// <returns>An F# Result</returns>
        public static FSharpResult<TSuccess, TFailure> ToFSharpResult<TSuccess, TFailure>(this Result<TSuccess, TFailure> result) where TFailure : Exception
        {
            if (result.IsSuccess)
                return FSharpResult<TSuccess, TFailure>.Success(result.Value);
            else
                return FSharpResult<TSuccess, TFailure>.Failure(result.Error);
        }

        /// <summary>
        /// Tries to execute a function and returns a result.
        /// </summary>
        /// <typeparam name="TSuccess">The success type</typeparam>
        /// <param name="func">The function to execute</param>
        /// <returns>A result containing the function's return value or an exception</returns>
        public static FSharpResult<TSuccess, Exception> TryExecute<TSuccess>(Func<TSuccess> func)
        {
            try
            {
                var result = func();
                return FSharpResult<TSuccess, Exception>.Success(result);
            }
            catch (Exception ex)
            {
                return FSharpResult<TSuccess, Exception>.Failure(ex);
            }
        }

        /// <summary>
        /// Tries to execute an async function and returns a result.
        /// </summary>
        /// <typeparam name="TSuccess">The success type</typeparam>
        /// <param name="func">The async function to execute</param>
        /// <returns>A task containing the result</returns>
        public static async Task<FSharpResult<TSuccess, Exception>> TryExecuteAsync<TSuccess>(Func<Task<TSuccess>> func)
        {
            try
            {
                var result = await func();
                return FSharpResult<TSuccess, Exception>.Success(result);
            }
            catch (Exception ex)
            {
                return FSharpResult<TSuccess, Exception>.Failure(ex);
            }
        }

        /// <summary>
        /// Combines two results, returning a tuple of the success values if both are successful,
        /// or the first failure if either fails.
        /// </summary>
        public static FSharpResult<(T1, T2), TFailure> Zip<T1, T2, TFailure>(
            this FSharpResult<T1, TFailure> result1,
            FSharpResult<T2, TFailure> result2)
            where TFailure : Exception
        {
            if (result1.IsSuccess && result2.IsSuccess)
            {
                return FSharpResult<(T1, T2), TFailure>.Success((result1.Value, result2.Value));
            }
            else if (result1.IsFailure)
            {
                return FSharpResult<(T1, T2), TFailure>.Failure(result1.Error);
            }
            else
            {
                return FSharpResult<(T1, T2), TFailure>.Failure(result2.Error);
            }
        }

        /// <summary>
        /// Applies a function to each element in the sequence and collects the results.
        /// If any operation fails, the first failure is returned.
        /// </summary>
        public static FSharpResult<IEnumerable<TResult>, TFailure> Traverse<T, TResult, TFailure>(
            this IEnumerable<T> source,
            Func<T, FSharpResult<TResult, TFailure>> func)
            where TFailure : Exception
        {
            var results = new List<TResult>();

            foreach (var item in source)
            {
                var result = func(item);
                if (result.IsFailure)
                {
                    return FSharpResult<IEnumerable<TResult>, TFailure>.Failure(result.Error);
                }

                results.Add(result.Value);
            }

            return FSharpResult<IEnumerable<TResult>, TFailure>.Success(results);
        }
    }
}
