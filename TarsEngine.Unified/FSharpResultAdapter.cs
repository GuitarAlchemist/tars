using System;
using System.Threading.Tasks;

namespace TarsEngine.Unified
{
    /// <summary>
    /// Represents a result that is either a success with a value or a failure with an error.
    /// </summary>
    /// <typeparam name="TSuccess">The success type</typeparam>
    /// <typeparam name="TFailure">The failure type (must be an Exception)</typeparam>
    public readonly struct FSharpResult<TSuccess, TFailure> where TFailure : Exception
    {
        private readonly bool _isSuccess;
        private readonly TSuccess _value;
        private readonly TFailure _error;

        private FSharpResult(TSuccess value, TFailure error, bool isSuccess)
        {
            _value = value;
            _error = error;
            _isSuccess = isSuccess;
        }



        /// <summary>
        /// Creates a success result.
        /// </summary>
        /// <param name="value">The success value</param>
        /// <returns>A success result</returns>
        public static FSharpResult<TSuccess, TFailure> Success(TSuccess value) =>
            new(value, default!, true);

        /// <summary>
        /// Creates a failure result.
        /// </summary>
        /// <param name="error">The error</param>
        /// <returns>A failure result</returns>
        public static FSharpResult<TSuccess, TFailure> Failure(TFailure error) =>
            new(default!, error, false);

        /// <summary>
        /// Returns true if the result is a success.
        /// </summary>
        public bool IsSuccess => _isSuccess;

        /// <summary>
        /// Returns true if the result is a failure.
        /// </summary>
        public bool IsFailure => !_isSuccess;

        /// <summary>
        /// Gets the success value or throws an exception if the result is a failure.
        /// </summary>
        public TSuccess Value => IsSuccess ? _value : throw new InvalidOperationException("Cannot get value from a failure result");

        /// <summary>
        /// Gets the error or throws an exception if the result is a success.
        /// </summary>
        public TFailure Error => IsFailure ? _error : throw new InvalidOperationException("Cannot get error from a success result");

        /// <summary>
        /// Gets the success value or a default value if the result is a failure.
        /// </summary>
        /// <param name="defaultValue">The default value</param>
        /// <returns>The success value or the default value</returns>
        public TSuccess GetValueOrDefault(TSuccess defaultValue) =>
            IsSuccess ? _value : defaultValue;

        /// <summary>
        /// Gets the success value or computes a default value if the result is a failure.
        /// </summary>
        /// <param name="defaultFn">The function to compute the default value</param>
        /// <returns>The success value or the computed default value</returns>
        public TSuccess GetValueOrElse(Func<TFailure, TSuccess> defaultFn) =>
            IsSuccess ? _value : defaultFn(_error);

        /// <summary>
        /// Maps the success value using the provided function.
        /// </summary>
        /// <typeparam name="TResult">The result type</typeparam>
        /// <param name="mapper">The mapping function</param>
        /// <returns>A new result with the mapped value</returns>
        public FSharpResult<TResult, TFailure> Map<TResult>(Func<TSuccess, TResult> mapper) =>
            IsSuccess ? FSharpResult<TResult, TFailure>.Success(mapper(_value)) : FSharpResult<TResult, TFailure>.Failure(_error);

        /// <summary>
        /// Maps the error using the provided function.
        /// </summary>
        /// <typeparam name="TNewFailure">The new error type</typeparam>
        /// <param name="mapper">The mapping function</param>
        /// <returns>A new result with the mapped error</returns>
        public FSharpResult<TSuccess, TNewFailure> MapError<TNewFailure>(Func<TFailure, TNewFailure> mapper) where TNewFailure : Exception =>
            IsSuccess ? FSharpResult<TSuccess, TNewFailure>.Success(_value) : FSharpResult<TSuccess, TNewFailure>.Failure(mapper(_error));

        /// <summary>
        /// Binds the success value using the provided function.
        /// </summary>
        /// <typeparam name="TResult">The result type</typeparam>
        /// <param name="binder">The binding function</param>
        /// <returns>A new result</returns>
        public FSharpResult<TResult, TFailure> Bind<TResult>(Func<TSuccess, FSharpResult<TResult, TFailure>> binder) =>
            IsSuccess ? binder(_value) : FSharpResult<TResult, TFailure>.Failure(_error);

        /// <summary>
        /// Applies a function to the success value and a function to the error.
        /// </summary>
        /// <typeparam name="TResult">The result type</typeparam>
        /// <param name="onSuccess">The function to apply to the success value</param>
        /// <param name="onFailure">The function to apply to the error</param>
        /// <returns>The result of applying the appropriate function</returns>
        public TResult Match<TResult>(Func<TSuccess, TResult> onSuccess, Func<TFailure, TResult> onFailure) =>
            IsSuccess ? onSuccess(_value) : onFailure(_error);

        /// <summary>
        /// Executes an action based on whether the result is a success or failure.
        /// </summary>
        /// <param name="onSuccess">The action to execute if the result is a success</param>
        /// <param name="onFailure">The action to execute if the result is a failure</param>
        public void Match(Action<TSuccess> onSuccess, Action<TFailure> onFailure)
        {
            if (IsSuccess)
                onSuccess(_value);
            else
                onFailure(_error);
        }

        /// <summary>
        /// Converts the result to a Task.
        /// </summary>
        /// <returns>A task that completes with the success value or fails with the error</returns>
        public Task<TSuccess> AsTask()
        {
            if (IsSuccess)
                return Task.FromResult(_value);
            else
                return Task.FromException<TSuccess>(_error);
        }

        /// <summary>
        /// Gets the underlying F# Result.
        /// </summary>
        public object AsFSharpResult() => this;
    }
}
