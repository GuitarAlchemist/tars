using System;

namespace TarsEngine.Monads
{
    /// <summary>
    /// Represents the result of an operation that might fail.
    /// Used to handle errors in a functional way.
    /// </summary>
    /// <typeparam name="T">The type of the success value</typeparam>
    /// <typeparam name="TError">The type of the error value</typeparam>
    public readonly struct Result<T, TError>
    {
        private readonly T _value;
        private readonly TError _error;
        private readonly bool _isSuccess;

        private Result(T value, TError error, bool isSuccess)
        {
            _value = value;
            _error = error;
            _isSuccess = isSuccess;
        }

        /// <summary>
        /// Creates a successful Result with a value
        /// </summary>
        public static Result<T, TError> Success(T value) =>
            new(value, default!, true);

        /// <summary>
        /// Creates a failed Result with an error
        /// </summary>
        public static Result<T, TError> Failure(TError error) =>
            new(default!, error, false);

        /// <summary>
        /// Returns true if the result is successful
        /// </summary>
        public bool IsSuccess => _isSuccess;

        /// <summary>
        /// Returns true if the result is a failure
        /// </summary>
        public bool IsFailure => !_isSuccess;

        /// <summary>
        /// Gets the value if successful, or throws an exception if not
        /// </summary>
        public T Value => _isSuccess ? _value : throw new InvalidOperationException("Cannot access value of a failed result");

        /// <summary>
        /// Gets the error if failed, or throws an exception if not
        /// </summary>
        public TError Error => !_isSuccess ? _error : throw new InvalidOperationException("Cannot access error of a successful result");

        /// <summary>
        /// Gets the value if successful, or returns the default value if not
        /// </summary>
        public T ValueOrDefault => _value;

        /// <summary>
        /// Gets the value if successful, or returns the specified default value if not
        /// </summary>
        public T ValueOr(T defaultValue) => _isSuccess ? _value : defaultValue;

        /// <summary>
        /// Gets the value if successful, or returns the result of the specified function if not
        /// </summary>
        public T ValueOr(Func<TError, T> defaultValueProvider) => _isSuccess ? _value : defaultValueProvider(_error);

        /// <summary>
        /// Applies a function to the value if successful, or returns a failure with the same error if not
        /// </summary>
        public Result<TResult, TError> Map<TResult>(Func<T, TResult> mapper) =>
            _isSuccess ? Result<TResult, TError>.Success(mapper(_value)) : Result<TResult, TError>.Failure(_error);

        /// <summary>
        /// Applies a function to the error if failed, or returns a success with the same value if not
        /// </summary>
        public Result<T, TNewError> MapError<TNewError>(Func<TError, TNewError> mapper) =>
            _isSuccess ? Result<T, TNewError>.Success(_value) : Result<T, TNewError>.Failure(mapper(_error));

        /// <summary>
        /// Applies a function that returns a Result to the value if successful, or returns a failure with the same error if not
        /// </summary>
        public Result<TResult, TError> Bind<TResult>(Func<T, Result<TResult, TError>> binder) =>
            _isSuccess ? binder(_value) : Result<TResult, TError>.Failure(_error);

        /// <summary>
        /// Applies one of two functions depending on whether the result is successful
        /// </summary>
        public TResult Match<TResult>(Func<T, TResult> success, Func<TError, TResult> failure) =>
            _isSuccess ? success(_value) : failure(_error);

        /// <summary>
        /// Performs an action if the result is successful
        /// </summary>
        public Result<T, TError> IfSuccess(Action<T> action)
        {
            if (_isSuccess)
            {
                action(_value);
            }
            return this;
        }

        /// <summary>
        /// Performs an action if the result is a failure
        /// </summary>
        public Result<T, TError> IfFailure(Action<TError> action)
        {
            if (!_isSuccess)
            {
                action(_error);
            }
            return this;
        }
    }

    /// <summary>
    /// A simpler version of Result that uses Exception as the error type
    /// </summary>
    public static class Result
    {
        /// <summary>
        /// Creates a successful Result with a value
        /// </summary>
        public static Result<T, Exception> Success<T>(T value) =>
            Result<T, Exception>.Success(value);

        /// <summary>
        /// Creates a failed Result with an exception
        /// </summary>
        public static Result<T, Exception> Failure<T>(Exception error) =>
            Result<T, Exception>.Failure(error);

        /// <summary>
        /// Tries to execute a function and returns a Result
        /// </summary>
        public static Result<T, Exception> Try<T>(Func<T> func)
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
    }
}
