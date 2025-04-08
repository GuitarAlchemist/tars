using System;
using System.Threading.Tasks;

namespace TarsEngine.Monads
{
    /// <summary>
    /// Static class with utility methods for working with monads
    /// </summary>
    public static class Monad
    {
        #region Option Monad

        /// <summary>
        /// Creates an Option with a value
        /// </summary>
        public static Option<T> Some<T>(T value) => Option<T>.Some(value);

        /// <summary>
        /// Creates an Option with no value
        /// </summary>
        public static Option<T> None<T>() => Option<T>.None;

        /// <summary>
        /// Converts a nullable value to an Option
        /// </summary>
        public static Option<T> FromNullable<T>(T? value) where T : class =>
            value != null ? Some(value) : None<T>();

        /// <summary>
        /// Converts a nullable value to an Option
        /// </summary>
        public static Option<T> FromNullable<T>(T? value) where T : struct =>
            value.HasValue ? Some(value.Value) : None<T>();

        #endregion

        #region Result Monad

        /// <summary>
        /// Creates a successful Result with a value
        /// </summary>
        public static Result<T, TError> Success<T, TError>(T value) =>
            Result<T, TError>.Success(value);

        /// <summary>
        /// Creates a failed Result with an error
        /// </summary>
        public static Result<T, TError> Failure<T, TError>(TError error) =>
            Result<T, TError>.Failure(error);

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

        #endregion

        #region AsyncResult Monad

        /// <summary>
        /// Creates an AsyncResult from a value
        /// </summary>
        public static AsyncResult<T> AsyncFromResult<T>(T result) =>
            AsyncResult<T>.FromResult(result);

        /// <summary>
        /// Creates an AsyncResult from a task
        /// </summary>
        public static AsyncResult<T> AsyncFromTask<T>(Task<T> task) =>
            AsyncResult<T>.FromTask(task);

        #endregion

        #region AsyncOption Monad

        /// <summary>
        /// Creates an AsyncOption with a value
        /// </summary>
        public static AsyncOption<T> AsyncSome<T>(T value) =>
            AsyncOption<T>.Some(value);

        /// <summary>
        /// Creates an AsyncOption with no value
        /// </summary>
        public static AsyncOption<T> AsyncNone<T>() =>
            AsyncOption<T>.None;

        #endregion

        #region AsyncResultError Monad

        /// <summary>
        /// Creates a successful AsyncResultError with a value
        /// </summary>
        public static AsyncResultError<T, TError> AsyncSuccess<T, TError>(T value) =>
            AsyncResultError<T, TError>.Success(value);

        /// <summary>
        /// Creates a failed AsyncResultError with an error
        /// </summary>
        public static AsyncResultError<T, TError> AsyncFailure<T, TError>(TError error) =>
            AsyncResultError<T, TError>.Failure(error);

        /// <summary>
        /// Creates a successful AsyncResultError with a value
        /// </summary>
        public static AsyncResultError<T, Exception> AsyncSuccess<T>(T value) =>
            AsyncResultError<T, Exception>.Success(value);

        /// <summary>
        /// Creates a failed AsyncResultError with an exception
        /// </summary>
        public static AsyncResultError<T, Exception> AsyncFailure<T>(Exception error) =>
            AsyncResultError<T, Exception>.Failure(error);

        /// <summary>
        /// Tries to execute a function asynchronously and returns an AsyncResultError
        /// </summary>
        public static AsyncResultError<T, Exception> TryAsync<T>(Func<Task<T>> func) =>
            AsyncResultError.TryAsync(func);

        #endregion

        #region Printable Monad

        /// <summary>
        /// Creates a Printable with a value and a custom printer function
        /// </summary>
        public static Printable<T> Print<T>(T value, Func<T, string> printer) =>
            Printable<T>.Create(value, printer);

        /// <summary>
        /// Creates a Printable with a value and a default printer function
        /// </summary>
        public static Printable<T> Print<T>(T value) =>
            Printable<T>.Create(value);

        #endregion

        #region Conversion Methods

        /// <summary>
        /// Converts an Option to a Result
        /// </summary>
        public static Result<T, TError> ToResult<T, TError>(this Option<T> option, TError error) =>
            option.Match(
                some: value => Success<T, TError>(value),
                none: () => Failure<T, TError>(error)
            );

        /// <summary>
        /// Converts a Result to an Option
        /// </summary>
        public static Option<T> ToOption<T, TError>(this Result<T, TError> result) =>
            result.Match(
                success: value => Some(value),
                failure: _ => None<T>()
            );

        /// <summary>
        /// Converts an AsyncResult to an AsyncOption
        /// </summary>
        public static AsyncOption<T> ToAsyncOption<T>(this AsyncResult<T> asyncResult) =>
            AsyncOption<T>.FromValueTask(asyncResult.RunAsync());

        /// <summary>
        /// Converts an AsyncOption to an AsyncResult
        /// </summary>
        public static AsyncResult<T> ToAsyncResult<T>(this AsyncOption<T> asyncOption, T defaultValue) =>
            AsyncResult<T>.FromTask(asyncOption.RunAsync().ContinueWith(t => t.Result.ValueOr(defaultValue)));

        /// <summary>
        /// Converts an AsyncOption to an AsyncResultError
        /// </summary>
        public static AsyncResultError<T, TError> ToAsyncResultError<T, TError>(this AsyncOption<T> asyncOption, TError error) =>
            AsyncResultError<T, TError>.FromTask(asyncOption.RunAsync().ContinueWith(t => t.Result.ToResult(error)));

        #endregion
    }
}
