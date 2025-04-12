using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace TarsEngine.Monads
{
    /// <summary>
    /// Enhanced version of AsyncResult monad with additional utility methods.
    /// Represents an asynchronous operation that will produce a result.
    /// Used to handle async operations in a functional way.
    /// </summary>
    /// <typeparam name="T">The type of the result</typeparam>
    public class EnhancedAsyncResult<T>
    {
        private readonly Task<T> _task;

        private EnhancedAsyncResult(Task<T> task)
        {
            _task = task;
        }

        /// <summary>
        /// Creates an AsyncResult from a value
        /// </summary>
        public static EnhancedAsyncResult<T> FromResult(T result) =>
            new(Task.FromResult(result));

        /// <summary>
        /// Creates an AsyncResult from a task
        /// </summary>
        public static EnhancedAsyncResult<T> FromTask(Task<T> task) =>
            new(task);

        /// <summary>
        /// Runs the async operation and returns the result
        /// </summary>
        public Task<T> RunAsync() => _task;

        /// <summary>
        /// Applies a function to the result when it completes
        /// </summary>
        public EnhancedAsyncResult<TResult> Map<TResult>(Func<T, TResult> mapper) =>
            new(_task.ContinueWith(t => mapper(t.Result)));

        /// <summary>
        /// Applies a function that returns an AsyncResult to the result when it completes
        /// </summary>
        public EnhancedAsyncResult<TResult> Bind<TResult>(Func<T, EnhancedAsyncResult<TResult>> binder) =>
            new(_task.ContinueWith(t => binder(t.Result)._task).Unwrap());

        /// <summary>
        /// Applies a function that returns a Task to the result when it completes
        /// </summary>
        public EnhancedAsyncResult<TResult> BindTask<TResult>(Func<T, Task<TResult>> binder) =>
            new(_task.ContinueWith(t => binder(t.Result)).Unwrap());

        /// <summary>
        /// Performs an action when the operation completes
        /// </summary>
        public EnhancedAsyncResult<T> Do(Action<T> action) =>
            new(_task.ContinueWith(t => {
                action(t.Result);
                return t.Result;
            }));

        /// <summary>
        /// Converts the AsyncResult to an AsyncOption
        /// </summary>
        public EnhancedAsyncOption<T> ToAsyncOption() =>
            EnhancedAsyncOption<T>.FromValueTask(_task);

        /// <summary>
        /// Implicitly converts from Task of T to AsyncResult of T
        /// </summary>
        public static implicit operator EnhancedAsyncResult<T>(Task<T> task) =>
            FromTask(task);

        /// <summary>
        /// Implicitly converts from T to AsyncResult of T
        /// </summary>
        public static implicit operator EnhancedAsyncResult<T>(T value) =>
            FromResult(value);
    }

    /// <summary>
    /// Enhanced version of AsyncOption monad with additional utility methods.
    /// Represents an asynchronous operation that will produce an optional value.
    /// Used to handle async operations that might return null in a functional way.
    /// </summary>
    /// <typeparam name="T">The type of the value</typeparam>
    public class EnhancedAsyncOption<T>
    {
        private readonly Task<EnhancedOption<T>> _task;

        internal EnhancedAsyncOption(Task<EnhancedOption<T>> task)
        {
            _task = task;
        }

        /// <summary>
        /// Creates an AsyncOption with a value
        /// </summary>
        public static EnhancedAsyncOption<T> Some(T value) =>
            new(Task.FromResult(EnhancedOption<T>.Some(value)));

        /// <summary>
        /// Creates an AsyncOption with no value
        /// </summary>
        public static EnhancedAsyncOption<T> None =>
            new(Task.FromResult(EnhancedOption<T>.None));

        /// <summary>
        /// Creates an AsyncOption from a task that returns an Option
        /// </summary>
        public static EnhancedAsyncOption<T> FromTask(Task<EnhancedOption<T>> task) =>
            new(task);

        /// <summary>
        /// Creates an AsyncOption from a task that returns a value
        /// </summary>
        public static EnhancedAsyncOption<T> FromValueTask(Task<T> task) =>
            new(task.ContinueWith(t =>
                t.IsCompletedSuccessfully && t.Result != null
                    ? EnhancedOption<T>.Some(t.Result)
                    : EnhancedOption<T>.None));

        /// <summary>
        /// Runs the async operation and returns the option
        /// </summary>
        public Task<EnhancedOption<T>> RunAsync() => _task;

        /// <summary>
        /// Applies a function to the value if present when the operation completes
        /// </summary>
        public EnhancedAsyncOption<TResult> Map<TResult>(Func<T, TResult> mapper) =>
            new(_task.ContinueWith(t => t.Result.Map(mapper)));

        /// <summary>
        /// Applies a function that returns an Option to the value if present when the operation completes
        /// </summary>
        public EnhancedAsyncOption<TResult> Bind<TResult>(Func<T, EnhancedOption<TResult>> binder) =>
            new(_task.ContinueWith(t => t.Result.Bind(binder)));

        /// <summary>
        /// Applies a function that returns an AsyncOption to the value if present when the operation completes
        /// </summary>
        public EnhancedAsyncOption<TResult> BindAsync<TResult>(Func<T, EnhancedAsyncOption<TResult>> binder) =>
            new(_task.ContinueWith(t =>
                t.Result.IsSome
                    ? binder(t.Result.Value).RunAsync()
                    : Task.FromResult(EnhancedOption<TResult>.None)).Unwrap());

        /// <summary>
        /// Performs an action if the option has a value when the operation completes
        /// </summary>
        public EnhancedAsyncOption<T> IfSome(Action<T> action) =>
            new(_task.ContinueWith(t => {
                t.Result.IfSome(action);
                return t.Result;
            }));

        /// <summary>
        /// Performs an action if the option has no value when the operation completes
        /// </summary>
        public EnhancedAsyncOption<T> IfNone(Action action) =>
            new(_task.ContinueWith(t => {
                t.Result.IfNone(action);
                return t.Result;
            }));

        /// <summary>
        /// Implicitly converts from Task of Option of T to AsyncOption of T
        /// </summary>
        public static implicit operator EnhancedAsyncOption<T>(Task<EnhancedOption<T>> task) =>
            FromTask(task);

        /// <summary>
        /// Implicitly converts from Option of T to AsyncOption of T
        /// </summary>
        public static implicit operator EnhancedAsyncOption<T>(EnhancedOption<T> option) =>
            new(Task.FromResult(option));
    }

    /// <summary>
    /// Enhanced version of AsyncResultError monad with additional utility methods.
    /// Represents an asynchronous operation that might fail.
    /// Used to handle async operations that might fail in a functional way.
    /// </summary>
    /// <typeparam name="T">The type of the success value</typeparam>
    /// <typeparam name="TError">The type of the error value</typeparam>
    public class EnhancedAsyncResultError<T, TError>
    {
        private readonly Task<Result<T, TError>> _task;

        internal EnhancedAsyncResultError(Task<Result<T, TError>> task)
        {
            _task = task;
        }

        /// <summary>
        /// Creates a successful AsyncResultError with a value
        /// </summary>
        public static EnhancedAsyncResultError<T, TError> Success(T value) =>
            new(Task.FromResult(Result<T, TError>.Success(value)));

        /// <summary>
        /// Creates a failed AsyncResultError with an error
        /// </summary>
        public static EnhancedAsyncResultError<T, TError> Failure(TError error) =>
            new(Task.FromResult(Result<T, TError>.Failure(error)));

        /// <summary>
        /// Creates an AsyncResultError from a task that returns a Result
        /// </summary>
        public static EnhancedAsyncResultError<T, TError> FromTask(Task<Result<T, TError>> task) =>
            new(task);

        /// <summary>
        /// Creates an AsyncResultError from a task that returns a value
        /// </summary>
        public static EnhancedAsyncResultError<T, TError> FromValueTask(Task<T> task, Func<Exception, TError> errorMapper) =>
            new(task.ContinueWith(t => {
                if (t.IsFaulted)
                    return Result<T, TError>.Failure(errorMapper(t.Exception.InnerException));
                if (t.IsCanceled)
                    return Result<T, TError>.Failure(errorMapper(new TaskCanceledException()));
                return Result<T, TError>.Success(t.Result);
            }));

        /// <summary>
        /// Runs the async operation and returns the result
        /// </summary>
        public Task<Result<T, TError>> RunAsync() => _task;

        /// <summary>
        /// Applies a function to the value if successful when the operation completes
        /// </summary>
        public EnhancedAsyncResultError<TResult, TError> Map<TResult>(Func<T, TResult> mapper) =>
            new(_task.ContinueWith(t => t.Result.Map(mapper)));

        /// <summary>
        /// Applies a function to the error if failed when the operation completes
        /// </summary>
        public EnhancedAsyncResultError<T, TNewError> MapError<TNewError>(Func<TError, TNewError> mapper) =>
            new(_task.ContinueWith(t => t.Result.MapError(mapper)));

        /// <summary>
        /// Applies a function that returns a Result to the value if successful when the operation completes
        /// </summary>
        public EnhancedAsyncResultError<TResult, TError> Bind<TResult>(Func<T, Result<TResult, TError>> binder) =>
            new(_task.ContinueWith(t => t.Result.Bind(binder)));

        /// <summary>
        /// Applies a function that returns an AsyncResultError to the value if successful when the operation completes
        /// </summary>
        public EnhancedAsyncResultError<TResult, TError> BindAsync<TResult>(Func<T, EnhancedAsyncResultError<TResult, TError>> binder) =>
            new(_task.ContinueWith(t =>
                t.Result.IsSuccess
                    ? binder(t.Result.Value).RunAsync()
                    : Task.FromResult(Result<TResult, TError>.Failure(t.Result.Error))).Unwrap());

        /// <summary>
        /// Performs an action if the result is successful when the operation completes
        /// </summary>
        public EnhancedAsyncResultError<T, TError> IfSuccess(Action<T> action) =>
            new(_task.ContinueWith(t => {
                t.Result.IfSuccess(action);
                return t.Result;
            }));

        /// <summary>
        /// Performs an action if the result is a failure when the operation completes
        /// </summary>
        public EnhancedAsyncResultError<T, TError> IfFailure(Action<TError> action) =>
            new(_task.ContinueWith(t => {
                t.Result.IfFailure(action);
                return t.Result;
            }));

        /// <summary>
        /// Implicitly converts from Task of Result of T, TError to AsyncResultError of T, TError
        /// </summary>
        public static implicit operator EnhancedAsyncResultError<T, TError>(Task<Result<T, TError>> task) =>
            FromTask(task);

        /// <summary>
        /// Implicitly converts from Result of T, TError to AsyncResultError of T, TError
        /// </summary>
        public static implicit operator EnhancedAsyncResultError<T, TError>(Result<T, TError> result) =>
            new(Task.FromResult(result));
    }

    /// <summary>
    /// Static helper class for creating EnhancedAsyncResult instances
    /// </summary>
    public static class EnhancedAsyncResult
    {
        /// <summary>
        /// Creates an AsyncResult from a value
        /// </summary>
        public static EnhancedAsyncResult<T> FromResult<T>(T result) =>
            EnhancedAsyncResult<T>.FromResult(result);

        /// <summary>
        /// Creates an AsyncResult from a task
        /// </summary>
        public static EnhancedAsyncResult<T> FromTask<T>(Task<T> task) =>
            EnhancedAsyncResult<T>.FromTask(task);

        /// <summary>
        /// Tries to execute a function asynchronously and returns an AsyncResult
        /// </summary>
        public static EnhancedAsyncResult<T> TryAsync<T>(Func<Task<T>> func)
        {
            try
            {
                return EnhancedAsyncResult<T>.FromTask(func());
            }
            catch (Exception)
            {
                return EnhancedAsyncResult<T>.FromTask(Task.FromException<T>(new Exception("Operation failed")));
            }
        }

        /// <summary>
        /// Creates an AsyncOption with a value
        /// </summary>
        public static EnhancedAsyncOption<T> Some<T>(T value) =>
            EnhancedAsyncOption<T>.Some(value);

        /// <summary>
        /// Creates an AsyncOption with no value
        /// </summary>
        public static EnhancedAsyncOption<T> None<T>() =>
            EnhancedAsyncOption<T>.None;

        /// <summary>
        /// Creates a successful AsyncResultError with a value
        /// </summary>
        public static EnhancedAsyncResultError<T, TError> Success<T, TError>(T value) =>
            EnhancedAsyncResultError<T, TError>.Success(value);

        /// <summary>
        /// Creates a failed AsyncResultError with an error
        /// </summary>
        public static EnhancedAsyncResultError<T, TError> Failure<T, TError>(TError error) =>
            EnhancedAsyncResultError<T, TError>.Failure(error);

        /// <summary>
        /// Creates a successful AsyncResultError with a value
        /// </summary>
        public static EnhancedAsyncResultError<T, Exception> Success<T>(T value) =>
            EnhancedAsyncResultError<T, Exception>.Success(value);

        /// <summary>
        /// Creates a failed AsyncResultError with an exception
        /// </summary>
        public static EnhancedAsyncResultError<T, Exception> Failure<T>(Exception error) =>
            EnhancedAsyncResultError<T, Exception>.Failure(error);

        /// <summary>
        /// Tries to execute a function asynchronously and returns an AsyncResultError
        /// </summary>
        public static EnhancedAsyncResultError<T, Exception> TryAsyncWithResult<T>(Func<Task<T>> func)
        {
            try
            {
                return EnhancedAsyncResultError<T, Exception>.FromValueTask(func(), ex => ex);
            }
            catch (Exception ex)
            {
                return Failure<T>(ex);
            }
        }

        /// <summary>
        /// Combines multiple async results into a single async result containing a tuple
        /// </summary>
        public static EnhancedAsyncResult<(T1, T2)> Zip<T1, T2>(
            EnhancedAsyncResult<T1> asyncResult1,
            EnhancedAsyncResult<T2> asyncResult2)
        {
            return EnhancedAsyncResult<(T1, T2)>.FromTask(
                Task.WhenAll(asyncResult1.RunAsync(), asyncResult2.RunAsync())
                    .ContinueWith(_ => (asyncResult1.RunAsync().Result, asyncResult2.RunAsync().Result)));
        }

        /// <summary>
        /// Combines multiple async results into a single async result containing a tuple
        /// </summary>
        public static EnhancedAsyncResult<(T1, T2, T3)> Zip<T1, T2, T3>(
            EnhancedAsyncResult<T1> asyncResult1,
            EnhancedAsyncResult<T2> asyncResult2,
            EnhancedAsyncResult<T3> asyncResult3)
        {
            return EnhancedAsyncResult<(T1, T2, T3)>.FromTask(
                Task.WhenAll(asyncResult1.RunAsync(), asyncResult2.RunAsync(), asyncResult3.RunAsync())
                    .ContinueWith(_ => (
                        asyncResult1.RunAsync().Result,
                        asyncResult2.RunAsync().Result,
                        asyncResult3.RunAsync().Result)));
        }

        /// <summary>
        /// Combines multiple async results using a combiner function
        /// </summary>
        public static EnhancedAsyncResult<TResult> Map2<T1, T2, TResult>(
            EnhancedAsyncResult<T1> asyncResult1,
            EnhancedAsyncResult<T2> asyncResult2,
            Func<T1, T2, TResult> mapper)
        {
            return Zip(asyncResult1, asyncResult2).Map(tuple => mapper(tuple.Item1, tuple.Item2));
        }

        /// <summary>
        /// Combines multiple async results using a combiner function
        /// </summary>
        public static EnhancedAsyncResult<TResult> Map3<T1, T2, T3, TResult>(
            EnhancedAsyncResult<T1> asyncResult1,
            EnhancedAsyncResult<T2> asyncResult2,
            EnhancedAsyncResult<T3> asyncResult3,
            Func<T1, T2, T3, TResult> mapper)
        {
            return Zip(asyncResult1, asyncResult2, asyncResult3)
                .Map(tuple => mapper(tuple.Item1, tuple.Item2, tuple.Item3));
        }

        /// <summary>
        /// Traverses a sequence of async results and returns an async result of sequence
        /// </summary>
        public static EnhancedAsyncResult<IEnumerable<T>> Sequence<T>(IEnumerable<EnhancedAsyncResult<T>> asyncResults)
        {
            var tasks = asyncResults.Select(ar => ar.RunAsync()).ToArray();
            return EnhancedAsyncResult<IEnumerable<T>>.FromTask(
                Task.WhenAll(tasks).ContinueWith(t => t.Result.AsEnumerable()));
        }

        /// <summary>
        /// Maps a sequence using a function that returns an async result and collects the results
        /// </summary>
        public static EnhancedAsyncResult<IEnumerable<TResult>> Traverse<T, TResult>(
            IEnumerable<T> source,
            Func<T, EnhancedAsyncResult<TResult>> mapper)
        {
            return Sequence(source.Select(mapper));
        }
    }

    /// <summary>
    /// Extension methods for EnhancedAsyncResult
    /// </summary>
    public static class EnhancedAsyncResultExtensions
    {
        /// <summary>
        /// Converts a Task of T to an EnhancedAsyncResult of T
        /// </summary>
        public static EnhancedAsyncResult<T> ToAsyncResult<T>(this Task<T> task) =>
            EnhancedAsyncResult<T>.FromTask(task);

        /// <summary>
        /// Converts a value to an EnhancedAsyncResult of T
        /// </summary>
        public static EnhancedAsyncResult<T> ToAsyncResult<T>(this T value) =>
            EnhancedAsyncResult<T>.FromResult(value);

        /// <summary>
        /// Converts a Task of T to an EnhancedAsyncOption of T
        /// </summary>
        public static EnhancedAsyncOption<T> ToAsyncOption<T>(this Task<T> task) =>
            EnhancedAsyncOption<T>.FromValueTask(task);

        /// <summary>
        /// Converts an EnhancedOption of T to an EnhancedAsyncOption of T
        /// </summary>
        public static EnhancedAsyncOption<T> ToAsync<T>(this EnhancedOption<T> option) =>
            new(Task.FromResult(option));

        /// <summary>
        /// Converts a Task of EnhancedOption of T to an EnhancedAsyncOption of T
        /// </summary>
        public static EnhancedAsyncOption<T> ToAsyncOption<T>(this Task<EnhancedOption<T>> task) =>
            EnhancedAsyncOption<T>.FromTask(task);

        /// <summary>
        /// Converts a Task of T to an EnhancedAsyncResultError of T, Exception
        /// </summary>
        public static EnhancedAsyncResultError<T, Exception> ToAsyncResultError<T>(this Task<T> task) =>
            EnhancedAsyncResultError<T, Exception>.FromValueTask(task, ex => ex);

        /// <summary>
        /// Converts a Result of T, TError to an EnhancedAsyncResultError of T, TError
        /// </summary>
        public static EnhancedAsyncResultError<T, TError> ToAsync<T, TError>(this Result<T, TError> result) =>
            new(Task.FromResult(result));

        /// <summary>
        /// Converts a Task of Result of T, TError to an EnhancedAsyncResultError of T, TError
        /// </summary>
        public static EnhancedAsyncResultError<T, TError> ToAsyncResultError<T, TError>(this Task<Result<T, TError>> task) =>
            EnhancedAsyncResultError<T, TError>.FromTask(task);

        /// <summary>
        /// Converts a synchronous method to an asynchronous one
        /// </summary>
        public static Task<T> AsTask<T>(this Func<T> func) =>
            Task.FromResult(func());

        /// <summary>
        /// Waits for all tasks to complete and returns their results as a tuple
        /// </summary>
        public static async Task<(T1, T2)> WhenAll<T1, T2>(this Task<T1> task1, Task<T2> task2)
        {
            await Task.WhenAll(task1, task2);
            return (task1.Result, task2.Result);
        }

        /// <summary>
        /// Waits for all tasks to complete and returns their results as a tuple
        /// </summary>
        public static async Task<(T1, T2, T3)> WhenAll<T1, T2, T3>(
            this Task<T1> task1, Task<T2> task2, Task<T3> task3)
        {
            await Task.WhenAll(task1, task2, task3);
            return (task1.Result, task2.Result, task3.Result);
        }

        /// <summary>
        /// Waits for all tasks to complete and returns their results as a tuple
        /// </summary>
        public static async Task<(T1, T2, T3, T4)> WhenAll<T1, T2, T3, T4>(
            this Task<T1> task1, Task<T2> task2, Task<T3> task3, Task<T4> task4)
        {
            await Task.WhenAll(task1, task2, task3, task4);
            return (task1.Result, task2.Result, task3.Result, task4.Result);
        }
    }
}
