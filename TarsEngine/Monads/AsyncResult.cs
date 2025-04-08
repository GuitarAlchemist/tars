using System;
using System.Threading.Tasks;

namespace TarsEngine.Monads
{
    /// <summary>
    /// Represents an asynchronous operation that will produce a result.
    /// Used to handle async operations in a functional way.
    /// </summary>
    /// <typeparam name="T">The type of the result</typeparam>
    public class AsyncResult<T>
    {
        private readonly Task<T> _task;

        private AsyncResult(Task<T> task)
        {
            _task = task;
        }

        /// <summary>
        /// Creates an AsyncResult from a value
        /// </summary>
        public static AsyncResult<T> FromResult(T result) =>
            new(Task.FromResult(result));

        /// <summary>
        /// Creates an AsyncResult from a task
        /// </summary>
        public static AsyncResult<T> FromTask(Task<T> task) =>
            new(task);

        /// <summary>
        /// Runs the async operation and returns the result
        /// </summary>
        public Task<T> RunAsync() => _task;

        /// <summary>
        /// Applies a function to the result when it completes
        /// </summary>
        public AsyncResult<TResult> Map<TResult>(Func<T, TResult> mapper) =>
            new(_task.ContinueWith(t => mapper(t.Result)));

        /// <summary>
        /// Applies a function that returns an AsyncResult to the result when it completes
        /// </summary>
        public AsyncResult<TResult> Bind<TResult>(Func<T, AsyncResult<TResult>> binder) =>
            new(_task.ContinueWith(t => binder(t.Result)._task).Unwrap());

        /// <summary>
        /// Applies a function that returns a Task to the result when it completes
        /// </summary>
        public AsyncResult<TResult> BindTask<TResult>(Func<T, Task<TResult>> binder) =>
            new(_task.ContinueWith(t => binder(t.Result)).Unwrap());

        /// <summary>
        /// Performs an action when the operation completes
        /// </summary>
        public AsyncResult<T> Do(Action<T> action) =>
            new(_task.ContinueWith(t => {
                action(t.Result);
                return t.Result;
            }));

        /// <summary>
        /// Implicitly converts from Task of T to AsyncResult of T
        /// </summary>
        public static implicit operator AsyncResult<T>(Task<T> task) =>
            FromTask(task);

        /// <summary>
        /// Implicitly converts from T to AsyncResult of T
        /// </summary>
        public static implicit operator AsyncResult<T>(T value) =>
            FromResult(value);
    }

    /// <summary>
    /// Extension methods for AsyncResult of T
    /// </summary>
    public static class AsyncResultExtensions
    {
        /// <summary>
        /// Converts a Task of T to an AsyncResult of T
        /// </summary>
        public static AsyncResult<T> ToAsyncResult<T>(this Task<T> task) =>
            AsyncResult<T>.FromTask(task);

        /// <summary>
        /// Converts a value to an AsyncResult of T
        /// </summary>
        public static AsyncResult<T> ToAsyncResult<T>(this T value) =>
            AsyncResult<T>.FromResult(value);

        /// <summary>
        /// Converts a synchronous method to an asynchronous one
        /// </summary>
        public static Task<T> AsTask<T>(this Func<T> func) =>
            Task.FromResult(func());
    }
}
