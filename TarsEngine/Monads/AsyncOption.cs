using System;
using System.Threading.Tasks;

namespace TarsEngine.Monads
{
    /// <summary>
    /// Represents an asynchronous operation that will produce an optional value.
    /// Used to handle async operations that might return null in a functional way.
    /// </summary>
    /// <typeparam name="T">The type of the value</typeparam>
    public class AsyncOption<T>
    {
        private readonly Task<Option<T>> _task;

        private AsyncOption(Task<Option<T>> task)
        {
            _task = task;
        }

        /// <summary>
        /// Creates an AsyncOption with a value
        /// </summary>
        public static AsyncOption<T> Some(T value) =>
            new(Task.FromResult(Option<T>.Some(value)));

        /// <summary>
        /// Creates an AsyncOption with no value
        /// </summary>
        public static AsyncOption<T> None =>
            new(Task.FromResult(Option<T>.None));

        /// <summary>
        /// Creates an AsyncOption from a task that returns an Option
        /// </summary>
        public static AsyncOption<T> FromTask(Task<Option<T>> task) =>
            new(task);

        /// <summary>
        /// Creates an AsyncOption from a task that returns a value
        /// </summary>
        public static AsyncOption<T> FromValueTask(Task<T> task) =>
            new(task.ContinueWith(t => t.Result != null ? Option<T>.Some(t.Result) : Option<T>.None));

        /// <summary>
        /// Runs the async operation and returns the option
        /// </summary>
        public Task<Option<T>> RunAsync() => _task;

        /// <summary>
        /// Applies a function to the value if present when the operation completes
        /// </summary>
        public AsyncOption<TResult> Map<TResult>(Func<T, TResult> mapper) =>
            new(_task.ContinueWith(t => t.Result.Map(mapper)));

        /// <summary>
        /// Applies a function that returns an Option to the value if present when the operation completes
        /// </summary>
        public AsyncOption<TResult> Bind<TResult>(Func<T, Option<TResult>> binder) =>
            new(_task.ContinueWith(t => t.Result.Bind(binder)));

        /// <summary>
        /// Applies a function that returns an AsyncOption to the value if present when the operation completes
        /// </summary>
        public AsyncOption<TResult> Bind<TResult>(Func<T, AsyncOption<TResult>> binder) =>
            new(_task.ContinueWith(t =>
                t.Result.Match(
                    some: value => binder(value)._task,
                    none: () => Task.FromResult(Option<TResult>.None)
                )).Unwrap());

        /// <summary>
        /// Performs an action on the value if present when the operation completes
        /// </summary>
        public AsyncOption<T> Do(Action<T> action) =>
            new(_task.ContinueWith(t => {
                t.Result.IfSome(action);
                return t.Result;
            }));
    }

    /// <summary>
    /// Extension methods for AsyncOption of T
    /// </summary>
    public static class AsyncOptionExtensions
    {
        /// <summary>
        /// Converts a Task of Option of T to an AsyncOption of T
        /// </summary>
        public static AsyncOption<T> ToAsyncOption<T>(this Task<Option<T>> task) =>
            AsyncOption<T>.FromTask(task);

        /// <summary>
        /// Converts a Task of T to an AsyncOption of T
        /// </summary>
        public static AsyncOption<T> ToAsyncOption<T>(this Task<T> task) =>
            AsyncOption<T>.FromValueTask(task);

        /// <summary>
        /// Converts an Option of T to an AsyncOption of T
        /// </summary>
        public static AsyncOption<T> ToAsync<T>(this Option<T> option) =>
            option.Match(
                some: value => AsyncOption<T>.Some(value),
                none: () => AsyncOption<T>.None
            );
    }
}
