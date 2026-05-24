using System;
using System.Threading.Tasks;

namespace TarsEngine.Monads
{
    /// <summary>
    /// Provides utility methods for working with asynchronous operations in a monadic way.
    /// Helps avoid CS1998 warnings (async method lacks 'await' operators).
    /// </summary>
    public static class AsyncMonad
    {
        /// <summary>
        /// Creates a completed task with the specified result.
        /// Use this instead of writing async methods that don't await anything.
        /// </summary>
        /// <typeparam name="T">The type of the result.</typeparam>
        /// <param name="value">The result value.</param>
        /// <returns>A task that has already completed with the specified result.</returns>
        public static Task<T> Return<T>(T value) => Task.FromResult(value);

        /// <summary>
        /// Creates a completed task with no result.
        /// Use this instead of writing async methods that don't await anything.
        /// </summary>
        /// <returns>A task that has already completed.</returns>
        public static Task Return() => Task.CompletedTask;

        /// <summary>
        /// Executes a function that returns a task and ensures it's properly awaited.
        /// </summary>
        /// <typeparam name="T">The type of the result.</typeparam>
        /// <param name="func">The function to execute.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public static async Task<T> Bind<T>(Func<Task<T>> func)
        {
            return await func();
        }

        /// <summary>
        /// Executes a function that returns a task and ensures it's properly awaited.
        /// </summary>
        /// <param name="func">The function to execute.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public static async Task Bind(Func<Task> func)
        {
            await func();
        }

        /// <summary>
        /// Transforms the result of a task using a specified function.
        /// </summary>
        /// <typeparam name="TInput">The type of the input.</typeparam>
        /// <typeparam name="TOutput">The type of the output.</typeparam>
        /// <param name="task">The task to transform.</param>
        /// <param name="func">The transformation function.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public static async Task<TOutput> Map<TInput, TOutput>(Task<TInput> task, Func<TInput, TOutput> func)
        {
            var result = await task;
            return func(result);
        }

        /// <summary>
        /// Executes a CPU-bound operation asynchronously on a background thread.
        /// </summary>
        /// <typeparam name="T">The type of the result.</typeparam>
        /// <param name="func">The function to execute.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public static Task<T> RunAsync<T>(Func<T> func)
        {
            return Task.Run(func);
        }

        /// <summary>
        /// Executes a CPU-bound operation asynchronously on a background thread.
        /// </summary>
        /// <param name="action">The action to execute.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public static Task RunAsync(Action action)
        {
            return Task.Run(action);
        }
    }
}
