using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using TarsEngine.Monads;

namespace TarsEngine.Examples
{
    /// <summary>
    /// Examples of how to use the monad library
    /// </summary>
    public static class MonadExamples
    {
        #region Option Monad Examples

        /// <summary>
        /// Example of using the Option monad to handle nullable references
        /// </summary>
        public static void OptionExample()
        {
            // Instead of:
            // string name = GetName();
            // if (name != null)
            // {
            //     Console.WriteLine($"Hello, {name}!");
            // }
            // else
            // {
            //     Console.WriteLine("Hello, anonymous user!");
            // }

            // Use:
            Option<string> nameOption = GetNameOption();
            string greeting = nameOption.Match(
                some: name => $"Hello, {name}!",
                none: () => "Hello, anonymous user!"
            );
            Console.WriteLine(greeting);
        }

        private static Option<string> GetNameOption()
        {
            // Simulate getting a name that might be null
            string? name = null; // or "John"
            return name != null ? Option<string>.Some(name) : Option<string>.None;
        }

        #endregion

        #region Result Monad Examples

        /// <summary>
        /// Example of using the Result monad to handle errors
        /// </summary>
        public static void ResultExample()
        {
            // Instead of:
            // try
            // {
            //     int result = Divide(10, 0);
            //     Console.WriteLine($"Result: {result}");
            // }
            // catch (Exception ex)
            // {
            //     Console.WriteLine($"Error: {ex.Message}");
            // }

            // Use:
            Result<int, Exception> result = DivideResult(10, 0);
            string message = result.Match(
                success: value => $"Result: {value}",
                failure: ex => $"Error: {ex.Message}"
            );
            Console.WriteLine(message);
        }

        private static Result<int, Exception> DivideResult(int a, int b)
        {
            try
            {
                if (b == 0)
                    throw new DivideByZeroException();
                return Result<int, Exception>.Success(a / b);
            }
            catch (Exception ex)
            {
                return Result<int, Exception>.Failure(ex);
            }
        }

        #endregion

        #region AsyncResult Monad Examples

        /// <summary>
        /// Example of using the AsyncResult monad to handle async operations
        /// </summary>
        public static async Task AsyncResultExample()
        {
            // Instead of:
            // public async Task<bool> DeactivateAsync()
            // {
            //     if (!_isActive)
            //     {
            //         _logger.LogInformation("Connection discovery is already inactive");
            //         return true;
            //     }
            //
            //     try
            //     {
            //         _logger.LogInformation("Deactivating connection discovery");
            //         _isActive = false;
            //         return true;
            //     }
            //     catch (Exception ex)
            //     {
            //         _logger.LogError(ex, "Error deactivating connection discovery");
            //         return false;
            //     }
            // }

            // Use:
            AsyncResult<bool> result = DeactivateAsyncResult();
            bool success = await result.RunAsync();
            Console.WriteLine($"Deactivation {(success ? "succeeded" : "failed")}");
        }

        private static AsyncResult<bool> DeactivateAsyncResult()
        {
            bool isActive = true;

            if (!isActive)
            {
                Console.WriteLine("Connection discovery is already inactive");
                return AsyncResult<bool>.FromResult(true);
            }

            try
            {
                Console.WriteLine("Deactivating connection discovery");
                isActive = false;
                return AsyncResult<bool>.FromResult(true);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error deactivating connection discovery: {ex.Message}");
                return AsyncResult<bool>.FromResult(false);
            }
        }

        #endregion

        #region Printable Monad Examples

        /// <summary>
        /// Example of using the Printable monad to format values
        /// </summary>
        public static void PrintableExample()
        {
            // Instead of:
            // DateTime now = DateTime.Now;
            // Console.WriteLine($"Current time: {now:yyyy-MM-dd HH:mm:ss}");

            // Use:
            DateTime now = DateTime.Now;
            Printable<DateTime> printableNow = Printable<DateTime>.Create(now, dt => dt.ToString("yyyy-MM-dd HH:mm:ss"));
            Console.WriteLine($"Current time: {printableNow}");

            // Or with extension method:
            Console.WriteLine($"Current time: {now.ToPrintable(dt => dt.ToString("yyyy-MM-dd HH:mm:ss"))}");

            // Or with predefined printer:
            Console.WriteLine($"Current time: {now.ToPrintable(Printers.DateTime)}");
        }

        #endregion

        #region Combined Monad Examples

        /// <summary>
        /// Example of combining multiple monads
        /// </summary>
        public static async Task CombinedExample()
        {
            // Example of a complex operation that:
            // 1. Might return null (Option)
            // 2. Might fail with an exception (Result)
            // 3. Is asynchronous (AsyncResult)
            // 4. Needs custom formatting (Printable)

            AsyncResultError<Option<DateTime>, Exception> result = GetLastLoginTimeAsync("user123");

            await result.RunAsync().ContinueWith(t => {
                string message = t.Result.Match(
                    success: option => option.Match(
                        some: time => $"Last login: {time.ToPrintable(Printers.DateTime)}",
                        none: () => "No previous login found"
                    ),
                    failure: ex => $"Error retrieving login time: {ex.Message}"
                );

                Console.WriteLine(message);
            });
        }

        private static AsyncResultError<Option<DateTime>, Exception> GetLastLoginTimeAsync(string userId)
        {
            // Simulate an async operation that might fail and might return null
            try
            {
                // Simulate database lookup
                if (userId == "user123")
                {
                    // User exists and has logged in before
                    return AsyncResultError<Option<DateTime>, Exception>.Success(
                        Option<DateTime>.Some(DateTime.Now.AddDays(-1))
                    );
                }
                else if (userId == "newuser")
                {
                    // User exists but has never logged in
                    return AsyncResultError<Option<DateTime>, Exception>.Success(
                        Option<DateTime>.None
                    );
                }
                else
                {
                    // User doesn't exist
                    throw new ArgumentException($"User {userId} not found");
                }
            }
            catch (Exception ex)
            {
                return AsyncResultError<Option<DateTime>, Exception>.Failure(ex);
            }
        }

        #endregion
    }
}
