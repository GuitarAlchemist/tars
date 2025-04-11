using System;
using System.Collections.Generic;

namespace TarsEngine.Monads
{
    /// <summary>
    /// Represents a value that can be converted to a string representation.
    /// Used for consistent string formatting across different types.
    /// </summary>
    /// <typeparam name="T">The type of the value</typeparam>
    public readonly struct Printable<T>
    {
        private readonly T _value;
        private readonly Func<T, string> _printer;

        private Printable(T value, Func<T, string> printer)
        {
            _value = value;
            _printer = printer;
        }

        /// <summary>
        /// Creates a Printable with a value and a custom printer function
        /// </summary>
        public static Printable<T> Create(T value, Func<T, string> printer) =>
            new(value, printer);

        /// <summary>
        /// Creates a Printable with a value and a default printer function
        /// </summary>
        public static Printable<T> Create(T value) =>
            new(value, v => v?.ToString() ?? "null");

        /// <summary>
        /// Gets the underlying value
        /// </summary>
        public T Value => _value;

        /// <summary>
        /// Converts the value to a string using the printer function
        /// </summary>
        public string Print() => _printer(_value);

        /// <summary>
        /// Applies a function to the value and returns a new Printable
        /// </summary>
        public Printable<TResult> Map<TResult>(Func<T, TResult> mapper, Func<TResult, string>? printer = null) =>
            new(mapper(_value), printer ?? (v => v?.ToString() ?? "null"));

        /// <summary>
        /// Applies a function that returns a Printable to the value
        /// </summary>
        public Printable<TResult> Bind<TResult>(Func<T, Printable<TResult>> binder) =>
            binder(_value);

        /// <summary>
        /// Implicitly converts from T to Printable of T
        /// </summary>
        public static implicit operator Printable<T>(T value) =>
            Create(value);

        /// <summary>
        /// Explicitly converts from Printable of T to T
        /// </summary>
        public static explicit operator T(Printable<T> printable) => printable.Value;

        /// <summary>
        /// Converts the Printable to a string
        /// </summary>
        public override string ToString() => Print();
    }

    /// <summary>
    /// Extension methods for Printable of T
    /// </summary>
    public static class PrintableExtensions
    {
        /// <summary>
        /// Converts a value to a Printable with a custom printer function
        /// </summary>
        public static Printable<T> ToPrintable<T>(this T value, Func<T, string> printer) =>
            Printable<T>.Create(value, printer);

        /// <summary>
        /// Converts a value to a Printable with a default printer function
        /// </summary>
        public static Printable<T> ToPrintable<T>(this T value) =>
            Printable<T>.Create(value);

        /// <summary>
        /// Converts a collection to a Printable with a custom separator
        /// </summary>
        public static Printable<IEnumerable<T>> ToPrintable<T>(this IEnumerable<T> collection, string separator = ", ") =>
            Printable<IEnumerable<T>>.Create(collection, c => {
                if (c == null) return "null";
                return string.Join(separator, c);
            });

        /// <summary>
        /// Converts an Option to a Printable
        /// </summary>
        public static Printable<Option<T>> ToPrintable<T>(this Option<T> option) =>
            Printable<Option<T>>.Create(option, o => o.Match(
                some: v => $"Some({v})",
                none: () => "None"
            ));

        /// <summary>
        /// Converts a Result to a Printable
        /// </summary>
        public static Printable<Result<T, TError>> ToPrintable<T, TError>(this Result<T, TError> result) =>
            Printable<Result<T, TError>>.Create(result, r => r.Match(
                success: v => $"Success({v})",
                failure: e => $"Failure({e})"
            ));
    }

    /// <summary>
    /// Common printer functions for different types
    /// </summary>
    public static class Printers
    {
        /// <summary>
        /// Prints a DateTime in a standard format
        /// </summary>
        public static string DateTime(DateTime dateTime) =>
            dateTime.ToString("yyyy-MM-dd HH:mm:ss");

        /// <summary>
        /// Prints a TimeSpan in a standard format
        /// </summary>
        public static string TimeSpan(TimeSpan timeSpan) =>
            timeSpan.ToString(@"hh\:mm\:ss\.fff");

        /// <summary>
        /// Prints a decimal with a specified number of decimal places
        /// </summary>
        public static string Decimal(decimal value, int decimalPlaces = 2) =>
            value.ToString($"F{decimalPlaces}");

        /// <summary>
        /// Prints a double with a specified number of decimal places
        /// </summary>
        public static string Double(double value, int decimalPlaces = 2) =>
            value.ToString($"F{decimalPlaces}");

        /// <summary>
        /// Prints a collection with a custom separator and item formatter
        /// </summary>
        public static string Collection<T>(IEnumerable<T> collection, Func<T, string> itemFormatter = null, string separator = ", ") =>
            string.Join(separator, Enumerable.Select(collection, itemFormatter ?? (v => v?.ToString() ?? "null")));
    }
}
