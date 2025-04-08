using System;

namespace TarsEngine.Monads
{
    /// <summary>
    /// Represents an optional value that may or may not be present.
    /// Used to handle nullable references in a functional way.
    /// </summary>
    /// <typeparam name="T">The type of the value</typeparam>
    public readonly struct Option<T>
    {
        private readonly T _value;
        private readonly bool _hasValue;

        private Option(T value, bool hasValue)
        {
            _value = value;
            _hasValue = hasValue;
        }

        /// <summary>
        /// Creates an Option with a value
        /// </summary>
        public static Option<T> Some(T value) => new(value, true);

        /// <summary>
        /// Creates an Option with no value
        /// </summary>
        public static Option<T> None => new(default, false);

        /// <summary>
        /// Returns true if the option has a value
        /// </summary>
        public bool HasValue => _hasValue;

        /// <summary>
        /// Gets the value if present, or throws an exception if not
        /// </summary>
        public T Value => _hasValue ? _value : throw new InvalidOperationException("Option has no value");

        /// <summary>
        /// Gets the value if present, or returns the default value if not
        /// </summary>
        public T ValueOrDefault => _value;

        /// <summary>
        /// Gets the value if present, or returns the specified default value if not
        /// </summary>
        public T ValueOr(T defaultValue) => _hasValue ? _value : defaultValue;

        /// <summary>
        /// Gets the value if present, or returns the result of the specified function if not
        /// </summary>
        public T ValueOr(Func<T> defaultValueProvider) => _hasValue ? _value : defaultValueProvider();

        /// <summary>
        /// Applies a function to the value if present, or returns None if not
        /// </summary>
        public Option<TResult> Map<TResult>(Func<T, TResult> mapper) =>
            _hasValue ? Option<TResult>.Some(mapper(_value)) : Option<TResult>.None;

        /// <summary>
        /// Applies a function that returns an Option to the value if present, or returns None if not
        /// </summary>
        public Option<TResult> Bind<TResult>(Func<T, Option<TResult>> binder) =>
            _hasValue ? binder(_value) : Option<TResult>.None;

        /// <summary>
        /// Applies one of two functions depending on whether the option has a value
        /// </summary>
        public TResult Match<TResult>(Func<T, TResult> some, Func<TResult> none) =>
            _hasValue ? some(_value) : none();

        /// <summary>
        /// Performs an action if the option has a value
        /// </summary>
        public Option<T> IfSome(Action<T> action)
        {
            if (_hasValue)
            {
                action(_value);
            }
            return this;
        }

        /// <summary>
        /// Performs an action if the option has no value
        /// </summary>
        public Option<T> IfNone(Action action)
        {
            if (!_hasValue)
            {
                action();
            }
            return this;
        }

        /// <summary>
        /// Implicitly converts from T to Option of T
        /// </summary>
        public static implicit operator Option<T>(T value) =>
            value == null ? None : Some(value);

        /// <summary>
        /// Explicitly converts from Option of T to T, throwing if no value
        /// </summary>
        public static explicit operator T(Option<T> option) => option.Value;
    }

    /// <summary>
    /// Extension methods for Option of T
    /// </summary>
    public static class OptionExtensions
    {
        /// <summary>
        /// Converts a nullable value to an Option
        /// </summary>
        public static Option<T> ToOption<T>(this T? value) where T : class =>
            value != null ? Option<T>.Some(value) : Option<T>.None;

        /// <summary>
        /// Converts a nullable value to an Option
        /// </summary>
        public static Option<T> ToOption<T>(this T? value) where T : struct =>
            value.HasValue ? Option<T>.Some(value.Value) : Option<T>.None;
    }
}
