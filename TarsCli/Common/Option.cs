namespace TarsCli.Common;

/// <summary>
/// Represents an optional value, which may or may not be present.
/// This is a functional programming pattern to handle the absence of a value
/// without using null references.
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
    public static Option<T> Some(T value) =>
        value == null
            ? throw new ArgumentNullException(nameof(value), "Cannot create Some with a null value")
            : new Option<T>(value, true);

    /// <summary>
    /// Creates an Option with no value
    /// </summary>
    public static Option<T> None => new Option<T>(default, false);

    /// <summary>
    /// Returns true if the option has a value
    /// </summary>
    public bool IsSome => _hasValue;

    /// <summary>
    /// Returns true if the option has no value
    /// </summary>
    public bool IsNone => !_hasValue;

    /// <summary>
    /// Gets the value if present, otherwise throws an exception
    /// </summary>
    public T Value => _hasValue
        ? _value
        : throw new InvalidOperationException("Cannot access value of None");

    /// <summary>
    /// Maps the option to a new option with a different type
    /// </summary>
    public Option<TResult> Map<TResult>(Func<T, TResult> mapper) =>
        _hasValue
            ? Option<TResult>.Some(mapper(_value))
            : Option<TResult>.None;

    /// <summary>
    /// Maps the option to a new option with a different type asynchronously
    /// </summary>
    public async Task<Option<TResult>> MapAsync<TResult>(Func<T, Task<TResult>> mapper) =>
        _hasValue
            ? Option<TResult>.Some(await mapper(_value))
            : Option<TResult>.None;

    /// <summary>
    /// Binds the option to a new option with a different type
    /// </summary>
    public Option<TResult> Bind<TResult>(Func<T, Option<TResult>> binder) =>
        _hasValue
            ? binder(_value)
            : Option<TResult>.None;

    /// <summary>
    /// Binds the option to a new option with a different type asynchronously
    /// </summary>
    public async Task<Option<TResult>> BindAsync<TResult>(Func<T, Task<Option<TResult>>> binder) =>
        _hasValue
            ? await binder(_value)
            : Option<TResult>.None;

    /// <summary>
    /// Returns the value if present, otherwise returns the default value
    /// </summary>
    public T ValueOr(T defaultValue) =>
        _hasValue ? _value : defaultValue;

    /// <summary>
    /// Returns the value if present, otherwise returns the result of the function
    /// </summary>
    public T ValueOrElse(Func<T> defaultValueProvider) =>
        _hasValue ? _value : defaultValueProvider();

    /// <summary>
    /// Executes an action if the option has a value
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
    /// Executes an action if the option has no value
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
    /// Matches the option to one of two functions based on whether it has a value
    /// </summary>
    public TResult Match<TResult>(Func<T, TResult> some, Func<TResult> none) =>
        _hasValue ? some(_value) : none();

    /// <summary>
    /// Matches the option to one of two actions based on whether it has a value
    /// </summary>
    public void Match(Action<T> some, Action none)
    {
        if (_hasValue)
        {
            some(_value);
        }
        else
        {
            none();
        }
    }

    /// <summary>
    /// Converts the option to a nullable value (only works when T is a struct)
    /// </summary>
    public T? ToNullableIfStruct() =>
        _hasValue && _value is ValueType ? _value : default;

    /// <summary>
    /// Converts the option to an enumerable with zero or one elements
    /// </summary>
    public IEnumerable<T> ToEnumerable()
    {
        if (_hasValue)
        {
            yield return _value;
        }
    }

    public override string ToString() =>
        _hasValue ? $"Some({_value})" : "None";

    public override bool Equals(object? obj) =>
        obj is Option<T> other && Equals(other);

    public bool Equals(Option<T> other) =>
        _hasValue == other._hasValue &&
        (!_hasValue || EqualityComparer<T>.Default.Equals(_value, other._value));

    public override int GetHashCode() =>
        _hasValue ? (_value?.GetHashCode() ?? 0) : 0;

    public static bool operator ==(Option<T> left, Option<T> right) =>
        left.Equals(right);

    public static bool operator !=(Option<T> left, Option<T> right) =>
        !left.Equals(right);
}

/// <summary>
/// Static helper class for creating Option instances
/// </summary>
public static class Option
{
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
    public static Option<T> FromNullable<T>(T? value) where T : struct =>
        value.HasValue ? Some(value.Value) : None<T>();

    /// <summary>
    /// Converts a reference type to an Option
    /// </summary>
    public static Option<T> FromReference<T>(T value) where T : class =>
        value != null ? Some(value) : None<T>();

    /// <summary>
    /// Tries to execute a function and returns an Option with the result
    /// </summary>
    public static Option<T> Try<T>(Func<T> func)
    {
        try
        {
            return Some(func());
        }
        catch
        {
            return None<T>();
        }
    }

    /// <summary>
    /// Tries to execute a function asynchronously and returns an Option with the result
    /// </summary>
    public static async Task<Option<T>> TryAsync<T>(Func<Task<T>> func)
    {
        try
        {
            return Some(await func());
        }
        catch
        {
            return None<T>();
        }
    }
}