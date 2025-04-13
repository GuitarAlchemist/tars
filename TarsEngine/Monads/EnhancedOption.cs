namespace TarsEngine.Monads;

/// <summary>
/// Enhanced version of Option monad with additional utility methods.
/// Represents an optional value that may or may not be present.
/// Used to handle nullable references in a functional way.
/// </summary>
/// <typeparam name="T">The type of the value</typeparam>
public readonly struct EnhancedOption<T>
{
    private readonly T _value;
    private readonly bool _hasValue;

    private EnhancedOption(T value, bool hasValue)
    {
        _value = value;
        _hasValue = hasValue;
    }

    /// <summary>
    /// Creates an Option with a value
    /// </summary>
    public static EnhancedOption<T> Some(T value) =>
        value == null
            ? throw new ArgumentNullException(nameof(value), "Cannot create Some with a null value")
            : new EnhancedOption<T>(value, true);

    /// <summary>
    /// Creates an Option with no value
    /// </summary>
    public static EnhancedOption<T> None => new(default!, false);

    /// <summary>
    /// Returns true if the option has a value
    /// </summary>
    public bool IsSome => _hasValue;

    /// <summary>
    /// Returns true if the option has no value
    /// </summary>
    public bool IsNone => !_hasValue;

    /// <summary>
    /// Gets the value if present, or throws an exception if not
    /// </summary>
    public T Value => _hasValue ? _value : throw new InvalidOperationException("Option has no value");

    /// <summary>
    /// Gets the value if present, or returns the default value if not
    /// </summary>
    public T ValueOrDefault => _value;

    /// <summary>
    /// Maps the option to a new option with a different type
    /// </summary>
    public EnhancedOption<TResult> Map<TResult>(Func<T, TResult> mapper) =>
        _hasValue ? EnhancedOption<TResult>.Some(mapper(_value)) : EnhancedOption<TResult>.None;

    /// <summary>
    /// Applies a function that returns an Option to the value if present, or returns None if not
    /// </summary>
    public EnhancedOption<TResult> Bind<TResult>(Func<T, EnhancedOption<TResult>> binder) =>
        _hasValue ? binder(_value) : EnhancedOption<TResult>.None;

    /// <summary>
    /// Applies one of two functions depending on whether the option has a value
    /// </summary>
    public TResult Match<TResult>(Func<T, TResult> some, Func<TResult> none) =>
        _hasValue ? some(_value) : none();

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
    /// Performs an action if the option has a value
    /// </summary>
    public EnhancedOption<T> IfSome(Action<T> action)
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
    public EnhancedOption<T> IfNone(Action action)
    {
        if (!_hasValue)
        {
            action();
        }
        return this;
    }

    /// <summary>
    /// Filters the option based on a predicate
    /// </summary>
    public EnhancedOption<T> Filter(Func<T, bool> predicate) =>
        _hasValue && predicate(_value) ? this : None;

    /// <summary>
    /// Converts the option to an enumerable with zero or one elements
    /// </summary>
    public IEnumerable<T> AsEnumerable()
    {
        if (_hasValue)
        {
            yield return _value;
        }
    }

    /// <summary>
    /// Converts the option to a list with zero or one elements
    /// </summary>
    public List<T> ToList() => AsEnumerable().ToList();

    /// <summary>
    /// Converts the option to an array with zero or one elements
    /// </summary>
    public T[] ToArray() => AsEnumerable().ToArray();

    /// <summary>
    /// Converts the option to a nullable value
    /// </summary>
    public T? ToNullable<T1>() where T1 : struct =>
        _hasValue ? (T?)_value : default;

    /// <summary>
    /// Converts the option to a nullable reference
    /// </summary>
    public T? ToNullableReference<T1>() where T1 : class =>
        _hasValue ? _value : default;

    /// <summary>
    /// Implicitly converts from T to Option of T
    /// </summary>
    public static implicit operator EnhancedOption<T>(T value) =>
        value != null ? Some(value) : None;

    /// <summary>
    /// Explicitly converts from Option of T to T
    /// </summary>
    public static explicit operator T(EnhancedOption<T> option) =>
        option.Value;
}

/// <summary>
/// Static helper class for creating EnhancedOption instances
/// </summary>
public static class EnhancedOption
{
    /// <summary>
    /// Creates an Option with a value
    /// </summary>
    public static EnhancedOption<T> Some<T>(T value) => EnhancedOption<T>.Some(value);

    /// <summary>
    /// Creates an Option with no value
    /// </summary>
    public static EnhancedOption<T> None<T>() => EnhancedOption<T>.None;

    /// <summary>
    /// Converts a nullable value to an Option
    /// </summary>
    public static EnhancedOption<T> FromNullable<T>(T? value) where T : struct =>
        value.HasValue ? Some(value.Value) : None<T>();

    /// <summary>
    /// Converts a reference type to an Option
    /// </summary>
    public static EnhancedOption<T> FromReference<T>(T value) where T : class =>
        value != null ? Some(value) : None<T>();

    /// <summary>
    /// Tries to execute a function and returns an Option with the result
    /// </summary>
    public static EnhancedOption<T> Try<T>(Func<T> func)
    {
        try
        {
            var result = func();
            return result != null ? Some(result) : None<T>();
        }
        catch
        {
            return None<T>();
        }
    }

    /// <summary>
    /// Tries to execute a function asynchronously and returns an Option with the result
    /// </summary>
    public static async Task<EnhancedOption<T>> TryAsync<T>(Func<Task<T>> func)
    {
        try
        {
            var result = await func();
            return result != null ? Some(result) : None<T>();
        }
        catch
        {
            return None<T>();
        }
    }

    /// <summary>
    /// Combines multiple options into a single option containing a tuple
    /// </summary>
    public static EnhancedOption<(T1, T2)> Zip<T1, T2>(EnhancedOption<T1> option1, EnhancedOption<T2> option2) =>
        option1.IsSome && option2.IsSome
            ? Some((option1.Value, option2.Value))
            : None<(T1, T2)>();

    /// <summary>
    /// Combines multiple options into a single option containing a tuple
    /// </summary>
    public static EnhancedOption<(T1, T2, T3)> Zip<T1, T2, T3>(
        EnhancedOption<T1> option1,
        EnhancedOption<T2> option2,
        EnhancedOption<T3> option3) =>
        option1.IsSome && option2.IsSome && option3.IsSome
            ? Some((option1.Value, option2.Value, option3.Value))
            : None<(T1, T2, T3)>();

    /// <summary>
    /// Combines multiple options using a combiner function
    /// </summary>
    public static EnhancedOption<TResult> Map2<T1, T2, TResult>(
        EnhancedOption<T1> option1,
        EnhancedOption<T2> option2,
        Func<T1, T2, TResult> mapper) =>
        option1.IsSome && option2.IsSome
            ? Some(mapper(option1.Value, option2.Value))
            : None<TResult>();

    /// <summary>
    /// Combines multiple options using a combiner function
    /// </summary>
    public static EnhancedOption<TResult> Map3<T1, T2, T3, TResult>(
        EnhancedOption<T1> option1,
        EnhancedOption<T2> option2,
        EnhancedOption<T3> option3,
        Func<T1, T2, T3, TResult> mapper) =>
        option1.IsSome && option2.IsSome && option3.IsSome
            ? Some(mapper(option1.Value, option2.Value, option3.Value))
            : None<TResult>();

    /// <summary>
    /// Traverses a sequence of options and returns an option of sequence
    /// </summary>
    public static EnhancedOption<IEnumerable<T>> Sequence<T>(IEnumerable<EnhancedOption<T>> options)
    {
        var result = new List<T>();
        foreach (var option in options)
        {
            if (option.IsNone)
            {
                return None<IEnumerable<T>>();
            }
            result.Add(option.Value);
        }
        return Some<IEnumerable<T>>(result);
    }

    /// <summary>
    /// Maps a sequence using a function that returns an option and collects the results
    /// </summary>
    public static EnhancedOption<IEnumerable<TResult>> Traverse<T, TResult>(
        IEnumerable<T> source,
        Func<T, EnhancedOption<TResult>> mapper)
    {
        var result = new List<TResult>();
        foreach (var item in source)
        {
            var option = mapper(item);
            if (option.IsNone)
            {
                return None<IEnumerable<TResult>>();
            }
            result.Add(option.Value);
        }
        return Some<IEnumerable<TResult>>(result);
    }
}

/// <summary>
/// Extension methods for EnhancedOption
/// </summary>
public static class EnhancedOptionExtensions
{
    /// <summary>
    /// Converts a nullable value to an Option
    /// </summary>
    public static EnhancedOption<T> ToOption<T>(this T? value) where T : struct =>
        EnhancedOption.FromNullable(value);

    /// <summary>
    /// Converts a reference type to an Option
    /// </summary>
    public static EnhancedOption<T> ToOption<T>(this T value) where T : class =>
        EnhancedOption.FromReference(value);

    /// <summary>
    /// Converts a Task of T to an AsyncOption of T
    /// </summary>
    public static Task<EnhancedOption<T>> ToOptionAsync<T>(this Task<T> task) =>
        EnhancedOption.TryAsync(async () => await task);

    /// <summary>
    /// Filters a sequence to only the Some values
    /// </summary>
    public static IEnumerable<T> Choose<T>(this IEnumerable<EnhancedOption<T>> source) =>
        source.Where(option => option.IsSome).Select(option => option.Value);

    /// <summary>
    /// Maps a sequence using a function that returns an option and filters out the None results
    /// </summary>
    public static IEnumerable<TResult> Choose<T, TResult>(
        this IEnumerable<T> source,
        Func<T, EnhancedOption<TResult>> chooser) =>
        source.Select(chooser).Choose();

    /// <summary>
    /// Applies a function to each element of a sequence and returns the first Some result
    /// </summary>
    public static EnhancedOption<TResult> TryPick<T, TResult>(
        this IEnumerable<T> source,
        Func<T, EnhancedOption<TResult>> picker)
    {
        foreach (var item in source)
        {
            var result = picker(item);
            if (result.IsSome)
            {
                return result;
            }
        }
        return EnhancedOption<TResult>.None;
    }

    /// <summary>
    /// Returns the first element of a sequence that satisfies a predicate, or None if no such element exists
    /// </summary>
    public static EnhancedOption<T> TryFind<T>(
        this IEnumerable<T> source,
        Func<T, bool> predicate)
    {
        foreach (var item in source)
        {
            if (predicate(item))
            {
                return EnhancedOption<T>.Some(item);
            }
        }
        return EnhancedOption<T>.None;
    }

    /// <summary>
    /// Returns the first element of a sequence, or None if the sequence is empty
    /// </summary>
    public static EnhancedOption<T> TryFirst<T>(this IEnumerable<T> source)
    {
        using (var enumerator = source.GetEnumerator())
        {
            if (enumerator.MoveNext())
            {
                return EnhancedOption<T>.Some(enumerator.Current);
            }
            return EnhancedOption<T>.None;
        }
    }

    /// <summary>
    /// Returns the first element of a sequence that satisfies a predicate, or None if no such element exists
    /// </summary>
    public static EnhancedOption<T> TryFirst<T>(
        this IEnumerable<T> source,
        Func<T, bool> predicate) =>
        source.Where(predicate).TryFirst();

    /// <summary>
    /// Returns the last element of a sequence, or None if the sequence is empty
    /// </summary>
    public static EnhancedOption<T> TryLast<T>(this IEnumerable<T> source)
    {
        if (source is IList<T> list)
        {
            int count = list.Count;
            if (count > 0)
            {
                return EnhancedOption<T>.Some(list[count - 1]);
            }
            return EnhancedOption<T>.None;
        }
        else
        {
            using (var enumerator = source.GetEnumerator())
            {
                if (!enumerator.MoveNext())
                {
                    return EnhancedOption<T>.None;
                }

                T result = enumerator.Current;
                while (enumerator.MoveNext())
                {
                    result = enumerator.Current;
                }
                return EnhancedOption<T>.Some(result);
            }
        }
    }

    /// <summary>
    /// Returns the last element of a sequence that satisfies a predicate, or None if no such element exists
    /// </summary>
    public static EnhancedOption<T> TryLast<T>(
        this IEnumerable<T> source,
        Func<T, bool> predicate) =>
        source.Where(predicate).TryLast();

    /// <summary>
    /// Returns the single element of a sequence, or None if the sequence is empty or contains more than one element
    /// </summary>
    public static EnhancedOption<T> TrySingle<T>(this IEnumerable<T> source)
    {
        using (var enumerator = source.GetEnumerator())
        {
            if (!enumerator.MoveNext())
            {
                return EnhancedOption<T>.None;
            }

            T result = enumerator.Current;
            if (enumerator.MoveNext())
            {
                return EnhancedOption<T>.None;
            }
            return EnhancedOption<T>.Some(result);
        }
    }

    /// <summary>
    /// Returns the single element of a sequence that satisfies a predicate, or None if no such element exists or more than one such element exists
    /// </summary>
    public static EnhancedOption<T> TrySingle<T>(
        this IEnumerable<T> source,
        Func<T, bool> predicate) =>
        source.Where(predicate).TrySingle();

    /// <summary>
    /// Returns the element at a specified index in a sequence, or None if the index is out of range
    /// </summary>
    public static EnhancedOption<T> TryElementAt<T>(this IEnumerable<T> source, int index)
    {
        if (index < 0)
        {
            return EnhancedOption<T>.None;
        }

        if (source is IList<T> list)
        {
            if (index < list.Count)
            {
                return EnhancedOption<T>.Some(list[index]);
            }
            return EnhancedOption<T>.None;
        }
        else
        {
            using (var enumerator = source.GetEnumerator())
            {
                while (index > 0)
                {
                    if (!enumerator.MoveNext())
                    {
                        return EnhancedOption<T>.None;
                    }
                    index--;
                }

                if (enumerator.MoveNext())
                {
                    return EnhancedOption<T>.Some(enumerator.Current);
                }
                return EnhancedOption<T>.None;
            }
        }
    }

    /// <summary>
    /// Gets the value associated with the specified key in a dictionary, or None if the key is not found
    /// </summary>
    public static EnhancedOption<TValue> TryGetValue<TKey, TValue>(
        this IDictionary<TKey, TValue> dictionary,
        TKey key)
    {
        if (dictionary.TryGetValue(key, out var value))
        {
            return EnhancedOption<TValue>.Some(value);
        }
        return EnhancedOption<TValue>.None;
    }

    /// <summary>
    /// Parses a string to an integer, or returns None if parsing fails
    /// </summary>
    public static EnhancedOption<int> TryParseInt(this string str)
    {
        if (int.TryParse(str, out var result))
        {
            return EnhancedOption<int>.Some(result);
        }
        return EnhancedOption<int>.None;
    }

    /// <summary>
    /// Parses a string to a double, or returns None if parsing fails
    /// </summary>
    public static EnhancedOption<double> TryParseDouble(this string str)
    {
        if (double.TryParse(str, out var result))
        {
            return EnhancedOption<double>.Some(result);
        }
        return EnhancedOption<double>.None;
    }

    /// <summary>
    /// Parses a string to a decimal, or returns None if parsing fails
    /// </summary>
    public static EnhancedOption<decimal> TryParseDecimal(this string str)
    {
        if (decimal.TryParse(str, out var result))
        {
            return EnhancedOption<decimal>.Some(result);
        }
        return EnhancedOption<decimal>.None;
    }

    /// <summary>
    /// Parses a string to a DateTime, or returns None if parsing fails
    /// </summary>
    public static EnhancedOption<DateTime> TryParseDateTime(this string str)
    {
        if (DateTime.TryParse(str, out var result))
        {
            return EnhancedOption<DateTime>.Some(result);
        }
        return EnhancedOption<DateTime>.None;
    }

    /// <summary>
    /// Parses a string to a Guid, or returns None if parsing fails
    /// </summary>
    public static EnhancedOption<Guid> TryParseGuid(this string str)
    {
        if (Guid.TryParse(str, out var result))
        {
            return EnhancedOption<Guid>.Some(result);
        }
        return EnhancedOption<Guid>.None;
    }

    /// <summary>
    /// Parses a string to an enum value, or returns None if parsing fails
    /// </summary>
    public static EnhancedOption<TEnum> TryParseEnum<TEnum>(this string str) where TEnum : struct
    {
        if (Enum.TryParse<TEnum>(str, out var result))
        {
            return EnhancedOption<TEnum>.Some(result);
        }
        return EnhancedOption<TEnum>.None;
    }
}