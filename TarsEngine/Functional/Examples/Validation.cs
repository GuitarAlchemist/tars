using System;
using System.Collections.Generic;
using System.Linq;

namespace TarsEngine.Functional.Examples
{
    /// <summary>
    /// Represents a validation result that can be either valid with a value or invalid with a list of errors.
    /// </summary>
    /// <typeparam name="T">The type of the value</typeparam>
    /// <typeparam name="TError">The type of the errors</typeparam>
    public abstract record Validation<T, TError> : DiscriminatedUnion
    {
        protected Validation() { }

        /// <summary>
        /// Creates a valid validation result
        /// </summary>
        public static Validation<T, TError> Valid(T value) => new Valid<T, TError>(value);

        /// <summary>
        /// Creates an invalid validation result with a single error
        /// </summary>
        public static Validation<T, TError> Invalid(TError error) =>
            new Invalid<T, TError>(new List<TError> { error });

        /// <summary>
        /// Creates an invalid validation result with multiple errors
        /// </summary>
        public static Validation<T, TError> Invalid(IEnumerable<TError> errors) =>
            new Invalid<T, TError>(errors.ToList());

        /// <summary>
        /// Pattern matches on the validation result, applying the appropriate function
        /// </summary>
        public TResult Match<TResult>(
            Func<T, TResult> validFunc,
            Func<IReadOnlyList<TError>, TResult> invalidFunc) =>
            this switch
            {
                Valid<T, TError> valid => validFunc(valid.Value),
                Invalid<T, TError> invalid => invalidFunc(invalid.Errors),
                _ => UnhandledCase<TResult>(this)
            };

        /// <summary>
        /// Performs an action based on whether the validation result is valid or invalid
        /// </summary>
        public void Match(
            Action<T> validAction,
            Action<IReadOnlyList<TError>> invalidAction)
        {
            switch (this)
            {
                case Valid<T, TError> valid:
                    validAction(valid.Value);
                    break;
                case Invalid<T, TError> invalid:
                    invalidAction(invalid.Errors);
                    break;
                default:
                    throw new InvalidOperationException($"Unhandled case: {GetType().Name}");
            }
        }

        /// <summary>
        /// Maps the value of a valid validation result using the given function
        /// </summary>
        public Validation<TResult, TError> Map<TResult>(Func<T, TResult> mapper) =>
            this switch
            {
                Valid<T, TError> valid => Validation<TResult, TError>.Valid(mapper(valid.Value)),
                Invalid<T, TError> invalid => Validation<TResult, TError>.Invalid(invalid.Errors),
                _ => throw new InvalidOperationException($"Unhandled case: {GetType().Name}")
            };

        /// <summary>
        /// Maps the errors of an invalid validation result using the given function
        /// </summary>
        public Validation<T, TNewError> MapError<TNewError>(Func<TError, TNewError> mapper) =>
            this switch
            {
                Valid<T, TError> valid => Validation<T, TNewError>.Valid(valid.Value),
                Invalid<T, TError> invalid => Validation<T, TNewError>.Invalid(invalid.Errors.Select(mapper)),
                _ => throw new InvalidOperationException($"Unhandled case: {GetType().Name}")
            };

        /// <summary>
        /// Applies a function to the value of a valid validation result
        /// </summary>
        public Validation<TResult, TError> Bind<TResult>(Func<T, Validation<TResult, TError>> binder) =>
            this switch
            {
                Valid<T, TError> valid => binder(valid.Value),
                Invalid<T, TError> invalid => Validation<TResult, TError>.Invalid(invalid.Errors),
                _ => throw new InvalidOperationException($"Unhandled case: {GetType().Name}")
            };

        /// <summary>
        /// Returns true if the validation result is valid
        /// </summary>
        public bool IsValid => this is Valid<T, TError>;

        /// <summary>
        /// Returns true if the validation result is invalid
        /// </summary>
        public bool IsInvalid => this is Invalid<T, TError>;

        /// <summary>
        /// Gets the value or throws an exception if the validation result is invalid
        /// </summary>
        public T Value => this is Valid<T, TError> valid
            ? valid.Value
            : throw new InvalidOperationException("Cannot get value from an invalid validation result");

        /// <summary>
        /// Gets the errors or throws an exception if the validation result is valid
        /// </summary>
        public IReadOnlyList<TError> Errors => this is Invalid<T, TError> invalid
            ? invalid.Errors
            : throw new InvalidOperationException("Cannot get errors from a valid validation result");

        /// <summary>
        /// Gets the value or a default value if the validation result is invalid
        /// </summary>
        public T ValueOrDefault(T? defaultValue = default) =>
            this is Valid<T, TError> valid ? valid.Value : defaultValue!;

        /// <summary>
        /// Gets the errors or an empty list if the validation result is valid
        /// </summary>
        public IReadOnlyList<TError> ErrorsOrEmpty() =>
            this is Invalid<T, TError> invalid ? invalid.Errors : Array.Empty<TError>();
    }

    /// <summary>
    /// Represents a valid validation result
    /// </summary>
    public sealed record Valid<T, TError> : Validation<T, TError>
    {
        public new T Value { get; }

        public Valid(T value)
        {
            Value = value;
        }
    }

    /// <summary>
    /// Represents an invalid validation result
    /// </summary>
    public sealed record Invalid<T, TError> : Validation<T, TError>
    {
        public new IReadOnlyList<TError> Errors { get; }

        public Invalid(IReadOnlyList<TError> errors)
        {
            Errors = errors;
        }
    }

    /// <summary>
    /// Extension methods for Validation
    /// </summary>
    public static class ValidationExtensions
    {
        /// <summary>
        /// Converts a value to a valid validation result
        /// </summary>
        public static Validation<T, TError> ToValid<T, TError>(this T value) =>
            Validation<T, TError>.Valid(value);

        /// <summary>
        /// Converts an error to an invalid validation result
        /// </summary>
        public static Validation<T, TError> ToInvalid<T, TError>(this TError error) =>
            Validation<T, TError>.Invalid(error);

        /// <summary>
        /// Converts a list of errors to an invalid validation result
        /// </summary>
        public static Validation<T, TError> ToInvalid<T, TError>(this IEnumerable<TError> errors) =>
            Validation<T, TError>.Invalid(errors);

        /// <summary>
        /// Tries to execute a function and returns a valid validation result with the result if successful,
        /// or an invalid validation result with the exception if not
        /// </summary>
        public static Validation<T, Exception> Try<T>(Func<T> func)
        {
            try
            {
                return Validation<T, Exception>.Valid(func());
            }
            catch (Exception ex)
            {
                return Validation<T, Exception>.Invalid(ex);
            }
        }

        /// <summary>
        /// Combines multiple validation results into a single validation result
        /// </summary>
        public static Validation<IEnumerable<T>, TError> Sequence<T, TError>(
            this IEnumerable<Validation<T, TError>> validations)
        {
            var values = new List<T>();
            var errors = new List<TError>();

            foreach (var validation in validations)
            {
                validation.Match<object>(
                    validFunc: value => { values.Add(value); return null!; },
                    invalidFunc: errs => { errors.AddRange(errs); return null!; }
                );
            }

            return errors.Any()
                ? Validation<IEnumerable<T>, TError>.Invalid(errors)
                : Validation<IEnumerable<T>, TError>.Valid(values);
        }

        /// <summary>
        /// Applies a validation function to each element of a sequence and combines the results
        /// </summary>
        public static Validation<IEnumerable<TResult>, TError> Traverse<T, TResult, TError>(
            this IEnumerable<T> source,
            Func<T, Validation<TResult, TError>> validator)
        {
            return source.Select(validator).Sequence();
        }

        /// <summary>
        /// Combines two validation results using a combiner function
        /// </summary>
        public static Validation<TResult, TError> Map2<T1, T2, TResult, TError>(
            Validation<T1, TError> validation1,
            Validation<T2, TError> validation2,
            Func<T1, T2, TResult> mapper)
        {
            if (validation1.IsValid && validation2.IsValid)
            {
                return Validation<TResult, TError>.Valid(mapper(validation1.Value, validation2.Value));
            }

            var errors = new List<TError>();
            if (validation1.IsInvalid)
            {
                errors.AddRange(validation1.Errors);
            }
            if (validation2.IsInvalid)
            {
                errors.AddRange(validation2.Errors);
            }

            return Validation<TResult, TError>.Invalid(errors);
        }

        /// <summary>
        /// Combines three validation results using a combiner function
        /// </summary>
        public static Validation<TResult, TError> Map3<T1, T2, T3, TResult, TError>(
            Validation<T1, TError> validation1,
            Validation<T2, TError> validation2,
            Validation<T3, TError> validation3,
            Func<T1, T2, T3, TResult> mapper)
        {
            if (validation1.IsValid && validation2.IsValid && validation3.IsValid)
            {
                return Validation<TResult, TError>.Valid(
                    mapper(validation1.Value, validation2.Value, validation3.Value));
            }

            var errors = new List<TError>();
            if (validation1.IsInvalid)
            {
                errors.AddRange(validation1.Errors);
            }
            if (validation2.IsInvalid)
            {
                errors.AddRange(validation2.Errors);
            }
            if (validation3.IsInvalid)
            {
                errors.AddRange(validation3.Errors);
            }

            return Validation<TResult, TError>.Invalid(errors);
        }

        /// <summary>
        /// Converts an Option to a Validation, using the provided error for None
        /// </summary>
        public static Validation<T, TError> ToValidation<T, TError>(
            this Monads.Option<T> option, TError error) =>
            option.Match(
                some: value => Validation<T, TError>.Valid(value),
                none: () => Validation<T, TError>.Invalid(error)
            );

        /// <summary>
        /// Converts a Validation to an Option, discarding the errors
        /// </summary>
        public static Monads.Option<T> ToOption<T, TError>(this Validation<T, TError> validation) =>
            validation.Match(
                validFunc: value => Monads.Option<T>.Some(value),
                invalidFunc: _ => Monads.Option<T>.None
            );

        /// <summary>
        /// Converts an Either to a Validation
        /// </summary>
        public static Validation<T, TError> ToValidation<T, TError>(this Either<TError, T> either) =>
            either.Match(
                leftFunc: error => Validation<T, TError>.Invalid(error),
                rightFunc: value => Validation<T, TError>.Valid(value)
            );

        /// <summary>
        /// Converts a Validation to an Either
        /// </summary>
        public static Either<IReadOnlyList<TError>, T> ToEither<T, TError>(this Validation<T, TError> validation) =>
            validation.Match(
                validFunc: value => Either<IReadOnlyList<TError>, T>.Right(value),
                invalidFunc: errors => Either<IReadOnlyList<TError>, T>.Left(errors)
            );
    }
}
