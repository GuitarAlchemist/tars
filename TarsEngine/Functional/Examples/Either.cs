using System;

namespace TarsEngine.Functional.Examples
{
    /// <summary>
    /// Represents a value of one of two possible types (a disjoint union).
    /// An instance of Either is either a Left or a Right.
    /// By convention, Left is used for failure and Right is used for success.
    /// </summary>
    /// <typeparam name="TLeft">The type of the Left value</typeparam>
    /// <typeparam name="TRight">The type of the Right value</typeparam>
    public abstract record Either<TLeft, TRight> : DiscriminatedUnion
    {
        protected Either() { }

        /// <summary>
        /// Creates a Left instance
        /// </summary>
        public static Either<TLeft, TRight> Left(TLeft value) => new Left<TLeft, TRight>(value);

        /// <summary>
        /// Creates a Right instance
        /// </summary>
        public static Either<TLeft, TRight> Right(TRight value) => new Right<TLeft, TRight>(value);

        /// <summary>
        /// Pattern matches on the Either, applying the appropriate function based on whether it's a Left or Right
        /// </summary>
        public TResult Match<TResult>(Func<TLeft, TResult> leftFunc, Func<TRight, TResult> rightFunc) =>
            this switch
            {
                Left<TLeft, TRight> left => leftFunc(left.Value),
                Right<TLeft, TRight> right => rightFunc(right.Value),
                _ => UnhandledCase<TResult>(this)
            };

        /// <summary>
        /// Performs an action based on whether the Either is a Left or Right
        /// </summary>
        public void Match(Action<TLeft> leftAction, Action<TRight> rightAction)
        {
            switch (this)
            {
                case Left<TLeft, TRight> left:
                    leftAction(left.Value);
                    break;
                case Right<TLeft, TRight> right:
                    rightAction(right.Value);
                    break;
                default:
                    throw new InvalidOperationException($"Unhandled case: {GetType().Name}");
            }
        }

        /// <summary>
        /// Maps the Right value of the Either using the given function
        /// </summary>
        public Either<TLeft, TNewRight> Map<TNewRight>(Func<TRight, TNewRight> mapper) =>
            this switch
            {
                Left<TLeft, TRight> left => Either<TLeft, TNewRight>.Left(left.Value),
                Right<TLeft, TRight> right => Either<TLeft, TNewRight>.Right(mapper(right.Value)),
                _ => throw new InvalidOperationException($"Unhandled case: {GetType().Name}")
            };

        /// <summary>
        /// Maps the Left value of the Either using the given function
        /// </summary>
        public Either<TNewLeft, TRight> MapLeft<TNewLeft>(Func<TLeft, TNewLeft> mapper) =>
            this switch
            {
                Left<TLeft, TRight> left => Either<TNewLeft, TRight>.Left(mapper(left.Value)),
                Right<TLeft, TRight> right => Either<TNewLeft, TRight>.Right(right.Value),
                _ => throw new InvalidOperationException($"Unhandled case: {GetType().Name}")
            };

        /// <summary>
        /// Applies a function to the Right value of the Either
        /// </summary>
        public Either<TLeft, TNewRight> Bind<TNewRight>(Func<TRight, Either<TLeft, TNewRight>> binder) =>
            this switch
            {
                Left<TLeft, TRight> left => Either<TLeft, TNewRight>.Left(left.Value),
                Right<TLeft, TRight> right => binder(right.Value),
                _ => throw new InvalidOperationException($"Unhandled case: {GetType().Name}")
            };

        /// <summary>
        /// Returns true if the Either is a Left
        /// </summary>
        public bool IsLeft => this is Left<TLeft, TRight>;

        /// <summary>
        /// Returns true if the Either is a Right
        /// </summary>
        public bool IsRight => this is Right<TLeft, TRight>;

        /// <summary>
        /// Gets the Left value or throws an exception if the Either is a Right
        /// </summary>
        public TLeft LeftValue => this is Left<TLeft, TRight> left
            ? left.Value
            : throw new InvalidOperationException("Cannot get Left value from a Right");

        /// <summary>
        /// Gets the Right value or throws an exception if the Either is a Left
        /// </summary>
        public TRight RightValue => this is Right<TLeft, TRight> right
            ? right.Value
            : throw new InvalidOperationException("Cannot get Right value from a Left");

        /// <summary>
        /// Gets the Left value or a default value if the Either is a Right
        /// </summary>
        public TLeft LeftValueOrDefault(TLeft? defaultValue = default) =>
            this is Left<TLeft, TRight> left ? left.Value : defaultValue!;

        /// <summary>
        /// Gets the Right value or a default value if the Either is a Left
        /// </summary>
        public TRight RightValueOrDefault(TRight? defaultValue = default) =>
            this is Right<TLeft, TRight> right ? right.Value : defaultValue!;
    }

    /// <summary>
    /// Represents the Left case of an Either
    /// </summary>
    public sealed record Left<TLeft, TRight> : Either<TLeft, TRight>
    {
        public TLeft Value { get; }

        public Left(TLeft value)
        {
            Value = value;
        }
    }

    /// <summary>
    /// Represents the Right case of an Either
    /// </summary>
    public sealed record Right<TLeft, TRight> : Either<TLeft, TRight>
    {
        public TRight Value { get; }

        public Right(TRight value)
        {
            Value = value;
        }
    }

    /// <summary>
    /// Extension methods for Either
    /// </summary>
    public static class EitherExtensions
    {
        /// <summary>
        /// Converts a value to a Right
        /// </summary>
        public static Either<TLeft, TRight> ToRight<TLeft, TRight>(this TRight value) =>
            Either<TLeft, TRight>.Right(value);

        /// <summary>
        /// Converts a value to a Left
        /// </summary>
        public static Either<TLeft, TRight> ToLeft<TLeft, TRight>(this TLeft value) =>
            Either<TLeft, TRight>.Left(value);

        /// <summary>
        /// Tries to execute a function and returns a Right with the result if successful, or a Left with the exception if not
        /// </summary>
        public static Either<Exception, TRight> Try<TRight>(Func<TRight> func)
        {
            try
            {
                return Either<Exception, TRight>.Right(func());
            }
            catch (Exception ex)
            {
                return Either<Exception, TRight>.Left(ex);
            }
        }

        /// <summary>
        /// Converts an Option to an Either, using the provided error value for None
        /// </summary>
        public static Either<TLeft, TRight> ToEither<TLeft, TRight>(
            this Monads.Option<TRight> option, TLeft errorValue) =>
            option.Match(
                some: value => Either<TLeft, TRight>.Right(value),
                none: () => Either<TLeft, TRight>.Left(errorValue)
            );

        /// <summary>
        /// Converts an EnhancedOption to an Either, using the provided error value for None
        /// </summary>
        public static Either<TLeft, TRight> ToEither<TLeft, TRight>(
            this Monads.EnhancedOption<TRight> option, TLeft errorValue) =>
            option.Match(
                some: value => Either<TLeft, TRight>.Right(value),
                none: () => Either<TLeft, TRight>.Left(errorValue)
            );

        /// <summary>
        /// Converts an Either to an Option, discarding the Left value
        /// </summary>
        public static Monads.Option<TRight> ToOption<TLeft, TRight>(this Either<TLeft, TRight> either) =>
            either.Match(
                leftFunc: _ => Monads.Option<TRight>.None,
                rightFunc: value => Monads.Option<TRight>.Some(value)
            );
    }
}
