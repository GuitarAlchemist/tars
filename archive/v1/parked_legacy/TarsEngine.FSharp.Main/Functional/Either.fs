namespace TarsEngine.FSharp.Main.Functional

open System
open TarsEngine.FSharp.Main.Monads

/// <summary>
/// Represents a value of one of two possible types (a disjoint union).
/// An instance of Either is either a Left or a Right.
/// By convention, Left is used for failure and Right is used for success.
/// </summary>
/// <typeparam name="'TLeft">The type of the Left value</typeparam>
/// <typeparam name="'TRight">The type of the Right value</typeparam>
type Either<'TLeft, 'TRight> =
    | Left of 'TLeft
    | Right of 'TRight

/// <summary>
/// Module containing functions for working with Either types
/// </summary>
module Either =
    /// <summary>
    /// Pattern matches on the Either, applying the appropriate function based on whether it's a Left or Right
    /// </summary>
    let match' leftFunc rightFunc either =
        match either with
        | Left value -> leftFunc value
        | Right value -> rightFunc value

    /// <summary>
    /// Performs an action based on whether the Either is a Left or Right
    /// </summary>
    let iter leftAction rightAction either =
        match either with
        | Left value -> leftAction value
        | Right value -> rightAction value

    /// <summary>
    /// Maps the Right value of the Either using the given function
    /// </summary>
    let map mapper either =
        match either with
        | Left value -> Left value
        | Right value -> Right (mapper value)

    /// <summary>
    /// Maps the Left value of the Either using the given function
    /// </summary>
    let mapLeft mapper either =
        match either with
        | Left value -> Left (mapper value)
        | Right value -> Right value

    /// <summary>
    /// Applies a function to the Right value of the Either
    /// </summary>
    let bind binder either =
        match either with
        | Left value -> Left value
        | Right value -> binder value

    /// <summary>
    /// Returns true if the Either is a Left
    /// </summary>
    let isLeft either =
        match either with
        | Left _ -> true
        | Right _ -> false

    /// <summary>
    /// Returns true if the Either is a Right
    /// </summary>
    let isRight either =
        match either with
        | Left _ -> false
        | Right _ -> true

    /// <summary>
    /// Gets the Left value or throws an exception if the Either is a Right
    /// </summary>
    let leftValue either =
        match either with
        | Left value -> value
        | Right _ -> raise (InvalidOperationException("Cannot get Left value from a Right"))

    /// <summary>
    /// Gets the Right value or throws an exception if the Either is a Left
    /// </summary>
    let rightValue either =
        match either with
        | Left _ -> raise (InvalidOperationException("Cannot get Right value from a Left"))
        | Right value -> value

    /// <summary>
    /// Gets the Left value or a default value if the Either is a Right
    /// </summary>
    let leftValueOrDefault defaultValue either =
        match either with
        | Left value -> value
        | Right _ -> defaultValue

    /// <summary>
    /// Gets the Right value or a default value if the Either is a Left
    /// </summary>
    let rightValueOrDefault defaultValue either =
        match either with
        | Left _ -> defaultValue
        | Right value -> value

    /// <summary>
    /// Tries to execute a function and returns a Right with the result if successful, or a Left with the exception if not
    /// </summary>
    let tryFunc (func: unit -> 'TRight) : Either<Exception, 'TRight> =
        try
            Right (func())
        with
        | ex -> Left ex

    /// <summary>
    /// Converts an Option to an Either, using the provided error value for None
    /// </summary>
    let ofOption errorValue option =
        match option with
        | Some value -> Right value
        | None -> Left errorValue

    /// <summary>
    /// Converts an Either to an Option, discarding the Left value
    /// </summary>
    let toOption either =
        match either with
        | Left _ -> None
        | Right value -> Some value
