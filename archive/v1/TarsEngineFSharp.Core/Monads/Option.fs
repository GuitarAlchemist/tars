namespace TarsEngineFSharp.Core.Monads

/// <summary>
/// Contains operations for the Option type.
/// </summary>
module Option =
    /// <summary>
    /// Creates a Some option with the given value.
    /// </summary>
    let some value = Some value

    /// <summary>
    /// Creates a None option.
    /// </summary>
    let none = None

    /// <summary>
    /// Returns true if the option is Some.
    /// </summary>
    let isSome = function
        | Some _ -> true
        | None -> false

    /// <summary>
    /// Returns true if the option is None.
    /// </summary>
    let isNone = function
        | Some _ -> false
        | None -> true

    /// <summary>
    /// Returns the value or raises an exception if the option is None.
    /// </summary>
    let getValue = function
        | Some value -> value
        | None -> failwith "Cannot get value from None"

    /// <summary>
    /// Returns the value or the provided default value if the option is None.
    /// </summary>
    let getValueOrDefault defaultValue = function
        | Some value -> value
        | None -> defaultValue

    /// <summary>
    /// Returns the value or computes a default value if the option is None.
    /// </summary>
    let getValueOrElse defaultFn = function
        | Some value -> value
        | None -> defaultFn()

    /// <summary>
    /// Applies a function to the value if the option is Some, otherwise returns None.
    /// </summary>
    let map f = function
        | Some value -> Some (f value)
        | None -> None

    /// <summary>
    /// Applies a function that returns an option to the value if the option is Some,
    /// otherwise returns None.
    /// </summary>
    let bind f = function
        | Some value -> f value
        | None -> None

    /// <summary>
    /// Applies a function to the value if the option is Some, otherwise returns the default value.
    /// </summary>
    let fold folder state = function
        | Some value -> folder state value
        | None -> state

    /// <summary>
    /// Applies a function to the value if the option is Some, otherwise does nothing.
    /// </summary>
    let iter f = function
        | Some value -> f value
        | None -> ()

    /// <summary>
    /// Filters an option based on a predicate.
    /// </summary>
    let filter predicate = function
        | Some value when predicate value -> Some value
        | _ -> None

    /// <summary>
    /// Converts a nullable value to an option.
    /// </summary>
    let ofNullable (nullable: 'T when 'T: null) =
        match box nullable with
        | null -> None
        | _ -> Some nullable

    /// <summary>
    /// Converts an option to a nullable value.
    /// </summary>
    let toNullable = function
        | Some value -> value
        | None -> null

    /// <summary>
    /// Converts a Result to an Option, discarding the error.
    /// </summary>
    let ofResult = function
        | Result.Success value -> Some value
        | Result.Failure _ -> None

    /// <summary>
    /// Converts an Option to a Result, using the provided error if the option is None.
    /// </summary>
    let toResult error = function
        | Some value -> Result.Success value
        | None -> Result.Failure error

    /// <summary>
    /// Combines two options, returning a tuple of the values if both are Some,
    /// or None if either is None.
    /// </summary>
    let zip option1 option2 =
        match option1, option2 with
        | Some value1, Some value2 -> Some (value1, value2)
        | _ -> None

    /// <summary>
    /// Combines a list of options into a single option with a list of values.
    /// </summary>
    let sequence options =
        let folder state option =
            match state, option with
            | Some values, Some value -> Some (value :: values)
            | _ -> None

        let initialState = Some []

        options
        |> List.fold folder initialState
        |> map List.rev

    /// <summary>
    /// Applies a function that returns an option to each element in the list and collects the results.
    /// </summary>
    let traverse f list =
        let folder state element =
            match state with
            | Some values ->
                match f element with
                | Some value -> Some (value :: values)
                | None -> None
            | None -> None

        let initialState = Some []

        list
        |> List.fold folder initialState
        |> map List.rev

    /// <summary>
    /// Converts an Option to a C# style option with pattern matching.
    /// </summary>
    let toCSharp option =
        match option with
        | Some value -> struct (true, value)
        | None -> struct (false, Unchecked.defaultof<_>)

/// <summary>
/// Contains extension methods for working with the Option type.
/// </summary>
[<AutoOpen>]
module OptionExtensions =
    /// <summary>
    /// Provides a fluent syntax for working with options.
    /// </summary>
    type OptionBuilder() =
        member _.Return(value) = Some value
        member _.ReturnFrom(option) = option
        member _.Bind(option, f) = Option.bind f option
        member _.Zero() = None
        member _.Delay(f) = f
        member _.Run(f) = f()

        member _.TryWith(body, handler) =
            try
                body()
            with
            | ex -> handler ex

        member _.TryFinally(body, compensation) =
            try
                body()
            finally
                compensation()

        member _.Using(disposable: #System.IDisposable, body) =
            using disposable body

        member this.While(guard, body) =
            if not (guard()) then
                Some ()
            else
                body() |> Option.bind (fun () -> this.While(guard, body))

        member _.For(sequence, body) =
            sequence
            |> Seq.fold (fun state element ->
                state |> Option.bind (fun () -> body element)
            ) (Some ())

        member _.Combine(option, f) =
            option |> Option.bind (fun () -> f())

        member _.MergeSources(option1, option2) =
            Option.zip option1 option2

    /// <summary>
    /// Creates a computation expression for working with options.
    /// </summary>
    let option = OptionBuilder()
