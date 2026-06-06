namespace TarsEngine.DSL

/// Module containing monadic abstractions for the TARS DSL
module Monads =
    /// Option monad for handling optional values
    type DslOption<'T> =
        | Some of 'T
        | None

    /// Option monad functions
    module DslOption =
        /// Return a value wrapped in Some
        let return' x = Some x

        /// Bind operation for Option monad
        let bind f opt =
            match opt with
            | Some x -> f x
            | None -> None

        /// Map operation for Option monad
        let map f opt =
            match opt with
            | Some x -> Some (f x)
            | None -> None

        /// Apply a function wrapped in an option to a value wrapped in an option
        let apply fOpt xOpt =
            match fOpt, xOpt with
            | Some f, Some x -> Some (f x)
            | _ -> None

        /// Convert a nullable value to an option
        let ofNullable (x: 'T when 'T : null) =
            if isNull x then None else Some x

        /// Convert an option to a nullable value
        let toNullable opt =
            match opt with
            | Some x -> x
            | None -> null

        /// Get the value from an option or a default value if None
        let defaultValue defaultValue opt =
            match opt with
            | Some x -> x
            | None -> defaultValue

        /// Get the value from an option or compute a default value if None
        let defaultWith f opt =
            match opt with
            | Some x -> x
            | None -> f()

    /// Result monad for handling success/error cases
    type Result<'T, 'E> =
        | Ok of 'T
        | Error of 'E

    /// Result monad functions
    module Result =
        /// Return a value wrapped in Ok
        let return' x = Ok x

        /// Bind operation for Result monad
        let bind f result =
            match result with
            | Ok x -> f x
            | Error e -> Error e

        /// Map operation for Result monad
        let map f result =
            match result with
            | Ok x -> Ok (f x)
            | Error e -> Error e

        /// Map the error value
        let mapError f result =
            match result with
            | Ok x -> Ok x
            | Error e -> Error (f e)

        /// Apply a function wrapped in a result to a value wrapped in a result
        let apply fResult xResult =
            match fResult, xResult with
            | Ok f, Ok x -> Ok (f x)
            | Error e, _ -> Error e
            | _, Error e -> Error e

        /// Convert a result to an option (discarding the error)
        let toOption result =
            match result with
            | Ok x -> Some x
            | Error _ -> None

        /// Get the value from a result or a default value if Error
        let defaultValue defaultValue result =
            match result with
            | Ok x -> x
            | Error _ -> defaultValue

        /// Get the value from a result or compute a default value if Error
        let defaultWith f result =
            match result with
            | Ok x -> x
            | Error e -> f e

    /// State monad for handling stateful computations
    type State<'S, 'T> = State of ('S -> 'T * 'S)

    /// State monad functions
    module State =
        /// Run a state computation with an initial state
        let run (State f) initialState = f initialState

        /// Return a value in the state monad
        let return' x = State (fun s -> (x, s))

        /// Bind operation for State monad
        let bind f (State m) =
            State (fun s ->
                let (x, s') = m s
                let (State m') = f x
                m' s')

        /// Map operation for State monad
        let map f (State m) =
            State (fun s ->
                let (x, s') = m s
                (f x, s'))

        /// Get the current state
        let get = State (fun s -> (s, s))

        /// Set the state to a new value
        let put newState = State (fun _ -> ((), newState))

        /// Modify the state with a function
        let modify f = State (fun s -> ((), f s))

    /// Reader monad for handling environment/configuration
    type Reader<'E, 'T> = Reader of ('E -> 'T)

    /// Reader monad functions
    module Reader =
        /// Run a reader computation with an environment
        let run (Reader f) env = f env

        /// Return a value in the reader monad
        let return' x = Reader (fun _ -> x)

        /// Bind operation for Reader monad
        let bind f (Reader m) =
            Reader (fun env ->
                let x = m env
                let (Reader m') = f x
                m' env)

        /// Map operation for Reader monad
        let map f (Reader m) =
            Reader (fun env -> f (m env))

        /// Get the environment
        let ask = Reader id

        /// Run a reader with a modified environment
        let local f (Reader m) = Reader (fun env -> m (f env))

    /// Writer monad for handling logging/output
    type Writer<'W, 'T> = Writer of 'T * 'W list

    /// Writer monad functions
    module Writer =
        /// Run a writer computation
        let run (Writer (x, w)) = (x, w)

        /// Return a value in the writer monad
        let return' x = Writer (x, [])

        /// Bind operation for Writer monad
        let bind f (Writer (x, w)) =
            let (Writer (y, w')) = f x
            Writer (y, w @ w')

        /// Map operation for Writer monad
        let map f (Writer (x, w)) =
            Writer (f x, w)

        /// Write a value to the log
        let tell w = Writer ((), [w])

        /// Get the current log
        let listen (Writer (x, w)) = Writer ((x, w), w)

        /// Apply a function to the log
        let pass (Writer ((x, f), w)) = Writer (x, f w)

    /// Computation expression builder for Option monad
    type DslOptionBuilder() =
        member _.Return(x) = DslOption.return' x
        member _.Bind(m, f) = DslOption.bind f m
        member _.Zero() = None
        member _.ReturnFrom(m) = m

    /// Computation expression builder for Result monad
    type ResultBuilder() =
        member _.Return(x) = Result.return' x
        member _.Bind(m, f) = Result.bind f m
        member _.Zero() = Error "Empty result"
        member _.ReturnFrom(m) = m

    /// Computation expression builder for State monad
    type StateBuilder() =
        member _.Return(x) = State.return' x
        member _.Bind(m, f) = State.bind f m
        member _.Zero() = State (fun s -> ((), s))
        member _.ReturnFrom(m) = m

    /// Computation expression builder for Reader monad
    type ReaderBuilder() =
        member _.Return(x) = Reader.return' x
        member _.Bind(m, f) = Reader.bind f m
        member _.Zero() = Reader (fun _ -> ())
        member _.ReturnFrom(m) = m

    /// Computation expression builder for Writer monad
    type WriterBuilder() =
        member _.Return(x) = Writer.return' x
        member _.Bind(m, f) = Writer.bind f m
        member _.Zero() = Writer ((), [])
        member _.ReturnFrom(m) = m

    /// Computation expression instances
    let dsloption = DslOptionBuilder()
    let result = ResultBuilder()
    let state = StateBuilder()
    let reader = ReaderBuilder()
    let writer = WriterBuilder()
