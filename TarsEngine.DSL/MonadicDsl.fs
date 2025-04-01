namespace TarsEngine.DSL

/// Module containing monadic extensions for the TARS DSL
module MonadicDsl =
    open System
    open System.Collections.Generic
    open TarsEngine.DSL.SimpleDsl
    open TarsEngine.DSL.Monads

    /// Result type for DSL execution
    type DslResult<'T> = Result<'T, string>

    /// Environment type for DSL execution
    type DslEnvironment = Dictionary<string, SimpleDsl.PropertyValue>

    /// State type for DSL execution
    type DslState = {
        Environment: DslEnvironment
        LastResult: SimpleDsl.PropertyValue
    }

    /// Computation type for DSL execution
    type DslComputation<'T> = State<DslState, DslResult<'T>>

    /// DSL monad functions
    module Dsl =
        /// Return a value in the DSL monad
        let return' x : DslComputation<'T> =
            State (fun s -> (Ok x, s))

        /// Return an error in the DSL monad
        let error msg : DslComputation<'T> =
            State (fun s -> (Error msg, s))

        /// Bind operation for DSL monad
        let bind (f: 'T -> DslComputation<'U>) (m: DslComputation<'T>) : DslComputation<'U> =
            State (fun s ->
                let (result, s') = State.run m s
                match result with
                | Ok x -> State.run (f x) s'
                | Error msg -> (Error msg, s'))

        /// Map operation for DSL monad
        let map (f: 'T -> 'U) (m: DslComputation<'T>) : DslComputation<'U> =
            State (fun s ->
                let (result, s') = State.run m s
                match result with
                | Ok x -> (Ok (f x), s')
                | Error msg -> (Error msg, s'))

        /// Get the current environment
        let getEnvironment : DslComputation<DslEnvironment> =
            State (fun s -> (Ok s.Environment, s))

        /// Set the environment
        let setEnvironment (env: DslEnvironment) : DslComputation<unit> =
            State (fun s -> (Ok (), { s with Environment = env }))

        /// Get a variable from the environment
        let getVariable (name: string) : DslComputation<SimpleDsl.PropertyValue> =
            State (fun s ->
                match s.Environment.TryGetValue(name) with
                | true, value -> (Ok value, s)
                | false, _ -> (Error $"Variable '{name}' not found", s))

        /// Set a variable in the environment
        let setVariable (name: string) (value: SimpleDsl.PropertyValue) : DslComputation<unit> =
            State (fun s ->
                s.Environment.[name] <- value
                (Ok (), s))

        /// Get the last result
        let getLastResult : DslComputation<SimpleDsl.PropertyValue> =
            State (fun s -> (Ok s.LastResult, s))

        /// Set the last result
        let setLastResult (value: SimpleDsl.PropertyValue) : DslComputation<unit> =
            State (fun s -> (Ok (), { s with LastResult = value }))

        /// Execute a block in the DSL monad
        let executeBlock (block: SimpleDsl.Block) : DslComputation<SimpleDsl.PropertyValue> =
            State (fun s ->
                match SimpleDsl.executeBlock block s.Environment with
                | SimpleDsl.Success value -> (Ok value, { s with LastResult = value })
                | SimpleDsl.Error msg -> (Error msg, s))

        /// Execute a program in the DSL monad
        let executeProgram (program: SimpleDsl.Program) : DslComputation<SimpleDsl.PropertyValue> =
            State (fun s ->
                match SimpleDsl.executeProgram program with
                | SimpleDsl.Success value -> (Ok value, { s with LastResult = value })
                | SimpleDsl.Error msg -> (Error msg, s))

        /// Run a DSL computation with an initial environment
        let run (comp: DslComputation<'T>) (env: DslEnvironment) =
            let initialState = { Environment = env; LastResult = SimpleDsl.StringValue("") }
            State.run comp initialState

    /// Computation expression builder for DSL monad
    type DslBuilder() =
        member _.Return(x) = Dsl.return' x
        member _.Bind(m, f) = Dsl.bind f m
        member _.Zero() = Dsl.return' ()
        member _.ReturnFrom(m) = m

        /// Let binding in computation expression
        member _.Let(x, f) = f x

        /// Sequential execution in computation expression
        member _.Combine(m1, m2) = Dsl.bind (fun _ -> m2) m1

        /// For loop in computation expression
        member _.For(xs: seq<'T>, f: 'T -> DslComputation<unit>) =
            let rec loop (xs: 'T list) =
                match xs with
                | [] -> Dsl.return' ()
                | x::xs -> Dsl.bind (fun _ -> loop xs) (f x)
            loop (Seq.toList xs)

        /// While loop in computation expression
        member _.While(guard, body) =
            let rec loop () =
                if guard() then
                    Dsl.bind (fun _ -> loop()) body
                else
                    Dsl.return' ()
            loop()

        /// Try-with in computation expression
        member _.TryWith(body, handler) =
            State (fun s ->
                try
                    State.run body s
                with
                | ex -> State.run (handler ex) s)

        /// Try-finally in computation expression
        member _.TryFinally(body, compensation) =
            State (fun s ->
                try
                    State.run body s
                finally
                    compensation())

        /// Using in computation expression
        member _.Using(resource: #IDisposable, body) =
            State (fun s ->
                try
                    State.run (body resource) s
                finally
                    if not (isNull (box resource)) then
                        resource.Dispose())

        /// Delay in computation expression
        member _.Delay(f) = State (fun s -> State.run (f()) s)

    /// Computation expression instance
    let dsl = DslBuilder()

    /// Example usage of the DSL monad
    let exampleDslComputation =
        dsl {
            // Get the environment
            let! env = Dsl.getEnvironment

            // Set a variable
            do! Dsl.setVariable "x" (SimpleDsl.NumberValue 42.0)

            // Get a variable
            let! x = Dsl.getVariable "x"

            // Execute a block
            let block = {
                Type = SimpleDsl.BlockType.Action
                Name = Option.None
                Content = ""
                Properties = Map.ofList [
                    "type", SimpleDsl.StringValue "log"
                    "message", SimpleDsl.StringValue "Hello, World!"
                ]
                NestedBlocks = []
            }
            let! result = Dsl.executeBlock block

            // Return the result
            return result
        }
