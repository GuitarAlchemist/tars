/// <summary>
/// Core algebraic structures for TARS functional programming.
/// Simplified version compatible with F#'s type system.
/// </summary>
/// <remarks>
/// Reference: docs/3_Roadmap/functional_patterns_proposal.md
/// </remarks>
namespace Tars.Core.Functional

/// <summary>
/// Semigroup: A type with an associative binary operation.
/// Laws:
/// - Associativity: combine (combine a b) c = combine a (combine b c)
/// </summary>
type Semigroup<'T> = { Combine: 'T -> 'T -> 'T }

/// <summary>
/// Monoid: A semigroup with an identity element.
/// Laws:
/// - Left identity: combine empty a = a
/// - Right identity: combine a empty = a
/// - Associativity: inherited from Semigroup
/// </summary>
type Monoid<'T> = { Combine: 'T -> 'T -> 'T; Empty: 'T }

/// <summary>
/// Standard semigroup instances
/// </summary>
module Semigroup =
    /// List concatenation semigroup
    let list<'T> : Semigroup<'T list> = { Combine = (@) }

    /// String concatenation semigroup
    let string: Semigroup<string> = { Combine = (+) }

    /// Map merge semigroup (right-biased for conflicts)
    let map<'K, 'V when 'K: comparison> : Semigroup<Map<'K, 'V>> =
        { Combine = fun m1 m2 -> Map.fold (fun acc k v -> Map.add k v acc) m1 m2 }

    /// First semigroup (take first non-None)
    let first<'T> : Semigroup<'T option> =
        { Combine =
            fun a b ->
                match a with
                | Some _ -> a
                | None -> b }

    /// Last semigroup (take last non-None)
    let last<'T> : Semigroup<'T option> =
        { Combine =
            fun a b ->
                match b with
                | Some _ -> b
                | None -> a }

/// <summary>
/// Standard monoid instances
/// </summary>
module Monoid =
    /// List concatenation monoid
    let list<'T> : Monoid<'T list> = { Combine = (@); Empty = [] }

    /// String concatenation monoid
    let string: Monoid<string> = { Combine = (+); Empty = "" }

    /// Map merge monoid
    let map<'K, 'V when 'K: comparison> : Monoid<Map<'K, 'V>> =
        { Combine = fun m1 m2 -> Map.fold (fun acc k v -> Map.add k v acc) m1 m2
          Empty = Map.empty }

    /// Unit monoid (trivial)
    let unit: Monoid<unit> =
        { Combine =
            fun () () ->
                ()
                Empty = () }

    /// Combine a sequence of values using a monoid
    let concat (monoid: Monoid<'T>) (xs: 'T seq) : 'T = Seq.fold monoid.Combine monoid.Empty xs

    /// Combine a list of values using a monoid
    let combineMany (monoid: Monoid<'T>) (xs: 'T list) : 'T =
        List.fold monoid.Combine monoid.Empty xs

/// <summary>
/// Extended option operations
/// </summary>
module OptionOps =
    /// Sequence a list of options into an option of list (all or nothing)
    let sequence (opts: 'T option list) : 'T list option =
        let folder opt acc =
            match opt, acc with
            | Some x, Some xs -> Some(x :: xs)
            | _ -> None

        List.foldBack folder opts (Some [])

    /// Traverse: map then sequence
    let traverse (f: 'A -> 'B option) (xs: 'A list) : 'B list option = xs |> List.map f |> sequence

/// <summary>
/// Extended result operations
/// </summary>
module ResultOps =
    /// Sequence a list of results into a result of list (fail-fast)
    let sequence (results: Result<'T, 'E> list) : Result<'T list, 'E> =
        let folder res acc =
            match res, acc with
            | Ok x, Ok xs -> Ok(x :: xs)
            | Error e, _ -> Error e
            | _, Error e -> Error e

        List.foldBack folder results (Ok [])

    /// Traverse: map then sequence
    let traverse (f: 'A -> Result<'B, 'E>) (xs: 'A list) : Result<'B list, 'E> = xs |> List.map f |> sequence

/// <summary>
/// Extended async operations
/// </summary>
module AsyncOps =
    /// Sequence a list of async operations
    let sequence (asyncs: Async<'T> list) : Async<'T list> =
        async {
            let! results = Async.Sequential(Array.ofList asyncs)
            return Array.toList results
        }

    /// Traverse: map then sequence
    let traverse (f: 'A -> Async<'B>) (xs: 'A list) : Async<'B list> = xs |> List.map f |> sequence

    /// Execute actions and discard results
    let sequence_ (asyncs: Async<'T> list) : Async<unit> =
        async {
            for a in asyncs do
                let! _ = a
                ()
        }

/// <summary>
/// Kleisli composition for Result monad
/// </summary>
module Kleisli =
    /// Compose two Result-returning functions
    let compose (f: 'A -> Result<'B, 'E>) (g: 'B -> Result<'C, 'E>) : 'A -> Result<'C, 'E> =
        fun a ->
            match f a with
            | Ok b -> g b
            | Error e -> Error e

    /// Kleisli composition operator (>=>)
    let (>=>) f g = compose f g

/// <summary>
/// Operators for working with functors
/// </summary>
module Operators =
    /// Functor map operator for Option
    let (<!>) f opt = Option.map f opt

    /// Functor map operator for Result
    let (<!^) f res = Result.map f res

    /// Applicative apply for Option
    let (<*>) fopt opt =
        match fopt, opt with
        | Some f, Some x -> Some(f x)
        | _ -> None

    /// Applicative apply for Result
    let (<*^) fres res =
        match fres, res with
        | Ok f, Ok x -> Ok(f x)
        | Error e, _ -> Error e
        | _, Error e -> Error e

    /// Bind operator for Option
    let (>>=) opt f = Option.bind f opt

    /// Bind operator for Result
    let (>>=^) res f = Result.bind f res
