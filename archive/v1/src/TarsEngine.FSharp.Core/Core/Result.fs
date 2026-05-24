namespace TarsEngine.FSharp.Core.Core

/// Result type for error handling
type Result<'T, 'E> =
    | Ok of 'T
    | Error of 'E

module Result =
    
    let map f = function
        | Ok x -> Ok (f x)
        | Error e -> Error e
    
    let bind f = function
        | Ok x -> f x
        | Error e -> Error e
    
    let isOk = function
        | Ok _ -> true
        | Error _ -> false
    
    let isError = function
        | Ok _ -> false
        | Error _ -> true
