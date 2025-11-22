DESCRIBE {
    name: "Enhanced FLUX v2.0"
    author: "TARS"
}

CONFIG {
    target: "fsharp"
    optimization: "high"
}

PATTERN result_type {
    type Result<'T, 'E> = Ok of 'T | Error of 'E
    let bind f = function | Ok v -> f v | Error e -> Error e
}

FSHARP {
    open System
    type Result<'T, 'E> = Ok of 'T | Error of 'E
    let bind f = function | Ok v -> f v | Error e -> Error e
    let map f = function | Ok v -> Ok (f v) | Error e -> Error e
}