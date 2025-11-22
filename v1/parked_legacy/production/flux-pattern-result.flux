PATTERN result_handling {
    type Result<'T, 'E> = Ok of 'T | Error of 'E
    let bind f = function | Ok v -> f v | Error e -> Error e
    let map f = function | Ok v -> Ok (f v) | Error e -> Error e
}