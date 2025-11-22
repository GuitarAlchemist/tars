// FLUX Compiled Code
// Generated: 2025-06-23 12:54:59 PM

open System
    type Result<'T, 'E> = Ok of 'T | Error of 'E
    let bind f = function | Ok v -> f v | Error e -> Error e
    let map f = function | Ok v -> Ok (f v) | Error e -> Error e