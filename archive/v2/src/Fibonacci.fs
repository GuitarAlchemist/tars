module Fibonacci

/// Computes the n‑th Fibonacci number using a tail‑recursive algorithm.
/// n must be non‑negative; otherwise an ArgumentException is raised.
let fib n =
    if n < 0 then invalidArg "n" "n must be non‑negative"
    let rec loop i a b =
        if i = n then a
        else loop (i + 1) b (a + b)
    loop 0 0 1

/// Computes the n‑th Lucas number using the same tail‑recursive pattern.
/// n must be non‑negative; otherwise an ArgumentException is raised.
let lucas n =
    if n < 0 then invalidArg "n" "n must be non‑negative"
    let rec loop i a b =
        if i = n then a
        else loop (i + 1) b (a + b)
    loop 0 2 1
