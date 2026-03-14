let rec fibonacci n =
    let rec helper a b count =
        match count with
        | 0 -> a
        | _ when count < 0 -> -1 // Handle negative inputs
        | _ -> helper b (a + b) (count - 1)
    if n < 0 then -1 else helper 0 1 n