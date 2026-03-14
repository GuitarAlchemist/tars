let rec sumEvenNumbers lst =
    match lst with
    | [] -> 0
    | x::xs ->
        if x % 2 = 0 then
            x + sumEvenNumbers xs
        else
            sumEvenNumbers xs