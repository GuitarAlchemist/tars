let rec factorial n =
    if n < 0 then
        failwith "Factorial is not defined for negative numbers"
    elif n = 0 then
        1
    else
        n * factorial (n - 1)

printfn "%d" (factorial 5)  // Expected output: 120
printfn "%d" (factorial 0)  // Expected output: 1