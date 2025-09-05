
// Original problematic code
let processData input =
    if input <> null then
        if input.Length > 0 then
            let result = []
            for i in 0..input.Length-1 do
                if input.[i] <> "" then
                    if input.[i].Contains("error") then
                        printfn "Error found"
                    else
                        if input.[i].Length > 10 then
                            printfn "Long string: %s" input.[i]
                        else
                            printfn "Short string: %s" input.[i]
            result
        else
            []
    else
        []
