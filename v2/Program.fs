let countCharOccurrences (input: string) (charToCount: char) =
    try
        let lowerInput = input.ToLower()
        let lowerCharToCount = charToCount.ToString().ToLower()

        if String.IsNullOrEmpty(lowerInput) || String.IsNullOrEmpty(lowerCharToCount) then
            logger.LogError("Invalid input or character.")
            failwithf "Invalid input or character."}

let main args =
    let input = "Hello World"
    let charToCount = 'o'
    let result = countCharOccurrences input charToCount
    printfn "Character '%c' found %d times in the string." charToCount result