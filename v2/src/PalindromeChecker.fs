let rec isPalindrome (s: string) =
    let lowerCaseS = s.ToLower()
    if String.length lowerCaseS <= 1 then
        true
    else
        let firstChar = lowerCaseS.[0]
        let lastChar = lowerCaseS.[String.length lowerCaseS - 1]
        if firstChar = lastChar then
            isPalindrome (lowerCaseS.Substring(1, String.length lowerCaseS - 2))
        else
            false

// Test cases
printfn "%b" (isPalindrome "racecar")    // True
printfn "%b" (isPalindrome "RaceCar")    // True
printfn "%b" (isPalindrome "hello")      // False
printfn "%b" (isPalindrome "a")          // True