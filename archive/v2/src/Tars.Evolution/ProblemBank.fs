namespace Tars.Evolution

/// Curated F# coding challenges for benchmark-driven evolution.
/// Problems are organized by difficulty and category with deterministic validation.
module ProblemBank =

    let private mkProblem id title desc diff cat sig_ hints timeout validation =
        { Id = id; Title = title; Description = desc; Difficulty = diff
          Category = cat; ExpectedSignature = sig_; Hints = hints
          TimeLimitSeconds = timeout; ValidationCode = validation }

    // =========================================================================
    // Basic
    // =========================================================================

    let private basicProblems =
        [ mkProblem "basic-reverse-string" "Reverse a String"
            "Write a function that reverses a string without using the built-in Reverse method."
            Beginner StringManipulation "let reverse (s: string) : string"
            ["You can convert to char array and work with indices"] 30
            "let mutable passed = true\nlet check inp exp =\n    let got = reverse inp\n    if got <> exp then printfn \"FAIL: reverse '%s' = '%s', expected '%s'\" inp got exp; passed <- false\ncheck \"hello\" \"olleh\"\ncheck \"\" \"\"\ncheck \"a\" \"a\"\ncheck \"abcdef\" \"fedcba\"\nif passed then printfn \"PASS\"\n"

          mkProblem "basic-fizzbuzz" "FizzBuzz"
            "Write a function that returns a list of strings for numbers 1 to n. For multiples of 3 return Fizz, for multiples of 5 return Buzz, for multiples of both return FizzBuzz, otherwise return the number as a string."
            Beginner Algorithms "let fizzBuzz (n: int) : string list"
            ["Use List.map with pattern matching on modulo"] 30
            "let result = fizzBuzz 15\nlet mutable passed = true\nlet check idx exp =\n    if result.[idx] <> exp then printfn \"FAIL: fizzBuzz(15).[%d] = '%s', expected '%s'\" idx result.[idx] exp; passed <- false\nif result.Length <> 15 then printfn \"FAIL: expected 15 elements, got %d\" result.Length; passed <- false\nelse\n    check 0 \"1\"\n    check 1 \"2\"\n    check 2 \"Fizz\"\n    check 4 \"Buzz\"\n    check 14 \"FizzBuzz\"\nif passed then printfn \"PASS\"\n"

          mkProblem "basic-count-vowels" "Count Vowels"
            "Write a function that counts the number of vowels (a, e, i, o, u, case-insensitive) in a string."
            Beginner StringManipulation "let countVowels (s: string) : int"
            ["Use Seq.filter or fold over characters"] 30
            "let mutable passed = true\nlet check inp exp =\n    let got = countVowels inp\n    if got <> exp then printfn \"FAIL: countVowels '%s' = %d, expected %d\" inp got exp; passed <- false\ncheck \"hello\" 2\ncheck \"AEIOU\" 5\ncheck \"rhythm\" 0\ncheck \"\" 0\nif passed then printfn \"PASS\"\n"

          mkProblem "basic-palindrome" "Palindrome Check"
            "Write a function that checks if a string is a palindrome (reads the same forwards and backwards, case-insensitive, ignoring spaces)."
            Beginner StringManipulation "let isPalindrome (s: string) : bool"
            ["Normalize to lowercase and remove spaces first"] 30
            "let mutable passed = true\nlet check inp exp =\n    let got = isPalindrome inp\n    if got <> exp then printfn \"FAIL: isPalindrome '%s' = %b, expected %b\" inp got exp; passed <- false\ncheck \"racecar\" true\ncheck \"hello\" false\ncheck \"\" true\ncheck \"ab\" false\nif passed then printfn \"PASS\"\n"

          mkProblem "basic-max-element" "Find Maximum Element"
            "Write a function that finds the maximum element in a non-empty integer list without using List.max."
            Beginner Algorithms "let findMax (xs: int list) : int"
            ["Use List.fold or recursion"] 30
            "let mutable passed = true\nlet check inp exp =\n    let got = findMax inp\n    if got <> exp then printfn \"FAIL: findMax %A = %d, expected %d\" inp got exp; passed <- false\ncheck [1; 3; 2] 3\ncheck [5] 5\ncheck [-1; -5; -2] -1\ncheck [10; 20; 30; 5] 30\nif passed then printfn \"PASS\"\n" ]

    // =========================================================================
    // Intermediate
    // =========================================================================

    let private intermediateProblems =
        [ mkProblem "inter-binary-search" "Binary Search"
            "Implement binary search on a sorted integer array. Return Some index if found, None if not."
            Intermediate Algorithms "let binarySearch (arr: int[]) (target: int) : int option"
            ["Use a recursive helper with low/high bounds"] 45
            "let mutable passed = true\nlet check arr target exp =\n    let got = binarySearch arr target\n    if got <> exp then printfn \"FAIL: binarySearch %A %d = %A, expected %A\" arr target got exp; passed <- false\ncheck [|1;3;5;7;9|] 5 (Some 2)\ncheck [|1;3;5;7;9|] 4 None\ncheck [|1;3;5;7;9|] 1 (Some 0)\ncheck [|1;3;5;7;9|] 9 (Some 4)\ncheck [||] 1 None\nif passed then printfn \"PASS\"\n"

          mkProblem "inter-flatten" "Flatten Nested Lists"
            "Write a function that flattens a list of lists into a single list."
            Intermediate DataStructures "let flatten (xss: 'a list list) : 'a list"
            ["Use List.collect or List.concat"] 30
            "let mutable passed = true\nlet got1 = flatten [[1;2];[3;4];[5]]\nif got1 <> [1;2;3;4;5] then printfn \"FAIL: flatten [[1;2];[3;4];[5]] = %A\" got1; passed <- false\nlet got2 : int list = flatten []\nif got2 <> [] then printfn \"FAIL: flatten [] not empty\"; passed <- false\nlet got3 = flatten [[1];[];[2;3]]\nif got3 <> [1;2;3] then printfn \"FAIL: flatten [[1];[];[2;3]] = %A\" got3; passed <- false\nif passed then printfn \"PASS\"\n"

          mkProblem "inter-safe-divide" "Safe Division with Result"
            "Write a function that divides two floats, returning Ok result or Error message for division by zero."
            Intermediate ErrorHandling "let safeDivide (a: float) (b: float) : Result<float, string>"
            ["Check if b is zero, return Error with descriptive message"] 30
            "let mutable passed = true\nmatch safeDivide 10.0 2.0 with | Ok v when abs(v - 5.0) < 0.001 -> () | other -> printfn \"FAIL: safeDivide 10.0 2.0 = %A\" other; passed <- false\nmatch safeDivide 1.0 0.0 with | Error _ -> () | other -> printfn \"FAIL: safeDivide 1.0 0.0 should be Error\" ; passed <- false\nmatch safeDivide 0.0 5.0 with | Ok v when abs(v) < 0.001 -> () | other -> printfn \"FAIL: safeDivide 0.0 5.0 = %A\" other; passed <- false\nif passed then printfn \"PASS\"\n"

          mkProblem "inter-group-by" "Group By Key"
            "Write a function that groups a list of (key, value) tuples into a Map where each key maps to a list of values."
            Intermediate DataStructures "let groupByKey (items: (string * int) list) : Map<string, int list>"
            ["Use List.groupBy or fold with Map"] 30
            "let mutable passed = true\nlet result = groupByKey [(\"a\", 1); (\"b\", 2); (\"a\", 3); (\"b\", 4); (\"c\", 5)]\nif result.Count <> 3 then printfn \"FAIL: expected 3 groups, got %d\" result.Count; passed <- false\nlet empty = groupByKey []\nif empty.Count <> 0 then printfn \"FAIL: empty input should give empty map\"; passed <- false\nif passed then printfn \"PASS\"\n"

          mkProblem "inter-roman-numerals" "Roman Numeral Converter"
            "Write a function that converts an integer (1-3999) to its Roman numeral representation."
            Intermediate Algorithms "let toRoman (n: int) : string"
            ["Use a lookup table of (value, numeral) pairs and subtract greedily"] 45
            "let mutable passed = true\nlet check n exp =\n    let got = toRoman n\n    if got <> exp then printfn \"FAIL: toRoman %d = '%s', expected '%s'\" n got exp; passed <- false\ncheck 1 \"I\"\ncheck 4 \"IV\"\ncheck 9 \"IX\"\ncheck 14 \"XIV\"\ncheck 42 \"XLII\"\ncheck 1994 \"MCMXCIV\"\ncheck 3999 \"MMMCMXCIX\"\nif passed then printfn \"PASS\"\n" ]

    // =========================================================================
    // Advanced
    // =========================================================================

    let private advancedProblems =
        [ mkProblem "adv-merge-sort" "Merge Sort"
            "Implement merge sort for an integer list. The function should return a new sorted list."
            Advanced Algorithms "let mergeSort (xs: int list) : int list"
            ["Split list in half, recursively sort each half, merge the two sorted halves"] 60
            "let mutable passed = true\nlet check inp exp =\n    let got = mergeSort inp\n    if got <> exp then printfn \"FAIL: mergeSort %A = %A, expected %A\" inp got exp; passed <- false\ncheck [5;3;1;4;2] [1;2;3;4;5]\ncheck [] []\ncheck [1] [1]\ncheck [3;1;2;3;1] [1;1;2;3;3]\ncheck [10;-1;5;0] [-1;0;5;10]\nif passed then printfn \"PASS\"\n"

          mkProblem "adv-balanced-parens" "Balanced Parentheses"
            "Write a function that checks if a string has balanced parentheses, brackets, and braces. Supports (), [], {}."
            Advanced PatternMatching "let isBalanced (s: string) : bool"
            ["Use a stack (list). Push on open, pop and check match on close."] 45
            "let mutable passed = true\nlet check inp exp =\n    let got = isBalanced inp\n    if got <> exp then printfn \"FAIL: isBalanced '%s' = %b, expected %b\" inp got exp; passed <- false\ncheck \"()\" true\ncheck \"()[]{}\" true\ncheck \"(]\" false\ncheck \"([{}])\" true\ncheck \"(((\" false\ncheck \"\" true\ncheck \"({)}\" false\nif passed then printfn \"PASS\"\n"

          mkProblem "adv-eval-rpn" "Evaluate RPN Expression"
            "Write a function that evaluates a Reverse Polish Notation expression. Tokens are integers or operators (+, -, *, /)."
            Advanced Algorithms "let evalRPN (tokens: string list) : int"
            ["Use a stack. Push numbers, pop two operands for operators."] 45
            "let mutable passed = true\nlet check tokens exp =\n    let got = evalRPN tokens\n    if got <> exp then printfn \"FAIL: evalRPN %A = %d, expected %d\" tokens got exp; passed <- false\ncheck [\"2\"; \"3\"; \"+\"] 5\ncheck [\"4\"; \"13\"; \"5\"; \"/\"; \"+\"] 6\ncheck [\"2\"; \"1\"; \"+\"; \"3\"; \"*\"] 9\nif passed then printfn \"PASS\"\n"

          mkProblem "adv-matrix-multiply" "Matrix Multiplication"
            "Implement matrix multiplication for two 2D float arrays. Return the product matrix."
            Advanced Algorithms "let matMul (a: float[,]) (b: float[,]) : float[,]"
            ["Result[i,j] = sum of a[i,k] * b[k,j] for all k"] 60
            "let mutable passed = true\nlet a = array2D [[1.0; 2.0]; [3.0; 4.0]]\nlet b = array2D [[5.0; 6.0]; [7.0; 8.0]]\nlet c = matMul a b\nif abs(c.[0,0] - 19.0) > 0.001 || abs(c.[0,1] - 22.0) > 0.001 then printfn \"FAIL: row 0\"; passed <- false\nif abs(c.[1,0] - 43.0) > 0.001 || abs(c.[1,1] - 50.0) > 0.001 then printfn \"FAIL: row 1\"; passed <- false\nif passed then printfn \"PASS\"\n" ]

    // =========================================================================
    // Expert
    // =========================================================================

    let private expertProblems =
        [ mkProblem "exp-result-ce" "Result Computation Expression"
            "Implement a computation expression builder for Result<'T, string> that supports let!, return, and return!. Bind should short-circuit on Error."
            Expert TypeDesign "type ResultBuilder() = ...\nlet result = ResultBuilder()"
            ["Implement Bind, Return, ReturnFrom members on the builder class"] 90
            "let mutable passed = true\nlet divide a b = if b = 0 then Error \"division by zero\" else Ok (a / b)\nlet test1 = result { let! x = divide 10 2 in let! y = divide x 1 in return y + 1 }\nif test1 <> Ok 6 then printfn \"FAIL: test1 = %A, expected Ok 6\" test1; passed <- false\nlet test2 = result { let! x = divide 10 0 in return x + 1 }\nmatch test2 with | Error _ -> () | other -> printfn \"FAIL: test2 = %A, expected Error\" other; passed <- false\nif passed then printfn \"PASS\"\n"

          mkProblem "exp-active-pattern" "Active Pattern for Parsing"
            "Create an active pattern that parses a string as either an Int, a Float, or a Word. Use it in a classify function returning 'int', 'float', or 'word'."
            Expert PatternMatching "let (|Int|Float|Word|) (s: string) = ...\nlet classify (s: string) : string"
            ["Try Int32.TryParse, then Double.TryParse, else Word"] 60
            "let mutable passed = true\nlet check inp exp =\n    let got = classify inp\n    if got <> exp then printfn \"FAIL: classify '%s' = '%s', expected '%s'\" inp got exp; passed <- false\ncheck \"42\" \"int\"\ncheck \"3.14\" \"float\"\ncheck \"hello\" \"word\"\ncheck \"-7\" \"int\"\nif passed then printfn \"PASS\"\n"

          mkProblem "exp-graph-cycle" "Graph Cycle Detection"
            "Given an adjacency list representing a directed graph, detect if the graph contains a cycle."
            Expert Algorithms "let hasCycle (graph: Map<int, int list>) : bool"
            ["Use DFS with a visiting set (gray nodes) and visited set (black nodes)"] 90
            "let mutable passed = true\nlet check desc graph exp =\n    let got = hasCycle graph\n    if got <> exp then printfn \"FAIL: hasCycle (%s) = %b, expected %b\" desc got exp; passed <- false\ncheck \"simple cycle\" (Map.ofList [(1, [2]); (2, [3]); (3, [1])]) true\ncheck \"no cycle\" (Map.ofList [(1, [2]); (2, [3]); (3, [])]) false\ncheck \"self loop\" (Map.ofList [(1, [1])]) true\ncheck \"empty\" Map.empty false\nif passed then printfn \"PASS\"\n"

          mkProblem "exp-memoize" "Generic Memoization"
            "Write a generic memoize function that caches results. Use it to memoize a recursive Fibonacci function."
            Expert TypeDesign "let memoize (f: 'a -> 'b) : ('a -> 'b)\nlet fib : int -> int64"
            ["Use a Dictionary for the cache. For recursive memoization, use a mutable ref."] 60
            "let mutable passed = true\nif fib 0 <> 0L then printfn \"FAIL: fib 0\"; passed <- false\nif fib 1 <> 1L then printfn \"FAIL: fib 1\"; passed <- false\nif fib 10 <> 55L then printfn \"FAIL: fib 10 = %d\" (fib 10); passed <- false\nif fib 30 <> 832040L then printfn \"FAIL: fib 30\"; passed <- false\nlet sw = System.Diagnostics.Stopwatch.StartNew()\nlet r = fib 40\nsw.Stop()\nif r <> 102334155L then printfn \"FAIL: fib 40 = %d\" r; passed <- false\nif sw.ElapsedMilliseconds > 1000L then printfn \"FAIL: fib 40 took %dms\" sw.ElapsedMilliseconds; passed <- false\nif passed then printfn \"PASS\"\n" ]

    // =========================================================================
    // Public API
    // =========================================================================

    /// All curated benchmark problems.
    let all () : BenchmarkProblem list =
        basicProblems @ intermediateProblems @ advancedProblems @ expertProblems

    /// Filter by difficulty.
    let byDifficulty (d: ProblemDifficulty) : BenchmarkProblem list =
        all () |> List.filter (fun p -> p.Difficulty = d)

    /// Filter by category.
    let byCategory (c: ProblemCategory) : BenchmarkProblem list =
        all () |> List.filter (fun p -> p.Category = c)

    /// Find a problem by ID.
    let tryFind (id: string) : BenchmarkProblem option =
        all () |> List.tryFind (fun p -> p.Id = id)

    /// Problem count summary.
    let summary () =
        let problems = all ()
        {| Total = problems.Length
           Basic = problems |> List.filter (fun p -> p.Difficulty = Beginner) |> List.length
           Intermediate = problems |> List.filter (fun p -> p.Difficulty = Intermediate) |> List.length
           Advanced = problems |> List.filter (fun p -> p.Difficulty = Advanced) |> List.length
           Expert = problems |> List.filter (fun p -> p.Difficulty = Expert) |> List.length |}
