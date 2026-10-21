namespace Tars.Evolution

/// Curated F# coding challenges for benchmark-driven evolution.
/// Problems are organized by difficulty and category with deterministic validation.
module ProblemBank =

    let private mkProblem id title desc diff cat sig_ hints timeout validation =
        { Id = id; Title = title; Description = desc; Difficulty = diff
          Category = cat; ExpectedSignature = sig_; Hints = hints
          TimeLimitSeconds = timeout; ValidationCode = validation; Examples = []
          PerfHarness = None; Properties = None }

    /// `problem`, checked by `examples` compared outside the solution's process.
    let private withExamples (examples: string list) (problem: BenchmarkProblem) =
        { problem with Examples = examples }

    // =========================================================================
    // Basic
    // =========================================================================

    let private basicProblems =
        [ withExamples
            [ "reverse \"hello\" = \"olleh\""; "reverse \"\" = \"\""; "reverse \"a\" = \"a\""; "reverse \"abcdef\" = \"fedcba\"" ]
            (mkProblem "basic-reverse-string" "Reverse a String"
                "Write a function that reverses a string without using the built-in Reverse method."
                Beginner StringManipulation "let reverse (s: string) : string"
                ["You can convert to char array and work with indices"] 30
                "")

          withExamples
            [ "(fizzBuzz 15).Length = 15"
              "(fizzBuzz 15).[0] = \"1\""
              "(fizzBuzz 15).[1] = \"2\""
              "(fizzBuzz 15).[2] = \"Fizz\""
              "(fizzBuzz 15).[4] = \"Buzz\""
              "(fizzBuzz 15).[14] = \"FizzBuzz\"" ]
            (mkProblem "basic-fizzbuzz" "FizzBuzz"
                "Write a function that returns a list of strings for numbers 1 to n. For multiples of 3 return Fizz, for multiples of 5 return Buzz, for multiples of both return FizzBuzz, otherwise return the number as a string."
                Beginner Algorithms "let fizzBuzz (n: int) : string list"
                ["Use List.map with pattern matching on modulo"] 30
                "")

          withExamples
            [ "countVowels \"hello\" = 2"
              "countVowels \"AEIOU\" = 5"
              "countVowels \"rhythm\" = 0"
              "countVowels \"\" = 0" ]
            (mkProblem "basic-count-vowels" "Count Vowels"
                "Write a function that counts the number of vowels (a, e, i, o, u, case-insensitive) in a string."
                Beginner StringManipulation "let countVowels (s: string) : int"
                ["Use Seq.filter or fold over characters"] 30
                "")

          withExamples
            [ "isPalindrome \"racecar\" = true"
              "isPalindrome \"hello\" = false"
              "isPalindrome \"\" = true"
              "isPalindrome \"ab\" = false" ]
            (mkProblem "basic-palindrome" "Palindrome Check"
                "Write a function that checks if a string is a palindrome (reads the same forwards and backwards, case-insensitive, ignoring spaces)."
                Beginner StringManipulation "let isPalindrome (s: string) : bool"
                ["Normalize to lowercase and remove spaces first"] 30
                "")

          withExamples
            [ "findMax [1; 3; 2] = 3"
              "findMax [5] = 5"
              "findMax [-1; -5; -2] = -1"
              "findMax [10; 20; 30; 5] = 30" ]
            (mkProblem "basic-max-element" "Find Maximum Element"
                "Write a function that finds the maximum element in a non-empty integer list without using List.max."
                Beginner Algorithms "let findMax (xs: int list) : int"
                ["Use List.fold or recursion"] 30
                "") ]

    // =========================================================================
    // Intermediate
    // =========================================================================

    let private intermediateProblems =
        [ withExamples
            [ "binarySearch [|1; 3; 5; 7; 9|] 5 = Some 2"
              "binarySearch [|1; 3; 5; 7; 9|] 4 = None"
              "binarySearch [|1; 3; 5; 7; 9|] 1 = Some 0"
              "binarySearch [|1; 3; 5; 7; 9|] 9 = Some 4"
              "binarySearch [||] 1 = None" ]
            (mkProblem "inter-binary-search" "Binary Search"
                "Implement binary search on a sorted integer array. Return Some index if found, None if not."
                Intermediate Algorithms "let binarySearch (arr: int[]) (target: int) : int option"
                ["Use a recursive helper with low/high bounds"] 45
                "")

          withExamples
            [ "flatten [[1; 2]; [3; 4]; [5]] = [1; 2; 3; 4; 5]"
              "flatten [] = ([]: int list)"
              "flatten [[1]; []; [2; 3]] = [1; 2; 3]" ]
            (mkProblem "inter-flatten" "Flatten Nested Lists"
                "Write a function that flattens a list of lists into a single list."
                Intermediate DataStructures "let flatten (xss: 'a list list) : 'a list"
                ["Use List.collect or List.concat"] 30
                "")

          withExamples
            [ "safeDivide 10.0 2.0 = Ok 5.0"
              "global.Microsoft.FSharp.Core.Result.isError (safeDivide 1.0 0.0)"
              "safeDivide 0.0 5.0 = Ok 0.0" ]
            (mkProblem "inter-safe-divide" "Safe Division with Result"
                "Write a function that divides two floats, returning Ok result or Error message for division by zero."
                Intermediate ErrorHandling "let safeDivide (a: float) (b: float) : Result<float, string>"
                ["Check if b is zero, return Error with descriptive message"] 30
                "")

          withExamples
            [ "(groupByKey [(\"a\", 1); (\"b\", 2); (\"a\", 3); (\"b\", 4); (\"c\", 5)]).Count = 3"
              "(groupByKey []).Count = 0" ]
            (mkProblem "inter-group-by" "Group By Key"
                "Write a function that groups a list of (key, value) tuples into a Map where each key maps to a list of values."
                Intermediate DataStructures "let groupByKey (items: (string * int) list) : Map<string, int list>"
                ["Use List.groupBy or fold with Map"] 30
                "")

          withExamples
            [ "toRoman 1 = \"I\""
              "toRoman 4 = \"IV\""
              "toRoman 9 = \"IX\""
              "toRoman 14 = \"XIV\""
              "toRoman 42 = \"XLII\""
              "toRoman 1994 = \"MCMXCIV\""
              "toRoman 3999 = \"MMMCMXCIX\"" ]
            (mkProblem "inter-roman-numerals" "Roman Numeral Converter"
                "Write a function that converts an integer (1-3999) to its Roman numeral representation."
                Intermediate Algorithms "let toRoman (n: int) : string"
                ["Use a lookup table of (value, numeral) pairs and subtract greedily"] 45
                "") ]

    // =========================================================================
    // Advanced
    // =========================================================================

    let private advancedProblems =
        [ withExamples
            [ "mergeSort [5; 3; 1; 4; 2] = [1; 2; 3; 4; 5]"
              "mergeSort [] = []"
              "mergeSort [1] = [1]"
              "mergeSort [3; 1; 2; 3; 1] = [1; 1; 2; 3; 3]"
              "mergeSort [10; -1; 5; 0] = [-1; 0; 5; 10]" ]
            (mkProblem "adv-merge-sort" "Merge Sort"
                "Implement merge sort for an integer list. The function should return a new sorted list."
                Advanced Algorithms "let mergeSort (xs: int list) : int list"
                ["Split list in half, recursively sort each half, merge the two sorted halves"] 60
                "")

          withExamples
            [ "isBalanced \"()\" = true"
              "isBalanced \"()[]{}\" = true"
              "isBalanced \"(]\" = false"
              "isBalanced \"([{}])\" = true"
              "isBalanced \"(((\" = false"
              "isBalanced \"\" = true"
              "isBalanced \"({)}\" = false" ]
            (mkProblem "adv-balanced-parens" "Balanced Parentheses"
                "Write a function that checks if a string has balanced parentheses, brackets, and braces. Supports (), [], {}."
                Advanced PatternMatching "let isBalanced (s: string) : bool"
                ["Use a stack (list). Push on open, pop and check match on close."] 45
                "")

          withExamples
            [ "evalRPN [\"2\"; \"3\"; \"+\"] = 5"
              "evalRPN [\"4\"; \"13\"; \"5\"; \"/\"; \"+\"] = 6"
              "evalRPN [\"2\"; \"1\"; \"+\"; \"3\"; \"*\"] = 9" ]
            (mkProblem "adv-eval-rpn" "Evaluate RPN Expression"
                "Write a function that evaluates a Reverse Polish Notation expression. Tokens are integers or operators (+, -, *, /)."
                Advanced Algorithms "let evalRPN (tokens: string list) : int"
                ["Use a stack. Push numbers, pop two operands for operators."] 45
                "")

          withExamples
            [ "matMul (array2D [[1.0; 2.0]; [3.0; 4.0]]) (array2D [[5.0; 6.0]; [7.0; 8.0]]) = array2D [[19.0; 22.0]; [43.0; 50.0]]" ]
            (mkProblem "adv-matrix-multiply" "Matrix Multiplication"
                "Implement matrix multiplication for two 2D float arrays. Return the product matrix."
                Advanced Algorithms "let matMul (a: float[,]) (b: float[,]) : float[,]"
                ["Result[i,j] = sum of a[i,k] * b[k,j] for all k"] 60
                "") ]

    // =========================================================================
    // Expert
    // =========================================================================

    let private expertProblems =
        [ withExamples
            [ "result { let! x = Ok 5 in let! y = Ok x in return y + 1 } = Ok 6"
              "result { let! x = Error \"division by zero\" in return x + 1 } = Error \"division by zero\"" ]
            (mkProblem "exp-result-ce" "Result Computation Expression"
                "Implement a computation expression builder for Result<'T, string> that supports let!, return, and return!. Bind should short-circuit on Error."
                Expert TypeDesign "type ResultBuilder() = ...\nlet result = ResultBuilder()"
                ["Implement Bind, Return, ReturnFrom members on the builder class"] 90
                "")

          withExamples
            [ "classify \"42\" = \"int\""
              "classify \"3.14\" = \"float\""
              "classify \"hello\" = \"word\""
              "classify \"-7\" = \"int\"" ]
            (mkProblem "exp-active-pattern" "Active Pattern for Parsing"
                "Create an active pattern that parses a string as either an Int, a Float, or a Word. Use it in a classify function returning 'int', 'float', or 'word'."
                Expert PatternMatching "let (|Int|Float|Word|) (s: string) = ...\nlet classify (s: string) : string"
                ["Try Int32.TryParse, then Double.TryParse, else Word"] 60
                "")

          withExamples
            [ "hasCycle (Map.ofList [(1, [2]); (2, [3]); (3, [1])]) = true"
              "hasCycle (Map.ofList [(1, [2]); (2, [3]); (3, [])]) = false"
              "hasCycle (Map.ofList [(1, [1])]) = true"
              "hasCycle Map.empty = false" ]
            (mkProblem "exp-graph-cycle" "Graph Cycle Detection"
                "Given an adjacency list representing a directed graph, detect if the graph contains a cycle."
                Expert Algorithms "let hasCycle (graph: Map<int, int list>) : bool"
                ["Use DFS with a visiting set (gray nodes) and visited set (black nodes)"] 90
                "")

          withExamples
            [ "fib 0 = 0L"
              "fib 1 = 1L"
              "fib 10 = 55L"
              "fib 30 = 832040L"
              // The first call of fib 40 is the timed one, so no earlier call has filled the cache.
              "(let sw = global.System.Diagnostics.Stopwatch.StartNew() in global.Microsoft.FSharp.Core.Operators.ignore (fib 40); global.Microsoft.FSharp.Core.Operators.(<) sw.ElapsedMilliseconds 1000L)"
              "fib 40 = 102334155L" ]
            (mkProblem "exp-memoize" "Generic Memoization"
                "Write a generic memoize function that caches results. Use it to memoize a recursive Fibonacci function."
                Expert TypeDesign "let memoize (f: 'a -> 'b) : ('a -> 'b)\nlet fib : int -> int64"
                ["Use a Dictionary for the cache. For recursive memoization, use a mutable ref."] 60
                "") ]

    // =========================================================================
    // Public API
    // =========================================================================

    // =========================================================================
    // Performance-rewarded problems (correctness gate + speed reward)
    // =========================================================================

    /// Problems that carry a PerfHarness: a validated solution is additionally
    /// timed over a representative workload, giving the loop a continuous speed
    /// gradient (not just PASS/FAIL). A naive correct impl and a fast one both
    /// validate — but rank differently on ExecutionNs.
    let private sumSquaresHarness =
        "let __input = Array.init 1000000 (fun i -> float (i % 100))\n"
        + "for _ in 1..3 do sumSquares __input |> ignore\n"            // warmup (JIT + caches)
        + "let __times = System.Collections.Generic.List<float>()\n"
        + "for _ in 1..15 do\n"
        + "    let __sw = System.Diagnostics.Stopwatch.StartNew()\n"
        + "    sumSquares __input |> ignore\n"
        + "    __sw.Stop()\n"
        + "    __times.Add __sw.Elapsed.TotalMilliseconds\n"
        + "let __sorted = __times |> Seq.sort |> Seq.toArray\n"
        + "let __median = __sorted.[__sorted.Length / 2]\n"            // median resists jitter
        + "printfn \"ELAPSED_NS: %d\" (int64 (__median * 1_000_000.0))\n"

    let private perfProblems =
        let baseProblem =
            withExamples
              [ "sumSquares [|1.0; 2.0; 3.0|] = 14.0"
                "sumSquares [||] = 0.0"
                "sumSquares [|2.0|] = 4.0"
                "sumSquares [|-3.0; 4.0|] = 25.0" ]
              (mkProblem "inter-perf-sum-squares" "Sum of Squares (timed)"
                  "Write a function that returns the sum of the squares of all elements in a float array. It must be correct AND fast: prefer a single pass with no intermediate allocations."
                  Intermediate Algorithms "let sumSquares (xs: float[]) : float"
                  ["Fold once: Array.fold (fun acc x -> acc + x*x) 0.0 — avoid Array.map (it allocates a second array)."] 45
                  "")
        [ { baseProblem with PerfHarness = Some sumSquaresHarness } ]

    // =========================================================================
    // Hard (ADR 0005, backlog item 3)
    // =========================================================================

    /// A validation script, written one line per item.
    let private script (lines: string list) : string = String.concat "\n" lines + "\n"

    /// Longer problems than the tiers above: each has several rules to get right at once.
    let private hardProblems =
        [ mkProblem "hard-lru-trace" "LRU Cache Trace"
            "Simulate a least-recently-used cache that holds at most `capacity` keys. Apply the operations in order. (\"put\", key, value) stores the value and makes the key the most recently used; when the key is new and the cache is full, the least recently used key is evicted first. (\"get\", key, _) gives the stored value, or -1 when the key is absent; a key that is present becomes the most recently used. Return the results of the get operations, in order. A cache of capacity 0 stores nothing."
            Expert DataStructures "let lruTrace (capacity: int) (ops: (string * int * int) list) : int list"
            [ "A put on a key that is already present changes its value and its recency, and evicts nothing." ] 30
            (script
                [ "let mutable passed = true"
                  "let check (name: string) (capacity: int) (ops: (string * int * int) list) (expected: int list) ="
                  "    let got = lruTrace capacity ops"
                  "    if got <> expected then printfn \"FAIL: %s gave %A, expected %A\" name got expected; passed <- false"
                  "check \"eviction\" 2 [ \"put\", 1, 1; \"put\", 2, 2; \"get\", 1, 0; \"put\", 3, 3; \"get\", 2, 0; \"put\", 4, 4; \"get\", 1, 0; \"get\", 3, 0; \"get\", 4, 0 ] [ 1; -1; -1; 3; 4 ]"
                  "check \"update refreshes\" 2 [ \"put\", 1, 1; \"put\", 2, 2; \"put\", 1, 10; \"put\", 3, 3; \"get\", 1, 0; \"get\", 2, 0; \"get\", 3, 0 ] [ 10; -1; 3 ]"
                  "check \"capacity 0\" 0 [ \"put\", 1, 1; \"get\", 1, 0 ] [ -1 ]"
                  "check \"capacity 1\" 1 [ \"put\", 1, 1; \"put\", 2, 2; \"get\", 1, 0; \"get\", 2, 0 ] [ -1; 2 ]"
                  "check \"no gets\" 3 [ \"put\", 1, 1; \"put\", 2, 2 ] []"
                  "check \"a miss changes nothing\" 2 [ \"put\", 1, 1; \"put\", 2, 2; \"get\", 9, 0; \"put\", 3, 3; \"get\", 1, 0; \"get\", 2, 0 ] [ -1; -1; 2 ]"
                  "check \"update when full evicts nothing\" 2 [ \"put\", 1, 1; \"put\", 2, 2; \"put\", 2, 20; \"get\", 1, 0; \"get\", 2, 0 ] [ 1; 20 ]"
                  "if passed then printfn \"PASS\"" ])

          mkProblem "hard-dijkstra" "Cheapest Path"
            "Find the cost of the cheapest path in a directed graph. Each edge is (from, to, weight), and weights are never negative. Return Some of the smallest total weight of a path from `source` to `target`, or None when `target` cannot be reached. A node reaches itself at cost 0. Two nodes can be joined by several edges."
            Expert Algorithms "let shortestPath (edges: (int * int * int) list) (source: int) (target: int) : int option"
            [ "The path with the fewest edges is not always the cheapest." ] 30
            (script
                [ "let mutable passed = true"
                  "let check (name: string) (edges: (int * int * int) list) (source: int) (target: int) (expected: int option) ="
                  "    let got = shortestPath edges source target"
                  "    if got <> expected then printfn \"FAIL: %s gave %A, expected %A\" name got expected; passed <- false"
                  "let g = [ 1, 2, 7; 1, 3, 9; 1, 6, 14; 2, 3, 10; 2, 4, 15; 3, 4, 11; 3, 6, 2; 4, 5, 6; 5, 6, 9 ]"
                  "check \"1 to 5\" g 1 5 (Some 26)"
                  "check \"1 to 6\" g 1 6 (Some 11)"
                  "check \"against the direction\" g 6 1 None"
                  "check \"a node to itself\" g 4 4 (Some 0)"
                  "check \"no edges, same node\" [] 1 1 (Some 0)"
                  "check \"no edges, other node\" [] 1 2 None"
                  "check \"parallel edges\" [ 1, 2, 5; 1, 2, 3 ] 1 2 (Some 3)"
                  "check \"more edges but cheaper\" [ 1, 2, 1; 2, 3, 1; 1, 3, 5; 3, 4, 1; 1, 4, 10 ] 1 4 (Some 3)"
                  "check \"zero weights\" [ 1, 2, 0; 2, 3, 0 ] 1 3 (Some 0)"
                  "check \"a cycle on the way\" [ 1, 2, 2; 2, 1, 2; 2, 3, 2 ] 1 3 (Some 4)"
                  "if passed then printfn \"PASS\"" ])

          mkProblem "hard-topo-order" "Smallest-First Task Order"
            "Order the tasks 0 to n-1 so that every dependency is respected. A pair (a, b) means that task a must come before task b. Whenever several tasks are ready, take the one with the smallest number, so there is exactly one right answer. Return None when the dependencies form a cycle."
            Expert Algorithms "let topoOrder (n: int) (deps: (int * int) list) : int list option"
            [ "A task that depends on itself is a cycle." ] 30
            (script
                [ "let mutable passed = true"
                  "let check (name: string) (n: int) (deps: (int * int) list) (expected: int list option) ="
                  "    let got = topoOrder n deps"
                  "    if got <> expected then printfn \"FAIL: %s gave %A, expected %A\" name got expected; passed <- false"
                  "check \"diamond\" 4 [ 1, 0; 2, 0; 3, 1; 3, 2 ] (Some [ 3; 1; 2; 0 ])"
                  "check \"no dependencies\" 3 [] (Some [ 0; 1; 2 ])"
                  "check \"cycle\" 3 [ 0, 1; 1, 2; 2, 0 ] None"
                  "check \"smallest ready first\" 5 [ 4, 0; 4, 1; 3, 1 ] (Some [ 2; 3; 4; 0; 1 ])"
                  "check \"self dependency\" 2 [ 0, 0 ] None"
                  "check \"no tasks\" 0 [] (Some [])"
                  "check \"duplicate pairs\" 2 [ 1, 0; 1, 0 ] (Some [ 1; 0 ])"
                  "check \"one blocked task\" 4 [ 3, 0 ] (Some [ 1; 2; 3; 0 ])"
                  "check \"a cycle beside free tasks\" 4 [ 1, 2; 2, 1 ] None"
                  "check \"a chain backwards\" 4 [ 3, 2; 2, 1; 1, 0 ] (Some [ 3; 2; 1; 0 ])"
                  "if passed then printfn \"PASS\"" ])

          mkProblem "hard-expr-eval" "Arithmetic Expression Evaluator"
            "Evaluate an integer arithmetic expression. It is made of non-negative integer literals, the binary operators + - * /, a unary minus, parentheses and spaces. * and / bind tighter than + and -, and operators of the same level apply from left to right. Division is integer division and truncates toward zero. Return Error, with any message, for a division by zero and for an expression that is not well formed: empty, unbalanced parentheses, a missing operand or an unknown character."
            Expert PatternMatching "let evaluate (expr: string) : Result<int, string>"
            [ "10 - 4 - 3 is 3, and 7 - -2 is 9." ] 30
            (script
                [ "let mutable passed = true"
                  "let value (expr: string) (expected: int) ="
                  "    match evaluate expr with"
                  "    | Ok v when v = expected -> ()"
                  "    | other -> printfn \"FAIL: evaluate \\\"%s\\\" gave %A, expected Ok %d\" expr other expected; passed <- false"
                  "let invalid (expr: string) ="
                  "    match evaluate expr with"
                  "    | Error _ -> ()"
                  "    | other -> printfn \"FAIL: evaluate \\\"%s\\\" gave %A, expected an Error\" expr other; passed <- false"
                  "value \"1 + 2 * 3\" 7"
                  "value \"(1 + 2) * 3\" 9"
                  "value \"10 - 4 - 3\" 3"
                  "value \"100 / 7 / 2\" 7"
                  "value \"-7 / 2\" -3"
                  "value \"2 * -3\" -6"
                  "value \"-(2 + 3) * 4\" -20"
                  "value \" 42 \" 42"
                  "value \"2*(3+(4-1))/3\" 4"
                  "value \"7 - -2\" 9"
                  "value \"8 / 3 * 3\" 6"
                  "value \"1 - (2 - 3) - 4\" -2"
                  "invalid \"1 / 0\""
                  "invalid \"1 / (2 - 2)\""
                  "invalid \"\""
                  "invalid \"(1 + 2\""
                  "invalid \"1 + 2)\""
                  "invalid \"1 +\""
                  "invalid \"2 $ 3\""
                  "invalid \"* 3\""
                  "if passed then printfn \"PASS\"" ])

          mkProblem "hard-wrap-text" "Word Wrap"
            "Wrap a text into lines of at most `width` characters. Words are separated by one or more spaces. A line holds as many whole words as fit, joined by single spaces, with no space at either end. A word longer than `width` is cut into pieces of exactly `width` characters, each alone on its line, and its last, shorter piece starts a line that the following words can share. A text with no words gives an empty list. `width` is at least 1."
            Expert StringManipulation "let wrapText (width: int) (text: string) : string list"
            [ "Words that exactly fill the width still fit." ] 30
            (script
                [ "let mutable passed = true"
                  "let check (width: int) (text: string) (expected: string list) ="
                  "    let got = wrapText width text"
                  "    if got <> expected then printfn \"FAIL: wrapText %d \\\"%s\\\" gave %A, expected %A\" width text got expected; passed <- false"
                  "check 10 \"the quick brown fox jumps\" [ \"the quick\"; \"brown fox\"; \"jumps\" ]"
                  "check 9 \"the quick brown\" [ \"the quick\"; \"brown\" ]"
                  "check 5 \"  hello   world  \" [ \"hello\"; \"world\" ]"
                  "check 4 \"ab abcdefghij z\" [ \"ab\"; \"abcd\"; \"efgh\"; \"ij z\" ]"
                  "check 7 \"\" []"
                  "check 7 \"     \" []"
                  "check 1 \"ab c\" [ \"a\"; \"b\"; \"c\" ]"
                  "check 3 \"a b c d\" [ \"a b\"; \"c d\" ]"
                  "check 6 \"abcdefghijkl\" [ \"abcdef\"; \"ghijkl\" ]"
                  "check 20 \"one\" [ \"one\" ]"
                  "check 5 \"aaaaa bb ccc\" [ \"aaaaa\"; \"bb\"; \"ccc\" ]"
                  "if passed then printfn \"PASS\"" ])

          mkProblem "hard-roman-parse" "Strict Roman Numeral Parser"
            "Parse a Roman numeral written in its standard form and return its value, for the values 1 to 3999. Return None for everything else: the empty string, lowercase letters, any other character, and a numeral that is not the standard way to write its value, such as IIII, VV, IC, IXI or MMMM."
            Expert StringManipulation "let parseRoman (s: string) : int option"
            [ "Each value from 1 to 3999 has exactly one standard numeral." ] 30
            (script
                [ "let mutable passed = true"
                  "let check (s: string) (expected: int option) ="
                  "    let got = parseRoman s"
                  "    if got <> expected then printfn \"FAIL: parseRoman \\\"%s\\\" gave %A, expected %A\" s got expected; passed <- false"
                  "check \"III\" (Some 3)"
                  "check \"IV\" (Some 4)"
                  "check \"IX\" (Some 9)"
                  "check \"XIV\" (Some 14)"
                  "check \"XLII\" (Some 42)"
                  "check \"CDXLIV\" (Some 444)"
                  "check \"MCMXCIV\" (Some 1994)"
                  "check \"MMMCMXCIX\" (Some 3999)"
                  "check \"\" None"
                  "check \"IIII\" None"
                  "check \"VV\" None"
                  "check \"IC\" None"
                  "check \"IL\" None"
                  "check \"IXI\" None"
                  "check \"VX\" None"
                  "check \"IIV\" None"
                  "check \"XM\" None"
                  "check \"CMCM\" None"
                  "check \"LXL\" None"
                  "check \"MMMM\" None"
                  "check \"xiv\" None"
                  "check \"X I\" None"
                  "check \"12\" None"
                  "if passed then printfn \"PASS\"" ])

          mkProblem "hard-rational" "Exact Rational Numbers"
            "Define exact fractions. `Rational` is a record with the integer fields `Num` and `Den`. `make n d` gives None when d is 0. Otherwise it gives the fraction in its normal form: lowest terms, a positive denominator, and zero written as 0/1, so that two equal fractions are equal records. `add` and `mul` return their result in that normal form. `show` gives \"Num/Den\", or only the numerator when the denominator is 1."
            Expert TypeDesign "type Rational = { Num: int; Den: int }\nlet make (n: int) (d: int) : Rational option\nlet add (a: Rational) (b: Rational) : Rational\nlet mul (a: Rational) (b: Rational) : Rational\nlet show (r: Rational) : string"
            [ "The sign belongs to the numerator." ] 30
            (script
                [ "let mutable passed = true"
                  "let expect (name: string) (got: 'a) (expected: 'a) ="
                  "    if got <> expected then printfn \"FAIL: %s gave %A, expected %A\" name got expected; passed <- false"
                  "let r (n: int) (d: int) : Rational = (make n d).Value"
                  "expect \"make 2 4\" (make 2 4) (Some { Num = 1; Den = 2 })"
                  "expect \"make 1 0\" (make 1 0) None"
                  "expect \"make 3 -6\" (make 3 -6) (Some { Num = -1; Den = 2 })"
                  "expect \"make -3 -6\" (make -3 -6) (Some { Num = 1; Den = 2 })"
                  "expect \"make 0 5\" (make 0 5) (Some { Num = 0; Den = 1 })"
                  "expect \"make 0 -5\" (make 0 -5) (Some { Num = 0; Den = 1 })"
                  "expect \"make 7 1\" (make 7 1) (Some { Num = 7; Den = 1 })"
                  "expect \"1/2 + 1/3\" (add (r 1 2) (r 1 3)) { Num = 5; Den = 6 }"
                  "expect \"1/2 + -1/2\" (add (r 1 2) (r -1 2)) { Num = 0; Den = 1 }"
                  "expect \"1/6 + 1/3\" (add (r 1 6) (r 1 3)) { Num = 1; Den = 2 }"
                  "expect \"2/3 * 3/4\" (mul (r 2 3) (r 3 4)) { Num = 1; Den = 2 }"
                  "expect \"-2/3 * 3/2\" (mul (r -2 3) (r 3 2)) { Num = -1; Den = 1 }"
                  "expect \"0 * 5/7\" (mul (r 0 3) (r 5 7)) { Num = 0; Den = 1 }"
                  "expect \"show 5/6\" (show (r 5 6)) \"5/6\""
                  "expect \"show -4/2\" (show (r -4 2)) \"-2\""
                  "expect \"show 0/9\" (show (r 0 9)) \"0\""
                  "expect \"show 1/-2\" (show (r 1 -2)) \"-1/2\""
                  "if passed then printfn \"PASS\"" ])

          mkProblem "hard-ledger" "Account Ledger"
            "Apply transactions to account balances, in order. `Deposit (account, amount)` adds to an account and creates it when it does not exist. `Withdraw (account, amount)` takes from an account. `Transfer (source, target, amount)` moves money and creates the target account when needed. Stop at the first transaction that cannot be applied and return Error with exactly one of these texts, where N is the position of the transaction, starting at 1. \"tx N: invalid amount\" when the amount is not positive; this is checked first. \"tx N: unknown account\" when the account money is taken from does not exist. \"tx N: insufficient funds\" when that account holds less than the amount. When every transaction applies, return Ok with the final balances."
            Expert ErrorHandling "type Tx =\n    | Deposit of string * int\n    | Withdraw of string * int\n    | Transfer of string * string * int\nlet settle (start: Map<string, int>) (txs: Tx list) : Result<Map<string, int>, string>"
            [ "Taking an account's whole balance is allowed and leaves it at 0." ] 30
            (script
                [ "let mutable passed = true"
                  "let check (name: string) (start: (string * int) list) (txs: Tx list) (expected: Result<(string * int) list, string>) ="
                  "    let got = settle (Map.ofList start) txs |> Result.map Map.toList"
                  "    if got <> expected then printfn \"FAIL: %s gave %A, expected %A\" name got expected; passed <- false"
                  "check \"all apply\" [ \"a\", 100 ] [ Deposit (\"a\", 50); Withdraw (\"a\", 30); Transfer (\"a\", \"b\", 70) ] (Ok [ \"a\", 50; \"b\", 70 ])"
                  "check \"unknown account\" [] [ Withdraw (\"x\", 10) ] (Error \"tx 1: unknown account\")"
                  "check \"insufficient funds\" [ \"a\", 10 ] [ Deposit (\"a\", 5); Withdraw (\"a\", 20) ] (Error \"tx 2: insufficient funds\")"
                  "check \"zero amount\" [] [ Deposit (\"a\", 0) ] (Error \"tx 1: invalid amount\")"
                  "check \"amount checked first\" [] [ Withdraw (\"nobody\", -5) ] (Error \"tx 1: invalid amount\")"
                  "check \"transfer to itself\" [ \"a\", 10 ] [ Transfer (\"a\", \"a\", 4) ] (Ok [ \"a\", 10 ])"
                  "check \"whole balance\" [ \"a\", 10 ] [ Withdraw (\"a\", 10) ] (Ok [ \"a\", 0 ])"
                  "check \"stops at the first error\" [ \"a\", 10 ] [ Withdraw (\"a\", 20); Deposit (\"a\", 100) ] (Error \"tx 1: insufficient funds\")"
                  "check \"no transactions\" [ \"a\", 1 ] [] (Ok [ \"a\", 1 ])"
                  "check \"transfer from unknown\" [ \"b\", 5 ] [ Deposit (\"b\", 1); Deposit (\"b\", 1); Transfer (\"a\", \"b\", 1) ] (Error \"tx 3: unknown account\")"
                  "check \"transfer too large\" [ \"a\", 5; \"b\", 0 ] [ Transfer (\"a\", \"b\", 6) ] (Error \"tx 1: insufficient funds\")"
                  "check \"deposit creates\" [] [ Deposit (\"new\", 3); Transfer (\"new\", \"other\", 3) ] (Ok [ \"new\", 0; \"other\", 3 ])"
                  "if passed then printfn \"PASS\"" ])

          mkProblem "hard-throttled-map" "Throttled Parallel Map"
            "Apply an asynchronous function to every item of a list, with at most `maxParallel` calls running at the same time, and return the results in the order of the items. The calls must really overlap: when `maxParallel` is greater than 1 and there are several items, more than one call runs at a time."
            Expert AsyncPatterns "let mapThrottled (maxParallel: int) (f: 'a -> Async<'b>) (xs: 'a list) : Async<'b list>"
            [ "The results come back in input order, whatever order the calls finish in." ] 30
            (script
                [ "let mutable passed = true"
                  "let running = [| 0 |]"
                  "let peak = [| 0 |]"
                  "let work (x: int) : Async<int> ="
                  "    async {"
                  "        let now = System.Threading.Interlocked.Increment(&running.[0])"
                  "        lock peak (fun () -> if now > peak.[0] then peak.[0] <- now)"
                  "        // Later items finish first, so results in completion order would be wrong."
                  "        do! Async.Sleep(120 - 10 * x)"
                  "        System.Threading.Interlocked.Decrement(&running.[0]) |> ignore"
                  "        return x * 2"
                  "    }"
                  "let run (maxParallel: int) (xs: int list) : int list * int ="
                  "    running.[0] <- 0"
                  "    peak.[0] <- 0"
                  "    let results = mapThrottled maxParallel work xs |> Async.RunSynchronously"
                  "    results, peak.[0]"
                  "let results3, peak3 = run 3 [ 1 .. 9 ]"
                  "if results3 <> [ 2; 4; 6; 8; 10; 12; 14; 16; 18 ] then printfn \"FAIL: results %A are not in input order\" results3; passed <- false"
                  "if peak3 > 3 then printfn \"FAIL: %d calls ran at once, the limit was 3\" peak3; passed <- false"
                  "if peak3 < 2 then printfn \"FAIL: the calls never overlapped\" ; passed <- false"
                  "let results1, peak1 = run 1 [ 1 .. 4 ]"
                  "if results1 <> [ 2; 4; 6; 8 ] then printfn \"FAIL: results %A with a limit of 1\" results1; passed <- false"
                  "if peak1 <> 1 then printfn \"FAIL: %d calls ran at once, the limit was 1\" peak1; passed <- false"
                  "let resultsNone, _ = run 4 []"
                  "if resultsNone <> [] then printfn \"FAIL: an empty list gave %A\" resultsNone; passed <- false"
                  "let resultsWide, peakWide = run 10 [ 1 .. 3 ]"
                  "if resultsWide <> [ 2; 4; 6 ] then printfn \"FAIL: results %A with a limit above the item count\" resultsWide; passed <- false"
                  "if peakWide > 3 then printfn \"FAIL: %d calls ran at once for 3 items\" peakWide; passed <- false"
                  "if passed then printfn \"PASS\"" ])

          mkProblem "hard-min-window" "Smallest Covering Substring"
            "Return the shortest substring of `s` that contains every character of `t`, repeats included: when `t` has the same character twice, the substring needs it twice. When several substrings have that shortest length, return the leftmost. Return the empty string when there is no such substring, and when `t` is empty. Upper and lower case are different characters."
            Expert Algorithms "let minWindow (s: string) (t: string) : string"
            [ "The characters of `t` can appear in any order in the substring." ] 30
            (script
                [ "let mutable passed = true"
                  "let check (s: string) (t: string) (expected: string) ="
                  "    let got = minWindow s t"
                  "    if got <> expected then printfn \"FAIL: minWindow \\\"%s\\\" \\\"%s\\\" gave \\\"%s\\\", expected \\\"%s\\\"\" s t got expected; passed <- false"
                  "check \"ADOBECODEBANC\" \"ABC\" \"BANC\""
                  "check \"a\" \"a\" \"a\""
                  "check \"a\" \"aa\" \"\""
                  "check \"aa\" \"aa\" \"aa\""
                  "check \"abc\" \"\" \"\""
                  "check \"\" \"a\" \"\""
                  "check \"xaybzbxa\" \"ab\" \"ayb\""
                  "check \"aA\" \"A\" \"A\""
                  "check \"aabdcab\" \"aab\" \"aab\""
                  "check \"cabwefgewcwaefgcf\" \"cae\" \"cwae\""
                  "check \"abcdef\" \"fa\" \"abcdef\""
                  "check \"abab\" \"ba\" \"ab\""
                  "check \"hello\" \"xyz\" \"\""
                  "if passed then printfn \"PASS\"" ])

          mkProblem "hard-regex-match" "Pattern Matching with . and *"
            "Decide whether a pattern matches a whole text, from its first character to its last. In the pattern, '.' matches any one character, and '*' means zero or more of the character or '.' that comes just before it. Every other character matches itself. The pattern is well formed: '*' is never first and never follows another '*'."
            Expert PatternMatching "let isMatch (text: string) (pattern: string) : bool"
            [ "A starred character can match nothing at all." ] 30
            (script
                [ "let mutable passed = true"
                  "let check (text: string) (pattern: string) (expected: bool) ="
                  "    let got = isMatch text pattern"
                  "    if got <> expected then printfn \"FAIL: isMatch \\\"%s\\\" \\\"%s\\\" gave %b, expected %b\" text pattern got expected; passed <- false"
                  "check \"aa\" \"a\" false"
                  "check \"aa\" \"a*\" true"
                  "check \"ab\" \".*\" true"
                  "check \"aab\" \"c*a*b\" true"
                  "check \"mississippi\" \"mis*is*p*.\" false"
                  "check \"mississippi\" \"mis*is*ip*.\" true"
                  "check \"\" \"\" true"
                  "check \"\" \"a*\" true"
                  "check \"\" \"a\" false"
                  "check \"a\" \"\" false"
                  "check \"abc\" \"a.c\" true"
                  "check \"abc\" \"a.\" false"
                  "check \"aaa\" \"a*a\" true"
                  "check \"ab\" \".*c\" false"
                  "check \"a\" \"ab*\" true"
                  "check \"bbbba\" \".*a*a\" true"
                  "check \"abcd\" \"d*\" false"
                  "check \"a*b\" \"a.b\" true"
                  "if passed then printfn \"PASS\"" ])

          mkProblem "hard-csv-line" "CSV Line Parser"
            "Split one line of CSV into its fields. Commas separate the fields. A field can be wrapped in double quotes: inside the quotes a comma is ordinary text, and two double quotes in a row stand for one double quote. The quotes that wrap a field are not part of its value. A field without quotes is taken as it is, spaces included. An empty line is one empty field."
            Expert StringManipulation "let parseCsvLine (line: string) : string list"
            [ "Two commas in a row hold an empty field between them." ] 30
            (script
                [ "let mutable passed = true"
                  "let check (line: string) (expected: string list) ="
                  "    let got = parseCsvLine line"
                  "    if got <> expected then printfn \"FAIL: parseCsvLine %A gave %A, expected %A\" line got expected; passed <- false"
                  "check \"a,b,c\" [ \"a\"; \"b\"; \"c\" ]"
                  "check \"\" [ \"\" ]"
                  "check \"a,,c\" [ \"a\"; \"\"; \"c\" ]"
                  "check \",,\" [ \"\"; \"\"; \"\" ]"
                  "check \"\\\"a,b\\\",c\" [ \"a,b\"; \"c\" ]"
                  "check \"\\\"he said \\\"\\\"hi\\\"\\\"\\\",x\" [ \"he said \\\"hi\\\"\"; \"x\" ]"
                  "check \" a , b \" [ \" a \"; \" b \" ]"
                  "check \"\\\"\\\"\" [ \"\" ]"
                  "check \"\\\"\\\",\\\"\\\"\" [ \"\"; \"\" ]"
                  "check \"x,\\\"\\\"\\\"\\\"\" [ \"x\"; \"\\\"\" ]"
                  "check \"one\" [ \"one\" ]"
                  "check \"\\\"a\\\",\\\"b,c\\\",d,\" [ \"a\"; \"b,c\"; \"d\"; \"\" ]"
                  "if passed then printfn \"PASS\"" ])

          mkProblem "hard-huffman" "Optimal Prefix Code"
            "Build a binary code that makes the encoded text as short as possible. The input gives each character with the number of times it occurs; the characters are distinct and the counts positive. Return a map from every character to its code, a non-empty string of '0' and '1'. No code can be the start of another one, and the sum over all characters of count times code length must be the smallest possible. A single character gets the code \"0\". An empty input gives an empty map."
            Expert DataStructures "let huffmanCodes (freqs: (char * int) list) : Map<char, string>"
            [ "Several codes are optimal; any of them is accepted." ] 30
            (script
                [ "let mutable passed = true"
                  "let check (name: string) (freqs: (char * int) list) (bestCost: int) ="
                  "    let codes = huffmanCodes freqs"
                  "    let all = codes |> Map.toList |> List.map snd"
                  "    let missing = freqs |> List.filter (fun (c, _) -> not (codes.ContainsKey c))"
                  "    if not missing.IsEmpty then printfn \"FAIL: %s has no code for %A\" name (missing |> List.map fst); passed <- false"
                  "    elif codes.Count <> freqs.Length then printfn \"FAIL: %s has %d codes for %d characters\" name codes.Count freqs.Length; passed <- false"
                  "    elif all |> List.exists (fun code -> code = \"\" || code |> Seq.exists (fun ch -> ch <> '0' && ch <> '1')) then"
                  "        printfn \"FAIL: %s has a code that is empty or not made of 0 and 1: %A\" name all; passed <- false"
                  "    elif all |> List.exists (fun a -> all |> List.exists (fun b -> a <> b && b.StartsWith a)) || (List.distinct all).Length <> all.Length then"
                  "        printfn \"FAIL: %s has a code that starts another one: %A\" name all; passed <- false"
                  "    else"
                  "        let cost = freqs |> List.sumBy (fun (c, n) -> n * codes.[c].Length)"
                  "        if cost <> bestCost then printfn \"FAIL: %s costs %d, the best is %d\" name cost bestCost; passed <- false"
                  "check \"textbook\" [ 'a', 5; 'b', 9; 'c', 12; 'd', 13; 'e', 16; 'f', 45 ] 224"
                  "check \"two\" [ 'a', 1; 'b', 1 ] 2"
                  "check \"four equal\" [ 'a', 1; 'b', 1; 'c', 1; 'd', 1 ] 8"
                  "check \"skewed\" [ 'a', 1; 'b', 2; 'c', 4; 'd', 8 ] 25"
                  "check \"three equal\" [ 'a', 3; 'b', 3; 'c', 3 ] 15"
                  "check \"one\" [ 'x', 7 ] 7"
                  "if huffmanCodes [ 'x', 7 ] <> Map.ofList [ 'x', \"0\" ] then printfn \"FAIL: a single character must get the code 0\"; passed <- false"
                  "if not (huffmanCodes []).IsEmpty then printfn \"FAIL: an empty input must give an empty map\"; passed <- false"
                  "if passed then printfn \"PASS\"" ])

          mkProblem "hard-lcs" "Longest Common Subsequence"
            "Return a longest common subsequence of two strings: a longest string whose characters appear in both inputs in the same order, though not necessarily next to each other. When several exist, any one of them is accepted. The inputs can be a few hundred characters long."
            Expert Algorithms "let lcs (a: string) (b: string) : string"
            [ "Trying every subsequence is far too slow for long inputs." ] 30
            (script
                [ "let mutable passed = true"
                  "let isSubsequence (sub: string) (s: string) : bool ="
                  "    let rec go (i: int) (j: int) = i = sub.Length || (j < s.Length && (if sub.[i] = s.[j] then go (i + 1) (j + 1) else go i (j + 1)))"
                  "    go 0 0"
                  "let check (a: string) (b: string) (length: int) ="
                  "    let got = lcs a b"
                  "    let show (s: string) = if s.Length > 40 then s.Substring(0, 40) + \"...\" else s"
                  "    if not (isSubsequence got a && isSubsequence got b) then"
                  "        printfn \"FAIL: \\\"%s\\\" is not a subsequence of both \\\"%s\\\" and \\\"%s\\\"\" (show got) (show a) (show b); passed <- false"
                  "    elif got.Length <> length then"
                  "        printfn \"FAIL: lcs \\\"%s\\\" \\\"%s\\\" has length %d, the longest has %d\" (show a) (show b) got.Length length; passed <- false"
                  "check \"ABCBDAB\" \"BDCABA\" 4"
                  "check \"AGGTAB\" \"GXTXAYB\" 4"
                  "check \"XMJYAUZ\" \"MZJAWXU\" 4"
                  "check \"\" \"abc\" 0"
                  "check \"abc\" \"\" 0"
                  "check \"abc\" \"abc\" 3"
                  "check \"abc\" \"def\" 0"
                  "check \"aaaa\" \"aa\" 2"
                  "check \"abcde\" \"ace\" 3"
                  "check (String.replicate 150 \"ab\") (String.replicate 150 \"ba\") 299"
                  "if passed then printfn \"PASS\"" ])

          mkProblem "hard-state-ce" "State Computation Expression"
            "Implement a computation expression that threads a state through a computation. `State<'s, 'a>` wraps a function from a state to a result and the next state. `runState` runs it from a starting state. `getState ()` gives the current state as its result. `putState s` replaces the state. The builder value `state` supports let!, do!, return and return!, and each step sees the state left by the step before it."
            Expert TypeDesign "type State<'s, 'a> = State of ('s -> 'a * 's)\nlet runState (m: State<'s, 'a>) (s: 's) : 'a * 's\nlet getState () : State<'s, 's>\nlet putState (s: 's) : State<'s, unit>\ntype StateBuilder() = ...\nlet state = StateBuilder()"
            [ "A block that ends with do! needs the builder's Zero member." ] 30
            (script
                [ "let mutable passed = true"
                  "let expect (name: string) (got: 'a) (expected: 'a) ="
                  "    if got <> expected then printfn \"FAIL: %s gave %A, expected %A\" name got expected; passed <- false"
                  "let next : State<int, int> ="
                  "    state {"
                  "        let! n = getState ()"
                  "        do! putState (n + 1)"
                  "        return n"
                  "    }"
                  "let twice : State<int, int> ="
                  "    state {"
                  "        let! a = next"
                  "        let! b = next"
                  "        return a + b"
                  "    }"
                  "let forwarded : State<int, int> = state { return! next }"
                  "let justPut : State<int, unit> = state { do! putState 99 }"
                  "let shout : State<string, int> ="
                  "    state {"
                  "        let! s = getState ()"
                  "        do! putState (s + \"!\")"
                  "        return s.Length"
                  "    }"
                  "let constant : State<int, string> = state { return \"k\" }"
                  "expect \"next from 10\" (runState next 10) (10, 11)"
                  "expect \"twice from 10\" (runState twice 10) (21, 12)"
                  "expect \"return! from 3\" (runState forwarded 3) (3, 4)"
                  "expect \"do! alone\" (runState justPut 0) ((), 99)"
                  "expect \"a string state\" (runState shout \"ab\") (2, \"ab!\")"
                  "expect \"return leaves the state\" (runState constant 5) (\"k\", 5)"
                  "if passed then printfn \"PASS\"" ])

          mkProblem "hard-life-step" "Game of Life on an Unbounded Grid"
            "Compute the next generation of Conway's Game of Life on a grid with no edges. The input is the set of live cells, as (x, y) pairs that can be negative. A live cell with 2 or 3 live neighbours stays alive. A dead cell with exactly 3 live neighbours becomes alive. Every other cell is dead in the next generation. A cell's neighbours are the 8 cells around it."
            Expert DataStructures "let lifeStep (alive: Set<int * int>) : Set<int * int>"
            [ "A cell that is dead now can be alive next, so the live cells alone are not enough to look at." ] 30
            (script
                [ "let mutable passed = true"
                  "let check (name: string) (alive: (int * int) list) (expected: (int * int) list) ="
                  "    let got = lifeStep (Set.ofList alive)"
                  "    if got <> Set.ofList expected then printfn \"FAIL: %s gave %A, expected %A\" name (Set.toList got) (List.sort expected); passed <- false"
                  "check \"blinker\" [ 1, 0; 1, 1; 1, 2 ] [ 0, 1; 1, 1; 2, 1 ]"
                  "check \"block\" [ 0, 0; 0, 1; 1, 0; 1, 1 ] [ 0, 0; 0, 1; 1, 0; 1, 1 ]"
                  "check \"nothing\" [] []"
                  "check \"one cell\" [ 5, 5 ] []"
                  "check \"two cells\" [ 0, 0; 1, 0 ] []"
                  "check \"glider\" [ 1, 0; 2, 1; 0, 2; 1, 2; 2, 2 ] [ 0, 1; 2, 1; 1, 2; 2, 2; 1, 3 ]"
                  "check \"blinker below zero\" [ -1, -1; -1, 0; -1, 1 ] [ -2, 0; -1, 0; 0, 0 ]"
                  "check \"corner birth\" [ 0, 0; 1, 0; 0, 1 ] [ 0, 0; 1, 0; 0, 1; 1, 1 ]"
                  "check \"crowded centre\" [ 0, 0; 1, 0; 2, 0; 0, 1; 1, 1; 2, 1; 0, 2; 1, 2; 2, 2 ] [ 0, 0; 2, 0; 0, 2; 2, 2; 1, -1; -1, 1; 3, 1; 1, 3 ]"
                  "let twoSteps = lifeStep (lifeStep (Set.ofList [ 1, 0; 1, 1; 1, 2 ]))"
                  "if twoSteps <> Set.ofList [ 1, 0; 1, 1; 1, 2 ] then printfn \"FAIL: two steps of a blinker gave %A\" (Set.toList twoSteps); passed <- false"
                  "if passed then printfn \"PASS\"" ]) ]

    /// All curated benchmark problems.
    let all () : BenchmarkProblem list =
        basicProblems @ intermediateProblems @ advancedProblems @ expertProblems @ perfProblems

    /// The hard bank (ADR 0005, backlog item 3). It is not part of `all ()`, so that bank and
    /// its recorded pass rates stay comparable.
    let hard () : BenchmarkProblem list = hardProblems

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
