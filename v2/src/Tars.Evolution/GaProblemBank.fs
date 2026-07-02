namespace Tars.Evolution

/// A benchmark fitness domain drawn from the sibling **Guitar Alchemist** (GA)
/// music-theory system. Where `ProblemBank` exercises generic F# coding skill,
/// these problems make TARS evolve against GA's actual domain mathematics:
/// pitch-class arithmetic, scale construction, triad quality, and Tₙ set-class
/// equivalence — the same computations GA's `GA.Domain.Core` performs.
///
/// Each problem's validation encodes GA's domain rules as ground truth, so a
/// passing solution is one that reasons correctly *in GA's domain*. This turns
/// the evolve/benchmark loop into a live fitness signal over music theory rather
/// than synthetic puzzles alone.
module GaProblemBank =

    let private mk id title desc diff sig_ hints validation : BenchmarkProblem =
        { Id = id
          Title = title
          Description = desc
          Difficulty = diff
          Category = MusicTheory
          ExpectedSignature = sig_
          Hints = hints
          TimeLimitSeconds = 45
          ValidationCode = validation
          PerfHarness = None
          Properties = None }

    let private problems =
        [ mk "ga-pc-interval" "Pitch-Class Interval"
            "Pitch classes are integers 0-11 (C=0, C#=1, ... B=11). Write a function that returns the ascending interval in semitones from pitch class a to pitch class b, as a value in 0-11."
            Beginner
            "let pcInterval (a: int) (b: int) : int"
            [ "Subtract, then wrap into 0-11 with a modulo that handles negatives." ]
            ("let mutable passed = true\n"
             + "let check a b exp =\n"
             + "    let got = pcInterval a b\n"
             + "    if got <> exp then printfn \"FAIL: pcInterval %d %d = %d, expected %d\" a b got exp; passed <- false\n"
             + "check 0 7 7\ncheck 7 0 5\ncheck 0 0 0\ncheck 11 1 2\ncheck 4 2 10\n"
             + "if passed then printfn \"PASS\"\n")

          mk "ga-transpose" "Transpose a Pitch-Class Set"
            "Write a function that transposes every pitch class in a list up by n semitones, wrapping each result into 0-11. Preserve order and length."
            Beginner
            "let transpose (pcs: int list) (n: int) : int list"
            [ "Map each element with (pc + n) reduced mod 12." ]
            ("let mutable passed = true\n"
             + "let check pcs n exp =\n"
             + "    let got = transpose pcs n\n"
             + "    if got <> exp then printfn \"FAIL: transpose %A %d = %A, expected %A\" pcs n got exp; passed <- false\n"
             + "check [0;4;7] 2 [2;6;9]\ncheck [0;4;7] 5 [5;9;0]\ncheck [11] 1 [0]\ncheck [] 3 []\n"
             + "if passed then printfn \"PASS\"\n")

          mk "ga-major-scale" "Major Scale Pitch Classes"
            "Write a function that returns the seven pitch classes of the major scale starting at the given root, in ascending scale order, each wrapped into 0-11. The major scale step pattern from the root is W W H W W W (2,2,1,2,2,2 semitones)."
            Intermediate
            "let majorScale (root: int) : int list"
            [ "Accumulate the offsets [0;2;4;5;7;9;11] from the root, each mod 12." ]
            ("let mutable passed = true\n"
             + "let check root exp =\n"
             + "    let got = majorScale root\n"
             + "    if got <> exp then printfn \"FAIL: majorScale %d = %A, expected %A\" root got exp; passed <- false\n"
             + "check 0 [0;2;4;5;7;9;11]\ncheck 7 [7;9;11;0;2;4;6]\ncheck 2 [2;4;6;7;9;11;1]\n"
             + "if passed then printfn \"PASS\"\n")

          mk "ga-triad-quality" "Triad Quality"
            "Given the three pitch classes of a triad (root, third, fifth, each 0-11), classify the chord. Return one of \"major\", \"minor\", \"diminished\", \"augmented\", or \"unknown\". Classify by the intervals above the root: major third = 4 & perfect fifth = 7 -> major; minor third = 3 & perfect fifth = 7 -> minor; minor third = 3 & diminished fifth = 6 -> diminished; major third = 4 & augmented fifth = 8 -> augmented."
            Intermediate
            "let triadQuality (root: int) (third: int) (fifth: int) : string"
            [ "Compute the two intervals above the root mod 12, then match the pair." ]
            ("let mutable passed = true\n"
             + "let check r t f exp =\n"
             + "    let got = triadQuality r t f\n"
             + "    if got <> exp then printfn \"FAIL: triadQuality %d %d %d = '%s', expected '%s'\" r t f got exp; passed <- false\n"
             + "check 0 4 7 \"major\"\ncheck 0 3 7 \"minor\"\ncheck 2 5 8 \"diminished\"\ncheck 0 4 8 \"augmented\"\ncheck 9 1 4 \"major\"\ncheck 0 5 7 \"unknown\"\n"
             + "if passed then printfn \"PASS\"\n")

          mk "ga-is-transposition" "Tₙ Set-Class Equivalence"
            "Two pitch-class sets are Tₙ-equivalent if one can be transposed by some number of semitones to produce the other (as sets — ignore order and duplicates). Write a function that returns true if set b is a transposition of set a. Pitch classes are 0-11."
            Advanced
            "let isTransposition (a: int list) (b: int list) : bool"
            [ "Reduce both to sorted distinct pitch classes. If sizes differ, false. Otherwise try all 12 transpositions of a and compare as sets." ]
            ("let mutable passed = true\n"
             + "let check a b exp =\n"
             + "    let got = isTransposition a b\n"
             + "    if got <> exp then printfn \"FAIL: isTransposition %A %A = %b, expected %b\" a b got exp; passed <- false\n"
             + "check [0;4;7] [2;6;9] true\ncheck [0;4;7] [0;3;7] false\ncheck [0;1;2] [5;6;7] true\ncheck [0;4;7] [0;4;7] true\ncheck [0;4] [0;4;7] false\n"
             + "if passed then printfn \"PASS\"\n") ]

    /// FsCheck property body for `transpose` (the runner supplies `#r`/`open` +
    /// the solution). Catches solutions that pass the small example cases but
    /// break an invariant on adversarial inputs (e.g. forgetting the mod-12 wrap):
    ///   - length preserved
    ///   - additivity: transpose by 5 then 7 ≡ identity (5+7 = 12 ≡ 0 mod 12)
    ///   - range: every output pitch class is in 0..11
    let private transposeProperties =
        "let arb = Arb.fromGen (Gen.listOf (Gen.choose (0, 11)))\n"
        + "let cfg = { Config.QuickThrowOnFailure with MaxTest = 200 }\n"
        + "try\n"
        + "    Check.One(cfg, Prop.forAll arb (fun xs -> (transpose xs 3).Length = xs.Length))\n"
        + "    Check.One(cfg, Prop.forAll arb (fun xs -> transpose (transpose xs 5) 7 = transpose xs 0))\n"
        + "    Check.One(cfg, Prop.forAll arb (fun xs -> transpose xs 4 |> List.forall (fun x -> x >= 0 && x <= 11)))\n"
        + "    printfn \"PROP PASS\"\n"
        + "with ex -> printfn \"PROP FAIL: %s\" (ex.Message.Split('\\n').[0])\n"

    /// All GA-domain music-theory benchmark problems.
    let all () : BenchmarkProblem list =
        problems
        |> List.map (fun p ->
            if p.Id = "ga-transpose" then { p with Properties = Some transposeProperties } else p)
