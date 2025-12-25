# 🐛 BUG: Silent Puzzle Demo Execution

**Issue**: `tars demo-puzzle --all --difficulty 3 --benchmark 5 --output json --export puzzle_baseline.json` runs silently with no output

**Date Reported**: 2024-12-24  
**Status**: 🐛 **CONFIRMED**  
**Priority**: 🔴 **HIGH** (poor UX)

---

## Problem Description

When running complex puzzle demo commands with multiple flags, TARS:
1. ❌ Doesn't parse the arguments correctly
2. ❌ Falls through to default help case
3. ❌ Doesn't print any output (not even help)
4. ❌ Returns silently, leaving user confused

### Expected Behavior
```bash
$ tars demo-puzzle --all --difficulty 3 --benchmark 5 --output json --export puzzle_baseline.json

🧩 TARS Puzzle Benchmark
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Running 9 puzzles (difficulty ≤ 3) with 5 iterations each...

[1/9] River Crossing (Difficulty: 1)
  Iteration 1/5... ✅ Solved in 2.3s (95% confidence)
  Iteration 2/5... ✅ Solved in 2.1s (97% confidence)
  ...
  
[2/9] Logic Grid (Difficulty: 2)
  ...

Results exported to: puzzle_baseline.json
```

### Actual Behavior
```bash
$ tars demo-puzzle --all --difficulty 3 --benchmark 5 --output json --export puzzle_baseline.json

$  # ← Silent! Nothing printed!
```

---

## Root Cause

**Location**: `src/Tars.Interface.Cli/Program.fs:164-174`

**Problem**: Simple pattern matching doesn't handle complex argument combinations

```fsharp
// Current (BROKEN)
| [| "demo-puzzle" |] -> return PuzzleDemo.listPuzzles ()
| [| "demo-puzzle"; "--all" |] -> return! PuzzleDemo.runAll logger false
| [| "demo-puzzle"; "--all"; "--verbose" |] -> return! PuzzleDemo.runAll logger true
| [| "demo-puzzle"; "--difficulty"; n |] -> ...
| [| "demo-puzzle"; name |] -> return! PuzzleDemo.runByName logger name false

// When args are: [| "demo-puzzle"; "--all"; "--difficulty"; "3"; "--benchmark"; "5"; ... |]
// None of the patterns match!
// Falls through to default case (line 363)
```

**Default case** (line 363-395):
```fsharp
| _ ->
    Tui.showSplashScreen ()
    printfn "Usage:"
    printfn "  tars chat..."
    // ... help text
    return 0
```

**Why it's silent**: The default case SHOULD print help, but something is preventing it (possibly async IO flushing, or early return).

---

## Solution

### Option 1: Proper Argument Parser (Recommended)

Use a real argument parser like `Argu` or create a custom one:

```fsharp
module PuzzleDemoArgs =
    type PuzzleOptions = {
        All: bool
        Difficulty: int option
        PuzzleName: string option
        Benchmark: int option
        Output: string option  // "json" | "table"
        Export: string option  // filename
        Verbose: bool
    }
    
    let parse (args: string[]) : Result<PuzzleOptions, string> =
        // Parse args into structured options
        ...

// In Program.fs:
| args when args.Length > 0 && args.[0] = "demo-puzzle" ->
    match PuzzleDemoArgs.parse (args |> Array.skip 1) with
    | Ok opts -> return! PuzzleDemo.runWithOptions logger opts
    | Error msg ->
        printfn "Error: %s" msg
        return 1
```

### Option 2: Pattern Matching with Guards

```fsharp
| args when args.Length > 0 && args.[0] = "demo-puzzle" ->
    let flags = args |> Array.skip 1 |> Set.ofArray
    let hasAll = flags.Contains("--all")
    let hasDiff = flags.Contains("--difficulty")
    // ... parse other flags
    
    return! PuzzleDemo.runWithFlags logger hasAll hasDiff ...
```

### Option 3: Delegate to PuzzleDemo Parser

```fsharp
| args when args.Length > 0 && args.[0] = "demo-puzzle" ->
    return! PuzzleDemo.run logger (args |> Array.skip 1)
```

---

## Additional Issues

### Missing: Progress Output

Even if args parsed correctly, `PuzzleDemo.runAll` doesn't print progress:

**Needed**:
```fsharp
let runPuzzle (logger: ILogger) (puzzle: Puzzle) (iteration: int) (total: int) =
    printfn "[%d/%d] %s (Difficulty: %d)" iteration total puzzle.Name puzzle.Difficulty
    printfn "  Solving with %s..." solver.Name
    
    let! result = solver.Solve(puzzle)
    
    match result with
    | Correct _ -> printfn "  ✅ Solved in %.1fs (%.0f%% confidence)" elapsed confidence
    | Incorrect _ -> printfn "  ❌ Failed: %s" reason
    | Timeout -> printfn "  ⏱️  Timeout after %ds" timeout
```

### Missing: JSON Export

The `--export` flag isn't implemented at all!

**Needed**:
```fsharp
let exportResults (results: PuzzleResult list) (filename: string) (format: string) =
    match format with
    | "json" ->
        let json = JsonSerializer.Serialize(results, jsonOptions)
        File.WriteAllText(filename, json)
        printfn "✅ Results exported to: %s" filename
    | "csv" ->
        // ... CSV export
    | _ ->
        printfn "❌ Unknown format: %s" format
```

---

## Quick Fix (Temporary)

**For now, use verbose flag explicitly**:
```bash
# This works (but limited)
tars demo-puzzle --all --verbose

# Or run individual puzzles
tars demo-puzzle river-crossing
```

---

## Proper Fix (TODO)

1. ✅ Add `Argu` NuGet package
2. ✅ Define `PuzzleDemoArgs` discriminated union
3. ✅ Parse args properly  
4. ✅ Add progress output to `runPuzzle`
5. ✅ Implement `--benchmark` (multiple iterations)
6. ✅ Implement `--export` (JSON/CSV output)
7. ✅ Add `--output table|json` formatting
8. ✅ Update help text with all options

---

## Test Cases (When Fixed)

```bash
# Test 1: List puzzles
tars demo-puzzle
# Expected: Table of all puzzles

# Test 2: Run all
tars demo-puzzle --all --verbose
# Expected: Progress for each puzzle

# Test 3: Filter by difficulty
tars demo-puzzle --difficulty 3
# Expected: Only difficulty ≤ 3

# Test 4: Benchmark mode
tars demo-puzzle --all --benchmark 10
# Expected: Each puzzle runs 10 times

# Test 5: Export results
tars demo-puzzle --all --export results.json
# Expected: results.json created

# Test 6: Combined
tars demo-puzzle --all --difficulty 3 --benchmark 5 --output json --export baseline.json
# Expected: Filter difficulty, run 5x each, export JSON
```

---

## Impact

**Severity**: HIGH  
**User Impact**: Can't use advanced puzzle benchmarking features  
**Workaround**: Use simple flags only  
**Estimated Fix Time**: 2-3 hours

---

## Related Issues

- #XXX: Add Argu argument parser
- #XXX: Puzzle demo needs progress indicators
- #XXX: Export puzzle results to multiple formats

---

*Reported by: User*  
*Date: 2024-12-24*  
*Status: Confirmed, not yet fixed*
