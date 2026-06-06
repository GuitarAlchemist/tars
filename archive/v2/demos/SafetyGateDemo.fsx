// TARS v2 SafetyGate Demo
// This demonstrates the safety checking pipeline

open System
open Tars.Kernel

[<EntryPoint>]
let main argv =
    printfn
        """
╭──────────────────────────────────────────────────────╮
│  TARS v2 SafetyGate Demonstration                    │
│  Phase 6.0: Architecture Hardening                   │
╰──────────────────────────────────────────────────────╯
"""

    let safetyGate = SafetyGate() :> ISafetyGate

    // Test 1: Static Check - Clean Code
    printfn "\n🔍 Test 1: Static Check (Clean Code)"
    printfn "Code: \"let x = 42\""
    let cleanCode = "let x = 42"
    let result1 = safetyGate.CheckStatic(cleanCode) |> Async.RunSynchronously

    match result1 with
    | Passed -> printfn "✅ Result: PASSED"
    | Failed reason -> printfn "❌ Result: FAILED - %s" reason

    // Test 2: Static Check - Code with TODO
    printfn "\n🔍 Test 2: Static Check (Code with TODO)"
    printfn "Code: \"let x = 42 // TODO: optimize\""
    let todoCode = "let x = 42 // TODO: optimize"
    let result2 = safetyGate.CheckStatic(todoCode) |> Async.RunSynchronously

    match result2 with
    | Passed -> printfn "✅ Result: PASSED"
    | Failed reason -> printfn "❌ Result: FAILED - %s" reason

    // Test 3: Sandbox Check - Safe Python Code
    printfn "\n🐳 Test 3: Sandbox Check (Safe Python)"
    printfn "Code: \"print('Hello from TARS sandbox!')\""
    let safePython = "print('Hello from TARS sandbox!')"
    let result3 = safetyGate.CheckSandbox(safePython) |> Async.RunSynchronously

    match result3 with
    | Passed -> printfn "✅ Result: PASSED - Code executed safely in Docker"
    | Failed reason -> printfn "❌ Result: FAILED - %s" reason

    // Test 4: Sandbox Check - Code that fails
    printfn "\n🐳 Test 4: Sandbox Check (Code with Error)"
    printfn "Code: \"import sys; sys.exit(1)\""
    let errorPython = "import sys; sys.exit(1)"
    let result4 = safetyGate.CheckSandbox(errorPython) |> Async.RunSynchronously

    match result4 with
    | Passed -> printfn "✅ Result: PASSED"
    | Failed reason -> printfn "❌ Result: FAILED - %s" reason

    printfn
        """
╭──────────────────────────────────────────────────────╮
│  SafetyGate Demo Complete                            │
│  All checks demonstrate Phase 6.0 capabilities       │
╰──────────────────────────────────────────────────────╯
"""

    0
