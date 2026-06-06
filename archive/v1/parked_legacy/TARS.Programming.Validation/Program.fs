module TARS.Programming.Validation.Program

open System
open TARS.Programming.Validation.ValidationSuite

[<EntryPoint>]
let main _ =
    let result = runSuite ()
    printfn "\n?? SUMMARY"
    printfn "  Programming learning: %s" (if result.ProgrammingLearning then "PASS" else "FAIL")
    printfn "  Metascript evolution: %s" (if result.MetascriptEvolution then "PASS" else "FAIL")
    printfn "  Autonomous improvement score: %.1f" result.AutonomousImprovement
    printfn "  Production integration: %s" (if result.ProductionIntegration then "PASS" else "FAIL")

    if result.ProgrammingLearning && result.MetascriptEvolution && result.AutonomousImprovement >= 50.0 && result.ProductionIntegration then 0 else 1
