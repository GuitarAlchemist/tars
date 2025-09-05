#r "TarsEngine.FSharp.Core.dll"
#r "Microsoft.Extensions.Logging.dll"
#r "Microsoft.Extensions.Logging.Console.dll"

open System
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.TarsEvolutionEngine

let loggerFactory = LoggerFactory.Create(fun builder -> builder.AddConsole() |> ignore)
let logger = loggerFactory.CreateLogger<TarsEvolutionEngineService>()

printfn "🧬 TARS REAL EVOLUTION SESSION STARTING"
printfn "======================================="

let engine = TarsEvolutionEngineService(logger)

// Run evolution session
let evolutionTask = engine.RunEvolutionSession("/app", None)
let result = evolutionTask |> Async.RunSynchronously

printfn "\n📊 EVOLUTION RESULTS:"
printfn "Session ID: %s" result.SessionId
printfn "Duration: %d ms" result.TotalDurationMs
printfn "Success: %b" result.OverallSuccess
printfn "Projects Analyzed: %d" result.ProjectsAnalyzed
printfn "Improvements Applied: %d" result.ImprovementsApplied

printfn "\n🔄 Steps:"
for step in result.Steps do
    let status = if step.Success then "✅" else "❌"
    printfn "  %s %s (%d ms)" status step.StepName step.ExecutionTimeMs

printfn "\n🎉 TARS Evolution Complete!"
