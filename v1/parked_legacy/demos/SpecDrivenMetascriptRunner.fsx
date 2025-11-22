#r "nuget: Microsoft.Extensions.Logging.Abstractions, 9.0.0"
#r "../TarsEngine.FSharp.Core/bin/Release/net9.0/TarsEngine.FSharp.Core.dll"

open System
open System.IO
open Microsoft.Extensions.Logging.Abstractions
open TarsEngine.FSharp.Core.Closures
open TarsEngine.FSharp.Core.Closures.UnifiedEvolutionaryClosureFactory
open TarsEngine.FSharp.Core.Services.MetascriptClosureIntegrationService

let factory = UnifiedEvolutionaryClosureFactory(NullLogger<UnifiedEvolutionaryClosureFactory>.Instance)
let service = MetascriptClosureIntegrationService(NullLogger<MetascriptClosureIntegrationService>.Instance, factory)

let specPath =
    Path.Combine(__SOURCE_DIRECTORY__, "..", "specs", "demos", "tiered_dynamic_spec.md")
    |> Path.GetFullPath

if not (File.Exists(specPath)) then
    failwithf "Spec file not found: %s" specPath

let commandLine =
    sprintf "CLOSURE_CREATE DYNAMIC_METASCRIPT \"TieredDynamic\" spec=\"%s\"" specPath

let command =
    match service.ParseClosureCommand(commandLine) with
    | Some cmd -> cmd
    | None -> failwithf "Failed to parse command: %s" commandLine

let result =
    service.ExecuteClosureCommand(command)
    |> Async.RunSynchronously

printfn "Output Summary:\n%s\n" result.OutputSummary

printfn "Evolution Data:"
result.EvolutionData
|> Map.toList
|> List.sortBy fst
|> List.iter (fun (key, value) -> printfn " - %s = %O" key value)

printfn "\nArtifacts:"
result.Artifacts |> List.iter (fun artifact -> printfn " - %s" artifact)

printfn "\nNext Steps:"
result.NextSteps |> List.iter (fun step -> printfn " - %s" step)

match result.Output with
| Some (:? UnifiedEvolutionaryClosureFactory.DynamicMetascriptSummary as summary) ->
    printfn "\nSpawned Agents:"
    summary.SpawnedAgents |> List.iter (fun (agent, count, strategy) -> printfn " - %A x%d @ %A" agent count strategy)
    printfn "\nConnections:"
    summary.Connections |> List.iter (fun (source, target, label) -> printfn " - %s -> %s (%s)" source target label)
| _ -> ()
