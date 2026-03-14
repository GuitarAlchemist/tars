module TARS.Programming.Validation.ProductionIntegration

open System
open System.IO

type ProductionIntegrationValidator() =

    member _.ValidateProductionDeployment() =
        let paths =
            [ "production/metascript-ecosystem"
              "production/autonomous-improvement"
              "production/blue-green-evolution" ]

        let results =
            paths
            |> List.map (fun path ->
                let exists = Directory.Exists path
                let fileCount = if exists then Directory.GetFiles(path, "*", SearchOption.AllDirectories).Length else 0
                printfn "  - %s: %s (%d files)" path (if exists then "present" else "missing") fileCount
                exists && fileCount > 0)

        List.forall id results

    member _.ValidateTarsCLIIntegration() =
        Directory.Exists "src/TarsEngine.FSharp.Cli"

    member _.ValidateFLUXIntegration() =
        let recursive = ".specify/meta/tier4/recursive-loop.trsx"
        let release = ".specify/meta/tier4/release-train.trsx"
        File.Exists recursive && File.Exists release

    member _.ValidateBlueGreenEnvironment() =
        Directory.Exists "production/blue-green-evolution"

    member _.ValidateMonitoringSystem() =
        Directory.Exists "production/learning-monitoring"
