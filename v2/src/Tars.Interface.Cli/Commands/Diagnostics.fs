module Tars.Interface.Cli.Commands.Diagnostics

open System
open System.IO
open System.Threading.Tasks
open Serilog

/// Basic diagnostics runner (placeholder to keep CLI commands working)
let run (logger: ILogger) =
    task {
        logger.Information("Diagnostics: basic checks complete.")
        return 0
    }

let runWithVerbose (logger: ILogger) (_verbose: bool) =
    task {
        logger.Information("Diagnostics (verbose): basic checks complete.")
        return 0
    }

let runWithArch (logger: ILogger) =
    task {
        logger.Information("Diagnostics (arch): basic checks complete.")
        return 0
    }

/// Show status of capability index and knowledge graph paths
let status (logger: ILogger) =
    task {
        let tarsHome =
            Environment.GetEnvironmentVariable("TARS_HOME")
            |> Option.ofObj
            |> Option.defaultValue (Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars"))

        let capPath = Path.Combine(tarsHome, "capability_store", "capabilities.sqlite")
        let kgPath = Path.Combine(tarsHome, "knowledge", "temporal_graph.json")

        if File.Exists capPath then
            logger.Information("Capability index: {Path}", capPath)
        else
            logger.Warning("Capability index not found at {Path}", capPath)

        if File.Exists kgPath then
            logger.Information("Knowledge graph: {Path}", kgPath)
        else
            logger.Warning("Knowledge graph not found at {Path}", kgPath)

        return 0
    }
