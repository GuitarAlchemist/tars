// Main.fs - Application entry point and Curriculum agent wiring
module Main
open System
open ToolFactory
open Types
open AgentFramework

/// Initialize the Curriculum agent with the set of available tools
let initializeCurriculumAgent () =
    // List of tool names the agent can invoke
    let availableTools = [ "run_static_analysis" ]
    // Resolve tool instances via the factory
    let toolInstances =
        availableTools
        |> List.choose (fun name ->
            match createTool name with
            | Ok t -> Some t
            | Error _ -> None)
    // Create the agent (actual constructor may differ)
    let agent = CurriculumAgent.create toolInstances
    agent

[<EntryPoint>]
let main argv =
    printfn "Starting TARS with Curriculum agent..."
    let curriculum = initializeCurriculumAgent()
    // Example invocation of static analysis at startup (no root path => repo root)
    match curriculum.InvokeTool "run_static_analysis" (box null) with
    | Ok resultObj ->
        let json = resultObj :?> string
        printfn "Static analysis report: %s" json
        // The Curriculum agent will parse this JSON and generate refactor tasks automatically
        0
    | Error err ->
        eprintfn "Error invoking run_static_analysis: %s" err
        1
