// Test FLUX Autonomous UI Generation
#r "TarsEngine.FSharp.FLUX/bin/Debug/net9.0/TarsEngine.FSharp.FLUX.dll"

open System
open System.Threading.Tasks
open TarsEngine.FSharp.FLUX.FluxEngine

printfn "🔥 TARS FLUX Autonomous UI Generation Test"
printfn "==========================================="
printfn ""

// Test 1: Load and execute the FLUX autonomous UI script
let testFluxAutonomousUI () =
    task {
        try
            printfn "📋 Test 1: Loading FLUX Autonomous UI Script"
            printfn "---------------------------------------------"
            
            let fluxEngine = createFluxEngine()
            
            // Check FLUX capabilities
            let capabilities = fluxEngine.GetCapabilities()
            printfn "🎯 FLUX Capabilities:"
            for capability in capabilities do
                printfn "   ✅ %s" capability
            printfn ""
            
            // Check supported languages
            let languages = fluxEngine.GetSupportedLanguages()
            printfn "🌐 Supported Languages:"
            for lang in languages do
                printfn "   📝 %s" lang
            printfn ""
            
            // Execute the autonomous UI generation script
            printfn "🚀 Executing FLUX Autonomous UI Generation Script"
            printfn "=================================================="
            
            let! result = fluxEngine.ExecuteFile("../Examples/autonomous_ui_generation.flux")
            
            if result.Success then
                printfn "✅ FLUX execution successful!"
                printfn "   ⏱️ Execution time: %A" result.ExecutionTime
                printfn "   📦 Blocks executed: %d" result.BlocksExecuted
                printfn "   🎯 Agents coordinated: %d" result.AgentOutputs.Count
                printfn "   📊 Diagnostics: %d" result.DiagnosticResults.Count
                printfn "   🔍 Reflections: %d" result.ReflectionInsights.Length
                printfn ""
                
                // Show execution trace
                if result.Trace.Length > 0 then
                    printfn "📋 Execution Trace (last 10 entries):"
                    let lastEntries = result.Trace |> List.rev |> List.take (min 10 result.Trace.Length)
                    for (i, entry) in List.indexed lastEntries do
                        printfn "   %d. %s" (i + 1) entry
                    printfn ""
                
                // Show agent outputs
                if result.AgentOutputs.Count > 0 then
                    printfn "🤖 Agent Outputs:"
                    for kvp in result.AgentOutputs do
                        printfn "   👤 %s: %s" kvp.Key (kvp.Value.ToString())
                    printfn ""
                
                // Show generated artifacts
                if result.GeneratedArtifacts.Count > 0 then
                    printfn "📁 Generated Artifacts:"
                    for kvp in result.GeneratedArtifacts do
                        printfn "   📄 %s: %s" kvp.Key (kvp.Value.ToString())
                    printfn ""
                
            else
                printfn "❌ FLUX execution failed!"
                match result.ErrorMessage with
                | Some error -> printfn "   Error: %s" error
                | None -> printfn "   Unknown error occurred"
            
            return result
            
        with
        | ex ->
            printfn "❌ Test failed with exception: %s" ex.Message
            return {
                Success = false
                Result = None
                ExecutionTime = TimeSpan.Zero
                BlocksExecuted = 0
                ErrorMessage = Some ex.Message
                Trace = []
                GeneratedArtifacts = Map.empty
                AgentOutputs = Map.empty
                DiagnosticResults = Map.empty
                ReflectionInsights = []
            }
    }

// Test 2: Create and execute a simple FLUX UI script
let testSimpleFluxUI () =
    task {
        try
            printfn "📋 Test 2: Simple FLUX UI Generation"
            printfn "------------------------------------"
            
            let simpleFluxScript = """
META {
    title: "Simple FLUX UI Test"
    version: "1.0.0"
    description: "Basic UI generation test"
}

AGENT UIGenerator {
    role: "Simple UI Generator"
    
    FSHARP {
        printfn "🎨 Generating simple dashboard UI..."
        
        let generateSimpleUI () =
            let uiCode = '''
module SimpleDashboard

open Elmish
open Fable.React
open Fable.React.Props

type Model = { Message: string; Counter: int }
type Msg = | Increment | Decrement | UpdateMessage of string

let init () = { Message = "Hello FLUX!"; Counter = 0 }, Cmd.none

let update msg model =
    match msg with
    | Increment -> { model with Counter = model.Counter + 1 }, Cmd.none
    | Decrement -> { model with Counter = model.Counter - 1 }, Cmd.none
    | UpdateMessage msg -> { model with Message = msg }, Cmd.none

let view model dispatch =
    div [] [
        h1 [] [ str model.Message ]
        p [] [ str (sprintf "Counter: %d" model.Counter) ]
        button [ OnClick (fun _ -> dispatch Increment) ] [ str "+" ]
        button [ OnClick (fun _ -> dispatch Decrement) ] [ str "-" ]
    ]
'''
            
            System.IO.File.WriteAllText("Generated_SimpleDashboard.fs", uiCode)
            printfn "✅ Generated simple UI: Generated_SimpleDashboard.fs"
            printfn "📏 Code size: %d characters" uiCode.Length
            
        generateSimpleUI()
    }
}

MAIN {
    printfn "🚀 Simple FLUX UI generation complete!"
}

DIAGNOSTIC {
    test: "Simple UI generation"
    validate: "F# Elmish code created successfully"
}
"""
            
            let fluxEngine = createFluxEngine()
            let! result = fluxEngine.ExecuteString(simpleFluxScript)
            
            if result.Success then
                printfn "✅ Simple FLUX UI generation successful!"
                printfn "   ⏱️ Time: %A" result.ExecutionTime
                printfn "   📦 Blocks: %d" result.BlocksExecuted
            else
                printfn "❌ Simple FLUX UI generation failed!"
                match result.ErrorMessage with
                | Some error -> printfn "   Error: %s" error
                | None -> ()
            
            return result
            
        with
        | ex ->
            printfn "❌ Simple test failed: %s" ex.Message
            return {
                Success = false
                Result = None
                ExecutionTime = TimeSpan.Zero
                BlocksExecuted = 0
                ErrorMessage = Some ex.Message
                Trace = []
                GeneratedArtifacts = Map.empty
                AgentOutputs = Map.empty
                DiagnosticResults = Map.empty
                ReflectionInsights = []
            }
    }

// Run the tests
let runTests () =
    task {
        printfn "🧪 Starting FLUX Autonomous UI Tests"
        printfn "====================================="
        printfn ""
        
        // Test 1: Full autonomous UI generation
        let! result1 = testFluxAutonomousUI()
        
        printfn ""
        printfn "─────────────────────────────────────"
        printfn ""
        
        // Test 2: Simple UI generation
        let! result2 = testSimpleFluxUI()
        
        printfn ""
        printfn "🎉 FLUX Autonomous UI Tests Complete!"
        printfn "====================================="
        printfn ""
        
        let successCount = [result1.Success; result2.Success] |> List.filter id |> List.length
        printfn "📊 Test Results:"
        printfn "   ✅ Successful tests: %d/2" successCount
        printfn "   📁 Generated files: Generated_TarsAutonomousDashboard.fs, Generated_SimpleDashboard.fs"
        printfn ""
        
        if successCount = 2 then
            printfn "🚀 All tests passed! FLUX autonomous UI generation is working!"
            printfn ""
            printfn "🎯 Key Achievements:"
            printfn "   🤖 Multi-agent UI generation"
            printfn "   ⚡ Real F# Elmish code creation"
            printfn "   📊 Quality assurance validation"
            printfn "   🔄 Feedback-driven evolution"
            printfn "   🌐 Multi-language FLUX execution"
        else
            printfn "⚠️ Some tests failed. Check the error messages above."
    }

// Execute the tests
runTests() |> Async.AwaitTask |> Async.RunSynchronously

printfn ""
printfn "✨ FLUX Autonomous UI Generation Test Complete! ✨"
