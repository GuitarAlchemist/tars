namespace TarsEngine.FSharp.FLUX

open System
open System.Threading.Tasks
open TarsEngine.FSharp.FLUX.Ast.FluxAst
open TarsEngine.FSharp.FLUX.Parser.FluxParser
open TarsEngine.FSharp.FLUX.Execution.FluxRuntime
open TarsEngine.FSharp.FLUX.Grammar.FluxGrammar

/// FLUX Engine - Main entry point for FLUX language system
module FluxEngine =

    /// FLUX Engine
    type FluxEngine() =

        /// Execute FLUX script from file
        member this.ExecuteFile(filePath: string) : Task<FluxExecutionResult> =
            task {
                match Parser.FluxParser.parseScriptFromFile filePath with
                | Ok script ->
                    let engine = Execution.FluxRuntime.FluxExecutionEngine()
                    return! engine.ExecuteScript(script)
                | Error errorMsg ->
                    return {
                        Success = false
                        Result = None
                        ExecutionTime = TimeSpan.Zero
                        BlocksExecuted = 0
                        ErrorMessage = Some errorMsg
                        Trace = []
                        GeneratedArtifacts = Map.empty
                        AgentOutputs = Map.empty
                        DiagnosticResults = Map.empty
                        ReflectionInsights = []
                    }
            }

        /// Execute FLUX script from string
        member this.ExecuteString(content: string) : Task<FluxExecutionResult> =
            task {
                match Parser.FluxParser.parseScript content with
                | Ok script ->
                    let engine = Execution.FluxRuntime.FluxExecutionEngine()
                    return! engine.ExecuteScript(script)
                | Error errorMsg ->
                    return {
                        Success = false
                        Result = None
                        ExecutionTime = TimeSpan.Zero
                        BlocksExecuted = 0
                        ErrorMessage = Some errorMsg
                        Trace = []
                        GeneratedArtifacts = Map.empty
                        AgentOutputs = Map.empty
                        DiagnosticResults = Map.empty
                        ReflectionInsights = []
                    }
            }

        /// Parse FLUX script
        member this.ParseScript(content: string) : Result<FluxScript, string> =
            Parser.FluxParser.parseScript content

        /// Get supported languages
        member this.GetSupportedLanguages() : string list =
            ["FSHARP"; "CSHARP"; "PYTHON"; "JAVASCRIPT"; "MERMAID"; "SQL"]

        /// Get FLUX capabilities
        member this.GetCapabilities() : string list =
            [
                "Multi-language execution"
                "Agent orchestration"
                "Dynamic grammar fetching"
                "Reflection and reasoning"
                "Diagnostic testing"
                "Vector operations"
                "I/O operations"
                "Computation expression generation"
            ]

        /// Create test script
        member this.CreateTestScript() : string =
            """
META {
    title: "FLUX Test Script"
    version: "1.0.0"
    description: "Test script demonstrating FLUX capabilities"
}

FSHARP {
    let message = "Hello from FLUX!"
    printfn "%s" message
    let result = 2 + 3
    printfn "2 + 3 = %d" result
}

AGENT TestAgent {
    role: "Test Agent"
    capabilities: ["testing", "validation"]

    FSHARP {
        printfn "Agent executing F# code"
    }
}

DIAGNOSTIC {
    test: "Basic functionality test"
    validate: "System is operational"
}

REASONING {
    This is a test script that demonstrates the basic capabilities
    of the FLUX metascript language system.
}
"""

    /// Create FLUX engine instance
    let createFluxEngine() = FluxEngine()

    /// Execute FLUX file (convenience function)
    let executeFile (filePath: string) : Task<FluxExecutionResult> =
        let engine = createFluxEngine()
        engine.ExecuteFile(filePath)

    /// Execute FLUX string (convenience function)
    let executeString (content: string) : Task<FluxExecutionResult> =
        let engine = createFluxEngine()
        engine.ExecuteString(content)

    printfn "🔥 FLUX Engine Loaded"
    printfn "====================="
    printfn "✅ Parser ready"
    printfn "✅ Runtime ready"
    printfn "✅ Grammar system ready"
    printfn "✅ Multi-language support"
    printfn "✅ Internet grammar fetching"
    printfn ""
    printfn "🎯 FLUX - Revolutionary Multi-Modal Metascript Language!"
    printfn "   Ready to execute .flux metascripts!"
