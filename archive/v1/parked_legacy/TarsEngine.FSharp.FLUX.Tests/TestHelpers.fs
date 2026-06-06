namespace TarsEngine.FSharp.FLUX.Tests

open System
open System.IO
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open Unquote
open TarsEngine.FSharp.FLUX.Ast.FluxAst
open TarsEngine.FSharp.FLUX.Parser.FluxParser
open TarsEngine.FSharp.FLUX.Execution.FluxRuntime
open TarsEngine.FSharp.FLUX.FluxEngine

/// Test helpers and utilities for FLUX testing
module TestHelpers =
    
    /// Create a temporary test file
    let createTempFile (content: string) (extension: string) : string =
        let tempFile = Path.GetTempFileName() + extension
        File.WriteAllText(tempFile, content)
        tempFile
    
    /// Clean up temporary file
    let cleanupTempFile (filePath: string) : unit =
        if File.Exists(filePath) then
            File.Delete(filePath)
    
    /// Create test FLUX script
    let createTestFluxScript (blocks: FluxBlock list) : FluxScript =
        {
            Blocks = blocks
            FileName = None
            ParsedAt = DateTime.UtcNow
            Version = "1.0.0"
            Metadata = Map.empty
        }
    
    /// Create test meta block
    let createTestMetaBlock (properties: (string * FluxValue) list) : FluxBlock =
        MetaBlock {
            Properties = properties |> List.map (fun (name, value) -> { Name = name; Value = value })
            LineNumber = 1
        }
    
    /// Create test language block
    let createTestLanguageBlock (language: string) (content: string) : FluxBlock =
        LanguageBlock {
            Language = language
            Content = content
            LineNumber = 1
            Variables = Map.empty
        }
    
    /// Create test agent block
    let createTestAgentBlock (name: string) (properties: AgentProperty list) (langBlocks: LanguageBlock list) : FluxBlock =
        AgentBlock {
            Name = name
            Properties = properties
            LanguageBlocks = langBlocks
            LineNumber = 1
        }
    
    /// Create test diagnostic block
    let createTestDiagnosticBlock (operations: DiagnosticOperation list) : FluxBlock =
        DiagnosticBlock {
            Operations = operations
            LineNumber = 1
        }
    
    /// Create test reflection block
    let createTestReflectionBlock (operations: ReflectionOperation list) : FluxBlock =
        ReflectionBlock {
            Operations = operations
            LineNumber = 1
        }
    
    /// Create test reasoning block
    let createTestReasoningBlock (content: string) : FluxBlock =
        ReasoningBlock {
            Content = content
            LineNumber = 1
            ThinkingBudget = None
            ReasoningQuality = None
        }
    
    /// Execute FLUX script and return result
    let executeFluxScript (script: FluxScript) : Task<FluxExecutionResult> =
        let engine = FluxExecutionEngine()
        engine.ExecuteScript(script)
    
    /// Execute FLUX script from string
    let executeFluxString (content: string) : Task<FluxExecutionResult> =
        task {
            match parseScript content with
            | Ok script -> return! executeFluxScript script
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
    
    /// Assert execution success
    let assertExecutionSuccess (result: FluxExecutionResult) : unit =
        result.Success |> should equal true
        result.ErrorMessage |> should equal None
    
    /// Assert execution failure
    let assertExecutionFailure (result: FluxExecutionResult) : unit =
        result.Success |> should equal false
        result.ErrorMessage |> should not' (equal None)
    
    /// Assert execution time within bounds
    let assertExecutionTime (maxTime: TimeSpan) (result: FluxExecutionResult) : unit =
        result.ExecutionTime |> should be (lessThan maxTime)
    
    /// Assert blocks executed count
    let assertBlocksExecuted (expectedCount: int) (result: FluxExecutionResult) : unit =
        result.BlocksExecuted |> should equal expectedCount
    
    /// Assert diagnostic result
    let assertDiagnosticResult (key: string) (expected: bool) (result: FluxExecutionResult) : unit =
        result.DiagnosticResults |> Map.containsKey key |> should equal true
        result.DiagnosticResults.[key] |> should equal expected
    
    /// Assert reflection insight exists
    let assertReflectionInsight (insight: string) (result: FluxExecutionResult) : unit =
        result.ReflectionInsights |> List.contains insight |> should equal true
    
    /// Assert agent output exists
    let assertAgentOutput (agentName: string) (result: FluxExecutionResult) : unit =
        result.AgentOutputs |> Map.containsKey agentName |> should equal true
    
    /// Assert generated artifact exists
    let assertGeneratedArtifact (artifactName: string) (result: FluxExecutionResult) : unit =
        result.GeneratedArtifacts |> Map.containsKey artifactName |> should equal true
    
    /// Create performance test data
    let createPerformanceTestData (blockCount: int) : FluxScript =
        let blocks = 
            [1..blockCount]
            |> List.map (fun i -> 
                createTestLanguageBlock "FSHARP" (sprintf "let x%d = %d" i i))
        createTestFluxScript blocks
    
    /// Measure execution time
    let measureExecutionTime (action: unit -> 'T) : 'T * TimeSpan =
        let stopwatch = System.Diagnostics.Stopwatch.StartNew()
        let result = action()
        stopwatch.Stop()
        (result, stopwatch.Elapsed)
    
    /// Create test FLUX file content
    let createTestFluxFileContent () : string =
        """
META {
    title: "Test FLUX Script"
    version: "1.0.0"
    description: "Integration test script"
}

FSHARP {
    let message = "Hello from FLUX!"
    printfn "%s" message
    let result = 2 + 3
    printfn "Result: %d" result
}

AGENT TestAgent {
    role: "Test Agent"
    capabilities: ["testing", "validation"]
    
    FSHARP {
        printfn "Agent executing F# code"
        let agentResult = "Agent completed successfully"
        printfn "%s" agentResult
    }
}

DIAGNOSTIC {
    test: "Basic functionality test"
    validate: "System is operational"
    benchmark: "Performance test"
}

REASONING {
    This is a comprehensive test script that validates
    the core functionality of the FLUX metascript system.
    It tests language execution, agent orchestration,
    diagnostic capabilities, and reasoning blocks.
}
"""
    
    /// Property-based test generators using FsCheck
    module Generators =
        open FsCheck
        
        /// Generate random FluxValue
        let fluxValueGen : Gen<FluxValue> =
            Gen.oneof [
                Gen.map StringValue Arb.generate<string>
                Gen.map NumberValue Arb.generate<float>
                Gen.map BooleanValue Arb.generate<bool>
                Gen.constant NullValue
            ]
        
        /// Generate random language name
        let languageGen : Gen<string> =
            Gen.elements ["FSHARP"; "CSHARP"; "PYTHON"; "JAVASCRIPT"; "SQL"]
        
        /// Generate random language block
        let languageBlockGen : Gen<LanguageBlock> =
            Gen.map2 (fun lang content -> {
                Language = lang
                Content = content
                LineNumber = 1
                Variables = Map.empty
            }) languageGen Arb.generate<string>
        
        /// Generate random FLUX script
        let fluxScriptGen : Gen<FluxScript> =
            Gen.map (fun blocks -> {
                Blocks = blocks |> List.map LanguageBlock
                FileName = None
                ParsedAt = DateTime.UtcNow
                Version = "1.0.0"
                Metadata = Map.empty
            }) (Gen.listOf languageBlockGen)

    /// Benchmark helpers using NBomber
    module Benchmarks =
        open NBomber.FSharp
        
        /// Create FLUX execution scenario
        let createFluxExecutionScenario (name: string) (script: FluxScript) =
            Scenario.create name (fun context ->
                task {
                    let! result = executeFluxScript script
                    return if result.Success then Response.ok() else Response.fail()
                }
            )
        
        /// Run performance benchmark
        let runFluxBenchmark (scenarios: NBomber.Contracts.Scenario list) =
            NBomberRunner
                .registerScenarios(scenarios)
                .run()

    // Test helpers loaded successfully
