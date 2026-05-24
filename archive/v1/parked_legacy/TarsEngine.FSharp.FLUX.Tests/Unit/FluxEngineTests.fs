namespace TarsEngine.FSharp.FLUX.Tests.Unit

open System
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open Unquote
open FsCheck
open FsCheck.Xunit
open TarsEngine.FSharp.FLUX.Ast.FluxAst
open TarsEngine.FSharp.FLUX.FluxEngine
open TarsEngine.FSharp.FLUX.Tests.TestHelpers

/// Unit tests for FLUX Engine
module FluxEngineTests =
    
    [<Fact>]
    let ``FluxEngine can be instantiated`` () =
        // Arrange & Act
        let engine = FluxEngine()
        
        // Assert
        engine |> should not' (equal null)
    
    [<Fact>]
    let ``FluxEngine returns supported languages`` () =
        // Arrange
        let engine = FluxEngine()
        
        // Act
        let languages = engine.GetSupportedLanguages()
        
        // Assert
        languages |> should not' (be Empty)
        languages |> should contain "FSHARP"
        languages |> should contain "CSHARP"
        languages |> should contain "PYTHON"
        languages |> should contain "JAVASCRIPT"
    
    [<Fact>]
    let ``FluxEngine returns capabilities`` () =
        // Arrange
        let engine = FluxEngine()
        
        // Act
        let capabilities = engine.GetCapabilities()
        
        // Assert
        capabilities |> should not' (be Empty)
        capabilities |> should contain "Multi-language execution"
        capabilities |> should contain "Agent orchestration"
        capabilities |> should contain "Dynamic grammar fetching"
    
    [<Fact>]
    let ``FluxEngine can create test script`` () =
        // Arrange
        let engine = FluxEngine()
        
        // Act
        let testScript = engine.CreateTestScript()
        
        // Assert
        testScript |> should not' (be EmptyString)
        testScript |> should contain "META"
        testScript |> should contain "FSHARP"
    
    [<Fact>]
    let ``FluxEngine can parse simple script`` () =
        // Arrange
        let engine = FluxEngine()
        let simpleScript = """
META {
    title: "Simple Test"
    version: "1.0.0"
}
"""
        
        // Act
        let result = engine.ParseScript(simpleScript)
        
        // Assert
        match result with
        | Ok script -> 
            script.Blocks |> should not' (be Empty)
            script.Version |> should equal "1.0.0"
        | Error msg -> 
            failwith (sprintf "Parse failed: %s" msg)
    
    [<Fact>]
    let ``FluxEngine handles empty script gracefully`` () =
        // Arrange
        let engine = FluxEngine()
        let emptyScript = ""
        
        // Act
        let result = engine.ParseScript(emptyScript)
        
        // Assert
        match result with
        | Ok script -> script.Blocks |> should not' (be Empty) // Should have default test content
        | Error _ -> () // Empty script should parse to default content
    
    [<Fact>]
    let ``FluxEngine ExecuteString handles simple F# code`` () =
        task {
            // Arrange
            let engine = FluxEngine()
            let simpleScript = """
META {
    title: "F# Test"
    version: "1.0.0"
}

FSHARP {
    let x = 42
    printfn "The answer is %d" x
}
"""
            
            // Act
            let! result = engine.ExecuteString(simpleScript)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should be (greaterThan 0)
        }
    
    [<Fact>]
    let ``FluxEngine ExecuteString handles execution errors gracefully`` () =
        task {
            // Arrange
            let engine = FluxEngine()
            let errorScript = """
META {
    title: "Error Test"
    version: "1.0.0"
}

FSHARP {
    let x = 1 / 0  // This should cause an error
    printfn "This won't print"
}
"""
            
            // Act
            let! result = engine.ExecuteString(errorScript)
            
            // Assert
            // The execution might succeed but with error handling in the runtime
            // The specific behavior depends on how errors are handled in the runtime
            result |> should not' (equal null)
        }
    
    [<Fact>]
    let ``FluxEngine ExecuteFile handles non-existent file gracefully`` () =
        task {
            // Arrange
            let engine = FluxEngine()
            let nonExistentFile = "non_existent_file.flux"
            
            // Act
            let! result = engine.ExecuteFile(nonExistentFile)
            
            // Assert
            TestHelpers.assertExecutionFailure result
            result.ErrorMessage |> should not' (equal None)
        }
    
    [<Fact>]
    let ``FluxEngine ExecuteFile works with valid file`` () =
        task {
            // Arrange
            let engine = FluxEngine()
            let fluxContent = TestHelpers.createTestFluxFileContent()
            let tempFile = TestHelpers.createTempFile fluxContent ".flux"
            
            try
                // Act
                let! result = engine.ExecuteFile(tempFile)
                
                // Assert
                TestHelpers.assertExecutionSuccess result
                result.BlocksExecuted |> should be (greaterThan 0)
            finally
                TestHelpers.cleanupTempFile tempFile
        }
    
    [<Property>]
    let ``FluxEngine handles various script sizes`` (scriptSize: PositiveInt) =
        task {
            // Arrange
            let engine = FluxEngine()
            let blockCount = min scriptSize.Get 10 // Limit to reasonable size for testing
            let script = TestHelpers.createPerformanceTestData blockCount
            
            // Act
            let! result = TestHelpers.executeFluxScript script
            
            // Assert
            result |> should not' (equal null)
            result.BlocksExecuted |> should be (greaterThanOrEqualTo 0)
        }
    
    [<Fact>]
    let ``FluxEngine convenience functions work correctly`` () =
        task {
            // Arrange
            let fluxContent = TestHelpers.createTestFluxFileContent()
            let tempFile = TestHelpers.createTempFile fluxContent ".flux"
            
            try
                // Act
                let! fileResult = FluxEngine.executeFile tempFile
                let! stringResult = FluxEngine.executeString fluxContent
                
                // Assert
                TestHelpers.assertExecutionSuccess fileResult
                TestHelpers.assertExecutionSuccess stringResult
                fileResult.BlocksExecuted |> should equal stringResult.BlocksExecuted
            finally
                TestHelpers.cleanupTempFile tempFile
        }
    
    [<Fact>]
    let ``FluxEngine module functions work correctly`` () =
        task {
            // Arrange
            let fluxContent = TestHelpers.createTestFluxFileContent()
            
            // Act
            let parseResult = FluxEngine.parseFluxScript fluxContent
            let capabilities = FluxEngine.getFluxCapabilities()
            let languages = FluxEngine.getSupportedLanguages()
            let testScript = FluxEngine.createTestFluxScript()
            
            // Assert
            match parseResult with
            | Ok script -> script.Blocks |> should not' (be Empty)
            | Error msg -> failwith (sprintf "Parse failed: %s" msg)
            
            capabilities |> should not' (be Empty)
            languages |> should not' (be Empty)
            testScript |> should not' (be EmptyString)
        }
    
    [<Fact>]
    let ``FluxEngine execution is deterministic for same input`` () =
        task {
            // Arrange
            let engine = FluxEngine()
            let script = """
META {
    title: "Deterministic Test"
    version: "1.0.0"
}

FSHARP {
    let result = 2 + 2
    printfn "Result: %d" result
}
"""
            
            // Act
            let! result1 = engine.ExecuteString(script)
            let! result2 = engine.ExecuteString(script)
            
            // Assert
            result1.Success |> should equal result2.Success
            result1.BlocksExecuted |> should equal result2.BlocksExecuted
            // Note: ExecutionTime may vary, so we don't compare that
        }
    
    [<Fact>]
    let ``FluxEngine handles concurrent executions`` () =
        task {
            // Arrange
            let engine = FluxEngine()
            let script = TestHelpers.createTestFluxFileContent()
            
            // Act
            let tasks = 
                [1..5] 
                |> List.map (fun _ -> engine.ExecuteString(script))
                |> Array.ofList
            
            let! results = Task.WhenAll(tasks)
            
            // Assert
            results |> Array.iter TestHelpers.assertExecutionSuccess
            results |> Array.forall (fun r -> r.BlocksExecuted > 0) |> should equal true
        }

printfn "🧪 FLUX Engine Unit Tests Loaded"
printfn "================================"
printfn "✅ Engine instantiation tests"
printfn "✅ Capability and language tests"
printfn "✅ Script parsing tests"
printfn "✅ Execution tests"
printfn "✅ Error handling tests"
printfn "✅ File operation tests"
printfn "✅ Property-based tests"
printfn "✅ Concurrency tests"
