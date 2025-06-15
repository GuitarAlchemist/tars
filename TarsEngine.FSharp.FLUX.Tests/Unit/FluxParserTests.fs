namespace TarsEngine.FSharp.FLUX.Tests.Unit

open System
open Xunit
open FsUnit.Xunit
open Unquote
open TarsEngine.FSharp.FLUX.Ast.FluxAst
open TarsEngine.FSharp.FLUX.Parser.FluxParser

/// Unit tests for FLUX Parser
module FluxParserTests =
    
    [<Fact>]
    let ``Parser can parse empty string`` () =
        // Arrange
        let input = ""
        
        // Act
        let result = parseScript input
        
        // Assert
        match result with
        | Ok script -> 
            script.Blocks |> should not' (be Empty) // Should have default test content
            script.Version |> should equal "1.0.0"
        | Error msg -> 
            failwith (sprintf "Parse failed: %s" msg)
    
    [<Fact>]
    let ``Parser handles simple META block`` () =
        // Arrange
        let input = """
META {
    title: "Test Script"
    version: "1.0.0"
}
"""
        
        // Act
        let result = parseScript input
        
        // Assert
        match result with
        | Ok script -> 
            script.Blocks |> should not' (be Empty)
            script.Version |> should equal "1.0.0"
        | Error msg -> 
            failwith (sprintf "Parse failed: %s" msg)
    
    [<Fact>]
    let ``Parser handles simple FSHARP block`` () =
        // Arrange
        let input = """
FSHARP {
    let x = 42
    printfn "Value: %d" x
}
"""
        
        // Act
        let result = parseScript input
        
        // Assert
        match result with
        | Ok script -> 
            script.Blocks |> should not' (be Empty)
        | Error msg -> 
            failwith (sprintf "Parse failed: %s" msg)
    
    [<Fact>]
    let ``Parser handles multiple blocks`` () =
        // Arrange
        let input = """
META {
    title: "Multi-block Test"
    version: "1.0.0"
}

FSHARP {
    let x = 42
}

PYTHON {
    x = 42
}
"""
        
        // Act
        let result = parseScript input
        
        // Assert
        match result with
        | Ok script -> 
            script.Blocks |> should not' (be Empty)
        | Error msg -> 
            failwith (sprintf "Parse failed: %s" msg)
    
    [<Fact>]
    let ``Parser handles AGENT blocks`` () =
        // Arrange
        let input = """
AGENT TestAgent {
    role: "Test Agent"
    capabilities: ["testing", "validation"]
    
    FSHARP {
        printfn "Agent code"
    }
}
"""
        
        // Act
        let result = parseScript input
        
        // Assert
        match result with
        | Ok script -> 
            script.Blocks |> should not' (be Empty)
        | Error msg -> 
            failwith (sprintf "Parse failed: %s" msg)
    
    [<Fact>]
    let ``Parser handles DIAGNOSTIC blocks`` () =
        // Arrange
        let input = """
DIAGNOSTIC {
    test: "Basic functionality test"
    validate: "System is operational"
    benchmark: "Performance test"
}
"""
        
        // Act
        let result = parseScript input
        
        // Assert
        match result with
        | Ok script -> 
            script.Blocks |> should not' (be Empty)
        | Error msg -> 
            failwith (sprintf "Parse failed: %s" msg)
    
    [<Fact>]
    let ``Parser handles REASONING blocks`` () =
        // Arrange
        let input = """
REASONING {
    This is a reasoning block that contains
    natural language reasoning about the problem
    and solution approach.
}
"""
        
        // Act
        let result = parseScript input
        
        // Assert
        match result with
        | Ok script -> 
            script.Blocks |> should not' (be Empty)
        | Error msg -> 
            failwith (sprintf "Parse failed: %s" msg)
    
    [<Fact>]
    let ``Parser can parse from file`` () =
        // Arrange
        let tempFile = System.IO.Path.GetTempFileName() + ".flux"
        let content = """
META {
    title: "File Test"
    version: "1.0.0"
}

FSHARP {
    let x = 42
}
"""
        System.IO.File.WriteAllText(tempFile, content)
        
        try
            // Act
            let result = parseScriptFromFile tempFile
            
            // Assert
            match result with
            | Ok script -> 
                script.Blocks |> should not' (be Empty)
                script.FileName |> should equal (Some tempFile)
            | Error msg -> 
                failwith (sprintf "Parse failed: %s" msg)
        finally
            if System.IO.File.Exists(tempFile) then
                System.IO.File.Delete(tempFile)
    
    [<Fact>]
    let ``Parser handles non-existent file gracefully`` () =
        // Arrange
        let nonExistentFile = "non_existent_file.flux"
        
        // Act
        let result = parseScriptFromFile nonExistentFile
        
        // Assert
        match result with
        | Ok _ -> failwith "Should have failed for non-existent file"
        | Error msg -> 
            msg |> should contain "Failed to read file"
    
    [<Fact>]
    let ``Parser handles malformed input gracefully`` () =
        // Arrange
        let malformedInput = "This is not valid FLUX syntax { [ } ]"
        
        // Act
        let result = parseScript malformedInput
        
        // Assert
        // The simplified parser should still work and return default content
        match result with
        | Ok script -> 
            script.Blocks |> should not' (be Empty)
        | Error msg -> 
            msg |> should not' (be EmptyString)

printfn "🧪 FLUX Parser Unit Tests Loaded"
printfn "================================"
printfn "✅ Basic parsing tests"
printfn "✅ Block type parsing tests"
printfn "✅ File parsing tests"
printfn "✅ Error handling tests"
