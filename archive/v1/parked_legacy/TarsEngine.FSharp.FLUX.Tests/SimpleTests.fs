namespace TarsEngine.FSharp.FLUX.Tests

open System
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.FLUX.Ast.FluxAst
open TarsEngine.FSharp.FLUX.FluxEngine

/// Simple tests to validate FLUX system
module SimpleTests =
    
    [<Fact>]
    let ``FLUX Engine can be created`` () =
        // Arrange & Act
        let engine = FluxEngine()
        
        // Assert
        engine |> should not' (equal null)
    
    [<Fact>]
    let ``FLUX Engine returns supported languages`` () =
        // Arrange
        let engine = FluxEngine()
        
        // Act
        let languages = engine.GetSupportedLanguages()
        
        // Assert
        languages |> should not' (be Empty)
        languages |> should contain "FSHARP"
    
    [<Fact>]
    let ``FLUX Engine returns capabilities`` () =
        // Arrange
        let engine = FluxEngine()
        
        // Act
        let capabilities = engine.GetCapabilities()
        
        // Assert
        capabilities |> should not' (be Empty)
        capabilities |> should contain "Multi-language execution"
    
    [<Fact>]
    let ``FLUX Engine can create test script`` () =
        // Arrange
        let engine = FluxEngine()
        
        // Act
        let testScript = engine.CreateTestScript()
        
        // Assert
        testScript |> should not' (be EmptyString)
        testScript.Contains("META") |> should equal true
    
    [<Fact>]
    let ``FLUX Engine can execute simple script`` () =
        async {
            // Arrange
            let engine = FluxEngine()
            let simpleScript = """
META {
    title: "Simple Test"
    version: "1.0.0"
}

FSHARP {
    let x = 42
    printfn "The answer is %d" x
}
"""
            
            // Act
            let! result = engine.ExecuteString(simpleScript) |> Async.AwaitTask
            
            // Assert
            result |> should not' (equal null)
            result.Success |> should equal true
            result.BlocksExecuted |> should be (greaterThan 0)
        }
    
    [<Fact>]
    let ``FluxValue can represent different types`` () =
        // Arrange & Act
        let stringValue = StringValue "test"
        let numberValue = NumberValue 42.0
        let boolValue = BooleanValue true
        let nullValue = NullValue
        
        // Assert
        match stringValue with
        | StringValue s -> s |> should equal "test"
        | _ -> failwith "Expected StringValue"
        
        match numberValue with
        | NumberValue n -> n |> should equal 42.0
        | _ -> failwith "Expected NumberValue"
        
        match boolValue with
        | BooleanValue b -> b |> should equal true
        | _ -> failwith "Expected BooleanValue"
        
        match nullValue with
        | NullValue -> () // Success
        | _ -> failwith "Expected NullValue"
    
    [<Fact>]
    let ``FLUX Parser can parse simple script`` () =
        // Arrange
        let engine = FluxEngine()
        let simpleScript = """
META {
    title: "Parser Test"
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
