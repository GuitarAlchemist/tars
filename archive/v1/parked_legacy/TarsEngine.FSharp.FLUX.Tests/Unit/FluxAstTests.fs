namespace TarsEngine.FSharp.FLUX.Tests.Unit

open System
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.FLUX.Ast.FluxAst

/// Unit tests for FLUX AST
module FluxAstTests =
    
    [<Fact>]
    let ``FluxValue can represent string values`` () =
        // Arrange & Act
        let stringValue = StringValue "test"
        
        // Assert
        match stringValue with
        | StringValue s -> s |> should equal "test"
        | _ -> failwith "Expected StringValue"
    
    [<Fact>]
    let ``FluxValue can represent number values`` () =
        // Arrange & Act
        let numberValue = NumberValue 42.0
        
        // Assert
        match numberValue with
        | NumberValue n -> n |> should equal 42.0
        | _ -> failwith "Expected NumberValue"
    
    [<Fact>]
    let ``FluxValue can represent boolean values`` () =
        // Arrange & Act
        let boolValue = BooleanValue true
        
        // Assert
        match boolValue with
        | BooleanValue b -> b |> should equal true
        | _ -> failwith "Expected BooleanValue"
    
    [<Fact>]
    let ``FluxValue can represent array values`` () =
        // Arrange & Act
        let arrayValue = ArrayValue [StringValue "a"; NumberValue 1.0]
        
        // Assert
        match arrayValue with
        | ArrayValue items -> 
            items.Length |> should equal 2
            items.[0] |> should equal (StringValue "a")
            items.[1] |> should equal (NumberValue 1.0)
        | _ -> failwith "Expected ArrayValue"
    
    [<Fact>]
    let ``FluxValue can represent object values`` () =
        // Arrange & Act
        let objectValue = ObjectValue (Map.ofList [("key", StringValue "value")])
        
        // Assert
        match objectValue with
        | ObjectValue props -> 
            props.Count |> should equal 1
            props.["key"] |> should equal (StringValue "value")
        | _ -> failwith "Expected ObjectValue"
    
    [<Fact>]
    let ``FluxValue can represent null values`` () =
        // Arrange & Act
        let nullValue = NullValue
        
        // Assert
        match nullValue with
        | NullValue -> () // Success
        | _ -> failwith "Expected NullValue"
    
    [<Fact>]
    let ``LanguageBlock has required properties`` () =
        // Arrange & Act
        let langBlock = {
            Language = "FSHARP"
            Content = "let x = 42"
            LineNumber = 1
            Variables = Map.empty
        }
        
        // Assert
        langBlock.Language |> should equal "FSHARP"
        langBlock.Content |> should equal "let x = 42"
        langBlock.LineNumber |> should equal 1
        langBlock.Variables |> should equal Map.empty
    
    [<Fact>]
    let ``MetaBlock has required properties`` () =
        // Arrange & Act
        let metaBlock = {
            Properties = [{ Name = "title"; Value = StringValue "Test" }]
            LineNumber = 1
        }
        
        // Assert
        metaBlock.Properties.Length |> should equal 1
        metaBlock.Properties.[0].Name |> should equal "title"
        metaBlock.Properties.[0].Value |> should equal (StringValue "Test")
        metaBlock.LineNumber |> should equal 1
    
    [<Fact>]
    let ``AgentBlock has required properties`` () =
        // Arrange & Act
        let agentBlock = {
            Name = "TestAgent"
            Properties = [Role "Test Role"]
            LanguageBlocks = []
            LineNumber = 1
        }
        
        // Assert
        agentBlock.Name |> should equal "TestAgent"
        agentBlock.Properties.Length |> should equal 1
        match agentBlock.Properties.[0] with
        | Role r -> r |> should equal "Test Role"
        | _ -> failwith "Expected Role property"
        agentBlock.LanguageBlocks |> should equal []
        agentBlock.LineNumber |> should equal 1
    
    [<Fact>]
    let ``FluxScript has required properties`` () =
        // Arrange & Act
        let script = {
            Blocks = []
            FileName = Some "test.flux"
            ParsedAt = DateTime.UtcNow
            Version = "1.0.0"
            Metadata = Map.empty
        }
        
        // Assert
        script.Blocks |> should equal []
        script.FileName |> should equal (Some "test.flux")
        script.Version |> should equal "1.0.0"
        script.Metadata |> should equal Map.empty
    
    [<Fact>]
    let ``FluxExecutionResult has required properties`` () =
        // Arrange & Act
        let result = {
            Success = true
            Result = Some (StringValue "test")
            ExecutionTime = TimeSpan.FromSeconds(1.0)
            BlocksExecuted = 5
            ErrorMessage = None
            Trace = []
            GeneratedArtifacts = Map.empty
            AgentOutputs = Map.empty
            DiagnosticResults = Map.empty
            ReflectionInsights = []
        }
        
        // Assert
        result.Success |> should equal true
        result.Result |> should equal (Some (StringValue "test"))
        result.ExecutionTime |> should equal (TimeSpan.FromSeconds(1.0))
        result.BlocksExecuted |> should equal 5
        result.ErrorMessage |> should equal None
        result.Trace |> should equal []
        result.GeneratedArtifacts |> should equal Map.empty
        result.AgentOutputs |> should equal Map.empty
        result.DiagnosticResults |> should equal Map.empty
        result.ReflectionInsights |> should equal []

    // AST tests completed
