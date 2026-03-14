module Tars.Tests.SelfExtensionTests

open Xunit
open FsUnit
open System
open System.IO
open Tars.Core.SelfExtension

// =============================================================================
// TOOL SERIALIZATION TESTS (Phase 17 Fix)
// =============================================================================

[<Fact>]
let ``ToolSerialization roundtrips FSharpScript implementation`` () =
    let original: DynamicToolDefinition =
        { Name = "test_tool"
          Description = "A test tool"
          Version = "1.0"
          InputSchema = Some """{"type": "string"}"""
          OutputSchema = Some """{"type": "string"}"""
          Implementation = FSharpScript "printfn \"Hello\"" }

    let json = ToolSerialization.serializeToJson original

    match ToolSerialization.deserializeFromJson json with
    | Ok deserialized ->
        deserialized.Name |> should equal original.Name
        deserialized.Description |> should equal original.Description
        deserialized.Version |> should equal original.Version

        match deserialized.Implementation with
        | FSharpScript code -> code |> should equal "printfn \"Hello\""
        | _ -> failwith "Wrong implementation type"
    | Error e -> failwithf "Deserialization failed: %s" e

[<Fact>]
let ``ToolSerialization roundtrips MetascriptRef implementation`` () =
    let original: DynamicToolDefinition =
        { Name = "meta_tool"
          Description = "Uses a metascript"
          Version = "2.0"
          InputSchema = None
          OutputSchema = None
          Implementation = MetascriptRef "my_metascript" }

    let json = ToolSerialization.serializeToJson original

    match ToolSerialization.deserializeFromJson json with
    | Ok deserialized ->
        match deserialized.Implementation with
        | MetascriptRef name -> name |> should equal "my_metascript"
        | _ -> failwith "Wrong implementation type"
    | Error e -> failwithf "Deserialization failed: %s" e

[<Fact>]
let ``ToolSerialization roundtrips ExternalCommand implementation`` () =
    let original: DynamicToolDefinition =
        { Name = "shell_tool"
          Description = "Runs a shell command"
          Version = "1.0"
          InputSchema = Some "{}"
          OutputSchema = Some "{}"
          Implementation = ExternalCommand "echo test" }

    let json = ToolSerialization.serializeToJson original

    match ToolSerialization.deserializeFromJson json with
    | Ok deserialized ->
        match deserialized.Implementation with
        | ExternalCommand cmd -> cmd |> should equal "echo test"
        | _ -> failwith "Wrong implementation type"
    | Error e -> failwithf "Deserialization failed: %s" e

[<Fact>]
let ``SerializableTool has correct field names`` () =
    let tool: DynamicToolDefinition =
        { Name = "field_test"
          Description = "Tests field names"
          Version = "1.0"
          InputSchema = Some "input"
          OutputSchema = Some "output"
          Implementation = FSharpScript "code" }

    let json = ToolSerialization.serializeToJson tool

    // The JSON should contain readable field names
    json |> should contain "\"Name\": \"field_test\""
    json |> should contain "\"ImplementationType\": \"FSharpScript\""
    json |> should contain "\"ImplementationValue\": \"code\""

// =============================================================================
// EXTENSION TYPES TESTS
// =============================================================================

[<Fact>]
let ``ExtensionType serialization is correct`` () =
    let ext: Extension =
        { Id = Guid.NewGuid()
          Name = "test"
          Type = ExtensionType.DynamicTool
          Description = "desc"
          CreatedAt = DateTimeOffset.UtcNow
          CreatedBy = "test"
          Version = "1.0"
          FilePath = Some "/path/to/file"
          Status = ExtensionStatus.Active }

    ext.Type |> should equal ExtensionType.DynamicTool

[<Fact>]
let ``BlockDefinition can be created`` () =
    let block: BlockDefinition =
        { Name = "INVARIANT"
          Description = "A constraint block"
          Version = "1.0"
          Parameters =
            [ { Name = "id"
                Type = "string"
                Required = false
                Default = None }
              { Name = "confidence"
                Type = "float"
                Required = false
                Default = Some "1.0" } ]
          ContentType = BlockContentType.Freeform
          EbnfRule = "invariant_block ::= 'INVARIANT' ..."
          CompileToIR = "// compile code" }

    block.Name |> should equal "INVARIANT"
    block.Parameters.Length |> should equal 2
