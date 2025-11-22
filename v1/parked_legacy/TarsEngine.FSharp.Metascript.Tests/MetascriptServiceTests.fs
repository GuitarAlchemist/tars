module TarsEngine.FSharp.Metascript.Tests.MetascriptServiceTests

open System
open System.Threading.Tasks
open Microsoft.FSharp.Control
open Microsoft.FSharp.Reflection
open Microsoft.Extensions.Logging.Abstractions
open TarsEngine.FSharp.Metascript
open TarsEngine.FSharp.Metascript.BlockHandlers
open TarsEngine.FSharp.Metascript.Services
open Xunit

let private createMetascriptService () =
    let registryLogger = NullLogger<BlockHandlerRegistry>.Instance
    let registry = BlockHandlerRegistry(registryLogger)

    let register handler =
        registry.RegisterHandler(handler :> IBlockHandler)

    register (FSharpBlockHandler(NullLogger<FSharpBlockHandler>.Instance))
    register (CommandBlockHandler(NullLogger<CommandBlockHandler>.Instance))
    register (TextBlockHandler(NullLogger<TextBlockHandler>.Instance))

    let executorLogger = NullLogger<MetascriptExecutor>.Instance
    let executor = MetascriptExecutor(executorLogger, registry)

    let serviceLogger = NullLogger<MetascriptService>.Instance
    MetascriptService(serviceLogger, executor :> IMetascriptExecutor)

let private tryUnwrapString (value: obj) =
    match value with
    | :? string as s -> Some s
    | _ ->
        let valueType = value.GetType()
        if FSharpType.IsUnion(valueType) && valueType.FullName.StartsWith("Microsoft.FSharp.Core.FSharpOption", StringComparison.Ordinal) then
            let caseInfo, fields = FSharpValue.GetUnionFields(value, valueType)
            if caseInfo.Name = "Some" && fields.Length = 1 then
                match fields.[0] with
                | :? string as s -> Some s
                | _ -> None
            else
                None
        else
            None

[<Fact>]
let ``Parse simple metascript yields expected block sequence`` () : Task =
    task {
        let service = createMetascriptService ()

        let metascriptText = """
```fsharp
printfn "FSharp block executed"; "OK"
```
```command
echo metascript-command
```
```text
Plain text block
```
"""

        let! metascript = service.ParseMetascriptAsync(metascriptText, ?name = Some "test-script")

        Assert.Equal("test-script", metascript.Name)
        Assert.Equal(3, metascript.Blocks.Length)

        let actualTypes = metascript.Blocks |> List.map (fun block -> block.Type) |> List.toArray
        let expectedTypes =
            [| MetascriptBlockType.FSharp
               MetascriptBlockType.Command
               MetascriptBlockType.Text |]

        Assert.Equal<MetascriptBlockType[]>(expectedTypes, actualTypes)
    }

[<Fact>]
let ``Execute metascript returns combined output and block results`` () : Task =
    task {
        let service = createMetascriptService ()

        let metascriptText = """
```fsharp
printfn "FSharp block executed"; "OK"
```
```command
echo metascript-command
```
```text
Plain text block
```
"""

        let! result = service.ExecuteMetascriptTextAsync(metascriptText, ?name = Some "execution-test")

        Assert.Equal(MetascriptExecutionStatus.Success, result.Status)
        Assert.Equal(3, result.BlockResults.Length)

        let commandOutput = result.BlockResults.[1].Output
        let textOutput = result.BlockResults.[2].Output

        Assert.Contains("metascript-command", commandOutput, StringComparison.OrdinalIgnoreCase)
        Assert.Equal("Plain text block", textOutput.Trim())

        match result.BlockResults.[0].ReturnValue with
        | Some value ->
            match tryUnwrapString value with
            | Some s -> Assert.Equal("OK", s)
            | None -> failwith "Expected F# block return value to unwrap to string"
        | None -> failwith "Expected F# block to produce a return value"

        Assert.True(result.ExecutionTimeMs > 0.0)
    }
