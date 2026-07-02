module Tars.Tests.ToolInvokerTests

open System
open Xunit
open Tars.Core
open Tars.Core.WorkflowOfThought
open Tars.Tools

// The enriched IToolInvoker surfaces typed outcomes and records through an
// injected IToolRecorder — both testable without the global ledger.

type private StubRecorder() =
    member val Records = ResizeArray<string * bool * string>()

    interface IToolRecorder with
        member this.Record(name, _input, _output, _dur, success, category) =
            this.Records.Add((name, success, string category))

let private mkTool name (exec: string -> Async<Result<string, string>>) : Tool =
    { Name = name
      Description = ""
      Version = "1.0.0"
      ParentVersion = None
      CreatedAt = DateTime.UtcNow
      Execute = exec }

let private invokerWith (tools: Tool list) (recorder: IToolRecorder) : IToolInvoker =
    let reg = ToolRegistry()

    for t in tools do
        reg.Register t

    ToolInvoker(reg :> IToolRegistry, recorder) :> IToolInvoker

[<Fact>]
let ``Invoke returns NotFound for an unregistered tool`` () =
    let recorder = StubRecorder()
    let invoker = invokerWith [] recorder
    let outcome = invoker.Invoke("nope", Map.empty) |> Async.RunSynchronously
    Assert.Equal(ToolOutcome.NotFound, outcome)

[<Fact>]
let ``Invoke returns Succeeded and records a success`` () =
    let recorder = StubRecorder()
    let tool = mkTool "echo" (fun input -> async { return Result.Ok $"got:{input}" })
    let invoker = invokerWith [ tool ] recorder

    let outcome = invoker.Invoke("echo", Map.ofList [ "k", "v" ]) |> Async.RunSynchronously

    match outcome with
    | ToolOutcome.Succeeded out -> Assert.StartsWith("got:", out)
    | other -> failwith $"expected Succeeded, got %A{other}"

    Assert.True(recorder.Records |> Seq.exists (fun (n, s, _) -> n = "echo" && s))

[<Fact>]
let ``Invoke returns Failed with a classified category and records a failure`` () =
    let recorder = StubRecorder()
    let tool = mkTool "boom" (fun _ -> async { return Result.Error "network down" })
    let invoker = invokerWith [ tool ] recorder

    let outcome = invoker.Invoke("boom", Map.empty) |> Async.RunSynchronously

    match outcome with
    | ToolOutcome.Failed(category, message) ->
        Assert.Equal("DependencyFailure", category) // "network" → DependencyFailure
        Assert.Equal("network down", message)
    | other -> failwith $"expected Failed, got %A{other}"

    Assert.True(recorder.Records |> Seq.exists (fun (n, s, _) -> n = "boom" && not s))
