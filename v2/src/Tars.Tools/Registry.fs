namespace Tars.Tools

open System
open System.Threading.Tasks
open Tars.Core

type ITool =
    abstract member Name: string
    abstract member Description: string
    abstract member ExecuteAsync: args: Map<string, string> -> Task<string>

type ToolRegistry() =
    let tools = System.Collections.Concurrent.ConcurrentDictionary<string, ITool>()

    member this.Register(tool: ITool) = tools.TryAdd(tool.Name, tool) |> ignore

    member this.Get(name: string) =
        match tools.TryGetValue(name) with
        | true, tool -> Some tool
        | _ -> None

    member this.GetAll() = tools.Values |> Seq.toList
