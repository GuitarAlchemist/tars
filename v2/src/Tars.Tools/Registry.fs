namespace Tars.Tools

open System
open System.Text.Json
open System.Reflection
open System.Threading.Tasks
open Tars.Core

type ToolRegistry() =
    let tools =
        System.Collections.Concurrent.ConcurrentDictionary<string, Tars.Core.Tool>()

    member this.Register(tool: Tars.Core.Tool) = tools.TryAdd(tool.Name, tool) |> ignore

    member this.RegisterAssembly(assembly: Assembly) =
        let methods =
            assembly.GetTypes()
            |> Array.collect (fun t ->
                t.GetMethods(BindingFlags.Public ||| BindingFlags.Static ||| BindingFlags.Instance))
            |> Array.choose (fun m ->
                let attrs = m.GetCustomAttributes<TarsToolAttribute>()
                if Seq.isEmpty attrs then None else Some(m, Seq.head attrs))

        let convertValue (element: JsonElement) (t: Type) =
            try
                if t = typeof<string> then
                    box (element.GetString())
                elif t = typeof<int> then
                    box (element.GetInt32())
                elif t = typeof<int64> then
                    box (element.GetInt64())
                elif t = typeof<bool> then
                    box (element.GetBoolean())
                elif t = typeof<float> then
                    box (element.GetDouble() |> float)
                elif t = typeof<double> then
                    box (element.GetDouble())
                elif t = typeof<Guid> then
                    box (Guid.Parse(element.GetString()))
                else
                    // fallback to raw text
                    box (element.GetRawText())
            with _ ->
                // final fallback: try change type
                try
                    element.GetString() |> box
                with _ ->
                    box null

        let buildArgs (input: string) (parameters: ParameterInfo[]) =
            if parameters.Length = 0 then
                [||]
            elif parameters.Length = 1 && parameters[0].ParameterType = typeof<string> then
                [| input :> obj |]
            else
                let parsed =
                    try
                        JsonDocument.Parse(input).RootElement
                    with _ ->
                        JsonDocument.Parse("{}").RootElement

                if parsed.ValueKind = JsonValueKind.Object then
                    parameters
                    |> Array.map (fun p ->
                        let mutable prop = Unchecked.defaultof<JsonElement>
                        if parsed.TryGetProperty(p.Name, &prop) then
                            convertValue prop p.ParameterType
                        else
                            box null)
                else
                    // allow positional array when names not provided
                    if parsed.ValueKind = JsonValueKind.Array then
                        let values = parsed.EnumerateArray() |> Seq.toArray
                        parameters
                        |> Array.mapi (fun i p ->
                            if i < values.Length then
                                convertValue values[i] p.ParameterType
                            else
                                box null)
                    else
                        [| input :> obj |]

        for (m, attr) in methods do
            let execute (input: string) : Async<Result<string, string>> =
                async {
                    try
                        let parameters = m.GetParameters()

                        let args = buildArgs input parameters

                        let instance =
                            if m.IsStatic then
                                null
                            else
                                Activator.CreateInstance(m.DeclaringType)

                        let result = m.Invoke(instance, args)

                        match result with
                        | :? Task<string> as t ->
                            let! r = Async.AwaitTask t
                            return Result.Ok r
                        | :? string as s -> return Result.Ok s
                        | null -> return Result.Ok "null"
                        | _ -> return Result.Ok(result.ToString())
                    with ex ->
                        return Result.Error ex.Message
                }

            let tool: Tars.Core.Tool =
                { Name = attr.Name
                  Description = attr.Description
                  Version = "1.0.0"
                  ParentVersion = None
                  CreatedAt = DateTime.UtcNow
                  Execute = execute }

            this.Register(tool)

    member this.Get(name: string) =
        match tools.TryGetValue(name) with
        | true, tool -> Some tool
        | _ -> None

    member this.GetAll() = tools.Values |> Seq.toList
