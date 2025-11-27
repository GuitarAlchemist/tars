namespace Tars.Tools

open System
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

        for (m, attr) in methods do
            let execute (input: string) : Async<Result<string, string>> =
                async {
                    try
                        let parameters = m.GetParameters()

                        let args =
                            if parameters.Length = 1 && parameters[0].ParameterType = typeof<string> then
                                [| input :> obj |]
                            elif parameters.Length = 0 then
                                [||]
                            else
                                // TODO: Implement JSON parsing for complex arguments
                                [| input :> obj |]

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
