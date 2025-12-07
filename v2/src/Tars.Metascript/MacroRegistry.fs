namespace Tars.Metascript

open System
open System.IO
open System.Threading.Tasks
open System.Text.Json
open Tars.Metascript.Domain

/// Interface for managing reusable workflow macros
type IMacroRegistry =
    abstract member Register: Workflow -> Task<unit>
    abstract member Get: string -> Task<Workflow option>
    abstract member List: unit -> Task<Workflow list>
    abstract member Delete: string -> Task<bool>

/// File-based implementation of macro registry
type FileMacroRegistry(storageDir: string) =

    let ensureDir () =
        if not (Directory.Exists(storageDir)) then
            Directory.CreateDirectory(storageDir) |> ignore

    let getFilePath (name: string) =
        Path.Combine(storageDir, $"{name.ToLower()}.json")

    interface IMacroRegistry with

        member _.Register(workflow: Workflow) =
            task {
                ensureDir ()
                let filePath = getFilePath workflow.Name
                let json = Parser.toJson workflow
                do! File.WriteAllTextAsync(filePath, json)
            }

        member _.Get(name: string) =
            task {
                try
                    ensureDir ()
                    let filePath = getFilePath name

                    if File.Exists(filePath) then
                        let! json = File.ReadAllTextAsync(filePath)

                        match Parser.parseJson json with
                        | Parser.Success w -> return Some w
                        | _ -> return None
                    else
                        return None
                with _ ->
                    return None
            }

        member _.List() =
            task {
                try
                    ensureDir ()
                    let files = Directory.GetFiles(storageDir, "*.json")
                    let results = ResizeArray<Workflow>()

                    for file in files do
                        let! json = File.ReadAllTextAsync(file)

                        match Parser.parseJson json with
                        | Parser.Success w -> results.Add(w)
                        | _ -> () // Skip invalid files

                    return results |> Seq.toList
                with _ ->
                    return []
            }

        member _.Delete(name: string) =
            task {
                try
                    ensureDir ()
                    let filePath = getFilePath name

                    if File.Exists(filePath) then
                        File.Delete(filePath)
                        return true
                    else
                        return false
                with _ ->
                    return false
            }
