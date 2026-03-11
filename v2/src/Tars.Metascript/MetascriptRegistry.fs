namespace Tars.Metascript

open System.IO
open System.Threading.Tasks
open Tars.Metascript.V1

/// Interface for managing reusable V1 metascript macros
type IMetascriptRegistry =
    abstract member Register: Metascript -> Task<unit>
    abstract member Get: string -> Task<Metascript option>
    abstract member List: unit -> Task<Metascript list>
    abstract member Delete: string -> Task<bool>

/// File-based implementation of metascript registry using .tars files
type FileMetascriptRegistry(storageDir: string) =

    let ensureDir () =
        if not (Directory.Exists(storageDir)) then
            Directory.CreateDirectory(storageDir) |> ignore

    let getFilePath (name: string) =
        Path.Combine(storageDir, $"{name.ToLower()}.tars")

    interface IMetascriptRegistry with

        member _.Register(metascript: Metascript) =
            task {
                ensureDir ()
                let filePath = getFilePath metascript.Name
                let text = V1Parser.toMetascript metascript
                do! File.WriteAllTextAsync(filePath, text)
            }

        member _.Get(name: string) =
            task {
                try
                    ensureDir ()
                    let filePath = getFilePath name

                    if File.Exists(filePath) then
                        let! text = File.ReadAllTextAsync(filePath)
                        let metascript = V1Parser.parseMetascript text name (Some filePath)
                        return Some metascript
                    else
                        return None
                with _ ->
                    return None
            }

        member _.List() =
            task {
                try
                    ensureDir ()
                    let files = Directory.GetFiles(storageDir, "*.tars")
                    let results = ResizeArray<Metascript>()

                    for file in files do
                        let! text = File.ReadAllTextAsync(file)
                        let name = Path.GetFileNameWithoutExtension(file)
                        let metascript = V1Parser.parseMetascript text name (Some file)
                        results.Add(metascript)

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
