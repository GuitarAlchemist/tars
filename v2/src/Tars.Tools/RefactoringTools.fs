namespace Tars.Tools.Standard

open System.IO
open System.Text.Json
open Tars.Core.HybridBrain
open Tars.Tools

/// <summary>
/// Production-grade refactoring tools for F#.
/// Bridges the HybridBrain's ActionExecutor logic to the TARS Tool Registry.
/// </summary>
module RefactoringTools =

    /// Helper to read all lines from a file
    let private readLines (filePath: string) =
        if File.Exists(filePath) then
            try
                File.ReadAllLines(filePath) |> Array.toList |> Some
            with _ -> None
        else
            None

    /// Helper to write all lines back to a file
    let private writeLines (filePath: string) (lines: string list) =
        try
            File.WriteAllLines(filePath, lines |> List.toArray)
            true
        with _ -> false

    [<TarsToolAttribute("refactor_extract_function", 
                        "Extracts a block of code into a new function. Input JSON: { \"path\": \"src/File.fs\", \"name\": \"new_func\", \"start\": 10, \"end\": 20 }. Returns success message or error.")>]
    let extractFunction (args: string) =
        task {
            try
                use doc = JsonDocument.Parse(args)
                let root = doc.RootElement
                let path = root.GetProperty("path").GetString()
                let name = root.GetProperty("name").GetString()
                let startLine = root.GetProperty("start").GetInt32()
                let endLine = root.GetProperty("end").GetInt32()
                
                let fullPath = Path.GetFullPath(path)
                match ActionExecutor.FileRefactoring.extractFunction fullPath name startLine endLine with
                | Result.Ok msg -> return Result.Ok msg
                | Result.Error err -> return Result.Error err
            with ex -> 
                return Result.Error $"Refactoring failed (JSON error?): {ex.Message}"
        }

    [<TarsToolAttribute("refactor_remove_dead_code", 
                        "Removes a block of lines from a file. Input JSON: { \"path\": \"src/File.fs\", \"start\": 10, \"end\": 15 }. Use for dead code removal.")>]
    let removeLines (args: string) =
        task {
            try
                use doc = JsonDocument.Parse(args)
                let root = doc.RootElement
                let path = root.GetProperty("path").GetString()
                let startLine = root.GetProperty("start").GetInt32()
                let endLine = root.GetProperty("end").GetInt32()
                
                let fullPath = Path.GetFullPath(path)
                match ActionExecutor.FileRefactoring.removeLines fullPath startLine endLine with
                | Result.Ok msg -> return Result.Ok msg
                | Result.Error err -> return Result.Error err
            with ex -> 
                return Result.Error $"Refactoring failed: {ex.Message}"
        }

    [<TarsToolAttribute("refactor_rename_symbol", 
                        "Renames a symbol throughout a file. Input JSON: { \"path\": \"src/File.fs\", \"old\": \"oldName\", \"new\": \"newName\" }. Ensures all occurrences are updated.")>]
    let renameSymbol (args: string) =
        task {
            try
                use doc = JsonDocument.Parse(args)
                let root = doc.RootElement
                let path = root.GetProperty("path").GetString()
                let oldName = root.GetProperty("old").GetString()
                let newName = root.GetProperty("new").GetString()
                
                let fullPath = Path.GetFullPath(path)
                match ActionExecutor.FileRefactoring.renameSymbol fullPath oldName newName with
                | Result.Ok msg -> return Result.Ok msg
                | Result.Error err -> return Result.Error err
            with ex -> 
                return Result.Error $"Refactoring failed: {ex.Message}"
        }

    [<TarsToolAttribute("refactor_add_documentation", 
                        "Adds XML documentation to a specific line. Input JSON: { \"path\": \"src/File.fs\", \"line\": 10, \"text\": \"Summary of the function\" }.")>]
    let addDocumentation (args: string) =
        task {
            try
                use doc = JsonDocument.Parse(args)
                let root = doc.RootElement
                let path = root.GetProperty("path").GetString()
                let line = root.GetProperty("line").GetInt32()
                let text = root.GetProperty("text").GetString()
                
                let fullPath = Path.GetFullPath(path)
                match ActionExecutor.FileRefactoring.addDocumentation fullPath line text with
                | Result.Ok msg -> return Result.Ok msg
                | Result.Error err -> return Result.Error err
            with ex -> 
                return Result.Error $"Refactoring failed: {ex.Message}"
        }

    [<TarsToolAttribute("refactor_inline_value", 
                        "Inlines a value/binding usage on a specific line. Input JSON: { \"path\": \"src/File.fs\", \"name\": \"x\", \"line\": 15 }.")>]
    let inlineValue (args: string) =
        task {
            try
                use doc = JsonDocument.Parse(args)
                let root = doc.RootElement
                let path = root.GetProperty("path").GetString()
                let name = root.GetProperty("name").GetString()
                let line = root.GetProperty("line").GetInt32()
                
                let fullPath = Path.GetFullPath(path)
                match ActionExecutor.FileRefactoring.inlineValue fullPath name line with
                | Result.Ok msg -> return Result.Ok msg
                | Result.Error err -> return Result.Error err
            with ex -> 
                return Result.Error $"Refactoring failed: {ex.Message}"
        }
