namespace TarsEngine.FSharp.Core.Metascript

open System
open System.IO
open TarsEngine.FSharp.Core.Metascript.Types

/// Simple metascript parser that works
module Parser =
    
    let parseMetascript (filePath: string) : ParsedMetascript =
        let content = File.ReadAllText(filePath)
        let lines = content.Split('\n') |> Array.mapi (fun i line -> (i + 1, line.Trim()))
        
        let mutable blocks = []
        let mutable currentBlock = None
        let mutable currentContent = []
        
        for (lineNum, line) in lines do
            match line with
            | line when line.StartsWith("meta {") ->
                currentBlock <- Some (Meta, lineNum)
                currentContent <- []
            | line when line.StartsWith("reasoning {") ->
                currentBlock <- Some (Reasoning, lineNum)
                currentContent <- []
            | line when line.StartsWith("FSHARP {") ->
                currentBlock <- Some (FSharp, lineNum)
                currentContent <- []
            | line when line = "}" ->
                match currentBlock with
                | Some (blockType, startLine) ->
                    let content = String.Join("\n", List.rev currentContent)
                    let block = {
                        Type = blockType
                        Content = content
                        LineNumber = startLine
                    }
                    blocks <- block :: blocks
                    currentBlock <- None
                    currentContent <- []
                | None -> ()
            | _ ->
                match currentBlock with
                | Some _ -> currentContent <- line :: currentContent
                | None -> ()
        
        let meta = 
            blocks 
            |> List.tryFind (fun b -> b.Type = Meta)
            |> Option.map (fun b -> {
                Name = "parsed_metascript"
                Version = "1.0"
                Description = "Parsed metascript"
                Author = "TARS"
                Created = DateTime.Now.ToString("yyyy-MM-dd")
                Tags = []
                ExecutionMode = "standard"
                OutputDirectory = ".tars/output"
                TraceEnabled = true
            })
        
        {
            Meta = meta
            Blocks = List.rev blocks
            FilePath = filePath
        }
