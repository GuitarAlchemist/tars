namespace TarsEngine.FSharpToRust

module Program =
    open System
    open System.IO

    [<EntryPoint>]
    let main args =
        match args with
        | [| "--example" |] ->
            // Run the example
            let rustCode = Examples.runExample()
            0
        
        | [| inputFile; outputFile |] ->
            // Transpile the specified file
            printfn "Transpiling %s to %s..." inputFile outputFile
            
            if FSharpToRustTranspiler.transpileFile inputFile outputFile then
                printfn "Transpilation successful!"
                0
            else
                printfn "Transpilation failed."
                1
        
        | [| inputFile |] ->
            // Transpile the specified file to a default output file
            let outputFile = Path.ChangeExtension(inputFile, ".rs")
            printfn "Transpiling %s to %s..." inputFile outputFile
            
            if FSharpToRustTranspiler.transpileFile inputFile outputFile then
                printfn "Transpilation successful!"
                0
            else
                printfn "Transpilation failed."
                1
        
        | _ ->
            // Show usage
            printfn "Usage:"
            printfn "  TarsEngine.FSharpToRust --example"
            printfn "  TarsEngine.FSharpToRust <input-file> [<output-file>]"
            1
