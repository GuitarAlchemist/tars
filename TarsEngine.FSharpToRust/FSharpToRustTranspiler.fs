namespace TarsEngine.FSharpToRust

open System
open System.IO
open System.Text.RegularExpressions

/// F# to Rust transpiler
module FSharpToRustTranspiler =

    /// Type mapping from F# to Rust
    let rec mapType (fsType: string) =
        match fsType.ToLower() with
        | "int" | "int32" -> "i32"
        | "int64" | "long" -> "i64"
        | "float" | "double" -> "f64"
        | "float32" | "single" -> "f32"
        | "bool" | "boolean" -> "bool"
        | "string" -> "String"
        | "char" -> "char"
        | "unit" -> "()"
        | "byte" | "uint8" -> "u8"
        | "uint16" -> "u16"
        | "uint" | "uint32" -> "u32"
        | "uint64" -> "u64"
        | t when t.StartsWith("option<") ->
            let innerType = t.Substring(7, t.Length - 8)
            sprintf "Option<%s>" (mapType innerType)
        | t when t.StartsWith("list<") ->
            let innerType = t.Substring(5, t.Length - 6)
            sprintf "Vec<%s>" (mapType innerType)
        | t when t.StartsWith("array<") ->
            let innerType = t.Substring(6, t.Length - 7)
            sprintf "Vec<%s>" (mapType innerType)
        | _ -> sprintf "/* Unmapped type: %s */ ()" fsType

    /// Simple transpilation for basic F# constructs (for proof of concept)
    let simpleTranspile (fsharpCode: string) =
        // This is a simplified approach using regex for the proof of concept
        let mutable rustCode = fsharpCode

        // Replace F# function definitions
        let functionRegex = Regex(@"let\s+(\w+)(?:\s*\(([^)]*)\))?\s*(?::\s*([^=]+))?\s*=\s*(.+?)(?=let|\z)", RegexOptions.Singleline)
        rustCode <- functionRegex.Replace(rustCode, fun m ->
            let functionName = m.Groups.[1].Value
            let parameters =
                if m.Groups.[2].Success then
                    m.Groups.[2].Value.Split([|','|], StringSplitOptions.RemoveEmptyEntries)
                    |> Array.map (fun p ->
                        let parts = p.Trim().Split(':')
                        if parts.Length > 1 then
                            sprintf "%s: %s" (parts.[0].Trim()) (mapType (parts.[1].Trim()))
                        else
                            sprintf "%s: /* unknown type */" (parts.[0].Trim()))
                    |> String.concat ", "
                else ""

            let returnType =
                if m.Groups.[3].Success then
                    mapType (m.Groups.[3].Value.Trim())
                else "/* unknown return type */"

            let body = m.Groups.[4].Value.Trim()

            sprintf "fn %s(%s) -> %s {\n    %s\n}" functionName parameters returnType body)

        // Replace F# if/then/else
        let ifRegex = Regex(@"if\s+(.+?)\s+then\s+(.+?)(?:\s+else\s+(.+?))?(?=\s*let|\s*if|\z)", RegexOptions.Singleline)
        rustCode <- ifRegex.Replace(rustCode, fun m ->
            let condition = m.Groups.[1].Value.Trim()
            let thenBranch = m.Groups.[2].Value.Trim()

            if m.Groups.[3].Success then
                let elseBranch = m.Groups.[3].Value.Trim()
                sprintf "if %s { %s } else { %s }" condition thenBranch elseBranch
            else
                sprintf "if %s { %s }" condition thenBranch)

        // Replace F# match expressions
        let matchRegex = Regex(@"match\s+(.+?)\s+with\s+\|\s+(.+?)(?=let|\z)", RegexOptions.Singleline)
        rustCode <- matchRegex.Replace(rustCode, fun m ->
            let expr = m.Groups.[1].Value.Trim()
            let cases = m.Groups.[2].Value.Trim()

            // Process match cases
            let caseRegex = Regex(@"\|\s+(.+?)\s+->\s+(.+?)(?=\||\z)", RegexOptions.Singleline)
            let processedCases = caseRegex.Replace(cases, fun c ->
                let pattern = c.Groups.[1].Value.Trim()
                let result = c.Groups.[2].Value.Trim()
                sprintf "%s => { %s }," pattern result)

            sprintf "match %s {\n%s\n}" expr processedCases)

        // Replace F# types
        let typeRegex = Regex(@":\s*(\w+(?:<[^>]+>)?)")
        rustCode <- typeRegex.Replace(rustCode, fun m ->
            let fsType = m.Groups.[1].Value.Trim()
            sprintf ": %s" (mapType fsType))

        rustCode

    /// Transpile F# file to Rust
    let transpileFile (inputPath: string) (outputPath: string) =
        try
            // Read the F# code
            let fsharpCode = File.ReadAllText(inputPath)

            // Transpile to Rust
            let rustCode = simpleTranspile fsharpCode

            // Write the Rust code
            File.WriteAllText(outputPath, rustCode)

            true
        with
        | ex ->
            printfn "Error transpiling file: %s" ex.Message
            false
