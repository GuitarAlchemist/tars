#!/usr/bin/env dotnet fsi

// TARS Grammar CLI Tool
// Usage: dotnet fsi tarscli_grammar.fsx <command> [args]

#r "nuget: System.Text.Json"

open System
open System.IO
open System.Text.Json

// Simulate the grammar types and functions
type GrammarIndexEntry = {
    Id: string
    File: string
    Origin: string
    Version: string option
    Hash: string option
    LastModified: DateTime option
}

let grammarsDirectory = ".tars/evolution/grammars/base"
let grammarIndexFile = ".tars/evolution/grammars/base/grammar_index.json"

let ensureGrammarsDirectory () =
    if not (Directory.Exists(grammarsDirectory)) then
        Directory.CreateDirectory(grammarsDirectory) |> ignore

let loadGrammarIndex () =
    try
        if File.Exists(grammarIndexFile) then
            let json = File.ReadAllText(grammarIndexFile)
            JsonSerializer.Deserialize<GrammarIndexEntry[]>(json)
            |> Array.toList
        else
            []
    with
    | ex ->
        printfn $"Warning: Failed to load grammar index: {ex.Message}"
        []

let saveGrammarIndex (entries: GrammarIndexEntry list) =
    try
        ensureGrammarsDirectory()
        let json = JsonSerializer.Serialize(entries, JsonSerializerOptions(WriteIndented = true))
        File.WriteAllText(grammarIndexFile, json)
    with
    | ex ->
        printfn $"Warning: Failed to save grammar index: {ex.Message}"

let listGrammars () =
    ensureGrammarsDirectory()
    let index = loadGrammarIndex()
    
    printfn "📚 Available TARS Grammars:"
    printfn "=========================="
    
    if List.isEmpty index then
        printfn "No grammars found. Create some with 'extract-from-meta' or 'generate-from-examples'."
    else
        for entry in index do
            let version = entry.Version |> Option.defaultValue "unknown"
            let lastMod = entry.LastModified |> Option.map (fun d -> d.ToString("yyyy-MM-dd")) |> Option.defaultValue "unknown"
            printfn "📄 %s (%s) - v%s - %s" entry.Id entry.Origin version lastMod

            let filePath = Path.Combine(grammarsDirectory, entry.File)
            if File.Exists(filePath) then
                let fileInfo = FileInfo(filePath)
                printfn "   📁 %s (%d bytes)" filePath fileInfo.Length
            else
                printfn "   ❌ File not found: %s" filePath
            printfn ""

let extractFromMeta metaFile =
    printfn "🔧 Extracting inline grammars from: %s" metaFile

    if not (File.Exists(metaFile)) then
        printfn "❌ File not found: %s" metaFile
        exit 1

    let content = File.ReadAllText(metaFile)

    // Simple pattern matching for grammar blocks (this would be more sophisticated in real implementation)
    if content.Contains("grammar {") && content.Contains("LANG(\"EBNF\")") then
        printfn "✅ Found inline grammar block"
        printfn "💡 In a real implementation, this would:"
        printfn "   1. Parse the .tars file"
        printfn "   2. Extract grammar definitions"
        printfn "   3. Save to .tars/grammars/"
        printfn "   4. Update grammar index"
        printfn "   5. Replace inline grammar with use_grammar() reference"
    else
        printfn "ℹ️ No inline grammar blocks found in file"

let inlineGrammar grammarName =
    printfn "📥 Inlining grammar: %s" grammarName

    let grammarFile = Path.Combine(grammarsDirectory, grammarName + ".tars")
    if not (File.Exists(grammarFile)) then
        printfn "❌ Grammar not found: %s" grammarName
        exit 1

    let content = File.ReadAllText(grammarFile)
    printfn "✅ Grammar content (for inlining):"
    printfn "=================================="
    printfn "%s" content

let hashGrammar grammarName =
    printfn "🔐 Computing hash for grammar: %s" grammarName

    let grammarFile = Path.Combine(grammarsDirectory, grammarName + ".tars")
    if not (File.Exists(grammarFile)) then
        printfn "❌ Grammar not found: %s" grammarName
        exit 1

    let content = File.ReadAllText(grammarFile)
    use sha256 = System.Security.Cryptography.SHA256.Create()
    let hash = sha256.ComputeHash(System.Text.Encoding.UTF8.GetBytes(content))
    let hashString = Convert.ToHexString(hash).ToLowerInvariant()

    printfn "✅ SHA256: %s" hashString
    printfn "📊 Size: %d characters" content.Length
    printfn "📏 Lines: %d" (content.Split('\n').Length)

let generateFromExamples examplesFile =
    printfn "🤖 Generating grammar from examples: %s" examplesFile

    if not (File.Exists(examplesFile)) then
        printfn "❌ Examples file not found: %s" examplesFile
        exit 1

    printfn "💡 In a real implementation, this would:"
    printfn "   1. Analyze example inputs"
    printfn "   2. Use ML/pattern recognition to infer grammar rules"
    printfn "   3. Generate EBNF grammar"
    printfn "   4. Save to .tars/grammars/"
    printfn "   5. Validate against examples"

let downloadRFC rfcId =
    printfn "🌐 Downloading and processing RFC: %s" rfcId
    printfn "💡 In a real implementation, this would:"
    printfn "   1. Download from https://datatracker.ietf.org/doc/html/%s" rfcId
    printfn "   2. Extract BNF/ABNF rules"
    printfn "   3. Convert to EBNF format"
    printfn "   4. Save as %s_grammar.tars" rfcId
    printfn "   5. Update grammar index"

let showHelp () =
    printfn "🚀 TARS Grammar CLI Tool"
    printfn "======================="
    printfn ""
    printfn "Commands:"
    printfn "  list                           - List all available grammars"
    printfn "  extract-from-meta <file>       - Extract inline grammars from .tars file"
    printfn "  inline <grammar>               - Show grammar content for inlining"
    printfn "  hash <grammar>                 - Compute hash of grammar"
    printfn "  generate-from-examples <file>  - Generate grammar from example inputs"
    printfn "  download-rfc <rfcId>           - Download and process RFC grammar"
    printfn "  help                           - Show this help"
    printfn ""
    printfn "Examples:"
    printfn "  dotnet fsi tarscli_grammar.fsx list"
    printfn "  dotnet fsi tarscli_grammar.fsx extract-from-meta agent.tars"
    printfn "  dotnet fsi tarscli_grammar.fsx inline MiniQuery"
    printfn "  dotnet fsi tarscli_grammar.fsx hash RFC3986_URI"
    printfn "  dotnet fsi tarscli_grammar.fsx download-rfc rfc3986"

// Main command processing
let args = fsi.CommandLineArgs |> Array.skip 1

match args with
| [||] | [|"help"|] -> showHelp()
| [|"list"|] -> listGrammars()
| [|"extract-from-meta"; metaFile|] -> extractFromMeta metaFile
| [|"inline"; grammarName|] -> inlineGrammar grammarName
| [|"hash"; grammarName|] -> hashGrammar grammarName
| [|"generate-from-examples"; examplesFile|] -> generateFromExamples examplesFile
| [|"download-rfc"; rfcId|] -> downloadRFC rfcId
| _ -> 
    printfn "❌ Unknown command or invalid arguments"
    printfn "Use 'help' to see available commands"
    exit 1
