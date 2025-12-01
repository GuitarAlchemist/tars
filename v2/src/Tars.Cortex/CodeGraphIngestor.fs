namespace Tars.Cortex

open System
open System.IO
open System.Text.RegularExpressions
open Tars.Core

module CodeGraphIngestor =

    let private readFile (path: string) =
        try
            File.ReadAllText(path)
        with _ -> ""

    let private extractMatches (pattern: string) (content: string) =
        Regex.Matches(content, pattern, RegexOptions.Multiline)
        |> Seq.cast<Match>
        |> Seq.map (fun m -> m.Groups.[1].Value)
        |> Seq.toList

    /// Ingests a single F# file into the Knowledge Graph
    let ingestFile (graph: KnowledgeGraph) (filePath: string) =
        let content = readFile filePath
        let fileName = Path.GetFileName(filePath)
        let fileNode = FileNode fileName // Use filename for cleaner graph, or relative path
        graph.AddNode(fileNode)

        // 1. Extract Namespace/Module
        // namespace Tars.Cortex or module Tars.Cortex
        let namespaceMatch = Regex.Match(content, @"^(?:namespace|module)\s+([\w\.]+)", RegexOptions.Multiline)
        let moduleNode =
            if namespaceMatch.Success then
                let name = namespaceMatch.Groups.[1].Value
                let node = ModuleNode name
                graph.AddNode(node)
                graph.AddEdge(fileNode, node, Contains)
                Some node
            else
                None

        let parentNode = Option.defaultValue fileNode moduleNode

        // 2. Extract Types
        // type MyType = ... or type MyType(args) = ...
        // Regex handles: type Name, type Name<T>, type Name(args)
        let types = extractMatches @"^\s*type\s+([\w]+)(?:[<|\(].*?[>|\)])?\s*=" content
        for typeName in types do
            let typeNode = TypeNode typeName
            graph.AddNode(typeNode)
            graph.AddEdge(parentNode, typeNode, Contains)

        // 3. Extract Top-Level Functions
        // let MyFunc ... =
        // Exclude 'let private', 'let mutable' inside functions (hard with regex, but ^\s*let helps)
        let functions = extractMatches @"^\s*let\s+(?:rec\s+)?(?:inline\s+)?([\w]+)\s+" content
        for funcName in functions do
            // Filter out common keywords that might be matched if regex is loose
            if funcName <> "mutable" && funcName <> "private" then
                let funcNode = FunctionNode funcName
                graph.AddNode(funcNode)
                graph.AddEdge(parentNode, funcNode, Contains)

    /// Recursively ingests all F# files in a directory
    let ingestDirectory (graph: KnowledgeGraph) (rootPath: string) =
        if Directory.Exists(rootPath) then
            let files = Directory.GetFiles(rootPath, "*.fs", SearchOption.AllDirectories)
            for file in files do
                // Skip obj and bin folders
                if not (file.Contains(Path.DirectorySeparatorChar.ToString() + "obj" + Path.DirectorySeparatorChar.ToString()) || 
                        file.Contains(Path.DirectorySeparatorChar.ToString() + "bin" + Path.DirectorySeparatorChar.ToString())) then
                    ingestFile graph file
