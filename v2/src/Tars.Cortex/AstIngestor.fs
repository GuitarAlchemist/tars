namespace Tars.Cortex

open System
open System.IO
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open Tars.Core

module AstIngestor =

    let private checker = FSharpChecker.Create()

    let private getLongIdentName (lid: LongIdent) =
        String.Join(".", lid |> List.map (fun i -> i.idText))

    let private getSynLongIdentName (synLid: SynLongIdent) =
        match synLid with
        | SynLongIdent(id, _, _) -> getLongIdentName id

    let rec private walkSynModuleDecl (graph: KnowledgeGraph) (parentNode: GraphNode) (decl: SynModuleDecl) =
        match decl with
        | SynModuleDecl.NestedModule(moduleInfo=moduleInfo; decls=decls) ->
            match moduleInfo with
            | SynComponentInfo(longId=id) ->
                let name = getLongIdentName id
                let node = ModuleNode name
                graph.AddNode(node)
                graph.AddEdge(parentNode, node, Contains)
                for d in decls do walkSynModuleDecl graph node d

        | SynModuleDecl.Types(typeDefns, _) ->
            for typeDefn in typeDefns do
                match typeDefn with
                | SynTypeDefn(typeInfo=typeInfo; typeRepr=_repr; members=members) ->
                    match typeInfo with
                    | SynComponentInfo(longId=id) ->
                        let name = getLongIdentName id
                        let node = TypeNode name
                        graph.AddNode(node)
                        graph.AddEdge(parentNode, node, Contains)

                        // Members
                        for m in members do
                            match m with
                            | SynMemberDefn.Member(memberDefn=binding) ->
                                match binding with
                                | SynBinding(headPat=pat) ->
                                    match pat with
                                    | SynPat.LongIdent(longDotId=longDotId) ->
                                        let funcName = getSynLongIdentName longDotId
                                        let funcNode = FunctionNode funcName
                                        graph.AddNode(funcNode)
                                        graph.AddEdge(node, funcNode, Contains)
                                    | _ -> ()
                            | _ -> ()

        | SynModuleDecl.Let(bindings=bindings) ->
            for binding in bindings do
                match binding with
                | SynBinding(headPat=pat) ->
                    match pat with
                    | SynPat.LongIdent(longDotId=longDotId) ->
                        let funcName = getSynLongIdentName longDotId
                        let funcNode = FunctionNode funcName
                        graph.AddNode(funcNode)
                        graph.AddEdge(parentNode, funcNode, Contains)
                    | SynPat.Named(ident=ident) ->
                         let (SynIdent(ident, _)) = ident
                         let node = FunctionNode ident.idText
                         graph.AddNode(node)
                         graph.AddEdge(parentNode, node, Contains)
                    | _ -> ()

        | _ -> ()

    let ingestFile (graph: KnowledgeGraph) (filePath: string) =
        async {
            try
                let content = File.ReadAllText(filePath)
                let sourceText = SourceText.ofString content
                let! parseRes = checker.ParseFile(filePath, sourceText, FSharpParsingOptions.Default)
                
                let tree = parseRes.ParseTree
                match tree with
                | ParsedInput.ImplFile(parsedImplFileInput) ->
                    let fileName = Path.GetFileName(filePath)
                    let fileNode = FileNode fileName
                    graph.AddNode(fileNode)

                    for moduleOrNs in parsedImplFileInput.Contents do
                        match moduleOrNs with
                        | SynModuleOrNamespace(longId=id; decls=decls) ->
                            let name = getLongIdentName id
                            let node = ModuleNode name
                            graph.AddNode(node)
                            graph.AddEdge(fileNode, node, Contains)

                            for d in decls do walkSynModuleDecl graph node d
                | _ -> ()
            with ex ->
                printfn "Failed to parse %s: %s" filePath ex.Message
        }

    let ingestDirectory (graph: KnowledgeGraph) (rootPath: string) =
        async {
            if Directory.Exists(rootPath) then
                let files = Directory.GetFiles(rootPath, "*.fs", SearchOption.AllDirectories)
                for file in files do
                    if not (file.Contains(Path.DirectorySeparatorChar.ToString() + "obj" + Path.DirectorySeparatorChar.ToString()) || 
                            file.Contains(Path.DirectorySeparatorChar.ToString() + "bin" + Path.DirectorySeparatorChar.ToString())) then
                        do! ingestFile graph file
        }
