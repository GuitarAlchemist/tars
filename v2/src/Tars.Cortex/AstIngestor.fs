namespace Tars.Cortex

open System
open System.IO
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open Tars.Core

/// AST Ingestor for parsing F# source code and ingesting into the knowledge graph.
module AstIngestor =

    let private checker = FSharpChecker.Create()

    let private getLongIdentName (lid: LongIdent) =
        String.Join(".", lid |> List.map (fun i -> i.idText))

    let private getSynLongIdentName (synLid: SynLongIdent) =
        match synLid with
        | SynLongIdent(id, _, _) -> getLongIdentName id

    /// Extracted entity with its parent
    type private ExtractedEntity =
        { Entity: TarsEntity
          Parent: TarsEntity option }

    /// Recursively extracts entities from module declarations (synchronous collection)
    let rec private collectFromModuleDecl (parentNode: TarsEntity) (decl: SynModuleDecl) : ExtractedEntity list =
        match decl with
        | SynModuleDecl.NestedModule(componentInfo, _, decls, _, _, _) ->
            match componentInfo with
            | SynComponentInfo(_, _, _, id, _, _, _, _) ->
                let name = getLongIdentName id

                let node =
                    CodeModuleE
                        { Path = name
                          Namespace = name
                          Dependencies = []
                          Complexity = 0.0
                          LineCount = 0 }

                let current =
                    { Entity = node
                      Parent = Some parentNode }

                let children = decls |> List.collect (collectFromModuleDecl node)
                current :: children

        | SynModuleDecl.Types(typeDefns, _) ->
            typeDefns
            |> List.collect (fun typeDefn ->
                match typeDefn with
                | SynTypeDefn(typeInfo, _, members, _, _, _) ->
                    match typeInfo with
                    | SynComponentInfo(_, _, _, id, _, _, _, _) ->
                        let name = getLongIdentName id

                        let node =
                            ConceptE
                                { Name = name
                                  Description = "Type"
                                  RelatedConcepts = [] }

                        let current =
                            { Entity = node
                              Parent = Some parentNode }

                        let memberEntities =
                            members
                            |> List.choose (fun m ->
                                match m with
                                | SynMemberDefn.Member(binding, _) ->
                                    match binding with
                                    | SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _) ->
                                        match pat with
                                        | SynPat.LongIdent(longDotId, _, _, _, _, _) ->
                                            let funcName = getSynLongIdentName longDotId

                                            Some
                                                { Entity = FunctionE funcName
                                                  Parent = Some node }
                                        | _ -> None
                                | _ -> None)

                        current :: memberEntities)

        | SynModuleDecl.Let(_, bindings, _) ->
            bindings
            |> List.choose (fun binding ->
                match binding with
                | SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _) ->
                    match pat with
                    | SynPat.LongIdent(longDotId, _, _, _, _, _) ->
                        let funcName = getSynLongIdentName longDotId

                        Some
                            { Entity = FunctionE funcName
                              Parent = Some parentNode }
                    | SynPat.Named(synIdent, _, _, _) ->
                        // SynIdent contains an Ident we can access
                        let identName =
                            match synIdent with
                            | SynIdent(ident, _) -> ident.idText

                        Some
                            { Entity = FunctionE identName
                              Parent = Some parentNode }
                    | _ -> None)

        | _ -> []

    /// Ingests a single F# file into the knowledge graph
    let ingestFile (graph: IGraphService) (filePath: string) =
        async {
            if File.Exists(filePath) then
                try
                    let content = File.ReadAllText(filePath)
                    let sourceText = SourceText.ofString content

                    let options =
                        { FSharpParsingOptions.Default with
                            SourceFiles = [| filePath |] }

                    let! parseRes = checker.ParseFile(filePath, sourceText, options)

                    match parseRes.ParseTree with
                    | ParsedInput.ImplFile(impl) ->
                        let fileName = impl.FileName
                        let modules = impl.Contents

                        let shortName = Path.GetFileName(fileName: string)
                        let fileNode = FileE shortName
                        do! graph.AddNodeAsync(fileNode) |> Async.AwaitTask |> Async.Ignore

                        // Collect all entities first (synchronous)
                        let allEntities =
                            modules
                            |> List.collect (fun moduleOrNs ->
                                match moduleOrNs with
                                | SynModuleOrNamespace(id, _, _, decls, _, _, _, _, _) ->
                                    let name = getLongIdentName id

                                    let moduleNode =
                                        CodeModuleE
                                            { Path = name
                                              Namespace = name
                                              Dependencies = []
                                              Complexity = 0.0
                                              LineCount = 0 }

                                    let current =
                                        { Entity = moduleNode
                                          Parent = Some fileNode }

                                    let children = decls |> List.collect (collectFromModuleDecl moduleNode)
                                    current :: children)

                        // Then ingest them (async)
                        for entity in allEntities do
                            do! graph.AddNodeAsync(entity.Entity) |> Async.AwaitTask |> Async.Ignore

                            match entity.Parent with
                            | Some parent ->
                                do!
                                    graph.AddFactAsync(TarsFact.Contains(parent, entity.Entity))
                                    |> Async.AwaitTask
                                    |> Async.Ignore
                            | None -> ()
                    | _ -> ()
                with ex ->
                    printfn "Failed to parse %s: %s" filePath ex.Message
        }

    /// Recursively ingests all F# files in a directory
    let ingestDirectory (graph: IGraphService) (rootPath: string) =
        async {
            if Directory.Exists(rootPath) then
                let files =
                    Directory.GetFiles(rootPath, "*.fs", SearchOption.AllDirectories)
                    |> Array.filter (fun f -> not (f.Contains("obj") || f.Contains("bin")))

                for file in files do
                    do! ingestFile graph file
        }
