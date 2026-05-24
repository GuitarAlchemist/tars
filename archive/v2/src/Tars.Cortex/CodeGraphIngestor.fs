namespace Tars.Cortex

open System.IO
open Tars.Core

module CodeGraphIngestor =

    /// Ingests a single F# file into the Knowledge Graph using AST parsing
    let ingestFile (graph: IGraphService) (filePath: string) =
        async { do! AstIngestor.ingestFile graph filePath }

    /// Recursively ingests all F# files in a directory
    let ingestDirectory (graph: IGraphService) (rootPath: string) =
        async {
            if Directory.Exists(rootPath) then
                do! AstIngestor.ingestDirectory graph rootPath
        }
