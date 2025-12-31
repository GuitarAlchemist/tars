namespace Tars.Tools.Standard

open System
open System.IO
open System.Threading.Tasks
open Tars.Tools
open Tars.Core
open Tars.Knowledge
open Tars.LinkedData

module SymbolicTools =

    let private parseRelationType (s: string) =
        match s.ToLowerInvariant().Trim('\'', '"', '.', ' ', '(', ')') with
        | "isa" | "is_a" -> RelationType.IsA
        | "partof" | "part_of" -> RelationType.PartOf
        | "hasproperty" | "has_property" -> RelationType.HasProperty
        | "supports" -> RelationType.Supports
        | "contradicts" -> RelationType.Contradicts
        | "derivedfrom" | "derived_from" -> RelationType.DerivedFrom
        | "causes" -> RelationType.Causes
        | "prevents" -> RelationType.Prevents
        | "enables" -> RelationType.Enables
        | "precedes" -> RelationType.Precedes
        | "supersedes" -> RelationType.Supersedes
        | "mentions" -> RelationType.Mentions
        | "cites" -> RelationType.Cites
        | "implements" -> RelationType.Implements
        | _ -> RelationType.Custom s

    let private cleanArgument (args: string) =
        let s = args.Trim()
        // First, strip standard quotes at start/end
        let stripped = s.Trim('\'', '"')
        // Handle comments like "path.ttl (comment)" after stripping quotes
        let paren = stripped.IndexOf(" (")
        let final = if paren > 0 then stripped.Substring(0, paren).Trim() else stripped.Trim()
        // Also handle path with backslashes that might have trailing quotes
        final.Trim('\'', '"')

    /// Create the ingest_rdf tool
    let createIngestRdfTool (ledger: KnowledgeLedger) =
        Tool.Create(
            "ingest_rdf",
            "Ingests knowledge from an RDF file (Turtle format). Input: path to the .ttl file (string).",
            fun (args: string) ->
                task {
                    try
                        let rawPath = 
                            match ToolHelpers.tryParseStringArg args "path" with
                            | Some p -> p
                            | None -> args
                        let path = cleanArgument rawPath
                        
                        if String.IsNullOrWhiteSpace(path) then
                            return Microsoft.FSharp.Core.Result.Error "Please provide a file path."
                        else
                            let fullPath = 
                                if Path.IsPathRooted(path) then path
                                else Path.GetFullPath(path)

                            if not (File.Exists(fullPath)) then
                                return Microsoft.FSharp.Core.Result.Error (sprintf "File not found: %s. Please provide a valid path. Current directory: %s" path Environment.CurrentDirectory)
                            else
                                let! res = RdfParser.importFile ledger fullPath |> Async.StartAsTask
                                match res with
                                | Microsoft.FSharp.Core.Result.Ok count ->
                                    return Microsoft.FSharp.Core.Result.Ok (sprintf "Successfully ingested %d triples from %s." count path)
                                | Microsoft.FSharp.Core.Result.Error err ->
                                    return Microsoft.FSharp.Core.Result.Error (sprintf "Ingestion failed: %s" err)
                    with ex ->
                        return Microsoft.FSharp.Core.Result.Error (sprintf "Error ingesting RDF: %s" ex.Message)
                }
        )

    /// Create the query_ledger tool
    let createQueryLedgerTool (ledger: KnowledgeLedger) =
        Tool.Create(
            "query_ledger",
            "Queries the symbolic knowledge ledger for beliefs. Input: 'subject predicate object' (wildcard '?'). Or JSON.",
            fun (args: string) ->
                task {
                    try
                        let mutable subject: string option = None
                        let mutable predicate: RelationType option = None
                        let mutable obj: string option = None

                        // Try JSON first
                        match ToolHelpers.tryParseStringArg args "subject" with
                        | Some s when not (System.String.IsNullOrWhiteSpace s) -> subject <- Some s
                        | _ -> ()
                        
                        match ToolHelpers.tryParseStringArg args "predicate" with
                        | Some p when not (System.String.IsNullOrWhiteSpace p) -> predicate <- Some (parseRelationType p)
                        | _ -> ()

                        match ToolHelpers.tryParseStringArg args "object" with
                        | Some o when not (System.String.IsNullOrWhiteSpace o) -> obj <- Some o
                        | _ -> ()

                        // If all None, try parsing as 3/2/1-part space separated string
                        if subject.IsNone && predicate.IsNone && obj.IsNone then
                            let cleaned = cleanArgument args
                            let parts = cleaned.Split([| ' '; ','; '\t' |], StringSplitOptions.RemoveEmptyEntries)
                            if parts.Length >= 1 then
                                if parts.[0] <> "?" && not (parts.[0].StartsWith("?")) then subject <- Some (parts.[0].Trim('\'', '"'))
                            if parts.Length >= 2 then
                                if parts.[1] <> "?" && not (parts.[1].StartsWith("?")) then predicate <- Some (parseRelationType parts.[1])
                            if parts.Length >= 3 then
                                if parts.[2] <> "?" && not (parts.[2].StartsWith("?")) then obj <- Some (parts.[2].Trim('\'', '"', '.'))

                        let results = ledger.Query(?subject = subject, ?predicate = predicate, ?obj = obj)
                        
                        if Seq.isEmpty results then
                            return Microsoft.FSharp.Core.Result.Ok (sprintf "No matching beliefs found for query '%A %A %A'. Try a broader search using '?'." subject predicate obj)
                        else
                            let lines = 
                                results 
                                |> Seq.map (fun (b: Belief) -> sprintf "- %s %s %s (Confidence: %.2f)" (b.Subject.ToString()) (b.Predicate.ToString()) (b.Object.ToString()) b.Confidence)
                                |> String.concat "\n"
                            return Microsoft.FSharp.Core.Result.Ok (sprintf "Found %d matches:\n%s" (Seq.length results) lines)
                    with ex ->
                        return Microsoft.FSharp.Core.Result.Error (sprintf "Error querying ledger: %s" ex.Message)
                }
        )
