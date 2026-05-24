namespace Tars.LinkedData

open System
open VDS.RDF
open VDS.RDF.Parsing
open Tars.Core
open Tars.Knowledge

/// Parse RDF files (Turtle, N-Triples, RDF/XML) into beliefs
module RdfParser =
    
    /// Parse an RDF file and extract triples
    let parseFile (filePath: string) : Microsoft.FSharp.Core.Result<seq<string * string * string>, string> =
        try
            let graph = new Graph()
            
            // Auto-detect format and parse
            let parser =
                if filePath.EndsWith(".ttl") || filePath.EndsWith(".turtle") then
                    new TurtleParser() :> IRdfReader
                elif filePath.EndsWith(".nt") || filePath.EndsWith(".ntriples") then
                    new NTriplesParser() :> IRdfReader
                elif filePath.EndsWith(".rdf") || filePath.EndsWith(".xml") then
                    new RdfXmlParser() :> IRdfReader
                else
                    new TurtleParser() :> IRdfReader  // Default to Turtle
            
            parser.Load(graph, filePath)
            
            let triples =
                graph.Triples
                |> Seq.map (fun triple ->
                    let subject = triple.Subject.ToString()
                    let predicate = triple.Predicate.ToString()
                    // Extract literal values properly
                    let obj = 
                        match triple.Object with
                        | :? VDS.RDF.ILiteralNode as lit -> lit.Value
                        | node -> node.ToString()
                    (subject, predicate, obj))
            
            Microsoft.FSharp.Core.Result.Ok triples
        with ex ->
            Microsoft.FSharp.Core.Result.Error $"Failed to parse RDF file: {ex.Message}"
    
    /// Clean up URIs - extract local names or literal values
    let cleanUri (uri: string) =
        // Handle literals like "value"^^xsd:type or "value"@lang
        if uri.StartsWith("\"") then
            let endQuote = uri.IndexOf("\"", 1)
            if endQuote > 0 then uri.Substring(1, endQuote - 1) else uri.Trim('"')
        else
            // Handle URIs
            let uri = uri.Trim('<', '>')
            let hash = uri.LastIndexOf('#')
            let slash = uri.LastIndexOf('/')
            if hash >= 0 then uri.Substring(hash + 1)
            elif slash >= 0 then uri.Substring(slash + 1)
            else uri

    /// Convert RDF triples to TARS beliefs
    let triplesToBeliefs (sourceUri: Uri) (triples: seq<string * string * string>) : Belief list =
        triples
        |> Seq.map (fun (subject, predicate, obj) ->
            let cleanSubject = cleanUri subject
            let cleanPredicate = cleanUri predicate
            let cleanObject = cleanUri obj
            
            // Map common RDF predicates to TARS relation types
            let relationType =
                match cleanPredicate.ToLowerInvariant() with
                | "type" | "rdf:type" | "a" -> RelationType.IsA
                | "subclassof" -> RelationType.IsA
                | "hasproperty" | "property" -> RelationType.HasProperty
                | "partof" -> RelationType.PartOf
                | "derivedfrom" | "basedon" -> RelationType.DerivedFrom
                | "causes" -> RelationType.Causes
                | "prevents" -> RelationType.Prevents
                | "supports" -> RelationType.Supports
                | "opposes" -> RelationType.Contradicts
                | _ -> RelationType.Custom cleanPredicate
            
            // Create provenance from source URI
            let provenance = Provenance.FromExternal(sourceUri, None, 0.95)
            
            Belief.create cleanSubject relationType cleanObject provenance)
        |> Seq.toList
    
    /// Import RDF file directly into knowledge ledger
    let importFile 
        (ledger: KnowledgeLedger) 
        (filePath: string)
        : Async<Microsoft.FSharp.Core.Result<int, string>> =
        async {
            let log = Logging.withCategory "RDF"
            log.Info $"📥 Parsing RDF file: {filePath}"
            
            match parseFile filePath with
            | Microsoft.FSharp.Core.Result.Error err -> 
                log.Error($"Parse failed: {err}", null)
                return Microsoft.FSharp.Core.Result.Error err
                
            | Microsoft.FSharp.Core.Result.Ok triples ->
                let tripleList = triples |> Seq.toList
                log.Info $"   Found {tripleList.Length} triples"
                
                let sourceUri = Uri($"file:///{filePath}")
                let beliefs = triplesToBeliefs sourceUri (Seq.ofList tripleList)
                
                log.Info $"   Converting to {beliefs.Length} beliefs..."
                
                let mutable successCount = 0
                let mutable errorCount = 0
                
                for belief in beliefs do
                    // Explicitly await the Task Result
                    let! assertionTask = ledger.Assert(belief, AgentId.System) |> Async.AwaitTask
                    
                    // The result from Assert is a Result<BeliefId, string>
                    // We need to match it explicitly and map to our qualified Result type if needed
                    // But here we just use it for control flow
                    match assertionTask with
                    | Microsoft.FSharp.Core.Result.Ok _ -> successCount <- successCount + 1
                    | Microsoft.FSharp.Core.Result.Error err -> 
                        errorCount <- errorCount + 1
                        if errorCount <= 5 then  // Only log first 5 errors
                            log.Warn $"Failed to assert: {err}"
                
                log.Info $"✅ Imported {successCount} beliefs, {errorCount} errors"
                return Microsoft.FSharp.Core.Result.Ok successCount
        }
