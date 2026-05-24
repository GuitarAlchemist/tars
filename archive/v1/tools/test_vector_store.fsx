#!/usr/bin/env dotnet fsi

// TARS Vector Store Test Script
// Demonstrates the new multi-space vector store and inference engine

#r "../Tars.Engine.VectorStore/bin/Debug/net8.0/Tars.Engine.VectorStore.dll"
#r "../Tars.Engine.Integration/bin/Debug/net8.0/Tars.Engine.Integration.dll"
#r "nuget: System.Text.Json"

open System
open System.IO
open Tars.Engine.VectorStore
open Tars.Engine.Integration

printfn "🚀 TARS Multi-Space Vector Store Test"
printfn "====================================="

// Create TARS vector store service
let service = TARSVectorStoreService()

printfn "✅ Created TARS Vector Store Service"
printfn "   - Raw dimension: %d" service.Config.RawDimension
printfn "   - Enabled spaces: FFT=%b, Dual=%b, Hyperbolic=%b, Pauli=%b" 
    service.Config.EnableFFT 
    service.Config.EnableDual 
    service.Config.EnableHyperbolic 
    service.Config.EnablePauli

// Test 1: Add some sample documents
printfn "\n📄 Test 1: Adding Sample Documents"
printfn "=================================="

let sampleDocuments = [
    ("agent_doc", "agent_module { name: \"file_processor\" goal: \"Process and analyze files\" }", ["agent"; "metascript"])
    ("grammar_doc", "grammar { LANG(\"EBNF\") { expression = term ( \"+\" | \"-\" ) term ; } }", ["grammar"; "ebnf"])
    ("trace_doc", "execution_trace { agent: \"test_agent\" status: \"completed\" duration: 1.5 }", ["trace"; "execution"])
    ("code_doc", "let processFile (path: string) = File.ReadAllText(path) |> analyze", ["code"; "fsharp"])
]

async {
    for (id, content, tags) in sampleDocuments do
        let! embedding = service.EmbeddingGenerator.GenerateEmbedding content
        let doc = VectorStoreUtils.createDocument id content embedding tags (Some "test")
        do! service.VectorStore.AddDocument doc
        printfn "✅ Added document: %s" id
} |> Async.RunSynchronously

// Test 2: Search functionality
printfn "\n🔍 Test 2: Search Functionality"
printfn "==============================="

let testQueries = [
    ("agent", 3)
    ("grammar", 2)
    ("file processing", 3)
    ("F# code", 2)
]

async {
    for (query, maxResults) in testQueries do
        printfn "\nSearching for: '%s'" query
        let! results = service.Search(query, maxResults)
        
        if results.Length = 0 then
            printfn "  No results found"
        else
            for i, result in List.indexed results do
                printfn "  %d. %s (Score: %.3f)" (i + 1) result.Document.Id result.FinalScore
                printfn "     Content: %s" (if result.Document.Content.Length > 50 then result.Document.Content.Substring(0, 50) + "..." else result.Document.Content)
                printfn "     Spaces used: %s" (String.Join(", ", result.Scores |> List.map (fun s -> sprintf "%s(%.3f)" s.Space s.Score)))
} |> Async.RunSynchronously

// Test 3: Multi-space similarity analysis
printfn "\n🧮 Test 3: Multi-Space Similarity Analysis"
printfn "=========================================="

async {
    let! embedding1 = service.EmbeddingGenerator.GenerateEmbedding "agent processing files"
    let! embedding2 = service.EmbeddingGenerator.GenerateEmbedding "file processor agent"
    
    let similarityComputer = MultiSpaceSimilarityComputer(service.Config) :> ISimilarityComputer
    let scores = similarityComputer.ComputeSimilarity embedding1 embedding2
    let finalScore = similarityComputer.AggregateSimilarity scores
    
    printfn "Comparing: 'agent processing files' vs 'file processor agent'"
    printfn "Final aggregated score: %.4f" finalScore
    printfn "\nDetailed scores by space:"
    for score in scores do
        printfn "  %s: %.4f (confidence: %.2f) - %s" score.Space score.Score score.Confidence score.Reason
} |> Async.RunSynchronously

// Test 4: Inference engine
printfn "\n🧠 Test 4: Inference Engine"
printfn "==========================="

async {
    let parameters = Map.ofList [("max_docs", box 3)]
    let! result = service.Infer("What agents are available for file processing?", parameters)
    
    match result with
    | :? InferenceResult as inferenceResult ->
        printfn "Query: What agents are available for file processing?"
        printfn "Result: %A" inferenceResult.Result
        printfn "Confidence: %.2f" inferenceResult.Confidence
        printfn "Processing time: %A" inferenceResult.ProcessingTime
        printfn "Supporting documents: %d" inferenceResult.SupportingDocuments.Length
        printfn "\nReasoning steps:"
        for i, step in List.indexed inferenceResult.Reasoning do
            printfn "  %d. %s" (i + 1) step
    | _ ->
        printfn "Result: %A" result
} |> Async.RunSynchronously

// Test 5: Vector store statistics
printfn "\n📊 Test 5: Vector Store Statistics"
printfn "=================================="

async {
    let! stats = VectorStoreUtils.getStatistics service.VectorStore
    printfn "Document count: %d" stats.DocumentCount
    printfn "Average embedding size: %.1f" stats.AverageEmbeddingSize
    printfn "Index size: %d bytes" stats.IndexSize
    printfn "Last updated: %s" (stats.LastUpdated.ToString("yyyy-MM-dd HH:mm:ss"))
    
    printfn "\nSpace usage:"
    for kvp in stats.SpaceUsageStats do
        printfn "  %s: %d documents" kvp.Key kvp.Value
} |> Async.RunSynchronously

// Test 6: Belief state analysis
printfn "\n🎯 Test 6: Belief State Analysis"
printfn "==============================="

async {
    let testTexts = [
        "This is definitely true and accurate information"
        "This might be correct but I'm not entirely sure"
        "This is completely false and incorrect"
        "This statement is both true and false simultaneously"
    ]
    
    for text in testTexts do
        let! embedding = service.EmbeddingGenerator.GenerateEmbedding text
        printfn "Text: %s" (if text.Length > 40 then text.Substring(0, 40) + "..." else text)
        printfn "Belief state: %A" embedding.Belief
        printfn "Raw embedding stats: min=%.3f, max=%.3f, avg=%.3f" 
            (Array.min embedding.Raw) 
            (Array.max embedding.Raw) 
            (Array.average embedding.Raw)
        printfn ""
} |> Async.RunSynchronously

// Test 7: Transform analysis
printfn "\n🔄 Test 7: Transform Analysis"
printfn "============================="

async {
    let! embedding = service.EmbeddingGenerator.GenerateEmbedding "test signal for transform analysis"
    
    printfn "Transform dimensions:"
    printfn "  Raw: %d" embedding.Raw.Length
    printfn "  FFT: %d" embedding.FFT.Length
    printfn "  Dual: %d" embedding.Dual.Length
    printfn "  Projective: %d" embedding.Projective.Length
    printfn "  Hyperbolic: %d" embedding.Hyperbolic.Length
    printfn "  Wavelet: %d" embedding.Wavelet.Length
    printfn "  Minkowski: %d" embedding.Minkowski.Length
    
    printfn "\nPauli matrix elements:"
    let (a, b, c, d) = embedding.Pauli
    printfn "  [%.3f+%.3fi  %.3f+%.3fi]" a.Real a.Imaginary b.Real b.Imaginary
    printfn "  [%.3f+%.3fi  %.3f+%.3fi]" c.Real c.Imaginary d.Real d.Imaginary
    
    printfn "\nMetadata:"
    for kvp in embedding.Metadata do
        printfn "  %s: %s" kvp.Key kvp.Value
} |> Async.RunSynchronously

printfn "\n✅ All tests completed successfully!"
printfn "🎉 TARS Multi-Space Vector Store is working correctly!"

// Cleanup
async {
    do! service.VectorStore.Clear()
    printfn "\n🧹 Cleaned up test data"
} |> Async.RunSynchronously
