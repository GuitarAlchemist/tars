namespace Tars.Cortex

open System
open System.Collections.Concurrent
open System.IO
open System.Text.Json

/// Stores corpus-wide term frequencies to enable real BM25/IDF scoring
type TermFrequencyStore(storagePath: string) =
    let mutable documentCount = 0L
    let termDocs = ConcurrentDictionary<string, int64>()
    
    let load() =
        if File.Exists storagePath then
            try
                let json = File.ReadAllText storagePath
                let data = JsonSerializer.Deserialize<{| DocCount: int64; TermDocs: Map<string, int64> |}>(json)
                documentCount <- data.DocCount
                for kv in data.TermDocs do termDocs.[kv.Key] <- kv.Value
            with _ -> ()
            
    let save() =
        try
            let data = {| DocCount = documentCount; TermDocs = termDocs |> Seq.map (fun kv -> kv.Key, kv.Value) |> Map.ofSeq |}
            File.WriteAllText(storagePath, JsonSerializer.Serialize(data))
        with _ -> ()

    do load()

    /// Increment document count and update term presence
    member _.RecordDocument(text: string) =
        let words = 
            text.ToLowerInvariant().Split([| ' '; ','; '.'; '!'; '?'; ';'; ':' |], StringSplitOptions.RemoveEmptyEntries)
            |> Array.distinct
        
        System.Threading.Interlocked.Increment(&documentCount) |> ignore
        for word in words do
            termDocs.AddOrUpdate(word, 1L, (fun _ count -> count + 1L)) |> ignore
        save()

    /// Get IDF for a term
    member _.GetIdf(term: string) =
        let termCount = match termDocs.TryGetValue(term) with | true, v -> v | _ -> 0L
        if termCount = 0L || documentCount = 0L then 1.0f
        else
            // Standard BM25 IDF: log( (N - n + 0.5) / (n + 0.5) + 1 )
            float32 (Math.Log( (float documentCount - float termCount + 0.5) / (float termCount + 0.5) + 1.0 ))

    member _.DocumentCount = documentCount
    member _.VocabularySize = termDocs.Count
