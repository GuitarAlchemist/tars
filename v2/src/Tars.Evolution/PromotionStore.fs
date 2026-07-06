namespace Tars.Evolution

/// Storage seam for the promotion pipeline.
///
/// `IPromotionStore` owns the three promotion stores — recurrence records,
/// lineage records, and probabilistic weights — behind one interface so
/// callers can swap the process-global disk store for an isolated in-memory
/// one. The in-memory adapter is what ends test cross-contamination: each
/// test gets its own store instead of racing the shared `~/.tars` state.

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open System.Collections.Concurrent

/// The promotion pipeline's storage abstraction: recurrence + lineage + weights.
type IPromotionStore =
    /// All recurrence records currently known to the store.
    abstract member GetRecurrence: unit -> RecurrenceRecord list
    /// Look up a recurrence record by its pattern name.
    abstract member TryGetRecurrence: name: string -> RecurrenceRecord option
    /// Insert or replace a recurrence record (keyed by PatternName).
    abstract member UpsertRecurrence: record: RecurrenceRecord -> unit
    /// All lineage records currently known to the store.
    abstract member GetLineage: unit -> LineageRecord list
    /// Append a lineage record (keyed by Id).
    abstract member AddLineage: record: LineageRecord -> unit
    /// Load the probabilistic weights associated with this store.
    abstract member LoadWeights: unit -> WeightedGrammar.WeightedRule list
    /// Persist the probabilistic weights associated with this store.
    abstract member SaveWeights: weights: WeightedGrammar.WeightedRule list -> unit
    /// Flush recurrence + lineage state to durable storage (no-op in memory).
    abstract member Flush: unit -> unit

/// In-memory adapter — hermetic and parallel-safe. Holds all state in
/// process memory and never touches disk. This is the adapter tests inject
/// to avoid contaminating the shared promotion store.
type InMemoryPromotionStore() =
    let recurrence = ConcurrentDictionary<string, RecurrenceRecord>()
    let lineage = ConcurrentDictionary<string, LineageRecord>()
    let mutable weights : WeightedGrammar.WeightedRule list = []

    interface IPromotionStore with
        member _.GetRecurrence() = recurrence.Values |> Seq.toList
        member _.TryGetRecurrence(name) =
            match recurrence.TryGetValue name with
            | true, r -> Some r
            | false, _ -> None
        member _.UpsertRecurrence(record) = recurrence.[record.PatternName] <- record
        member _.GetLineage() = lineage.Values |> Seq.toList
        member _.AddLineage(record) = lineage.[record.Id] <- record
        member _.LoadWeights() = weights
        member _.SaveWeights(w) = weights <- w
        member _.Flush() = ()

/// Disk adapter — the process-global store rooted at ~/.tars/promotion.
/// Lazily loads recurrence + lineage from disk on first access and flushes
/// them back after each decision, preserving the pipeline's prior behavior.
/// Weights are delegated to `WeightedGrammar` (weights.json in the same dir).
type DiskPromotionStore() =
    let recurrence = ConcurrentDictionary<string, RecurrenceRecord>()
    let lineage = ConcurrentDictionary<string, LineageRecord>()

    let jsonOptions =
        let o = JsonSerializerOptions(JsonSerializerDefaults.General)
        o.Converters.Add(JsonFSharpConverter())
        o.WriteIndented <- true
        o

    let getPromotionDir () =
        let dir =
            Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".tars", "promotion")
        if not (Directory.Exists dir) then
            Directory.CreateDirectory dir |> ignore
        dir

    let load () =
        try
            let dir = getPromotionDir ()
            let recurrencePath = Path.Combine(dir, "recurrence.json")
            let lineagePath = Path.Combine(dir, "lineage.json")

            if File.Exists recurrencePath then
                let json = File.ReadAllText recurrencePath
                let records = JsonSerializer.Deserialize<RecurrenceRecord list>(json, jsonOptions)
                for r in records do
                    recurrence.[r.PatternName] <- r

            if File.Exists lineagePath then
                let json = File.ReadAllText lineagePath
                let records = JsonSerializer.Deserialize<LineageRecord list>(json, jsonOptions)
                for r in records do
                    lineage.[r.Id] <- r

            Ok (recurrence.Count, lineage.Count)
        with ex ->
            Error $"Failed to load promotion state: {ex.Message}"

    let initialized =
        lazy (
            match load () with
            | Ok (r, l) ->
                if r > 0 || l > 0 then
                    Console.Error.WriteLine($"[Promotion] Loaded {r} recurrence records, {l} lineage records from disk")
            | Error err ->
                Console.Error.WriteLine($"[Promotion] {err}"))

    let ensureLoaded () = initialized.Force()

    let flush () =
        try
            let dir = getPromotionDir ()
            let recurrenceData = recurrence.Values |> Seq.toList
            let lineageData = lineage.Values |> Seq.toList
            File.WriteAllText(
                Path.Combine(dir, "recurrence.json"),
                JsonSerializer.Serialize(recurrenceData, jsonOptions))
            File.WriteAllText(
                Path.Combine(dir, "lineage.json"),
                JsonSerializer.Serialize(lineageData, jsonOptions))
        with _ -> () // best-effort, matches prior save semantics

    interface IPromotionStore with
        member _.GetRecurrence() =
            ensureLoaded ()
            recurrence.Values |> Seq.toList
        member _.TryGetRecurrence(name) =
            ensureLoaded ()
            match recurrence.TryGetValue name with
            | true, r -> Some r
            | false, _ -> None
        member _.UpsertRecurrence(record) =
            ensureLoaded ()
            recurrence.[record.PatternName] <- record
        member _.GetLineage() =
            ensureLoaded ()
            lineage.Values |> Seq.toList
        member _.AddLineage(record) =
            ensureLoaded ()
            lineage.[record.Id] <- record
        member _.LoadWeights() = WeightedGrammar.load ()
        member _.SaveWeights(w) = WeightedGrammar.save w
        member _.Flush() = flush ()
