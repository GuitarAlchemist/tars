// Adaptive Memoization and Query Support for TARS Closure Factory
// Implements predicate-based memoization, LINQ-equivalent queries, and advanced caching strategies

namespace TarsEngine.FSharp.Core.Mathematics

open System
open System.Collections.Concurrent
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// Adaptive memoization cache entry
type MemoizationEntry<'T> = {
    Value: 'T
    Timestamp: DateTime
    AccessCount: int64
    ComputationCost: TimeSpan
    Predicate: obj -> bool
    IsValid: bool
}

/// Cache eviction strategy
type EvictionStrategy =
    | LeastRecentlyUsed
    | LeastFrequentlyUsed
    | TimeToLive of TimeSpan
    | PredicateBased of (obj -> bool)
    | CostBased of float

/// Query operation types for closure results
type QueryOperation<'T> =
    | Where of ('T -> bool)
    | Select of ('T -> 'U)
    | SelectMany of ('T -> seq<'U>)
    | GroupBy of ('T -> 'K)
    | OrderBy of ('T -> 'K)
    | Take of int
    | Skip of int
    | Aggregate of ('T -> 'T -> 'T)

/// Query builder for closure results
type ClosureQueryBuilder() =
    member _.Yield(x) = Seq.singleton x
    member _.YieldFrom(xs) = xs
    member _.For(xs, f) = Seq.collect f xs
    member _.Combine(xs, ys) = Seq.append xs ys
    member _.Zero() = Seq.empty
    member _.Delay(f) = f
    member _.Run(f) = f()

/// Adaptive Memoization and Query Support Module
module AdaptiveMemoizationAndQuerySupport =
    
    let closureQuery = ClosureQueryBuilder()
    
    // ============================================================================
    // ADAPTIVE MEMOIZATION
    // ============================================================================
    
    /// Create adaptive memoization cache
    let createAdaptiveMemoizationCache<'TKey, 'TValue when 'TKey : comparison> 
        (maxSize: int) 
        (evictionStrategy: EvictionStrategy) 
        (logger: ILogger) =
        
        let cache = ConcurrentDictionary<'TKey, MemoizationEntry<'TValue>>()
        let accessTimes = ConcurrentDictionary<'TKey, DateTime>()
        let mutable totalHits = 0L
        let mutable totalMisses = 0L
        
        let shouldEvict (key: 'TKey) (entry: MemoizationEntry<'TValue>) =
            match evictionStrategy with
            | LeastRecentlyUsed -> 
                accessTimes.TryGetValue(key) 
                |> function 
                | (true, time) -> DateTime.UtcNow - time > TimeSpan.FromMinutes(30.0)
                | _ -> true
            | LeastFrequentlyUsed -> entry.AccessCount < 5L
            | TimeToLive ttl -> DateTime.UtcNow - entry.Timestamp > ttl
            | PredicateBased predicate -> not (predicate (box entry.Value))
            | CostBased threshold -> entry.ComputationCost.TotalMilliseconds < threshold
        
        let evictIfNeeded () =
            if cache.Count >= maxSize then
                let candidatesForEviction = 
                    cache 
                    |> Seq.filter (fun kvp -> shouldEvict kvp.Key kvp.Value)
                    |> Seq.take (maxSize / 4) // Evict 25% when full
                    |> Seq.map (fun kvp -> kvp.Key)
                    |> Seq.toList
                
                for key in candidatesForEviction do
                    cache.TryRemove(key) |> ignore
                    accessTimes.TryRemove(key) |> ignore
                
                logger.LogInformation("🗑️ Evicted {Count} entries from adaptive cache", candidatesForEviction.Length)
        
        {|
            Get = fun (key: 'TKey) ->
                match cache.TryGetValue(key) with
                | (true, entry) when entry.IsValid ->
                    accessTimes.[key] <- DateTime.UtcNow
                    let updatedEntry = { entry with AccessCount = entry.AccessCount + 1L }
                    cache.[key] <- updatedEntry
                    System.Threading.Interlocked.Increment(&totalHits) |> ignore
                    Some entry.Value
                | _ ->
                    System.Threading.Interlocked.Increment(&totalMisses) |> ignore
                    None
            
            Set = fun (key: 'TKey) (value: 'TValue) (computationCost: TimeSpan) (predicate: obj -> bool) ->
                evictIfNeeded()
                let entry = {
                    Value = value
                    Timestamp = DateTime.UtcNow
                    AccessCount = 1L
                    ComputationCost = computationCost
                    Predicate = predicate
                    IsValid = true
                }
                cache.[key] <- entry
                accessTimes.[key] <- DateTime.UtcNow
                logger.LogDebug("💾 Cached result for key with cost {Cost}ms", computationCost.TotalMilliseconds)
            
            Invalidate = fun (predicate: 'TKey -> 'TValue -> bool) ->
                let keysToInvalidate = 
                    cache 
                    |> Seq.filter (fun kvp -> predicate kvp.Key kvp.Value.Value)
                    |> Seq.map (fun kvp -> kvp.Key)
                    |> Seq.toList
                
                for key in keysToInvalidate do
                    match cache.TryGetValue(key) with
                    | (true, entry) -> 
                        cache.[key] <- { entry with IsValid = false }
                    | _ -> ()
                
                logger.LogInformation("❌ Invalidated {Count} cache entries", keysToInvalidate.Length)
            
            GetStatistics = fun () -> {|
                TotalEntries = cache.Count
                HitRate = if totalHits + totalMisses > 0L then float totalHits / float (totalHits + totalMisses) else 0.0
                TotalHits = totalHits
                TotalMisses = totalMisses
                EvictionStrategy = evictionStrategy
                MaxSize = maxSize
            |}
            
            Clear = fun () ->
                cache.Clear()
                accessTimes.Clear()
                totalHits <- 0L
                totalMisses <- 0L
                logger.LogInformation("🧹 Cleared adaptive memoization cache")
        |}
    
    /// Create adaptive memoization closure
    let createAdaptiveMemoizedClosure<'TInput, 'TOutput when 'TInput : comparison> 
        (computation: 'TInput -> Async<'TOutput>) 
        (shouldMemoize: 'TInput -> 'TOutput -> bool)
        (maxCacheSize: int)
        (evictionStrategy: EvictionStrategy)
        (logger: ILogger) =
        
        let cache = createAdaptiveMemoizationCache<'TInput, 'TOutput> maxCacheSize evictionStrategy logger
        
        fun (input: 'TInput) ->
            async {
                match cache.Get input with
                | Some cachedResult ->
                    logger.LogDebug("🎯 Cache hit for adaptive memoized closure")
                    return cachedResult
                | None ->
                    let startTime = DateTime.UtcNow
                    let! result = computation input
                    let computationTime = DateTime.UtcNow - startTime
                    
                    if shouldMemoize input result then
                        let predicate = fun obj -> 
                            match obj with
                            | :? 'TOutput as output -> shouldMemoize input output
                            | _ -> false
                        cache.Set input result computationTime predicate
                        logger.LogDebug("💾 Stored result in adaptive cache")
                    else
                        logger.LogDebug("🚫 Result not memoized based on predicate")
                    
                    return result
            }
    
    // ============================================================================
    // LINQ-EQUIVALENT QUERY SUPPORT
    // ============================================================================
    
    /// Query operations for closure results
    let queryClosureResults<'T> (results: seq<'T>) =
        {|
            Where = fun (predicate: 'T -> bool) -> results |> Seq.filter predicate
            Select = fun (selector: 'T -> 'U) -> results |> Seq.map selector
            SelectMany = fun (selector: 'T -> seq<'U>) -> results |> Seq.collect selector
            GroupBy = fun (keySelector: 'T -> 'K when 'K : comparison) -> 
                results |> Seq.groupBy keySelector |> Seq.map (fun (k, vs) -> (k, Seq.toList vs))
            OrderBy = fun (keySelector: 'T -> 'K when 'K : comparison) -> 
                results |> Seq.sortBy keySelector
            OrderByDescending = fun (keySelector: 'T -> 'K when 'K : comparison) -> 
                results |> Seq.sortByDescending keySelector
            Take = fun (count: int) -> results |> Seq.take count
            Skip = fun (count: int) -> results |> Seq.skip count
            First = fun () -> results |> Seq.head
            FirstOrDefault = fun () -> results |> Seq.tryHead
            Last = fun () -> results |> Seq.last
            Count = fun () -> results |> Seq.length
            Any = fun (predicate: 'T -> bool) -> results |> Seq.exists predicate
            All = fun (predicate: 'T -> bool) -> results |> Seq.forall predicate
            Aggregate = fun (accumulator: 'T -> 'T -> 'T) -> results |> Seq.reduce accumulator
            Sum = fun (selector: 'T -> float) -> results |> Seq.sumBy selector
            Average = fun (selector: 'T -> float) -> results |> Seq.averageBy selector
            Min = fun (selector: 'T -> 'K when 'K : comparison) -> results |> Seq.minBy selector
            Max = fun (selector: 'T -> 'K when 'K : comparison) -> results |> Seq.maxBy selector
            Distinct = fun () -> results |> Seq.distinct
            Union = fun (other: seq<'T>) -> Seq.append results other |> Seq.distinct
            Intersect = fun (other: seq<'T>) -> 
                let otherSet = Set.ofSeq other
                results |> Seq.filter (fun x -> otherSet.Contains(x))
            Except = fun (other: seq<'T>) -> 
                let otherSet = Set.ofSeq other
                results |> Seq.filter (fun x -> not (otherSet.Contains(x)))
            ToList = fun () -> results |> Seq.toList
            ToArray = fun () -> results |> Seq.toArray
            ToMap = fun (keySelector: 'T -> 'K when 'K : comparison) -> 
                results |> Seq.map (fun x -> (keySelector x, x)) |> Map.ofSeq
        |}
    
    /// Advanced query builder with fluent interface
    let createAdvancedQueryBuilder<'T> (source: seq<'T>) =
        let mutable currentQuery = source
        
        {|
            Where = fun (predicate: 'T -> bool) ->
                currentQuery <- currentQuery |> Seq.filter predicate
                createAdvancedQueryBuilder currentQuery
            
            Select = fun (selector: 'T -> 'U) ->
                let newQuery = currentQuery |> Seq.map selector
                createAdvancedQueryBuilder newQuery
            
            SelectMany = fun (selector: 'T -> seq<'U>) ->
                let newQuery = currentQuery |> Seq.collect selector
                createAdvancedQueryBuilder newQuery
            
            GroupBy = fun (keySelector: 'T -> 'K when 'K : comparison) ->
                let newQuery = currentQuery |> Seq.groupBy keySelector
                createAdvancedQueryBuilder newQuery
            
            OrderBy = fun (keySelector: 'T -> 'K when 'K : comparison) ->
                currentQuery <- currentQuery |> Seq.sortBy keySelector
                createAdvancedQueryBuilder currentQuery
            
            Take = fun (count: int) ->
                currentQuery <- currentQuery |> Seq.take count
                createAdvancedQueryBuilder currentQuery
            
            Skip = fun (count: int) ->
                currentQuery <- currentQuery |> Seq.skip count
                createAdvancedQueryBuilder currentQuery
            
            Execute = fun () -> currentQuery
            
            ExecuteAsync = fun () -> async { return currentQuery }
            
            ToList = fun () -> currentQuery |> Seq.toList
            
            ToArray = fun () -> currentQuery |> Seq.toArray
            
            Count = fun () -> currentQuery |> Seq.length
            
            Any = fun () -> currentQuery |> Seq.isEmpty |> not
            
            First = fun () -> currentQuery |> Seq.head
            
            FirstOrDefault = fun () -> currentQuery |> Seq.tryHead
        |}
    
    // ============================================================================
    // PARALLEL QUERY SUPPORT
    // ============================================================================
    
    /// Parallel query operations for large datasets
    let createParallelQueryOperations<'T> (source: seq<'T>) (parallelism: int) =
        {|
            ParallelWhere = fun (predicate: 'T -> bool) ->
                source 
                |> Seq.chunkBySize (max 1 (Seq.length source / parallelism))
                |> Seq.map (fun chunk -> 
                    async { return chunk |> Array.filter predicate })
                |> Async.Parallel
                |> Async.map (Array.concat)
            
            ParallelSelect = fun (selector: 'T -> 'U) ->
                source 
                |> Seq.chunkBySize (max 1 (Seq.length source / parallelism))
                |> Seq.map (fun chunk -> 
                    async { return chunk |> Array.map selector })
                |> Async.Parallel
                |> Async.map (Array.concat)
            
            ParallelAggregate = fun (accumulator: 'T -> 'T -> 'T) ->
                async {
                    let chunks = source |> Seq.chunkBySize (max 1 (Seq.length source / parallelism))
                    let! partialResults = 
                        chunks
                        |> Seq.map (fun chunk -> 
                            async { return chunk |> Array.reduce accumulator })
                        |> Async.Parallel
                    
                    return partialResults |> Array.reduce accumulator
                }
            
            ParallelGroupBy = fun (keySelector: 'T -> 'K when 'K : comparison) ->
                async {
                    let chunks = source |> Seq.chunkBySize (max 1 (Seq.length source / parallelism))
                    let! partialGroups = 
                        chunks
                        |> Seq.map (fun chunk -> 
                            async { return chunk |> Array.groupBy keySelector })
                        |> Async.Parallel
                    
                    return 
                        partialGroups
                        |> Array.collect id
                        |> Array.groupBy fst
                        |> Array.map (fun (key, groups) -> 
                            (key, groups |> Array.collect snd))
                }
        |}
    
    // ============================================================================
    // CLOSURE RESULT CACHING AND QUERYING
    // ============================================================================
    
    /// Create queryable closure with caching
    let createQueryableClosure<'TInput, 'TOutput when 'TInput : comparison> 
        (computation: 'TInput -> Async<'TOutput>)
        (cacheSize: int)
        (logger: ILogger) =
        
        let cache = createAdaptiveMemoizationCache<'TInput, 'TOutput> cacheSize (TimeToLive (TimeSpan.FromHours(1.0))) logger
        let resultHistory = ConcurrentBag<'TInput * 'TOutput * DateTime>()
        
        let execute = fun (input: 'TInput) ->
            async {
                match cache.Get input with
                | Some cachedResult ->
                    return cachedResult
                | None ->
                    let startTime = DateTime.UtcNow
                    let! result = computation input
                    let computationTime = DateTime.UtcNow - startTime
                    
                    cache.Set input result computationTime (fun _ -> true)
                    resultHistory.Add((input, result, DateTime.UtcNow))
                    
                    return result
            }
        
        {|
            Execute = execute
            
            QueryResults = fun () ->
                let results = resultHistory |> Seq.map (fun (_, output, _) -> output)
                queryClosureResults results
            
            QueryHistory = fun () ->
                let history = resultHistory |> Seq.toList
                queryClosureResults history
            
            QueryByTimeRange = fun (startTime: DateTime) (endTime: DateTime) ->
                let filteredResults = 
                    resultHistory 
                    |> Seq.filter (fun (_, _, timestamp) -> timestamp >= startTime && timestamp <= endTime)
                    |> Seq.map (fun (_, output, _) -> output)
                queryClosureResults filteredResults
            
            GetCacheStatistics = cache.GetStatistics
            
            ClearCache = cache.Clear
            
            InvalidateCache = cache.Invalidate
        |}
