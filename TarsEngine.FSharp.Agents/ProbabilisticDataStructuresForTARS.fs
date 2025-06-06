// Probabilistic Data Structures for TARS Applications
// Efficient, memory-optimized data processing for large-scale autonomous systems

namespace TarsEngine.FSharp.Agents

open System
open System.Threading.Tasks
open System.Collections.Concurrent
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.ClosureFactory.AdvancedMathematicalClosureFactory

/// Probabilistic data structures applications for TARS
module ProbabilisticDataStructuresForTARS =
    
    /// TARS-specific Bloom filter for duplicate detection
    type TARSBloomFilter = {
        Filter: BloomFilter
        Category: string
        CreatedAt: DateTime
        LastUsed: DateTime
        HitCount: int64
        MissCount: int64
        MemoryUsage: int64
    }
    
    /// TARS frequency tracker using Count-Min Sketch
    type TARSFrequencyTracker = {
        Sketch: CountMinSketch
        Category: string
        WindowSize: TimeSpan
        CreatedAt: DateTime
        TotalQueries: int64
        AccuracyRate: float
    }
    
    /// TARS cardinality estimator using HyperLogLog
    type TARSCardinalityEstimator = {
        HLL: HyperLogLog
        Category: string
        EstimationHistory: (DateTime * int64) list
        AccuracyMetrics: float list
    }
    
    /// Probabilistic Data Structures Manager for TARS
    type TARSProbabilisticManager(logger: ILogger<TARSProbabilisticManager>) =
        
        let bloomFilters = ConcurrentDictionary<string, TARSBloomFilter>()
        let frequencyTrackers = ConcurrentDictionary<string, TARSFrequencyTracker>()
        let cardinalityEstimators = ConcurrentDictionary<string, TARSCardinalityEstimator>()
        
        /// Create optimized Bloom filter for TARS use case
        member this.CreateTARSBloomFilter(category: string, expectedElements: int, falsePositiveRate: float) = async {
            logger.LogInformation("🌸 Creating TARS Bloom filter for {Category} (expected: {Elements}, FPR: {Rate:P2})", 
                                category, expectedElements, falsePositiveRate)
            
            let! filter = createBloomFilter expectedElements falsePositiveRate
            
            let tarsFilter = {
                Filter = filter
                Category = category
                CreatedAt = DateTime.UtcNow
                LastUsed = DateTime.UtcNow
                HitCount = 0L
                MissCount = 0L
                MemoryUsage = int64 (filter.Size / 8)  // bits to bytes
            }
            
            bloomFilters.TryAdd(category, tarsFilter) |> ignore
            
            logger.LogInformation("✅ TARS Bloom filter created: {Size} bits, {HashFunctions} hash functions, {Memory} bytes", 
                                filter.Size, filter.HashFunctions, tarsFilter.MemoryUsage)
            
            return tarsFilter
        }
        
        /// Check for duplicates using TARS Bloom filter
        member this.CheckDuplicate(category: string, item: string) = async {
            match bloomFilters.TryGetValue(category) with
            | true, tarsFilter ->
                let! isPresent = checkBloomFilter tarsFilter.Filter item
                
                let updatedFilter = 
                    if isPresent then
                        { tarsFilter with 
                            HitCount = tarsFilter.HitCount + 1L
                            LastUsed = DateTime.UtcNow }
                    else
                        { tarsFilter with 
                            MissCount = tarsFilter.MissCount + 1L
                            LastUsed = DateTime.UtcNow }
                
                bloomFilters.TryUpdate(category, updatedFilter, tarsFilter) |> ignore
                
                logger.LogDebug("🔍 Duplicate check for {Category}: {Item} -> {Result}", category, item, isPresent)
                
                return {|
                    IsPresent = isPresent
                    Category = category
                    Item = item
                    HitRate = float updatedFilter.HitCount / float (updatedFilter.HitCount + updatedFilter.MissCount)
                    FilterEfficiency = updatedFilter.MemoryUsage
                |}
                
            | false, _ ->
                logger.LogWarning("Bloom filter for category {Category} not found", category)
                return {|
                    IsPresent = false
                    Category = category
                    Item = item
                    HitRate = 0.0
                    FilterEfficiency = 0L
                |}
        }
        
        /// Add item to TARS Bloom filter
        member this.AddToBloomFilter(category: string, item: string) = async {
            match bloomFilters.TryGetValue(category) with
            | true, tarsFilter ->
                let! updatedFilter = addToBloomFilter tarsFilter.Filter item
                
                let newTarsFilter = {
                    tarsFilter with 
                        Filter = updatedFilter
                        LastUsed = DateTime.UtcNow
                }
                
                bloomFilters.TryUpdate(category, newTarsFilter, tarsFilter) |> ignore
                
                logger.LogDebug("➕ Added {Item} to Bloom filter {Category}", item, category)
                return true
                
            | false, _ ->
                logger.LogWarning("Bloom filter for category {Category} not found", category)
                return false
        }
        
        /// Create frequency tracker for TARS analytics
        member this.CreateFrequencyTracker(category: string, epsilon: float, delta: float, windowSize: TimeSpan) = async {
            logger.LogInformation("📊 Creating TARS frequency tracker for {Category} (ε={Epsilon:F3}, δ={Delta:F3})", 
                                category, epsilon, delta)
            
            let sketch = createCountMinSketch epsilon delta
            
            let tracker = {
                Sketch = sketch
                Category = category
                WindowSize = windowSize
                CreatedAt = DateTime.UtcNow
                TotalQueries = 0L
                AccuracyRate = 1.0 - epsilon
            }
            
            frequencyTrackers.TryAdd(category, tracker) |> ignore
            
            logger.LogInformation("✅ TARS frequency tracker created: {Width}×{Depth} counters, {Accuracy:P1} accuracy", 
                                sketch.Width, sketch.Depth, tracker.AccuracyRate)
            
            return tracker
        }
        
        /// Track frequency using Count-Min Sketch
        member this.TrackFrequency(category: string, item: string, count: int) = async {
            match frequencyTrackers.TryGetValue(category) with
            | true, tracker ->
                let updatedSketch = addToCountMinSketch tracker.Sketch item count
                
                let newTracker = {
                    tracker with 
                        Sketch = updatedSketch
                        TotalQueries = tracker.TotalQueries + 1L
                }
                
                frequencyTrackers.TryUpdate(category, newTracker, tracker) |> ignore
                
                let estimatedFreq = estimateFrequency updatedSketch item
                
                logger.LogDebug("📈 Frequency tracked for {Category}: {Item} -> {Frequency}", category, item, estimatedFreq)
                
                return {|
                    Category = category
                    Item = item
                    EstimatedFrequency = estimatedFreq
                    TotalQueries = newTracker.TotalQueries
                    AccuracyRate = newTracker.AccuracyRate
                |}
                
            | false, _ ->
                logger.LogWarning("Frequency tracker for category {Category} not found", category)
                return {|
                    Category = category
                    Item = item
                    EstimatedFrequency = 0
                    TotalQueries = 0L
                    AccuracyRate = 0.0
                |}
        }
        
        /// Get frequency estimate
        member this.GetFrequencyEstimate(category: string, item: string) = async {
            match frequencyTrackers.TryGetValue(category) with
            | true, tracker ->
                let frequency = estimateFrequency tracker.Sketch item
                
                return {|
                    Category = category
                    Item = item
                    EstimatedFrequency = frequency
                    Confidence = tracker.AccuracyRate
                    WindowSize = tracker.WindowSize
                    TotalQueries = tracker.TotalQueries
                |}
                
            | false, _ ->
                logger.LogWarning("Frequency tracker for category {Category} not found", category)
                return {|
                    Category = category
                    Item = item
                    EstimatedFrequency = 0
                    Confidence = 0.0
                    WindowSize = TimeSpan.Zero
                    TotalQueries = 0L
                |}
        }
        
        /// Create cardinality estimator for TARS analytics
        member this.CreateCardinalityEstimator(category: string, precision: int) = async {
            logger.LogInformation("🔢 Creating TARS cardinality estimator for {Category} (precision: {Precision})", 
                                category, precision)
            
            let hll = createHyperLogLog precision
            
            let estimator = {
                HLL = hll
                Category = category
                EstimationHistory = []
                AccuracyMetrics = []
            }
            
            cardinalityEstimators.TryAdd(category, estimator) |> ignore
            
            let standardError = 1.04 / sqrt(float hll.BucketCount)
            
            logger.LogInformation("✅ TARS cardinality estimator created: {Buckets} buckets, {Error:P2} standard error", 
                                hll.BucketCount, standardError)
            
            return estimator
        }
        
        /// Add unique element to cardinality estimator
        member this.AddUniqueElement(category: string, element: string) = async {
            match cardinalityEstimators.TryGetValue(category) with
            | true, estimator ->
                let updatedHLL = addToHyperLogLog estimator.HLL element
                
                let newEstimator = {
                    estimator with 
                        HLL = updatedHLL
                        EstimationHistory = (DateTime.UtcNow, updatedHLL.EstimatedCardinality) :: estimator.EstimationHistory
                }
                
                cardinalityEstimators.TryUpdate(category, newEstimator, estimator) |> ignore
                
                logger.LogDebug("🔢 Cardinality updated for {Category}: estimated {Cardinality}", 
                              category, updatedHLL.EstimatedCardinality)
                
                return {|
                    Category = category
                    EstimatedCardinality = updatedHLL.EstimatedCardinality
                    HistoryLength = newEstimator.EstimationHistory.Length
                |}
                
            | false, _ ->
                logger.LogWarning("Cardinality estimator for category {Category} not found", category)
                return {|
                    Category = category
                    EstimatedCardinality = 0L
                    HistoryLength = 0
                |}
        }
        
        /// Get cardinality estimate
        member this.GetCardinalityEstimate(category: string) = async {
            match cardinalityEstimators.TryGetValue(category) with
            | true, estimator ->
                let standardError = 1.04 / sqrt(float estimator.HLL.BucketCount)
                
                return {|
                    Category = category
                    EstimatedCardinality = estimator.HLL.EstimatedCardinality
                    StandardError = standardError
                    Precision = int (log2 (float estimator.HLL.BucketCount))
                    HistoryLength = estimator.EstimationHistory.Length
                    MemoryUsage = estimator.HLL.BucketCount * 4  // 4 bytes per bucket
                |}
                
            | false, _ ->
                logger.LogWarning("Cardinality estimator for category {Category} not found", category)
                return {|
                    Category = category
                    EstimatedCardinality = 0L
                    StandardError = 1.0
                    Precision = 0
                    HistoryLength = 0
                    MemoryUsage = 0
                |}
        }
        
        /// Get comprehensive analytics for all probabilistic structures
        member this.GetTARSAnalytics() = async {
            let bloomStats = 
                bloomFilters.Values
                |> Seq.map (fun bf -> {|
                    Category = bf.Category
                    Type = "BloomFilter"
                    MemoryUsage = bf.MemoryUsage
                    HitRate = if bf.HitCount + bf.MissCount > 0L then 
                                float bf.HitCount / float (bf.HitCount + bf.MissCount) 
                              else 0.0
                    LastUsed = bf.LastUsed
                |})
                |> Seq.toList
            
            let frequencyStats = 
                frequencyTrackers.Values
                |> Seq.map (fun ft -> {|
                    Category = ft.Category
                    Type = "CountMinSketch"
                    MemoryUsage = int64 (ft.Sketch.Width * ft.Sketch.Depth * 4)  // 4 bytes per counter
                    AccuracyRate = ft.AccuracyRate
                    TotalQueries = ft.TotalQueries
                |})
                |> Seq.toList
            
            let cardinalityStats = 
                cardinalityEstimators.Values
                |> Seq.map (fun ce -> {|
                    Category = ce.Category
                    Type = "HyperLogLog"
                    MemoryUsage = int64 (ce.HLL.BucketCount * 4)
                    EstimatedCardinality = ce.HLL.EstimatedCardinality
                    StandardError = 1.04 / sqrt(float ce.HLL.BucketCount)
                |})
                |> Seq.toList
            
            let totalMemoryUsage = 
                (bloomStats |> List.sumBy (fun s -> s.MemoryUsage)) +
                (frequencyStats |> List.sumBy (fun s -> s.MemoryUsage)) +
                (cardinalityStats |> List.sumBy (fun s -> s.MemoryUsage))
            
            return {|
                BloomFilters = bloomStats
                FrequencyTrackers = frequencyStats
                CardinalityEstimators = cardinalityStats
                TotalStructures = bloomStats.Length + frequencyStats.Length + cardinalityStats.Length
                TotalMemoryUsage = totalMemoryUsage
                MemoryEfficiency = sprintf "%.2f MB total" (float totalMemoryUsage / 1024.0 / 1024.0)
                SystemHealth = if totalMemoryUsage < 100L * 1024L * 1024L then "Excellent" else "Good"  // < 100MB
            |}
        }
        
        /// Optimize probabilistic structures (cleanup, resize, etc.)
        member this.OptimizeProbabilisticStructures() = async {
            logger.LogInformation("🔧 Optimizing TARS probabilistic data structures...")
            
            let mutable optimizationsApplied = 0
            
            // Clean up old or unused structures
            let cutoffTime = DateTime.UtcNow.AddHours(-24.0)  // 24 hours ago
            
            let oldBloomFilters = 
                bloomFilters.Values
                |> Seq.filter (fun bf -> bf.LastUsed < cutoffTime)
                |> Seq.toList
            
            for oldFilter in oldBloomFilters do
                bloomFilters.TryRemove(oldFilter.Category) |> ignore
                optimizationsApplied <- optimizationsApplied + 1
                logger.LogInformation("🗑️ Removed old Bloom filter: {Category}", oldFilter.Category)
            
            logger.LogInformation("✅ Optimization completed: {Optimizations} structures optimized", optimizationsApplied)
            
            return {|
                OptimizationsApplied = optimizationsApplied
                StructuresRemoved = oldBloomFilters.Length
                MemoryFreed = oldBloomFilters |> List.sumBy (fun bf -> bf.MemoryUsage)
                OptimizationTime = DateTime.UtcNow
            |}
        }
