namespace TarsEngine.FSharp.Consciousness

open System
open TarsEngine.FSharp.Consciousness.HybridConsciousnessStorage

/// <summary>
/// Optimal storage configurations for different types of consciousness data
/// Balances performance, persistence, and memory usage based on data characteristics
/// </summary>
module ConsciousnessStorageConfig =
    
    /// <summary>
    /// Get optimal storage configurations for TARS consciousness system
    /// </summary>
    let getOptimalStorageConfigs() = [
        
        // VOLATILE TIER - Ultra-fast, temporary data
        {
            DataType = CurrentThoughts
            Tier = Volatile
            MaxVolatileItems = Some 10
            PersistenceInterval = None
            CompressionThreshold = None
            RetentionPeriod = Some (TimeSpan.FromMinutes(30.0))
        }
        
        {
            DataType = AttentionFocus
            Tier = Volatile
            MaxVolatileItems = Some 5
            PersistenceInterval = None
            CompressionThreshold = None
            RetentionPeriod = Some (TimeSpan.FromMinutes(15.0))
        }
        
        {
            DataType = EmotionalState
            Tier = Volatile
            MaxVolatileItems = Some 20
            PersistenceInterval = None
            CompressionThreshold = None
            RetentionPeriod = Some (TimeSpan.FromHours(1.0))
        }
        
        // CACHED TIER - Fast access with async persistence
        {
            DataType = WorkingMemory 0.5 // Medium importance
            Tier = Cached
            MaxVolatileItems = Some 50
            PersistenceInterval = Some (TimeSpan.FromSeconds(10.0))
            CompressionThreshold = Some 100
            RetentionPeriod = Some (TimeSpan.FromDays(7.0))
        }
        
        {
            DataType = WorkingMemory 0.8 // High importance
            Tier = Cached
            MaxVolatileItems = Some 100
            PersistenceInterval = Some (TimeSpan.FromSeconds(5.0))
            CompressionThreshold = Some 200
            RetentionPeriod = Some (TimeSpan.FromDays(30.0))
        }
        
        {
            DataType = AgentContributions
            Tier = Cached
            MaxVolatileItems = Some 200
            PersistenceInterval = Some (TimeSpan.FromSeconds(3.0))
            CompressionThreshold = Some 500
            RetentionPeriod = Some (TimeSpan.FromDays(14.0))
        }
        
        {
            DataType = ConversationContext
            Tier = Cached
            MaxVolatileItems = Some 30
            PersistenceInterval = Some (TimeSpan.FromSeconds(15.0))
            CompressionThreshold = Some 50
            RetentionPeriod = Some (TimeSpan.FromDays(3.0))
        }
        
        // PERSISTENT TIER - Immediate disk write for critical data
        {
            DataType = SelfAwareness
            Tier = Persistent
            MaxVolatileItems = None
            PersistenceInterval = None
            CompressionThreshold = None
            RetentionPeriod = None // Keep forever
        }
        
        {
            DataType = ConsciousnessLevel
            Tier = Persistent
            MaxVolatileItems = None
            PersistenceInterval = None
            CompressionThreshold = None
            RetentionPeriod = None // Keep forever
        }
        
        {
            DataType = PersonalityTraits
            Tier = Persistent
            MaxVolatileItems = None
            PersistenceInterval = None
            CompressionThreshold = None
            RetentionPeriod = None // Keep forever
        }
        
        // LONG-TERM TIER - Compressed archival storage
        {
            DataType = LongTermMemory 0.9 // Very significant memories
            Tier = LongTerm
            MaxVolatileItems = None
            PersistenceInterval = None
            CompressionThreshold = Some 1000
            RetentionPeriod = None // Keep forever
        }
        
        {
            DataType = WorkingMemory 0.9 // Promote high-importance working memory
            Tier = LongTerm
            MaxVolatileItems = None
            PersistenceInterval = None
            CompressionThreshold = Some 500
            RetentionPeriod = None // Keep forever
        }
    ]
    
    /// <summary>
    /// Performance-optimized configuration for high-throughput scenarios
    /// </summary>
    let getPerformanceOptimizedConfigs() = [
        
        // Maximize volatile storage for speed
        {
            DataType = CurrentThoughts
            Tier = Volatile
            MaxVolatileItems = Some 50
            PersistenceInterval = None
            CompressionThreshold = None
            RetentionPeriod = Some (TimeSpan.FromHours(2.0))
        }
        
        {
            DataType = WorkingMemory 0.5
            Tier = Volatile
            MaxVolatileItems = Some 200
            PersistenceInterval = None
            CompressionThreshold = None
            RetentionPeriod = Some (TimeSpan.FromHours(4.0))
        }
        
        {
            DataType = AgentContributions
            Tier = Cached
            MaxVolatileItems = Some 1000
            PersistenceInterval = Some (TimeSpan.FromMinutes(5.0))
            CompressionThreshold = Some 2000
            RetentionPeriod = Some (TimeSpan.FromDays(1.0))
        }
        
        // Only critical data gets immediate persistence
        {
            DataType = SelfAwareness
            Tier = Persistent
            MaxVolatileItems = None
            PersistenceInterval = None
            CompressionThreshold = None
            RetentionPeriod = None
        }
        
        {
            DataType = PersonalityTraits
            Tier = Persistent
            MaxVolatileItems = None
            PersistenceInterval = None
            CompressionThreshold = None
            RetentionPeriod = None
        }
    ]
    
    /// <summary>
    /// Durability-focused configuration for maximum data retention
    /// </summary>
    let getDurabilityOptimizedConfigs() = [
        
        // Minimize volatile storage, maximize persistence
        {
            DataType = CurrentThoughts
            Tier = Cached
            MaxVolatileItems = Some 5
            PersistenceInterval = Some (TimeSpan.FromSeconds(1.0))
            CompressionThreshold = Some 10
            RetentionPeriod = None // Keep forever
        }
        
        {
            DataType = WorkingMemory 0.3 // Even low importance gets cached
            Tier = Cached
            MaxVolatileItems = Some 10
            PersistenceInterval = Some (TimeSpan.FromSeconds(2.0))
            CompressionThreshold = Some 20
            RetentionPeriod = None // Keep forever
        }
        
        {
            DataType = AgentContributions
            Tier = Persistent
            MaxVolatileItems = None
            PersistenceInterval = None
            CompressionThreshold = None
            RetentionPeriod = None // Keep forever
        }
        
        {
            DataType = EmotionalState
            Tier = Cached
            MaxVolatileItems = Some 3
            PersistenceInterval = Some (TimeSpan.FromSeconds(5.0))
            CompressionThreshold = Some 50
            RetentionPeriod = None // Keep forever
        }
        
        // All critical data immediately persistent
        {
            DataType = SelfAwareness
            Tier = Persistent
            MaxVolatileItems = None
            PersistenceInterval = None
            CompressionThreshold = None
            RetentionPeriod = None
        }
        
        {
            DataType = ConsciousnessLevel
            Tier = Persistent
            MaxVolatileItems = None
            PersistenceInterval = None
            CompressionThreshold = None
            RetentionPeriod = None
        }
        
        {
            DataType = PersonalityTraits
            Tier = Persistent
            MaxVolatileItems = None
            PersistenceInterval = None
            CompressionThreshold = None
            RetentionPeriod = None
        }
    ]
    
    /// <summary>
    /// Memory-optimized configuration for resource-constrained environments
    /// </summary>
    let getMemoryOptimizedConfigs() = [
        
        // Minimal volatile storage
        {
            DataType = CurrentThoughts
            Tier = Volatile
            MaxVolatileItems = Some 3
            PersistenceInterval = None
            CompressionThreshold = None
            RetentionPeriod = Some (TimeSpan.FromMinutes(10.0))
        }
        
        {
            DataType = WorkingMemory 0.5
            Tier = Cached
            MaxVolatileItems = Some 10
            PersistenceInterval = Some (TimeSpan.FromSeconds(30.0))
            CompressionThreshold = Some 20
            RetentionPeriod = Some (TimeSpan.FromDays(1.0))
        }
        
        {
            DataType = AgentContributions
            Tier = Cached
            MaxVolatileItems = Some 20
            PersistenceInterval = Some (TimeSpan.FromMinutes(1.0))
            CompressionThreshold = Some 50
            RetentionPeriod = Some (TimeSpan.FromDays(3.0))
        }
        
        // Aggressive compression for long-term storage
        {
            DataType = LongTermMemory 0.8
            Tier = LongTerm
            MaxVolatileItems = None
            PersistenceInterval = None
            CompressionThreshold = Some 100
            RetentionPeriod = Some (TimeSpan.FromDays(90.0))
        }
        
        // Only essential data gets persistence
        {
            DataType = SelfAwareness
            Tier = Persistent
            MaxVolatileItems = None
            PersistenceInterval = None
            CompressionThreshold = None
            RetentionPeriod = None
        }
        
        {
            DataType = PersonalityTraits
            Tier = Persistent
            MaxVolatileItems = None
            PersistenceInterval = None
            CompressionThreshold = None
            RetentionPeriod = None
        }
    ]
    
    /// <summary>
    /// Get configuration based on deployment scenario
    /// </summary>
    let getConfigForScenario(scenario: string) =
        match scenario.ToLower() with
        | "performance" | "speed" | "realtime" -> getPerformanceOptimizedConfigs()
        | "durability" | "reliability" | "backup" -> getDurabilityOptimizedConfigs()
        | "memory" | "constrained" | "embedded" -> getMemoryOptimizedConfigs()
        | _ -> getOptimalStorageConfigs() // Default balanced approach
