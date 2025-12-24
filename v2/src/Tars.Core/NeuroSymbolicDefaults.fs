/// Global Neuro-Symbolic Configuration for TARS
/// Makes constraint-based AI the default across all interfaces
module Tars.Core.NeuroSymbolicDefaults

open System
open Tars.Core

/// Global configuration for neuro-symbolic features
type GlobalNeuroSymbolicConfig =
    {
        /// Enable neuro-symbolic features globally
        Enabled: bool

        /// Minimum score for accepting mutations (0.0-1.0)
        MinMutationScore: float

        /// Minimum score for accepting beliefs (0.0-1.0)
        MinBeliefScore: float

        /// Enable prompt shaping with warnings
        EnablePromptShaping: bool

        /// Enable agent selection biasing
        EnableAgentBiasing: bool

        /// Enable mutation filtering
        EnableMutationFiltering: bool

        /// Log metrics to console
        LogMetrics: bool

        /// Track performance history
        TrackPerformance: bool

        /// Max contradictions to remember
        MaxContradictionPatterns: int

        /// Max low-scoring patterns to remember
        MaxLowScoringPatterns: int
    }

/// Production-grade defaults (conservative but effective)
let productionDefaults =
    { Enabled = true
      MinMutationScore = 0.5
      MinBeliefScore = 0.7
      EnablePromptShaping = true
      EnableAgentBiasing = true
      EnableMutationFiltering = true
      LogMetrics = false // Don't spam logs in production
      TrackPerformance = true
      MaxContradictionPatterns = 10
      MaxLowScoringPatterns = 10 }

/// Development defaults (more lenient, more logging)
let developmentDefaults =
    { productionDefaults with
        MinMutationScore = 0.3
        MinBeliefScore = 0.5
        LogMetrics = true }

/// Aggressive defaults (strict constraints, research-grade)
let aggressiveDefaults =
    { productionDefaults with
        MinMutationScore = 0.7
        MinBeliefScore = 0.8
        MaxContradictionPatterns = 20
        MaxLowScoringPatterns = 20 }

/// Disabled (for comparison/debugging)
let disabledDefaults =
    { Enabled = false
      MinMutationScore = 0.0
      MinBeliefScore = 0.0
      EnablePromptShaping = false
      EnableAgentBiasing = false
      EnableMutationFiltering = false
      LogMetrics = false
      TrackPerformance = false
      MaxContradictionPatterns = 0
      MaxLowScoringPatterns = 0 }

/// Get configuration from environment or use defaults
let getConfig () : GlobalNeuroSymbolicConfig =
    let env = Environment.GetEnvironmentVariable("TARS_ENV")

    match env with
    | "production"
    | "prod" -> productionDefaults
    | "development"
    | "dev" -> developmentDefaults
    | "aggressive"
    | "research" -> aggressiveDefaults
    | "disabled"
    | "off" -> disabledDefaults
    | _ -> productionDefaults // Default to production settings

/// Singleton configuration instance
let mutable private currentConfig = getConfig ()

/// Get current global configuration
let getCurrentConfig () = currentConfig

/// Update global configuration
let setConfig (config: GlobalNeuroSymbolicConfig) = currentConfig <- config

/// Enable neuro-symbolic features globally
let enable () =
    setConfig { currentConfig with Enabled = true }

/// Disable neuro-symbolic features globally
let disable () =
    setConfig { currentConfig with Enabled = false }

/// Check if features are enabled
let isEnabled () = currentConfig.Enabled

/// Get human-readable summary of current configuration
let getConfigSummary () =
    let config = getCurrentConfig ()

    if not config.Enabled then
        "🔴 Neuro-Symbolic AI: DISABLED"
    else
        sprintf
            """🟢 Neuro-Symbolic AI: ENABLED
├─ Mutation Score Threshold: %.1f
├─ Belief Score Threshold: %.1f
├─ Prompt Shaping: %s
├─ Agent Biasing: %s
├─ Mutation Filtering: %s
├─ Performance Tracking: %s
└─ Metrics Logging: %s"""
            config.MinMutationScore
            config.MinBeliefScore
            (if config.EnablePromptShaping then "✅" else "❌")
            (if config.EnableAgentBiasing then "✅" else "❌")
            (if config.EnableMutationFiltering then "✅" else "❌")
            (if config.TrackPerformance then "✅" else "❌")
            (if config.LogMetrics then "✅" else "❌")

/// Print current configuration to console
let printConfig () = printfn "%s" (getConfigSummary ())
