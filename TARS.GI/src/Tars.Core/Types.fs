// TARS.GI Core Types - Hybrid General Intelligence Architecture
// Four-valued logic, world model, and formal verification types
// Based on the concrete blueprint for non-LLM-centric intelligence

namespace Tars.Core

open System
open System.Collections.Generic

/// Four-valued logic for belief representation (Belnap/FDE)
type Belnap = 
    | True 
    | False 
    | Both      // Contradiction
    | Unknown   // No information

/// Belief with provenance and confidence
type Belief = {
    Id: string
    Proposition: string
    Truth: Belnap
    Confidence: float
    Provenance: string list
    Timestamp: DateTime
}

/// Latent state for world model (predictive coding)
type Latent = { 
    Mean: float[]
    Cov: float[][]
}

/// Observation from environment
type Observation = float[]

/// Action with formal specification
type Action = { 
    Name: string
    Args: Map<string, obj> 
}

/// Skill specification with formal contracts
type SkillSpec = {
    Name: string
    Pre: Belief list
    Post: Belief list
    Checker: unit -> bool   // property tests
    Cost: float
}

/// Plan step with skill and arguments
type PlanStep = { 
    Skill: SkillSpec
    Args: Map<string, obj> 
}

/// Plan as sequence of steps
type Plan = PlanStep list

/// Meta-cognition metrics
type Metrics = {
    PredError: float
    BeliefEntropy: float
    SpecViolations: int
    ReplanCount: int
}

/// Task for HTN planning
type Task =
    | Achieve of string * Belief list  // Goal with context
    | Execute of SkillSpec * Map<string, obj>  // Direct skill execution

/// POMDP state for partial observability
type POMDPState = {
    StateId: string
    Probability: float
    Features: Map<string, float>
}

/// POMDP action with transition probabilities
type POMDPAction = {
    Action: Action
    TransitionProbs: Map<string, float>
    ObservationProbs: Map<string, float>
    Reward: float
}

/// Meta-cognition thresholds for reflection triggers
type ReflectionThresholds = {
    MaxPredictionError: float
    MaxBeliefEntropy: float
    MaxSpecViolations: int
    MaxUncertainty: float
}

/// Reflection action for meta-cognition
type ReflectionAction =
    | UpdateWorldModel
    | ResolveBeliefContradictions
    | ReviewSkillSpecs
    | GatherMoreObservations
    | ReplanCurrentGoal

/// VSA (Vector-Symbolic Architecture) binding for symbol-vector mapping
type VSABinding = {
    Symbol: string
    Vector: float[]
    BindingStrength: float
    LastUsed: DateTime
}

/// Audit trail entry for explainability
type AuditEntry = {
    Id: string
    Timestamp: DateTime
    Action: string
    Justification: string
    BeliefState: Belief list
    WorldState: Latent
    Outcome: string
}

/// System configuration
type TarsConfig = {
    WorldModelDimensions: int
    MaxBeliefs: int
    ReflectionThresholds: ReflectionThresholds
    SkillTimeout: TimeSpan
    AuditTrailSize: int
    VSAVectorSize: int
}
