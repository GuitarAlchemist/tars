# V1 Grammar & DSL Adaptation Research for V2

**Last Updated:** 2025-01-29  
**Status:** Research Complete  
**Related Documents:**
- [Graphiti Integration Research](./graphiti_integration_research.md)
- [Learn to Learn Architecture](../Architecture/learn_to_learn.md)

---

## 1. Executive Summary

This document analyzes the grammar and DSL infrastructure from TARS v1 to identify components that can be adapted for v2. The v1 codebase contains **extensive grammar evolution capabilities** that are largely absent from v2's minimal implementation.

### Key Findings

| Aspect | V1 | V2 | Gap |
|--------|----|----|-----|
| **Grammar Types** | Fractal, Tiered (1-16), RFC-based | JSON pattern detection only | Critical |
| **Evolution** | Multi-agent, emergent tier evolution | None | Critical |
| **DSL Support** | FLUX, Metascript, TRSX, multi-language | JSON workflows only | Significant |
| **EBNF Parsing** | Full tokenizer + production parser | Basic regex-based | Moderate |
| **Output Formats** | EBNF, ANTLR, JSON, XML, GraphViz, SVG | JSON only | Significant |

### Recommendation

**Port the following v1 components to v2:**
1. `FractalGrammar` types and engine (core pattern system)
2. `EmergentTierEvolution` (autonomous grammar advancement)
3. `GrammarSource` types (RFC integration, multi-source support)
4. `EbnfParser` (robust grammar parsing)
5. `FluxInterpreter` type system (multi-language execution)

---

## 2. V1 Grammar Architecture Inventory

### 2.1 Core Grammar Types (`Tars.Engine.Grammar`)

```
v1/src/Tars.Engine.Grammar/
├── GrammarSource.fs      # Grammar sources: Inline, External, EmbeddedRFC
├── FractalGrammar.fs     # Self-similar recursive grammars
├── FractalGrammarParser.fs
├── FractalGrammarIntegration.fs
├── RFCProcessor.fs       # Generate grammars from RFC specs
├── GrammarResolver.fs
└── LanguageDispatcher.fs
```

### 2.2 Grammar Evolution (`TarsEngine.FSharp.Core/Grammar`)

```
v1/src/TarsEngine.FSharp.Core/Grammar/
├── EbnfParser.fs              # Full EBNF tokenizer/parser
├── EmergentTierEvolution.fs   # Tier 1-16 evolution engine
├── UnifiedGrammarEvolution.fs # Hybrid fractal + tier evolution
├── ReasoningGrammarEvolution.fs
├── VectorStoreGrammarAnalyzer.fs
└── GrammarEvolutionDemo.fs
```

### 2.3 Team Grammar Evolution (`TarsEngine.FSharp.Core/Evolution`)

```
v1/src/TarsEngine.FSharp.Core/Evolution/
└── TeamGrammarEvolution.fs    # Multi-agent grammar evolution
```

### 2.4 FLUX DSL (`TarsEngine.FSharp.Core/FLUX`)

```
v1/src/TarsEngine.FSharp.Core/FLUX/
├── FluxInterpreter.fs         # Multi-language interpreter
├── FluxFractalArchitecture.fs
├── FluxIntegrationEngine.fs
├── UnifiedTrsxInterpreter.fs
├── Ast/FluxAst.fs
├── FractalGrammar/SimpleFractalGrammar.fs
├── FractalLanguage/
├── Mathematics/MathematicalEngine.fs
├── Refinement/CrossEntropyRefinement.fs
├── UnifiedFormat/TrsxMigrationTool.fs
└── VectorStore/SemanticVectorStore.fs
```

### 2.5 Metascript (`TarsEngine.FSharp.Core/Metascript`)

```
v1/src/TarsEngine.FSharp.Core/Metascript/
├── Types.fs                   # BlockType: Meta, Reasoning, FSharp, Lang, Mcp
├── Parser.fs
├── Services.fs
├── FractalGrammarMetascripts.fs
├── MetascriptGrammarRegistry.fs
└── MetascriptSpecLoader.fs
```

---

## 3. V2 Current Grammar Infrastructure

### 3.1 Grammar Distillation (`Tars.Core`)

**`GrammarDistill.fs`** (78 lines) - Minimal JSON-focused:
```fsharp
type GrammarSpec =
    { Fields: string list
      Required: string list
      Example: string
      PromptHint: string
      Validator: string -> bool }
```

**`GrammarPipeline.fs`** (264 lines) - Pattern detection:
```fsharp
type DetectedPattern =
    | JsonObject of fields: string list
    | XmlTags of tags: string list
    | CodeBlock of lang: string option
    | AnswerTag | ThinkingBlock | ToolCall | Unknown
```

### 3.2 EBNF Parser (`Tars.Cortex`)

**`Grammar.fs`** (218 lines) - Basic EBNF:
```fsharp
type Expr =
    | Terminal of string
    | NonTerminal of string
    | Sequence of Expr list
    | ZeroOrMore of Expr
    | BuiltIn of string
```

### 3.3 Metascript (`Tars.Metascript`)

**`Domain.fs`** (53 lines) - JSON workflows only:
```fsharp
type WorkflowStep =
    { Id: string; Type: string; Agent: string option; Tool: string option; ... }
```

---

## 4. Gap Analysis

### 4.1 Critical Gaps

| V1 Capability | V2 Status | Impact |
|---------------|-----------|--------|
| **Fractal Grammars** | Missing | Cannot express self-similar patterns |
| **Tier Evolution (1-16)** | Missing | No autonomous grammar advancement |
| **RFC Integration** | Missing | Cannot derive grammars from standards |
| **Multi-agent Evolution** | Missing | No collaborative grammar refinement |
| **FLUX Type System** | Missing | No dependent/linear/refined types |

### 4.2 Moderate Gaps

| V1 Capability | V2 Status | Impact |
|---------------|-----------|--------|
| **Full EBNF Parser** | Partial | Limited grammar complexity |
| **Multi-language Blocks** | Missing | F#/Python/Julia/CUDA not supported |
| **Grammar Visualization** | Missing | No GraphViz/SVG output |
| **Vector Store Analysis** | Missing | No semantic grammar search |

---

## 5. Recommended Adaptations

### 5.1 Phase 1: Core Types (Week 1-2)

**Port `GrammarSource.fs` types to `Tars.Core`:**

```fsharp
// v2/src/Tars.Core/GrammarTypes.fs (NEW)
namespace Tars.Core

type GrammarSource =
    | Inline of name: string * content: string
    | External of path: string
    | EmbeddedRFC of rfcId: string * ruleName: string
    | Generated of generator: string * seed: int

type GrammarMetadata =
    { Name: string
      Version: string option
      Source: string option  // "rfc", "generated", "manual", "evolved"
      Language: string       // "EBNF", "BNF", "GBNF"
      Created: DateTime option
      Hash: string option
      Tags: string list
      Tier: int option }     // 1-16 for evolved grammars

type Grammar =
    { Metadata: GrammarMetadata
      Source: GrammarSource
      Content: string }
```

### 5.2 Phase 2: Fractal Grammar System (Week 3-4)

**Port `FractalGrammar.fs` core types:**

```fsharp
// v2/src/Tars.Core/FractalGrammar.fs (NEW)
namespace Tars.Core

module FractalGrammar =

    type FractalProperties =
        { Dimension: float           // Fractal dimension (1.0-3.0)
          ScalingFactor: float       // Golden ratio default: 0.618
          IterationDepth: int        // Max recursion depth
          SelfSimilarityRatio: float // Pattern repetition ratio
          RecursionLimit: int        // Safety limit
          CompositionRules: string list }

    type FractalTransformation =
        | Scale of factor: float
        | Rotate of angle: float
        | Translate of x: float * y: float
        | Compose of FractalTransformation list
        | Recursive of depth: int * FractalTransformation
        | Conditional of predicate: string * ifTrue: FractalTransformation * ifFalse: FractalTransformation

    type FractalNode =
        { Id: string
          Name: string
          Pattern: string
          Children: FractalNode list
          Properties: FractalProperties
          Transformations: FractalTransformation list
          Level: int
          ParentId: string option }

    type FractalRule =
        { Name: string
          BasePattern: string
          RecursivePattern: string option
          TerminationCondition: string
          Transformations: FractalTransformation list
          Properties: FractalProperties
          Dependencies: string list }

    type FractalGrammar =
        { Name: string
          Version: string
          BaseGrammar: Grammar
          FractalRules: FractalRule list
          GlobalProperties: FractalProperties
          CompositionGraph: Map<string, string list>
          GenerationHistory: FractalNode list }
```

### 5.3 Phase 3: Tier Evolution System (Week 5-6)

**Port `EmergentTierEvolution.fs` concepts:**

```fsharp
// v2/src/Tars.Evolution/TierEvolution.fs (NEW)
namespace Tars.Evolution

module TierEvolution =

    let MAX_GRAMMAR_TIERS = 16
    let TIER_EVOLUTION_THRESHOLD = 0.7

    type GrammarTier =
        | Tier1_BasicCoordination
        | Tier2_ScientificDomain
        | Tier3_DomainSpecific
        | Tier4_AgentSpecialized
        | Tier5_SelfModifying
        | Tier6_MetaReasoning
        | Tier7_EmergentBehavior
        | Tier8_CrossDomain
        | Tier9_AutonomousEvolution
        | Tier10_CollectiveIntelligence
        | Tier11_AbstractReasoning
        | Tier12_SymbolicGrounding
        | Tier13_CausalInference
        | Tier14_CounterfactualReasoning
        | Tier15_MetaCognition
        | Tier16_UnifiedIntelligence

    type ProblemContext =
        { Domain: string
          Constraints: Map<string, float * float>  // (current, target)
          Tensions: (string * float) list          // (description, severity)
          RequiredCapabilities: string list
          CurrentTier: int
          SuccessMetrics: Map<string, float> }

    type EvolutionDirection =
        | ParameterSpace of string list
        | PhysicsModification of string
        | ComputationalExpression of string
        | MetaLanguageConstruct of string

    type GrammarEvolutionResult =
        { NewTier: int
          GeneratedExpressions: string list
          MetaConstructs: string list
          EvolutionReasoning: string
          ExpectedImprovement: float }

    /// Analyze problem context to identify evolution opportunities
    let analyzeProblemContext (context: ProblemContext) : EvolutionDirection list =
        let mutable directions = []

        // Analyze constraint tensions
        for kvp in context.Constraints do
            let (current, target) = kvp.Value
            let tension = abs(current - target) / max target 0.001
            if tension > 0.1 then
                match kvp.Key with
                | key when key.Contains("performance") ->
                    directions <- ComputationalExpression("PerformanceOptimization") :: directions
                | key when key.Contains("agent") ->
                    directions <- ComputationalExpression("AgentCoordination") :: directions
                | key when key.Contains("memory") ->
                    directions <- MetaLanguageConstruct("ResourceManagement") :: directions
                | _ ->
                    directions <- ParameterSpace([kvp.Key]) :: directions

        // Domain-specific evolution
        match context.Domain with
        | "SoftwareDevelopment" ->
            directions <- ComputationalExpression("AutonomousCodeGeneration") :: directions
            directions <- MetaLanguageConstruct("SelfImprovingArchitecture") :: directions
        | "AgentCoordination" ->
            directions <- ComputationalExpression("SemanticTaskRouting") :: directions
            directions <- MetaLanguageConstruct("DynamicTeamFormation") :: directions
        | "MachineLearning" ->
            directions <- ComputationalExpression("AdaptiveNeuralArchitecture") :: directions
        | _ ->
            directions <- ComputationalExpression("GenericOptimization") :: directions

        directions
```

### 5.4 Phase 4: Multi-Agent Grammar Evolution (Week 7-8)

**Port `TeamGrammarEvolution.fs` concepts:**

```fsharp
// v2/src/Tars.Evolution/TeamGrammarEvolution.fs (NEW)
namespace Tars.Evolution

module TeamGrammarEvolution =

    type EvolutionRole =
        | GrammarCreator    // Creates new grammar rules
        | GrammarMutator    // Modifies existing rules
        | GrammarValidator  // Tests and validates grammars
        | GrammarSynthesizer // Combines multiple grammars
        | GrammarOptimizer  // Improves performance

    type UniversityAgent =
        { Name: string
          Specialization: string
          Capabilities: string list
          OutputFormats: string list
          GrammarAffinity: string list
          EvolutionRole: EvolutionRole }

    type EvolvedGrammarRule =
        { RuleId: string
          OriginalGrammar: string
          RulePattern: string
          RuleBody: string
          CreatedBy: string
          FitnessScore: float
          GenerationNumber: int
          ParentRules: string list
          EvolutionHistory: string list
          UsageCount: int
          SuccessRate: float }

    type GrammarEvolutionSession =
        { SessionId: string
          TeamName: string
          ParticipatingAgents: UniversityAgent list
          BaseGrammars: Grammar list
          EvolvedRules: EvolvedGrammarRule list
          CurrentGeneration: int
          EvolutionGoal: string
          StartTime: DateTime
          LastActivity: DateTime
          IsActive: bool
          PerformanceMetrics: Map<string, float> }
```

### 5.5 Phase 5: FLUX Type System (Week 9-10)

**Port `FluxInterpreter.fs` type system:**

```fsharp
// v2/src/Tars.DSL/FluxTypes.fs (NEW)
namespace Tars.DSL

module FluxTypes =

    type FluxValue =
        | FluxString of string
        | FluxNumber of float
        | FluxBool of bool
        | FluxArray of FluxValue list
        | FluxObject of Map<string, FluxValue>
        | FluxFunction of (FluxValue list -> FluxValue)
        | FluxEffect of (unit -> FluxValue)
        | FluxAgent of AgentInstance

    and AgentInstance =
        { Name: string
          Type: string
          Tier: int
          Capabilities: string list
          LanguageBindings: string list
          State: Map<string, FluxValue> }

    type FluxType =
        | BasicType of string
        | DependentType of string * Map<string, FluxValue>
        | LinearType of string  // Used exactly once
        | RefinedType of string * (FluxValue -> bool)

    type FluxEffect =
        { Name: string
          Dependencies: string list
          Computation: unit -> FluxValue
          Memoized: bool
          Cache: Map<string, FluxValue> }

    type LanguageBinding =
        { Language: string  // "fsharp", "python", "julia", "cuda"
          Environment: string option
          Packages: string list
          Executor: string -> FluxValue }

    type FluxContext =
        { Variables: Map<string, FluxValue>
          Types: Map<string, FluxType>
          Effects: Map<string, FluxEffect>
          Agents: Map<string, AgentInstance>
          GrammarTier: int
          LanguageBindings: Map<string, LanguageBinding>
          TraceCapture: bool
          ExecutionLog: string list }
```

---

## 6. Integration with V2 Architecture

### 6.1 Module Placement

```
v2/src/
├── Tars.Core/
│   ├── GrammarTypes.fs      # NEW: Core grammar types
│   ├── FractalGrammar.fs    # NEW: Fractal grammar system
│   ├── GrammarDistill.fs    # EXISTING: Enhance with fractal support
│   └── GrammarPipeline.fs   # EXISTING: Add tier evolution hooks
│
├── Tars.Evolution/
│   ├── TierEvolution.fs     # NEW: Tier 1-16 evolution
│   ├── TeamGrammarEvolution.fs # NEW: Multi-agent evolution
│   └── Engine.fs            # EXISTING: Integrate grammar evolution
│
├── Tars.DSL/                # NEW PROJECT
│   ├── FluxTypes.fs         # FLUX type system
│   ├── FluxInterpreter.fs   # Multi-language execution
│   └── Tars.DSL.fsproj
│
└── Tars.Cortex/
    └── Grammar.fs           # EXISTING: Enhance EBNF parser
```

### 6.2 Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                        Tars.DSL                             │
│  (FluxTypes, FluxInterpreter, Multi-language execution)     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Tars.Evolution                          │
│  (TierEvolution, TeamGrammarEvolution, Engine)              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Tars.Core                             │
│  (GrammarTypes, FractalGrammar, GrammarPipeline)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Tars.Cortex                            │
│  (Grammar.fs - Enhanced EBNF parser)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Migration Strategy

### 7.1 Incremental Approach

| Phase | Duration | Deliverables | Risk |
|-------|----------|--------------|------|
| **1. Core Types** | 2 weeks | `GrammarTypes.fs`, `GrammarSource` | Low |
| **2. Fractal System** | 2 weeks | `FractalGrammar.fs`, basic engine | Medium |
| **3. Tier Evolution** | 2 weeks | `TierEvolution.fs`, problem analysis | Medium |
| **4. Team Evolution** | 2 weeks | `TeamGrammarEvolution.fs`, sessions | High |
| **5. FLUX Types** | 2 weeks | `FluxTypes.fs`, `Tars.DSL` project | High |

### 7.2 Breaking Changes

1. **`GrammarDistill.GrammarSpec`** → Extend with `Tier` and `FractalProperties`
2. **`GrammarPipeline.DetectedPattern`** → Add `FractalPattern` variant
3. **`Tars.Metascript.Domain.WorkflowStep`** → Add `GrammarTier` field

### 7.3 Backward Compatibility

- Keep existing `GrammarDistill.fromJsonExamples` function
- Add `fromJsonExamplesWithTier` for tier-aware distillation
- Existing metascript workflows continue to work (tier defaults to 1)

---

## 8. V1 Code Samples for Reference

### 8.1 Fractal Grammar Generation (v1)

```fsharp
// From v1/src/TarsEngine.FSharp.Core/Grammar/FractalGrammar.fs
member this.ApplyTransformation(pattern: string, transformation: FractalTransformation) =
    match transformation with
    | Scale factor ->
        if factor > 1.0 then
            pattern + " " + pattern.Substring(0, Math.Min(pattern.Length, int(factor * 10.0)))
        else
            pattern.Substring(0, Math.Max(1, int(float pattern.Length * factor)))
    | Rotate angle ->
        let words = pattern.Split(' ')
        let rotatedWords =
            words |> Array.mapi (fun i word ->
                let rotationIndex = (i + int(angle / 45.0)) % words.Length
                words.[rotationIndex])
        String.Join(" ", rotatedWords)
    | Compose transformations ->
        transformations |> List.fold (fun acc trans -> this.ApplyTransformation(acc, trans)) pattern
    | Recursive (depth, innerTransformation) ->
        if depth <= 0 then pattern
        else this.ApplyTransformation(this.ApplyTransformation(pattern, innerTransformation), Recursive(depth - 1, innerTransformation))
```

### 8.2 Tier Evolution Analysis (v1)

```fsharp
// From v1/src/TarsEngine.FSharp.Core/Grammar/EmergentTierEvolution.fs
let analyzeProblemContext (context: ProblemContext) : EvolutionDirection list =
    let mutable directions = []

    for kvp in context.Constraints do
        let (current, target) = kvp.Value
        let tension = abs(current - target) / target
        if tension > 0.1 then
            match kvp.Key with
            | key when key.Contains("performance") ->
                directions <- ComputationalExpression("PerformanceOptimizationEngine") :: directions
            | key when key.Contains("agent") ->
                directions <- ComputationalExpression("AgentCoordinationProtocol") :: directions
            | _ ->
                directions <- ParameterSpace([kvp.Key]) :: directions

    match context.Domain with
    | "SoftwareDevelopment" ->
        directions <- ComputationalExpression("AutonomousCodeGeneration") :: directions
        directions <- MetaLanguageConstruct("SelfImprovingArchitecture") :: directions
    | "AgentCoordination" ->
        directions <- ComputationalExpression("SemanticTaskRouting") :: directions
        directions <- MetaLanguageConstruct("DynamicTeamFormation") :: directions
    | _ ->
        directions <- ComputationalExpression("GenericOptimizationFramework") :: directions

    directions
```

### 8.3 FLUX Multi-Language Execution (v1)

```fsharp
// From v1/src/TarsEngine.FSharp.Core/FLUX/FluxInterpreter.fs
type FluxValue =
    | FluxString of string
    | FluxNumber of float
    | FluxBool of bool
    | FluxArray of FluxValue list
    | FluxObject of Map<string, FluxValue>
    | FluxFunction of (FluxValue list -> FluxValue)
    | FluxEffect of (unit -> FluxValue)
    | FluxAgent of AgentInstance

let executeFSharp (code: string) : FluxValue =
    try
        printfn "🔧 Executing F# code: %s" code
        FluxString("F# execution completed")
    with ex -> FluxString(sprintf "F# error: %s" ex.Message)

let executePython (code: string) : FluxValue =
    try
        printfn "🐍 Executing Python code: %s" code
        FluxString("Python execution completed")
    with ex -> FluxString(sprintf "Python error: %s" ex.Message)
```

---

## 9. Graphiti Integration Points

The adapted grammar system can integrate with Graphiti (from previous research):

### 9.1 Grammar as Temporal Entities

```fsharp
type GrammarEntity =
    { EntityId: string
      GrammarName: string
      Tier: int
      ValidFrom: DateTime
      ValidTo: DateTime option  // None = still valid
      EvolvedFrom: string option
      FitnessScore: float
      UsageCount: int }
```

### 9.2 Grammar Evolution as Episodes

```fsharp
type GrammarEvolutionEpisode =
    { EpisodeId: string
      Timestamp: DateTime
      SourceGrammar: string
      TargetGrammar: string
      EvolutionType: string  // "tier_advance", "mutation", "synthesis"
      Agent: string
      Reasoning: string
      PerformanceImpact: float }
```

### 9.3 Grammar Communities

```fsharp
type GrammarCommunity =
    { CommunityId: string
      Name: string
      MemberGrammars: string list
      DominantTier: int
      CommonPatterns: string list
      Summary: string }
```

---

## 10. Open Questions

| Question | Status | Notes |
|----------|--------|-------|
| Should we port the full EBNF parser from v1? | 🟡 Open | v2's regex-based parser may be sufficient |
| How to handle v1's RFC caching mechanism? | 🟡 Open | May need `.tars/cache/rfc` directory |
| Should FLUX support CUDA execution in v2? | 🟡 Open | Depends on GPU requirements |
| How to integrate with Graphiti temporal model? | 🟢 Resolved | See Section 9 |
| Should grammar tiers be persisted? | 🟡 Open | Knowledge base vs. file system |

---

## 11. Conclusion

The v1 grammar and DSL infrastructure represents **significant engineering investment** that should be leveraged in v2. The recommended 10-week phased approach allows incremental adoption while maintaining backward compatibility.

**Priority Adaptations:**
1. ✅ `GrammarTypes.fs` - Foundation for all grammar work
2. ✅ `FractalGrammar.fs` - Self-similar pattern expression
3. ✅ `TierEvolution.fs` - Autonomous grammar advancement
4. ⚠️ `TeamGrammarEvolution.fs` - Multi-agent collaboration (higher risk)
5. ⚠️ `FluxTypes.fs` - Advanced type system (highest risk)

The adapted system will enable TARS v2 to:
- Express complex, self-similar grammar patterns
- Autonomously evolve grammars through 16 tiers
- Collaborate across agents for grammar refinement
- Execute multi-language code blocks
- Integrate with Graphiti for temporal grammar tracking

