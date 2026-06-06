# TARS DSL Unification: .trsx, .flux, and Self-Extending Grammars

## Investigation Summary

### What Was Found

#### 1. `.trsx` Format (TARS eXtensible Script)

From `hybrid_brain_architecture.md` and V1 artifacts:

**Purpose**: The surface DSL for defining plans, tasks, agents, constraints, and workflows.

**Key Characteristics**:
- Human-readable, diff-friendly
- Versionable in git
- Compiles to typed F# IR (Intermediate Representation)
- Two-level grammar: Surface DSL → Strict IR

**V1 Examples Found**:
- `tars-self-improvement-cycle.trsx` - Self-improvement loop
- `autonomous_ui_evolution.trsx` - Dynamic UI generation
- `diagnostics-{traceId}.trsx` - Execution traces
- `sample.wot.trsx` - Workflow-of-Thought definitions

#### 2. `.flux` Format (FLUX Metaprogramming)

From `v2_ideas.md`:

**Purpose**: Language-agnostic metaprogramming layer for code generation and transformation.

**FLUX is for**:
- Cross-language code generation (Python, Rust, C#, F#, Java)
- DSL creation and evolution
- Grammar refinement
- Code synthesis from patterns

**Philosophy**:
> "If the Cortex is the mind, FLUX is the meta-mind—a way for TARS to reason about how it should reason."

#### 3. Current State in V2

| File Extension | Status | Parser Location |
|----------------|--------|-----------------|
| `.tars` | ✅ Implemented | `Tars.Metascript/V1/Parser.fs` |
| `.trsx` | ⚠️ Partially supported | `RunCommand.fs` (line 38) |
| `.flux` | ❌ Not implemented | Design only |
| `.wot.trsx` | ❌ Planned | Phase 14 |

---

## Recommendation: Unified DSL Architecture

### The Vision

> **"TARS should be able to extend its own grammars and DSLs."**

```
┌─────────────────────────────────────────────────────────────────┐
│                    TARS DSL HIERARCHY                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 0: CORE GRAMMAR (Immutable)                              │
│  ┌─────────────────────────────────────────────┐               │
│  │ Base primitives: meta{}, FSHARP{}, QUERY{}, │               │
│  │ COMMAND{}, TOOL{}, etc.                      │               │
│  └─────────────────────────────────────────────┘               │
│         ↓ extends                                                │
│                                                                  │
│  Layer 1: EXTENSION GRAMMARS (Self-Generated)                   │
│  ┌─────────────────────────────────────────────┐               │
│  │ BELIEF{}, INVARIANT{}, CONSTRAINT{},        │               │
│  │ WORKFLOW{}, DECISION{}, POLICY{}, etc.      │               │
│  └─────────────────────────────────────────────┘               │
│         ↓ defined in                                             │
│                                                                  │
│  Layer 2: FILE FORMATS                                          │
│  ┌─────────────────────────────────────────────┐               │
│  │ .tars   = Metascripts (workflows)           │               │
│  │ .trsx   = Rich scripts (plans + traces)     │               │
│  │ .flux   = Meta-scripts (code generation)    │               │
│  │ .wot    = Workflow-of-Thought (reasoning)   │               │
│  │ .ebnf   = Grammar definitions               │               │
│  └─────────────────────────────────────────────┘               │
│         ↓ compile to                                             │
│                                                                  │
│  Layer 3: TYPED IR (Intermediate Representation)                │
│  ┌─────────────────────────────────────────────┐               │
│  │ Plan<Draft> → Plan<Validated> → Executable  │               │
│  │ (F# types with phantom states)              │               │
│  │ **SINGLE EXECUTABLE IR for ALL DSLs**       │               │
│  └─────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Strategy

### Step 1: Define Core Grammar Extensions

TARS should be able to parse and execute these new block types:

```tars
// Current blocks (already implemented)
meta { name = "..." }
FSHARP(output="x") { ... }
QUERY(output="y", grammar="Name") { ... }
COMMAND { ... }
TOOL(name="...") { ... }

// NEW: Extension blocks (self-definable)
BELIEF(confidence=0.85, source="Wikipedia") {
    Paris is the capital of France.
}

INVARIANT(id="INV-001") {
    require: plan.Budget <= maxBudget
    message: "Budget constraint violated"
}

CONSTRAINT(type="temporal") {
    step[A] must_precede step[B]
    step[B] must_complete_before "5m"
}

WORKFLOW(name="self-improvement", version="1.0") {
    THOUGHT(id="analyze") { ... }
    DECISION(id="decide") { ... }
    ACTION(id="execute") { ... }
    VERIFY(id="check") { ... }
}
```

### Step 2: Grammar Self-Extension API

```fsharp
// In Tars.Core.SelfExtension
module GrammarExtension =
    
    /// A custom block type definition
    type BlockDefinition = {
        Name: string
        Parameters: Parameter list
        ContentType: ContentType
        Compiler: string -> Result<IR, ParseError>
    }
    
    /// Register a new block type at runtime
    let registerBlock (def: BlockDefinition) : unit
    
    /// Generate EBNF for a new block
    let generateEBNF (def: BlockDefinition) : string
    
    /// Compile extended grammar to parser
    let compileGrammar (ebnf: string) : Parser<IR>
```

### Step 3: The Self-Extending Grammar Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                  GRAMMAR EVOLUTION LOOP                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. LLM PROPOSES new grammar rule                               │
│     ↓                                                            │
│  2. PARSER validates EBNF syntax                                │
│     ↓                                                            │
│  3. VALIDATOR checks for conflicts with existing rules          │
│     ↓                                                            │
│  4. SANDBOX tests grammar with sample inputs                    │
│     ↓                                                            │
│  5. REGISTRY adds grammar to extension layer                    │
│     ↓                                                            │
│  6. TARS can now use new grammar in future scripts!             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Step 4: .trsx vs .tars vs .flux Distinction

| Format | Purpose | Execution Model |
|--------|---------|-----------------|
| `.tars` | Simple metascripts | Sequential block execution |
| `.trsx` | Rich scripts + traces | Typed IR + state machine |
| `.flux` | Meta-metascripts | Code generation + transformation |
| `.wot.trsx` | Workflow-of-Thought | Graph-based reasoning |

```
.tars:  Execute blocks sequentially
.trsx:  Compile to IR, validate, execute with provenance
.flux:  Generate code in any language, evolve grammars
.wot:   Execute as reasoning graph with knowledge integration
```

---

## Proposed File Structure

```
v2/
├── grammars/
│   ├── core/                    # Immutable core grammars
│   │   ├── metascript.ebnf      # Base .tars grammar
│   │   ├── trsx.ebnf            # Extended .trsx grammar
│   │   ├── flux.ebnf            # Meta-script grammar
│   │   └── wot.ebnf             # Workflow-of-Thought grammar
│   ├── extensions/              # Self-generated extensions
│   │   ├── belief.ebnf
│   │   ├── invariant.ebnf
│   │   └── manifest.json        # Extension registry
│   └── generated/               # TARS-created grammars
├── metascripts/
│   ├── core/                    # System metascripts
│   ├── generated/               # TARS-created metascripts  
│   └── examples/                # Reference implementations
├── wot/                         # Workflow-of-Thought definitions
│   ├── self_improvement.wot.trsx
│   ├── belief_validation.wot.trsx
│   └── grammar_evolution.wot.trsx
└── src/
    └── Tars.DSL/                # NEW: Unified DSL project
        ├── CoreGrammar.fs       # Base grammar parser
        ├── ExtensionParser.fs   # Extension block parser
        ├── GrammarEvolver.fs    # Self-extending grammar engine
        └── UnifiedCompiler.fs   # Compiles all formats to IR
```

---

## Implementation Priority

### Phase 1: Unify Existing Formats (Week 1-2)
1. Consolidate `.tars` and `.trsx` under one parser
2. Add proper extension recognition
3. Update `tars run` to handle all formats

### Phase 2: Core Extension Blocks (Week 3-4)
1. Implement `BELIEF{}`, `INVARIANT{}`, `CONSTRAINT{}` blocks
2. Add them to extension grammar
3. Wire to HybridBrain IR

### Phase 3: Grammar Self-Extension (Week 5-6)
1. Implement `GrammarEvolver.fs`
2. Create `tars extend grammar` enhancement
3. Enable TARS to propose new block types

### Phase 4: FLUX Metaprogramming (Week 7-8)
1. Implement `.flux` parser
2. Add code generation to any language
3. Enable grammar-to-grammar transformation

---

## Key Insight

> **"The safest form of self-improvement is grammar evolution, not code modification."**

Instead of TARS modifying its own F# code:
1. TARS creates new DSL constructs (grammars)
2. TARS writes metascripts using those constructs
3. The core F# engine interprets the constructs safely

**This is how TARS becomes smarter without breaking itself.**

---

## Next Steps

1. **Create `Tars.DSL` project** with unified parser
2. **Port V1 `.trsx` examples** from `.tars` folder
3. **Implement core extension blocks** (BELIEF, INVARIANT, etc.)
4. **Add `tars extend block` command** for self-extension
5. **Create demonstration** of grammar evolution

---

*"TARS doesn't modify its brain. It extends its vocabulary."*
