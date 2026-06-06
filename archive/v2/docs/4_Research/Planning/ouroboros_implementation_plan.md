# The Ouroboros Loop: Implementing Self-Reinforcing Code Evolution

**Date:** November 22, 2025
**Status:** Implementation Plan
**Dependencies:** `leveraging_codewiki.md`, `v1_component_reusability_analysis.md`

---

## 1. The Concept

The **Ouroboros Loop** is a self-reinforcing cycle where the system's output (Code) becomes its input (Context & Constraints) for the next iteration.

In TARS v2, we close this loop using two key technologies:

1. **Google Code Wiki**: Converts Code → Knowledge (Docs/Graphs).
2. **Grammar Distillation**: Converts Code/Knowledge → Constraints (Grammars).

This ensures that as TARS writes more code, it becomes *easier* to write consistent code, not harder.

---

## 2. The Cycle Architecture

### Step 1: Synthesis (The Writer)

* **Actor**: `Tars.Agents.Developer`
* **Action**: Generates F# code to implement a feature.
* **Input**: User Request + Existing Context.

### Step 2: Analysis (The Observer)

* **Actor**: `Code Wiki` (External Service)
* **Action**: Automatically detects the commit, updates the Wiki, redraws architecture diagrams, and indexes the new symbols.
* **Output**: Updated English documentation and Knowledge Graph.

### Step 3: Distillation (The Legislator)

* **Actor**: `Tars.Cortex.Grammar` (Reused v1 Component)
* **Action**:
    1. Reads the new Code Wiki summary.
    2. Analyzes the AST of the new code.
    3. **Distills a Grammar**: "I see you used the `Result<T,E>` pattern here. I will now create a GBNF grammar rule that *enforces* this pattern for all future error handling."
* **Output**: Updated `system_grammar.gbnf`.

### Step 4: Constraint (The Guide)

* **Actor**: `Tars.Cortex.Inference`
* **Action**: When the Developer Agent tries to write the *next* feature, the Inference Engine applies the `system_grammar.gbnf`.
* **Result**: The LLM is mathematically incapable of deviating from the established patterns.

---

## 3. Concrete Implementation Plan

### Phase 1: The "Manual" Ouroboros (Epic 2/3)

Since Code Wiki is external and Grammar Distillation needs porting, we start with a manual loop.

**Components:**

1. **`Tars.Skills.CodeWiki` (Mock)**:
    * Reads local `v2/docs/knowledge_base/*.md` files (which we manually update from Code Wiki).
2. **`Tars.Cortex.Grammar` (Ported)**:
    * Extract `GrammarDistillation` from v1.
    * Implement `DistillFromSource(file_path)`: Simple regex/AST extraction of function signatures.

**Workflow:**

1. TARS writes `VectorStore.fs`.
2. We manually copy Code Wiki summary to `docs/knowledge_base/VectorStore.md`.
3. TARS reads `docs/knowledge_base/VectorStore.md` before writing `VectorStoreTests.fs`.

### Phase 2: The Automated Pipeline (Epic 6 - Evolution)

**Components:**

1. **`CodeWikiSkill` (Real)**:
    * Uses Gemini CLI or API to fetch live wiki pages.
2. **`GrammarEvolutionAgent`**:
    * A specialized agent that wakes up after every merge.
    * Prompt: "Review the Code Wiki changes. Update the `style_guide.gbnf` to reflect new patterns."

**Workflow:**

1. **Commit Trigger**: TARS pushes code.
2. **Wiki Update**: Code Wiki updates automatically.
3. **Evolution Trigger**: `GrammarEvolutionAgent` reads the new Wiki page.
4. **Rule Update**: Agent updates `Tars.Cortex.Grammar/constraints/style.gbnf`.
    * *Example*: "New rule: All async functions must return `Task<Result<T, AppError>>`."

### Phase 3: The Real-Time Loop (Epic 8 - FLUX)

**Components:**

1. **`FLUX` Metascript**:
    * A script that runs *during* generation.
    * It queries Code Wiki for the *intent* of a module, then generates the Grammar on the fly.

**Workflow:**

1. User: "Add a new method to `VectorStore`."
2. TARS: "Wait, let me check the Ouroboros."
3. Query: `CodeWiki.GetPattern("VectorStore")` -> Returns "Uses ConcurrentDictionary and File Persistence".
4. Distill: Convert that description into a Grammar.
5. Generate: Write code that perfectly matches the description.

---

## 4. Technical Integration Details

### The Grammar Distiller (`Tars.Cortex.Grammar`)

We will reuse the v1 `GrammarDistillation` logic but simplify it.

**Interface:**

```fsharp
type IGrammarDistiller =
    /// Takes code or documentation and produces a GBNF grammar fragment
    abstract Distill: input: string -> sourceType: SourceType -> Async<string>

type SourceType =
    | RawCode      // AST analysis
    | WikiSummary  // NLP extraction via LLM
```

**Logic:**

1. **From Code**: Use F# Compiler Services (FCS) to get the AST. Extract public signatures. Convert to GBNF.
2. **From Wiki**: Feed the Wiki summary to a small LLM (GPT-4o-mini) with a prompt: "Convert this architectural description into a JSON schema or GBNF grammar that enforces this structure."

### The Code Wiki Skill (`Tars.Skills.CodeWiki`)

**Interface:**

```fsharp
type ICodeWikiSkill =
    abstract GetSummary: modulePath: string -> Async<string>
    abstract GetDiagram: modulePath: string -> Async<string>
    abstract GetPatterns: modulePath: string -> Async<string list>
```

---

## 5. Success Metrics

How do we know the Ouroboros is working?

1. **Consistency Score**: Run a linter. Does the new code match the style of the old code?
2. **Context Reduction**: Can we write a feature using only the Wiki summary (2k tokens) instead of the raw code (100k tokens)?
3. **Drift Detection**: Does the Grammar Engine reject code that violates the Wiki's architectural description?

---

## 6. Next Steps

1. **Epic 2**: Port `GrammarDistillation` from v1.
2. **Epic 4**: Build the `CodeWikiSkill` (initially reading local markdown).
3. **Epic 6**: Wire them together into the `GrammarEvolutionAgent`.
