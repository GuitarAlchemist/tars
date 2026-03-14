# Grammar × Knowledge Integration

## Overview

This document describes the integration between TARS's Grammar Distillation and Knowledge modules, enabling self-improving prompt patterns, output validation, and error learning.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TARS Meta-Learning Loop                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │  Task    │────▶│  Execute     │────▶│  Validate    │    │
│  │  Input   │     │  with LLM    │     │  Output      │    │
│  └──────────┘     └──────────────┘     └──────────────┘    │
│       ▲                  │                 │      │        │
│       │                  │                 │      │        │
│       │                  ▼                 │      ▼        │
│       │           ┌──────────────┐    ┌────┴─────────┐     │
│       │           │   Grammar    │    │   Success?   │     │
│       │           │  Distiller   │    └────┬────┬────┘     │
│       │           └──────────────┘         │    │          │
│       │                  │              Yes│    │No        │
│       │                  ▼                 ▼    ▼          │
│       │           ┌──────────────┐  ┌──────┐ ┌──────┐      │
│       │           │  Knowledge   │◀─│Prompt│ │Error │      │
│       │           │    Base      │  │Pattern│ │Pattern│     │
│       │           └──────────────┘  └──────┘ └──────┘      │
│       │                  │                                  │
│       │                  ▼                                  │
│       │           ┌──────────────┐                         │
│       └───────────│   Retrieve   │                         │
│                   │   Patterns   │                         │
│                   └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Foundation ✅ COMPLETE
- [x] Add `Grammar` and `PatternKind` fields to Knowledge entries
- [x] Implement grammar-based validation (`GrammarValidation.fs`)
- [x] Add error pattern recognition (`ErrorPatterns.fs`)
- [x] Wire into Evolution module (success/failure pattern recording)
- [x] Add 13 validation tests

### Phase 2: Core Loop ✅ COMPLETE
- [x] Evolution → Grammar → Knowledge pipeline (`GrammarPipeline.fs`)
- [x] Pattern detection (JSON, XML, code blocks, answer tags, thinking blocks, tool calls)
- [x] Grammar distillation and storage
- [x] Prompt pattern library with retrieval
- [x] Auto-retry with grammar hints
- [x] Full pipeline integration
- [x] Add 15 pipeline tests (28 total validation tests)

### Phase 3: Intelligence ✅ COMPLETE
- [x] Model-specific pattern adaptation (`GrammarIntelligence.fs`)
- [x] Grammar-aware RAG retrieval with similarity scoring
- [x] Confidence-weighted pattern selection
- [x] Pattern stats tracking (success/failure per model)
- [x] Adaptive prompt building based on model history
- [x] Add 18 intelligence tests (46 total validation tests)

## Data Structures

### Extended Knowledge Entry
```fsharp
type Entry = {
    Id: string
    Title: string
    Category: Category
    Content: string
    Grammar: string option      // NEW: Distilled grammar pattern
    Confidence: Confidence
    Source: Source
    Tags: string list
    CreatedAt: DateTime
    UpdatedAt: DateTime
}
```

### Validation Result
```fsharp
type ValidationResult =
    | Valid of parsed: Map<string, obj>
    | Invalid of ValidationError

type ValidationError = {
    Expected: string
    Actual: string
    Error: string
    Suggestions: string list
}
```

### Pattern Types
```fsharp
type PatternKind =
    | PromptPattern    // Successful prompt structures
    | OutputPattern    // Expected output formats
    | AntiPattern      // Known failure modes
    | ReasoningPattern // Chain-of-thought templates
```

### Model-Specific Pattern Stats (Phase 3)
```fsharp
type PatternStats = {
    Model: string           // Model name (e.g., "llama3.2")
    SuccessCount: int       // Times pattern succeeded with this model
    FailureCount: int       // Times pattern failed with this model
    LastUsed: DateTime      // Last time pattern was used
    AverageLatency: float option  // Average response time
}

type ModelPattern = {
    Entry: Entry                      // The knowledge entry
    Stats: Map<string, PatternStats>  // Stats per model
}

type SelectionWeights = {
    ConfidenceWeight: float   // Weight for entry confidence
    SuccessRateWeight: float  // Weight for model success rate
    RecencyWeight: float      // Weight for recent usage
    RelevanceWeight: float    // Weight for query relevance
}
```

## Usage Examples

### 1. Validate LLM Output
```fsharp
let grammar = "<answer>{text}</answer>"
match Validation.validate grammar llmOutput with
| Valid parsed -> processAnswer parsed.["text"]
| Invalid err -> 
    ErrorPatterns.record kb task llmOutput err
    retryWithHint grammar
```

### 2. Retrieve Patterns for Task
```fsharp
let patterns = kb.Search($"pattern {taskType}")
let grammars = patterns |> List.choose (_.Grammar)
let prompt = buildPromptWithGrammars basePrompt grammars
```

### 3. Learn from Evolution
```fsharp
// After successful task completion
let grammar = GrammarDistiller.extract output
kb.Add(
    title = $"Pattern: {taskType}",
    content = output,
    grammar = Some grammar,
    category = Learned,
    tags = ["pattern"; taskType; model]
)
```

### 4. Detect Output Patterns (Phase 2)
```fsharp
open Tars.Core.GrammarPipeline

// Automatically detect what pattern an LLM output follows
let output = """{"name": "John", "age": 30}"""
match detectPattern output with
| JsonObject fields -> printfn "JSON with fields: %A" fields
| XmlTags tags -> printfn "XML with tags: %A" tags
| CodeBlock lang -> printfn "Code block: %A" lang
| AnswerTag content -> printfn "Answer: %s" content
| ThinkingBlock content -> printfn "Thinking: %s" content
| ToolCall name -> printfn "Tool call: %s" name
| Unknown -> printfn "Unknown pattern"
```

### 5. Distill and Store Patterns (Phase 2)
```fsharp
// Automatically distill grammar from output and store in KB
let entry = distillAndStore kb "Parse user data" output ["json"; "user"]
// entry.Grammar = Some "{ \"name\": {name}, \"age\": {age} }"
```

### 6. Retrieve Patterns for Prompts (Phase 2)
```fsharp
// Get relevant patterns from KB to enhance prompts
let patterns = retrievePatterns kb "Parse JSON response" (Some OutputPattern)
let hint = buildPromptHint patterns
// hint = "Expected output format:\n{ \"name\": {name}, \"age\": {age} }"
```

### 7. Auto-Retry with Grammar Hints (Phase 2)
```fsharp
// Execute with automatic retry and grammar hints
let config = { MaxAttempts = 3; GrammarHint = Some "<answer>{text}</answer>" }
let execute prompt = callLLM prompt
let validate output =
    if output.Contains("<answer>") then Result.Ok output
    else Result.Error "Missing answer tag"

match executeWithRetry execute validate basePrompt config with
| Success (result, attempts) -> printfn "Succeeded on attempt %d" attempts
| AllFailed errors -> printfn "All attempts failed: %A" errors
```

### 8. Full Pipeline (Phase 2)
```fsharp
// Run the complete pipeline: retrieve patterns → execute → validate → distill → store
let result = runPipeline kb "Extract user info" callLLM basePrompt expectedGrammar ["user"]
match result with
| Result.Ok r ->
    printfn "Output: %s" r.Output
    printfn "Attempts: %d" r.Attempts
    printfn "Distilled: %A" r.DistilledPattern
    printfn "Used patterns: %d" r.UsedPatterns.Length
| Result.Error e -> printfn "Failed: %s" e
```

### 9. Model-Specific Pattern Selection (Phase 3)
```fsharp
open Tars.Core.GrammarIntelligence

// Load patterns with model-specific stats
let patterns = loadPatternsWithStats kb

// Select best patterns for a specific model
let weights = { ConfidenceWeight = 0.3; SuccessRateWeight = 0.4; RecencyWeight = 0.2; RelevanceWeight = 0.1 }
let selected = selectPatterns "llama3.2" "Parse JSON" 3 weights patterns
// Returns patterns sorted by weighted score for this model
```

### 10. Grammar-Aware RAG Retrieval (Phase 3)
```fsharp
// Retrieve patterns that match a target grammar structure
let targetGrammar = Some "{ \"name\": {name}, \"age\": {age} }"
let results = retrieveWithGrammar kb "user data" targetGrammar (Some OutputPattern) 5
// Returns entries with similar grammar structures, scored by similarity
```

### 11. Adaptive Prompt Building (Phase 3)
```fsharp
// Build prompts that adapt based on model's historical success with patterns
let patterns = loadPatternsWithStats kb
let prompt = buildAdaptivePrompt "llama3.2" "Parse user info" basePrompt patterns weights
// Prompt includes hints from patterns that worked well for this model
```

### 12. Record Pattern Usage (Phase 3)
```fsharp
// Track success/failure per model for continuous improvement
let pattern = patterns |> List.head
let updated = recordPatternUsage kb "llama3.2" pattern true (Some 150.0)
// Updates stats: success count, last used time, average latency
```

## File Changes

| File | Changes |
|------|---------|
| `Tars.Core/Knowledge.fs` | Add Grammar field, pattern helpers |
| `Tars.Core/Validation.fs` | NEW: Grammar validation module |
| `Tars.Core/ErrorPatterns.fs` | NEW: Error pattern recording |
| `Tars.Core/GrammarPipeline.fs` | NEW: Full pipeline with pattern detection, distillation, retrieval, retry |
| `Tars.Core/GrammarIntelligence.fs` | NEW: Model-specific adaptation, grammar-aware RAG, confidence-weighted selection |
| `Tars.Evolution/Engine.fs` | Wire in pattern learning |
| `Tars.Tests/ValidationTests.fs` | NEW: 46 validation, pipeline, and intelligence tests |

## Success Metrics

- **Validation Rate**: % of outputs passing grammar validation
- **Retry Reduction**: Fewer retries needed over time
- **Pattern Reuse**: How often stored patterns are retrieved
- **Error Prevention**: Known anti-patterns avoided

