# TARS V2 Cost Budget & Token Accounting

**Date:** November 22, 2025
**Status:** Design
**Priority:** P0 (CRITICAL)
**Context:** Preventing runaway costs from infinite agent loops.

---

## The Cost Problem

**Scenario:** User asks TARS to "fix all bugs in the repository."

- TARS finds 100 bugs
- Each bug requires 5 iterations to fix (20K tokens avg)
- Total: 100 × 20K × $0.01/1K = **$200**
- If agent loops infinitely (bug → fix → new bug): **UNLIMITED COST**

**Risk:** Without budget enforcement, a single task could bankrupt the user.

---

## Budget Architecture

### 1. Budget Definition

```fsharp
type Budget = {
    MaxTokens: int option       // Hard limit (e.g., 100K tokens)
    MaxDollars: decimal option  // Hard limit (e.g., $5.00)
    MaxMinutes: int option      // Timeout (e.g., 30 minutes)
    WarnAt: decimal             // Warn at X% of budget (e.g., 0.8 = 80%)
}

let defaultBudget = {
    MaxTokens = Some 100_000
    MaxDollars = Some 5.0m
    MaxMinutes = Some 30
    WarnAt = 0.8m
}
```

### 2. Token Accounting

```fsharp
type TokenUsage = {
    ModelName: string           // "gpt-4o" | "claude-3.5-sonnet"
    InputTokens: int
    OutputTokens: int
    EstimatedCost: decimal
    Timestamp: DateTime
}

type ITokenAccountant =
    abstract member RecordUsage: TokenUsage -> Async<unit>
    abstract member GetTotalCost: unit -> Async<decimal>
    abstract member GetRemainingBudget: Budget -> Async<decimal>

// Implementation
type TokenAccountant(logger: ILogger) =
    let mutable usage: TokenUsage list = []
    
    interface ITokenAccountant with
        member _.RecordUsage(t) = async {
            usage <- t :: usage
            logger.LogInformation($"Token usage: {t.InputTokens + t.OutputTokens} tokens, ${t.EstimatedCost}")
        }
        
        member _.GetTotalCost() = async {
            return usage |> List.sumBy (_.EstimatedCost)
        }
        
        member _.GetRemainingBudget(budget) = async {
            let! totalCost = (this :> ITokenAccountant).GetTotalCost()
            return budget.MaxDollars |> Option.map (fun max -> max - totalCost) |> Option.defaultValue 0m
        }
```

---

## Pricing Table (Nov 2025)

| Model | Input ($/1M) | Output ($/1M) |
| :--- | ---: | ---: |
| **GPT-4o** | $2.50 | $10.00 |
| **GPT-4o-mini** | $0.15 | $0.60 |
| **Claude 3.5 Sonnet** | $3.00 | $15.00 |
| **Claude 3.5 Haiku** | $0.80 | $4.00 |
| **Gemini 1.5 Pro** | $1.25 | $5.00 |
| **Gemini 1.5 Flash** | $0.075 | $0.30 |

**Cost Calculation:**

```fsharp
let modelPricing = Map [
    ("gpt-4o", (2.50m, 10.00m))
    ("gpt-4o-mini", (0.15m, 0.60m))
    ("claude-3.5-sonnet", (3.00m, 15.00m))
    ("claude-3.5-haiku", (0.80m, 4.00m))
]

let calculateCost (model: string) (inputTokens: int) (outputTokens: int) : decimal =
    let (inputPrice, outputPrice) = modelPricing.[model]
    let inputCost = decimal inputTokens / 1_000_000m * inputPrice
    let outputCost = decimal outputTokens / 1_000_000m * outputPrice
    inputCost + outputCost
```

---

## Budget Enforcement Strategy

### Phase 1: Soft Limits (Warnings)

```fsharp
let checkBudget (budget: Budget) (accountant: ITokenAccountant) = async {
    let! totalCost = accountant.GetTotalCost()
    let! remaining = accountant.GetRemainingBudget(budget)
    
    match budget.MaxDollars with
    | Some maxDollars ->
        let percentUsed = totalCost / maxDollars
        if percentUsed >= budget.WarnAt then
            return Warning $"⚠️  Budget {percentUsed * 100m:F0}%% used (${totalCost:F2} / ${maxDollars:F2})"
        else
            return Ok()
    | None -> return Ok()
}
```

**UI Display:**

```
┌─────────────────────────────────────────────┐
│ Budget Status                               │
├─────────────────────────────────────────────┤
│ Used:   $4.20 / $5.00  [████████░░] 84%     │
│ Tokens: 98,432 / 100,000                    │
│ Time:   22 min / 30 min                     │
│                                             │
│ ⚠️  Approaching budget limit!               │
└─────────────────────────────────────────────┘
```

### Phase 2: Hard Limits (Kill Switch)

```fsharp
let enforceHardLimit (budget: Budget) (accountant: ITokenAccountant) = async {
    let! totalCost = accountant.GetTotalCost()
    
    match budget.MaxDollars with
    | Some maxDollars when totalCost >= maxDollars ->
        return Error "Budget exceeded. Stopping agent."
    | _ -> return Ok()
}
```

**When budget exceeded:**

1. Agent immediately exits current loop
2. Saves session state to disk
3. Shows user: *"Budget limit reached. Session saved as `session-{id}.json`."*

---

## Model Fallback Strategy

**Dynamic pricing tiers based on remaining budget.**

```fsharp
type ModelTier = Primary | Fallback | Economy | ReadOnly

let selectModel (remainingBudget: decimal) : ModelTier * string =
    match remainingBudget with
    | r when r > 2.50m -> (Primary, "gpt-4o")           // Full power
    | r when r > 1.00m -> (Fallback, "gpt-4o-mini")      // Medium power
    | r when r > 0.50m -> (Economy, "claude-3.5-haiku") // Low power
    | _ -> (ReadOnly, "none")                            // Read-only mode
```

**Fallback Logic:**

```
┌──────────────────┐
│ Budget Remaining │
└────────┬─────────┘
         │
    > $2.50 ──► GPT-4o          (Primary)
         │
    > $1.00 ──► GPT-4o-mini     (Fallback)
         │
    > $0.50 ──► Claude Haiku    (Economy)
         │
    ≤ $0.50 ──► Read-Only Mode  (No LLM calls)
```

---

## Caution Mode (80-95% Budget)

**When budget is 80-95% consumed, enter "Caution Mode":**

1. **Prompt User Before Expensive Calls**

   ```
   ┌─────────────────────────────────────────────┐
   │ ⚠️  Expensive Operation                     │
   ├─────────────────────────────────────────────┤
   │ TARS wants to call GPT-4o                   │
   │ Estimated cost: ~$0.80                      │
   │ Remaining budget: $1.20                     │
   │                                             │
   │ Continue?                                   │
   │ [✓ Yes]  [✗ No, use cheaper model]         │
   └─────────────────────────────────────────────┘
   ```

2. **Use Smaller Context Windows**
   - Reduce prompt size by 50%
   - Summarize previous steps more aggressively

3. **Prefer Read-Only Operations**
   - Agent can still analyze code, but requires approval to modify files

---

## Budget Profiles

**Allow users to define budget templates.**

```fsharp
type BudgetProfile = {
    Name: string
    Budget: Budget
    Description: string
}

let budgetProfiles = [
    {
        Name = "Quick Task"
        Budget = { MaxDollars = Some 0.50m; MaxTokens = Some 10_000; MaxMinutes = Some 5; WarnAt = 0.8m }
        Description = "For simple, quick tasks"
    }
    {
        Name = "Standard Session"
        Budget = defaultBudget
        Description = "Default for most work"
    }
    {
        Name = "Deep Work"
        Budget = { MaxDollars = Some 20.0m; MaxTokens = Some 500_000; MaxMinutes = Some 120; WarnAt = 0.9m }
        Description = "For complex, long-running tasks"
    }
    {
        Name = "Unlimited (Development)"
        Budget = { MaxDollars = None; MaxTokens = None; MaxMinutes = None; WarnAt = 0.9m }
        Description = "⚠️  No limits (use with caution)"
    }
]
```

**Usage:**

```bash
$ tars fix-build --budget "Quick Task"
# Uses $0.50 budget

$ tars refactor-codebase --budget "Deep Work"
# Uses $20 budget
```

---

## Dashboard & Reporting

### Real-Time Cost Tracker

```
┌─────────────────────────────────────────────┐
│ TARS Cost Dashboard                         │
├─────────────────────────────────────────────┤
│ Current Session:                            │
│   Cost:  $3.42                              │
│   Calls: 12                                 │
│   Avg:   $0.29/call                         │
│                                             │
│ This Month:                                 │
│   Total: $42.18                             │
│   Sessions: 8                               │
│                                             │
│ Top Models:                                 │
│   1. GPT-4o       $25.10 (60%)              │
│   2. GPT-4o-mini   $8.40 (20%)              │
│   3. Claude Haiku  $8.68 (20%)              │
└─────────────────────────────────────────────┘
```

### Session Cost Report (Export to CSV)

```csv
SessionId,Task,StartTime,EndTime,TotalCost,TokensUsed,Model
sess-001,Fix build,2025-11-22 10:00,2025-11-22 10:15,$2.40,48000,gpt-4o
sess-002,Refactor,2025-11-22 11:00,2025-11-22 11:45,$8.20,164000,gpt-4o
```

---

## Cost Optimization Tips

### Tip 1: Use Cheaper Models for Scoring

```fsharp
// Instead of using GPT-4o to score ALL subtasks:
let scoreSubtasks (tasks: Task list) = async {
    // Use GPT-4o-mini for scoring (10x cheaper)
    let! scores = tasks |> List.map (fun t -> cheapModel.Score(t))
    
    // Only process top 3 with GPT-4o
    let topTasks = scores |> List.sortByDescending snd |> List.take 3
    return! topTasks |> List.map (fun t -> expensiveModel.Execute(t))
}
```

### Tip 2: Cache Repetitive Prompts

```fsharp
type PromptCache = Dictionary<string, string>

let cachedCompletion (cache: PromptCache) (prompt: string) = async {
    match cache.TryGetValue(prompt) with
    | true, result -> 
        logger.LogInfo("Cache hit! Saved $0.10")
        return result
    | false, _ ->
        let! result = llm.Complete(prompt)
        cache.[prompt] <- result
        return result
}
```

### Tip 3: Batch Similar Operations

```fsharp
// Instead of 10 separate LLM calls:
for file in files do
    let! summary = llm.Summarize(file)  // 10 calls

// Batch into 1 call:
let! summaries = llm.Summarize(files |> String.concat "\n---\n")  // 1 call
```

---

## Implementation Checklist

- [ ] **Week 1**: Implement `ITokenAccountant` interface
- [ ] **Week 2**: Add budget enforcement to F# Micro-Kernel
- [ ] **Week 3**: Build TUI dashboard for cost tracking
- [ ] **Week 4**: Add model fallback strategy
- [ ] **Week 5**: Implement caution mode + user prompts

---

## References

- [OpenAI Pricing](https://openai.com/pricing)
- [Anthropic Pricing](https://www.anthropic.com/pricing)
- [Google AI Pricing](https://ai.google.dev/pricing)
- [LangSmith Cost Tracking](https://docs.smith.langchain.com/evaluation/how_to_guides/measure_cost)
