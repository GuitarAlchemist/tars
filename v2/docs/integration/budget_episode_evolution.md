# TARS Evolution Command - Budget & Episode Integration

## Overview
Successfully integrated Budget Governance and Graphiti Episode Ingestion into the TARS Evolution command.

## Changes Made

### 1. Budget Governance ✅
- **Added Budget Configuration**
  - New field `Budget: decimal option` in `EvolveOptions` (Evolve.fs:27)
  - CLI parameter `--budget USD` for setting max monetary cost
  - Default budget: $10.00 USD if not specified
  - Budget integrated into Evolution context (Evolve.fs:379-389)

- **Usage Examples**
  ```bash
  # Use default $10 budget
  tars evolve --max-iterations 5
  
  # Set custom budget
  tars evolve --max-iterations 10 --budget 25.0
  
  # Minimal budget for testing
  tars evolve --max-iterations 1 --budget 1.0 --quiet
  ```

### 2. Graphiti Episode Ingestion ✅
- **Automatic Detection**
  - Checks `GRAPHITI_URL` environment variable
  - Falls back to `None` if not configured
  - Integrated into Evolution context (Evolve.fs:473-476)

- **Configuration**
  ```bash
  # Enable Graphiti integration
  $env:GRAPHITI_URL = "http://localhost:8000"
  tars evolve --max-iterations 5
  ```

### 3. MetascriptContext Updates ✅
All MetascriptContext initializations updated with `EpisodeService` field:
- `MacroDemo.fs` (line 158)
- `Run.fs` (line 96)
- `RunCommand.fs` (line 172)
- `RagDemo.fs` (lines 850, 1210)

### 4. Model Configuration ✅
- **Default Model**: Changed from `qwen2.5-coder:1.5b` to `qwen2.5-coder:7b`
  - Reason: 1.5b model not available in Ollama
  - Available models verified via `ollama list`

## Test Results

### Evolution Tests: **4/4 PASSING** ✅
```bash
dotnet test --filter Evolution --no-build
# Test summary: total: 4, failed: 0, succeeded: 4, skipped: 0
```

### Build Status: **SUCCESSFUL** ✅
```bash
dotnet build src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj
# Build succeeded in 7.1s
```

### Runtime Verification: **WORKING** ✅
```bash
dotnet run --project src/Tars.Interface.Cli -- evolve --max-iterations 1 --budget 1.0 --quiet
# Successfully generates tasks and executes with Ollama
```

## Available CLI Options

### Evolve Command
```bash
tars evolve [options]

Options:
  --max-iterations N     Set max generations (default: 5)
  --budget USD          Maximum monetary budget in USD (default: 10.0)
  --quiet               Suppress splash screen
  --demo                Use demo mode (disables epistemic governor)
  --verbose, -v         Show detailed logs
  --trace               Save execution trace
  --model <name>        Override default LLM model
```

## Integration Points

### Budget Governor
```fsharp
// Session budget for evolution
let budget =
    BudgetGovernor(
        { Budget.Infinite with
            MaxTokens = Some 1000000<token>
            MaxMoney = options.Budget 
                      |> Option.map (fun m -> m * 1m<usd>) 
                      |> Option.orElse (Some 10.0m<usd>) })
```

### Episode Service
```fsharp
EpisodeService = 
    match Environment.GetEnvironmentVariable("GRAPHITI_URL") with
    | null -> None
    | url -> Some (createServiceWithUrl url)
```

## Troubleshooting

### Issue: 404 Not Found
**Cause**: Ollama not running or model unavailable  
**Solution**: 
```bash
# Check Ollama status
ollama list

# Start Ollama
ollama serve

# Pull required model
ollama pull qwen2.5-coder:7b
```

### Issue: Budget exceeded too quickly
**Cause**: Complex tasks consuming tokens rapidly  
**Solution**: Increase budget or reduce iterations
```bash
tars evolve --max-iterations 3 --budget 20.0
```

### Issue: Episode ingestion not working
**Cause**: Graphiti URL not configured  
**Solution**: Set environment variable
```bash
$env:GRAPHITI_URL = "http://localhost:8000"
```

## Architecture Notes

### Budget Flow
1. CLI parses `--budget` parameter
2. `EvolveOptions.Budget` set
3. `BudgetGovernor` instantiated with limit
4. Passed to Evolution context
5. Tracked across LLM calls

### Episode Flow
1. Check `GRAPHITI_URL` env var
2. Create `EpisodeService` if available
3. Attach to Evolution context
4. Episodes logged during task execution
5. Batch sent to Graphiti for graph ingestion

## Performance Metrics

### Budget Consumption (1 iteration, qwen2.5-coder:7b)
- **Tokens**: ~2,000-5,000 tokens per task
- **Calls**: ~3-5 LLM calls per iteration
- **Cost**: <$0.01 USD (with Ollama, local)

### Memory Usage
- **Semantic Memory**: Stores past experiences
- **Vector Store**: ChromaDB or in-memory
- **Knowledge Graph**: Optional Graphiti integration

## Future Enhancements

### Planned (from phase6_integration_strategy.md)
- [ ] Speech Act validation in Evolution loop
- [ ] Budget-aware task prioritization
- [ ] Epistemic verification checkpoints
- [ ] Multi-agent collaboration patterns
- [ ] Enhanced episode attribution

### Possible Improvements
- [ ] Add `--demo` mode with mock LLM
- [ ] Budget visualization in CLI
- [ ] Episode replay/debugging tools
- [ ] Cost estimation before execution
- [ ] Budget alerts at thresholds

## Related Documentation
- `docs/3_Roadmap/1_Plans/phase6_integration_strategy.md` - Integration strategy
- `src/Tars.Core/Budget.fs` - Budget types and governor
- `src/Tars.Connectors/EpisodeIngestion.fs` - Episode service
- `src/Tars.Evolution/Engine.fs` - Evolution engine core

## Contributors
- Integration: Antigravity AI (2025-12-21)
- Architecture: TARS Core Team
