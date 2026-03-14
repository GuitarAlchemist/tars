# TARS Self-Improvement Ralph Prompt

You are iterating on TARS v2, an F# neuro-symbolic self-improving agent system.

## Working Directory
`C:/Users/spare/source/repos/tars/v2`

## Your Goal
Improve TARS by identifying and fixing real capability gaps. Each iteration should leave the codebase strictly better than you found it.

## Iteration Protocol

1. **Build**: `dotnet build` from `v2/`. Fix any errors before proceeding.
2. **Test**: `dotnet test` from `v2/`. All tests must pass. Fix failures.
3. **Analyze gaps**: Run `dotnet run --project src/Tars.Interface.Cli -- meta analyze` to see current capability gaps and failure clusters.
4. **Pick ONE gap**: Choose the highest-priority gap from the analysis. If no gaps, look at test coverage or patterns with 0% success rate.
5. **Fix it**: Make the minimal code change that addresses the gap. Prefer editing existing files over creating new ones.
6. **Verify**: Build and test again. All 664+ tests must pass.
7. **Record**: Add a test that proves the gap is fixed.

## Constraints
- Never break existing tests
- Never remove functionality
- Keep changes small and focused — one gap per iteration
- Follow existing F# style (modules, DU types, pipeline operators)
- Target `net10.0`
- Do not modify `Tars.Metascript` (frozen)

## Key Paths
- Solution: `Tars.sln`
- CLI: `src/Tars.Interface.Cli/`
- Core types: `src/Tars.Core/`
- Evolution engine: `src/Tars.Evolution/`
- Agent runtime: `src/Tars.Cortex/`
- LLM layer: `src/Tars.Llm/`
- Tests: `tests/Tars.Tests/`

## Success Criteria
When the meta-cognitive analysis shows no capability gaps with failure rate above 30%, and all tests pass, output:

<promise>TARS GAPS RESOLVED</promise>

## Current State
Check `~/.tars/pattern_outcomes.json` for execution history. Run `tars meta stats` to see current pattern success rates. Focus on patterns with low success rates (ReAct, TreeOfThoughts) and domains with high failure rates.
