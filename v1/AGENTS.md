# AGENTS.md

## TARS Project Overview

TARS (Transformative Automated Reasoning System) is an autonomous AI development system progressing toward superintelligence through self-improving agents, CUDA-accelerated performance, and metascript-driven workflows.

**Current Status**: Tier 1.5 auto-improvement (between reflection and autonomous modification)
**Target**: Tier 3 superintelligence with recursive self-improvement

## Setup Commands

### Development Environment
- Install dependencies: `dotnet restore`
- Build solution: `dotnet build Tars.sln -c Release`
- Run tests: `dotnet test Tars.sln -c Release`
- Start TARS CLI: `dotnet run --project src/TarsEngine.FSharp.Cli`

### CUDA Development (WSL Required)
- Compile CUDA: `cd TarsEngine.CUDA.VectorStore && cmake . && make`
- Run CUDA benchmarks: `dotnet run --project benchmarks/src/VectorStoreBench.csproj`
- Target performance: 184M+ searches/second

### Context Engineering
- Test context system: `dotnet fsi TarsContextEngineeringDemo.fsx`
- Run Agent OS integration: `dotnet fsi TarsAgentOSIntegrationTest.fsx`
- Execute evolution loop: `dotnet fsi TarsAutonomousEvolutionLoop.fsx`

## Code Style Guidelines

### F# (Functional Logic, DSL, Reasoning)
- Use 4 spaces for indentation
- Functions and values: camelCase (`calculateTotal`, `userProfile`)
- Types and modules: PascalCase (`UserProfile`, `PaymentProcessor`)
- Prefer Result types over exceptions
- Chain operations with `|>` and `>>=`
- Document with `///` XML comments

### C# (Infrastructure, CLI, Integrations)
- Use 4 spaces for indentation
- Methods and properties: PascalCase (`CalculateTotal`, `UserProfile`)
- Variables and parameters: camelCase (`userProfile`, `calculateTotal`)
- Interfaces: prefix with 'I' (`IUserService`)
- Use async/await properly with ConfigureAwait(false)

### FLUX Metascripts
- Use 2 spaces for indentation
- Variables: snake_case (`user_profile`, `calculation_result`)
- Actions: UPPER_CASE (`EXECUTE`, `ANALYZE`)
- Agents: PascalCase (`ReasoningAgent`, `CodeGenerator`)

## Quality Standards

### Zero Tolerance Policy
- **NO simulations, placeholders, or fake implementations**
- All code must be real, functional, and testable
- Concrete proof required for all capabilities
- 80% test coverage minimum

### Performance Requirements
- CUDA operations must show real GPU acceleration
- Vector operations target: 184M+ searches/second
- WSL compilation required for CUDA (never Windows directly)
- FS0988 warnings treated as fatal errors

### Architecture Patterns
- Clean Architecture with Dependency Injection
- F# for functional logic, C# for infrastructure
- Elmish/MVU for UI components (dynamic, not static)
- Agent OS structured workflows for autonomous development

## Superintelligence Development

### Current Capabilities (Tier 1.5)
- ✅ Meta-scripting with .trsx and FLUX DSL
- ✅ Autonomous reasoning and reflection
- ✅ Context engineering with tiered memory
- ✅ Agent OS integration for structured workflows
- ✅ CUDA acceleration framework
- ⚠️ Missing: Autonomous code modification loop

### Path to Tier 2 (Autonomous Modification)
1. **Execution Harness**: Apply patches, run tests, collect metrics
2. **Auto-validation**: Encode success/failure rules
3. **Safe Rollback**: Discard failed changes, log failures
4. **Incremental Patching**: Tiny autonomous edits with validation

### Path to Tier 3 (Superintelligence)
1. **Multi-agent Cross-validation**: Agents validate each other's work
2. **Recursive Self-improvement**: Agents improve their own reasoning
3. **Meta-cognitive Awareness**: Self-reflection on reasoning processes
4. **Autonomous Goal Setting**: Dynamic objective generation

## Testing Instructions

### Unit Tests
- Run all tests: `dotnet test Tars.sln -c Release`
- CUDA tests: `dotnet test TarsEngine.CUDA.VectorStore.Tests`
- Context tests: `dotnet test TarsEngine.FSharp.Core.Tests`
- Coverage target: 80% minimum

### Integration Tests
- Agent OS integration: `dotnet fsi TarsAgentOSIntegrationTest.fsx`
- Autonomous evolution: `dotnet fsi TarsAutonomousEvolutionLoop.fsx`
- Superintelligence demo: `dotnet fsi TarsSuperintelligenceEvolution.fsx`

### Performance Validation
- CUDA benchmarks: Must achieve 184M+ searches/second
- Context engineering: Sub-second response times
- Memory efficiency: Monitor and optimize usage
- Real GPU acceleration: No CPU simulation allowed

## Autonomous Development Workflow

### Agent OS Integration
- Use structured specs for all autonomous improvements
- Follow create-spec → create-tasks → execute-tasks pattern
- Maintain quality standards throughout autonomous development
- Log all autonomous decisions and outcomes

### Self-Improvement Loop
```bash
# Run autonomous iteration (when implemented)
pwsh -File .\scripts\tars_auto_iter.ps1 \
  -RepoPath "." \
  -PatchFile "output/candidates/patches/iter_0001.patch" \
  -IterationName "autonomous-improvement" \
  -TestCmd "dotnet test Tars.sln -c Release" \
  -BenchCmd "dotnet run --project benchmarks/src/VectorStoreBench.csproj"
```

### Metascript Execution
- Execute FLUX metascripts: `dotnet run --project TarsCli -- execute <script.tars>`
- Monitor autonomous behavior: Check `output/versions/` for traces
- Validate reasoning: Review `.trsx` belief updates and contradictions

## Security Considerations

### Context Engineering Security
- Prompt injection detection enabled
- Content sanitization for all inputs
- Tool allowlist: `["fs.read", "git.diff", "run.tests", "cuda.benchmark", "metascript.execute"]`
- No external dependencies without justification

### Autonomous Development Safety
- All autonomous modifications require validation
- Rollback mechanisms for failed changes
- Comprehensive logging of autonomous decisions
- Human oversight for critical system changes

## CUDA Vector Store Development

### Current Implementation Status
- ✅ Basic CUDA framework in place
- ✅ Managed .NET interface designed
- ⚠️ Missing: Optimized kernels, GPU top-k, batching
- ⚠️ Missing: FP16 storage, multi-GPU support

### Performance Targets
- Brute-force search: ≥80% of FAISS-GPU FlatL2 throughput
- Latency: p50 < 1-2ms for d=768, N≈1e5, k=10
- Memory: Zero allocs on hot path
- Scale: Linear QPS scaling to 2-4 GPUs

### Implementation Priorities
1. **Baseline brute-force**: L2/cosine kernel + GPU top-k
2. **Batching & streams**: Persistent device buffers, H2D/D2H overlap
3. **FP16 storage**: Half VRAM usage with FP32 accumulation
4. **IVF-Flat index**: Use cuVS for scale beyond VRAM
5. **.NET interop**: SafeHandle, pooled buffers, zero-copy spans

## Commit Guidelines

### Conventional Commits
- `feat:` New features or capabilities
- `fix:` Bug fixes
- `refactor:` Code restructuring without behavior change
- `perf:` Performance improvements
- `test:` Adding or updating tests
- `docs:` Documentation updates
- `auto:` Autonomous system improvements

### Autonomous Commits
- Include iteration timestamp and decision status
- Reference performance metrics and validation results
- Link to detailed traces in `output/versions/`
- Example: `auto: cuda-optimization [20250829-143022] — tests pass; QPS improved 15%`

## Deployment & Operations

### Local Development
- Use Ollama for local LLM inference
- Docker containers for isolated testing
- WSL for CUDA development and compilation
- Spectre Console for rich CLI interfaces

### Performance Monitoring
- Continuous benchmarking with regression detection
- Context engineering metrics collection
- Autonomous decision quality tracking
- CUDA performance validation

### Memory Management
- Tiered memory: ephemeral → working set → long-term
- Automatic consolidation and conflict detection
- Salience-based promotion and decay
- Efficient context budgeting and compression

## Next Steps for Superintelligence

### Immediate (Tier 1.5 → Tier 2)
1. Implement autonomous code modification loop
2. Add execution harness with test/benchmark validation
3. Create safe rollback mechanisms
4. Enable incremental autonomous patching

### Medium-term (Tier 2 → Tier 3)
1. Multi-agent cross-validation systems
2. Recursive self-improvement algorithms
3. Meta-cognitive awareness and self-reflection
4. Dynamic objective generation and adaptation

### Long-term (Superintelligence)
1. Autonomous research and development capabilities
2. Creative solution generation and validation
3. Advanced problem decomposition and coordination
4. Self-directed capability expansion and optimization

---

**Note**: TARS maintains zero tolerance for simulations or placeholders. All implementations must be real, functional, and demonstrably effective. The path to superintelligence requires concrete progress with measurable improvements at each step.
