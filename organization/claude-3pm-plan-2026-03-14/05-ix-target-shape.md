# Target Shape: ix

Produced: 2026-03-14, Phase 3 of execution plan.

## Target Directory Structure

```text
ix/
├── README.md
├── CLAUDE.md
├── MIGRATION.md
├── Cargo.toml                    # workspace root
├── rust-toolchain.toml
├── .github/workflows/ci.yml
├── .mcp.json                     # ix-mcp server registration
├── .claude/
│   ├── settings.local.json
│   ├── skills/                   # 20+ Claude Code skills
│   │   ├── ix-optimize/
│   │   ├── ix-search/
│   │   ├── ix-rotation/
│   │   └── ...
│   └── agents/                   # future subagents
├── crates/
│   ├── ix-math/                  # core math: linalg, stats, distance, activation
│   ├── ix-optimize/              # SGD, Adam, PSO, simulated annealing
│   ├── ix-supervised/            # regression, classification
│   ├── ix-unsupervised/          # clustering, dimensionality reduction
│   ├── ix-ensemble/              # random forest, boosting
│   ├── ix-nn/                    # layers, loss, attention, transformer blocks
│   ├── ix-rl/                    # bandits, Q-learning
│   ├── ix-evolution/             # GA, differential evolution
│   ├── ix-graph/                 # Markov, HMM, state spaces, routing
│   ├── ix-probabilistic/         # Bloom, CMS, HyperLogLog, Cuckoo
│   ├── ix-io/                    # CSV, JSON, TCP, WebSocket
│   ├── ix-signal/                # FFT, wavelets, Kalman
│   ├── ix-chaos/                 # Lyapunov, bifurcation, attractors
│   ├── ix-game/                  # Nash, Shapley, auctions
│   ├── ix-search/                # A*, MCTS, minimax, BFS/DFS
│   ├── ix-gpu/                   # WGPU compute kernels
│   ├── ix-cache/                 # embedded Redis-like store
│   ├── ix-pipeline/              # DAG executor
│   ├── ix-adversarial/           # FGSM, PGD, defense
│   ├── ix-grammar/               # EBNF, weighted rules, replicator dynamics
│   ├── ix-rotation/              # quaternions, SLERP, Euler, rotation matrices
│   ├── ix-sedenion/              # sedenions, octonions, Cayley-Dickson
│   ├── ix-fractal/               # Takagi, IFS, L-systems, space-filling
│   ├── ix-number-theory/         # sieve, primality, modular arithmetic
│   ├── ix-topo/                  # simplicial complexes, persistent homology
│   ├── ix-category/              # monads, adjunctions
│   ├── ix-dynamics/              # differential equations (stub)
│   ├── ix-ktheory/               # K-theory for graphs (stub)
│   ├── ix-agent/                 # MCP server, tool registry, handlers
│   ├── ix-skill/                 # CLI exposing all algorithms
│   └── ix-demo/                  # egui interactive demo app
├── examples/                     # runnable examples per domain
│   ├── ga/                       # guitar alchemist integration examples
│   └── tars/                     # TARS integration examples
├── schemas/                      # JSON schemas for CLI/MCP contracts
│   ├── tool-result.schema.json
│   └── skill-manifest.schema.json
└── docs/
    ├── architecture.md
    ├── brainstorms/
    └── plans/
```

## Crate Mapping: machin-* → ix-*

### Keep (direct 1:1 rename)

All 31 current crates map directly. No merges needed for the first milestone.

| Current | Target | Status |
|---------|--------|--------|
| machin-math | ix-math | rename |
| machin-optimize | ix-optimize | rename |
| machin-supervised | ix-supervised | rename |
| machin-unsupervised | ix-unsupervised | rename |
| machin-ensemble | ix-ensemble | rename |
| machin-nn | ix-nn | rename |
| machin-rl | ix-rl | rename |
| machin-evolution | ix-evolution | rename |
| machin-graph | ix-graph | rename |
| machin-probabilistic | ix-probabilistic | rename |
| machin-io | ix-io | rename |
| machin-signal | ix-signal | rename |
| machin-chaos | ix-chaos | rename |
| machin-game | ix-game | rename |
| machin-search | ix-search | rename |
| machin-gpu | ix-gpu | rename |
| machin-cache | ix-cache | rename |
| machin-pipeline | ix-pipeline | rename |
| machin-adversarial | ix-adversarial | rename |
| machin-grammar | ix-grammar | rename |
| machin-rotation | ix-rotation | rename |
| machin-sedenion | ix-sedenion | rename |
| machin-fractal | ix-fractal | rename |
| machin-number-theory | ix-number-theory | rename |
| machin-topo | ix-topo | rename |
| machin-category | ix-category | rename |
| machin-dynamics | ix-dynamics | rename |
| machin-ktheory | ix-ktheory | rename |
| machin-agent | ix-agent | rename |
| machin-skill | ix-skill | rename |
| machin-demo | ix-demo | rename |

### Future consolidation candidates (NOT first milestone)

| Candidate | Into | Rationale |
|-----------|------|-----------|
| ix-dynamics + ix-chaos | ix-dynamical-systems | Both deal with ODE/attractor dynamics |
| ix-ktheory + ix-topo + ix-category | ix-algebra | All advanced abstract algebra |
| ix-supervised + ix-ensemble | ix-ml | Traditional ML could merge |

These are **deferred** — do not attempt in the rename pass.

### New crates to add later

| Crate | Purpose | When |
|-------|---------|------|
| ix-core | Shared types, errors, config, tracing | After rename stabilizes |
| ix-memory | Retrieval scoring, compression, vector-store adapters | When TARS integration needs it |
| ix-eval | Evaluation/scoring framework | When GA/TARS need standardized scoring |

## First Milestone Target

The first milestone is the **mechanical rename** — same crate count, same functionality, new names:

- 31 crates renamed `machin-*` → `ix-*`
- 2 binaries renamed: `ix` (CLI), `ix-mcp` (MCP server)
- 20 skills renamed
- All tests pass, clippy clean
- Tagged `v0.2.0-ix`

No new crates, no merges, no refactoring beyond the rename.
