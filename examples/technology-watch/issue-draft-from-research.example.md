# [IX] Integrate Policy-Based Pruning for MCTS

**Repo:** `GuitarAlchemist/ix`
**Parent Epic:** GuitarAlchemist/ix#200
**Related Issues:** GuitarAlchemist/ix#209

## Problem Statement
The current MCTS implementation explores unpromising branches extensively before pruning them, resulting in high latency and wasted CPU cycles, particularly in sparse reward environments.

## Goal
Implement policy-guided branch pruning to skip low-value branches early, reducing search latency while maintaining decision quality.

## Non-Goals
- Do not rewrite the MCTS tree structure.
- Do not train the policy network in this issue (assume a pre-trained lightweight evaluator is provided).

## Acceptance Criteria
- [ ] MCTS node expansion logic checks the policy network threshold.
- [ ] Nodes falling below `prune_threshold` are immediately terminated.
- [ ] Performance benchmarks show a latency reduction without a significant drop in win-rate.

## Test Plan
- Run `cargo bench` on the MCTS module before and after changes.
- Add unit tests verifying that branches under threshold are not expanded.

## Governance & Policy
**Risk Policy:** medium
**Budget Policy:**
- **Tier:** free-local
- **Max Cost USD:** 0
- **Max Runner Minutes:** 30
**AFK Readiness:** candidate

## Stop Conditions
- If latency reduction is less than 10% on the baseline benchmarks, halt and request human review.

## Evidence Bundle
**Digester ID:** `tars-researcher-v1`
**Digestion Timestamp:** `2024-06-25T14:30:00Z`
**Source Hash:** `sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
