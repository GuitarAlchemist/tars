# Skill: Evidence Bundle

## When to use it
Use this skill when you need to gather, compile, or synthesize evidence to support a claim, finding, or task execution. This is essential for maintaining a source-grounded second brain.

## Allowed files/surfaces
- `docs/knowledge/**/*.md`
- `examples/**/*.json` (specifically those defining evidence bundles or claims)
- `docs/research/**/*.md`

## Required checks
- Ensure all claims made are backed by explicit source references.
- Include necessary metadata, such as provenance, confidence levels, and staleness markers.
- Verify that the compiled evidence strictly follows the defined contract (e.g., `docs/contracts/evidence-bundle.contract.md`).

## Stop conditions
- Halt if you cannot find authoritative sources for a claim.
- Halt if you are instructed to hallucinate or fabricate evidence.
- Halt if the `governance/state/afk-halt.json` marker is present.

## Evidence expected in the PR body
- A summary of the claims synthesized and the sources used.
- A checklist indicating that provenance and confidence metadata have been properly populated.
- References to any related knowledge base artifacts.

## Common failure modes
- Making assertions without verifiable citations or provenance.
- Failing to properly format the evidence bundle according to the JSON contract.
- Ignoring staleness constraints, leading to outdated claims.
