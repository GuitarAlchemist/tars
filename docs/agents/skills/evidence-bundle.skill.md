# Skill: Evidence Bundle

## When to use
Use this skill when completing tasks that require source-grounded claims, such as creating research digests, technology-watch proposals, or any architectural design decisions. It must be used when `issue_meta.evidence_required` includes items that need concrete source backing, or when working on the second-brain memory objects.

## Allowed files/surfaces
- `docs/contracts/`
- `docs/design/`
- `examples/technology-watch/`
- `examples/second-brain/`
- Any explicitly targeted knowledge management files mentioned in the issue.

## Required checks
1. **Provenance Check:** Ensure every major claim or digest is explicitly linked to a source (e.g., via `source_uri`, `source_hash`, or `provenance` metadata).
2. **Contract Compliance:** Ensure the created evidence bundles match the strict schema requirements (e.g., JSON-first, includes structural metadata like confidence and privacy level) as defined in `docs/contracts/`.
3. **Fact Separation:** Verify that LLM-generated claims are explicitly separated from factual source evidence, and that staleness markers and contradiction candidates are properly defined if applicable.

## Stop conditions
- Stop when all required evidence files (JSON/MD) have been created and correctly reference their sources.
- Stop if the source material required for grounding cannot be accessed or verified.

## Evidence expected in the PR body
- Links to the newly created or updated evidence bundle JSON/MD files.
- A summary of the provenance sources used for the claims.
- A confirmation statement that no claims lack a source linkage.

## Common failure modes
- Generating hallucinatory claims without linking them to specific, verifiable source evidence (`source_uri` or `source_hash`).
- Failing to include mandatory metadata fields (like `digester_id`, `digestion_timestamp`) required by the TARS V2 second-brain memory object contracts.
- Embedding plain text reasoning inside JSON files instead of adhering to the structured data contract.
