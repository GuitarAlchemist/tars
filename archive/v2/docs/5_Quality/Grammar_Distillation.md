# Grammar Distillation (Notes)

Status: scaffolding landed; not yet wired into runtime prompts.

## What we have
- `src/Tars.Core/GrammarDistill.fs`
  - `fromJsonExamples`: derives a flat-field JSON spec (fields, required) from examples.
  - Produces `PromptHint` for constrained generation and `Validator` for post-check.
  - `metascriptHint`: placeholder for Metascript-shaped responses.
- Offline eval includes a tiny grammar sanity check (`grammar:json-distill`) to catch regressions.

## How to use (manual)
- Build a spec from examples:
  ```fsharp
  let spec =
      GrammarDistill.fromJsonExamples
          [ """{"goal":"g","constraints":[],"validation_criteria":"c"}"""
            """{"goal":"x","constraints":["a"],"validation_criteria":"y"}""" ]
  // spec.PromptHint -> use in LLM prompt
  // spec.Validator responseText -> bool
  ```
- Add `spec.PromptHint` to your LLM system/user prompt to request constrained JSON.
- After generation, run `spec.Validator responseText` to enforce shape before parsing.

## Future wiring ideas
- RAG / CLI: wrap tool-call or JSON responses with `PromptHint` and reject/repair if `Validator` fails.
- Metascript: add a Metascript-specific spec builder to constrain node outputs.
- CI: extend OfflineEval with conformance rate checks when running a small prompt set.
