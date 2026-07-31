---
date: 2026-07-30
category: integration-issues
component: Tars.Llm / Microsoft.Extensions.AI adapters
severity: high
symptom: constrained decoding silently unenforced on the MAF agent path
---

# `ChatResponseFormat.Json` is a singleton; `ForJsonSchema` is not

## The surprise

In Microsoft.Extensions.AI (verified on 10.3.0), these are both
`ChatResponseFormatJson` — same type, statically and at runtime:

```fsharp
ChatResponseFormat.Json                              // singleton, Schema = null
ChatResponseFormat.ForJsonSchema(element, "s", "d")  // NEW instance, Schema = set
```

So all three of these are true:

- `obj.ReferenceEquals(ChatResponseFormat.Json, ChatResponseFormat.Json)` — **true**,
  it really is a stable singleton, which is what makes a reference check look safe
- `obj.ReferenceEquals(ForJsonSchema(...), ChatResponseFormat.Json)` — **false**
- `ForJsonSchema(...).Equals(ChatResponseFormat.Json)` — **false**

A type test does not separate them either. The only thing that distinguishes
"JSON mode" from "JSON mode with a schema" is `.Schema.HasValue`.

## Why it bit

`ChatClientAdapter.fs` decided JSON mode with a reference comparison against the
singleton. That is false exactly when a schema is present — i.e. the check
inverted its own purpose: the *more* constrained the request, the *less*
constrained it went out. Requests from MAF agents were serialised with
`JsonMode = false` and no `ResponseFormat` at all.

Nothing errors when this happens. The model returns plausible free-form text, the
JSON parser mostly copes, and the failure looks like ordinary model flakiness.

## Second trap: AdditionalProperties is not a wire channel

The same file passed JSON schemas via
`ChatOptions.AdditionalProperties["structured_outputs_json"]`. No stock M.E.AI
provider forwards unknown additional properties, so the schema was set and never
applied.

`ForJsonSchema` is the channel providers actually enforce. EBNF and regex have no
M.E.AI equivalent and legitimately need `AdditionalProperties` — but only a
backend that explicitly reads those keys will honour them.

## Grep anchors

- `ChatClientMapping.fromChatOptions` — the reverse mapping that was missing
- `ChatClientMapping.impliesJsonMode` — keeps the legacy `JsonMode` flag in step
  with `ResponseFormat`, since backends read one or the other
- `tests/Tars.Tests/ChatClientFormatTests.fs` — round-trip coverage

## Transferable lesson

Bidirectional adapters need a round-trip test, not two one-way tests. Both halves
of this one were individually plausible and jointly lossy, and the pair had zero
coverage precisely because each direction looked obviously correct on its own.
