# `/teach`, Seldon, Streeling, IX, and Demerzel

## Purpose

This document defines how the `/teach` skill connects to the broader GuitarAlchemist learning architecture.

`/teach` is not a standalone teaching toy. It is the user-facing entry point for a learning loop across Streeling, Seldon, IX, TARS, and Demerzel.

## Architecture split

```text
Streeling
  owns the curriculum, concept graph, prerequisites, source artifacts, and learning paths.

TARS /teach
  owns the interactive command, session flow, and user-facing learning experience.

Seldon
  owns adaptive teaching behavior, assessment, retry, escalation, and learning-state updates.

IX
  owns mastery metrics, weak-area detection, spaced-review signals, and learning analytics.

Demerzel
  consumes decision-readiness signals for important architecture/governance decisions.
```

## Flow

```text
Streeling concept path
-> TARS /teach session
-> Seldon assessment and adaptation
-> IX mastery metrics
-> TARS learning-state artifact
-> Demerzel readiness signal when relevant
```

## Streeling responsibilities

Streeling should provide:

```text
concept_id
concept_name
prerequisites
source_artifacts
related_project_decisions
common_misconceptions
quiz_seed_material
next_concepts
```

Examples of concepts:

- abstraction wall;
- evidence ladder;
- verification horizon;
- anti-rubber-stamp review;
- Goodhart risk;
- causal scorecards;
- computational argumentation;
- human decision gates.

## Seldon responsibilities

Seldon should manage:

- lesson adaptation;
- quick-check generation;
- teach-back scoring;
- scenario exam prompts;
- retry plans;
- misconception tracking;
- escalation when a concept remains weak;
- learning-state update candidates.

Seldon should not auto-promote a concept as mastered from confidence alone.

## IX responsibilities

IX should produce interpretable learning metrics:

```text
retrieval_success_rate
teach_back_quality
scenario_transfer_score
misconception_recurrence
spaced_review_due_count
concept_mastery_delta
decision_readiness_signal
```

Metrics are evidence, not authority.

A high quiz score alone is not mastery.

## TARS responsibilities

TARS should:

- run the `/teach` session;
- maintain the session artifact;
- connect lessons to second-brain concepts;
- link teachings to relevant repo artifacts;
- surface weak concepts before important decisions;
- let the maintainer choose learning goals.

## Demerzel responsibilities

Demerzel may ask whether the maintainer has enough concept readiness for a high-impact decision.

Examples:

```text
Architecture boundary change -> ask if relevant concept has been taught
Metric-driven decision -> ask if Goodhart concept is understood
Review routing change -> ask if anti-rubber-stamp concept is understood
```

Demerzel must not block all decisions on learning metrics.

The signal is advisory and should be shown with evidence.

## Student-maintainer model

The maintainer is treated as a learner for complex governance and architecture concepts.

This does not reduce authority. It improves decision quality.

```text
The system teaches before asking for high-impact judgment.
```

## Non-goals

- Do not turn `/teach` into passive docs generation.
- Do not require a lesson before every decision.
- Do not shame or punish weak scores.
- Do not treat quiz score as decision authority.
- Do not store unnecessary personal data.
