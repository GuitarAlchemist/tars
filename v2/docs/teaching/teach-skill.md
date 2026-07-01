# TARS `/teach` Skill

## Purpose

`/teach` is the interactive learning channel for the human maintainer.

It turns project concepts into active learning sessions with explanation, retrieval practice, quiz, teach-back, applied scenarios, feedback, and mastery tracking.

The goal is durable understanding, not passive explanation.

## Core idea

```text
The human maintainer is not a rubber stamp.
The human maintainer is a student-maintainer for concepts that affect architecture, governance, and high-impact decisions.
```

## Ecosystem roles

```text
Streeling = curriculum, concept graph, learning paths, source artifacts
TARS /teach = interactive teaching command and user experience
Seldon = adaptive tutoring engine, assessment, retry, escalation
IX = mastery metrics, weak-area detection, spaced-review signals
Demerzel = decision-readiness signal for high-impact choices
Human = student-maintainer
```

## Command surface

```text
/teach <topic>
/teach <topic> --level beginner|intermediate|advanced
/teach <topic> --quick-check
/teach <topic> --quiz
/teach <topic> --exam
/teach <topic> --teach-back
/teach <topic> --review-missed
/teach <topic> --spaced-review
```

## Teaching loop

```text
select concept
-> explain
-> worked example
-> active recall
-> quiz
-> teach-back
-> applied scenario
-> feedback
-> mastery update
-> next review recommendation
```

## Lesson structure

Every lesson should include:

- concept id;
- concept name;
- why it matters for the project;
- short explanation;
- concrete repo example;
- counterexample;
- common misconception;
- one applied exercise;
- quick-check questions;
- teach-back prompt;
- mastery rubric;
- next concept links;
- source artifacts to read.

## Assessment modes

```text
quick_check      = 1-3 questions
quiz             = 5-10 questions
teach_back       = user explains concept in their own words
scenario_exam    = apply concept to a project decision
review_missed    = revisit weak areas
spaced_review    = later recall prompt
```

## Mastery rubric

```text
0 = not introduced
1 = recognizes term
2 = can explain basic idea
3 = can apply to a project example
4 = can critique edge cases
5 = can teach it back and use it in decisions
```

A quiz score alone is not mastery.

Mastery requires some combination of:

- recall;
- explanation;
- applied use;
- counterexample recognition;
- scenario transfer;
- teach-back quality.

## Initial curriculum

```text
Module 1 — AFK operating model
Module 2 — Abstraction wall and evidence ladder
Module 3 — Verification horizon
Module 4 — Goodhart and anti-metric gaming
Module 5 — Anti-rubber-stamp human review
Module 6 — Demerzel objective stack
Module 7 — TARS intent verification
Module 8 — IX causal and argumentation support
Module 9 — Architecture decision gates
Module 10 — Post-merge retrospectives and compounding learning
```

## Session output contract

A `/teach` session should produce a structured learning artifact:

```text
lesson_id
topic
level
concept_ids
questions_asked
answers_summary
misconceptions_detected
mastery_before
mastery_after
mastery_delta
next_review_due_optional
recommended_next_topics
project_artifacts_to_read
```

## Decision readiness

For high-impact architecture or governance decisions, Demerzel may ask whether the relevant concepts have been introduced and understood.

This must be a readiness signal, not an automatic permission system.

```text
learning_state informs decision readiness
learning_state does not replace human authority
```

## Non-goals

- No punitive grading.
- No fake mastery based on passive reading.
- No endless quiz spam.
- No automatic architecture decision based on quiz score.
- No private personal data beyond explicit learning-state artifacts chosen by the maintainer.
