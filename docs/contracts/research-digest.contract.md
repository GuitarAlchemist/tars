# Technology-Watch Digest Contracts

This document defines the contracts for research paper and thesis digests consumed by the TARS second-brain technology-watch loop.

## Overview

The purpose of these digests is to capture structured metadata from external research sources (papers, theses) in a format actionable by the TARS/IX/Demerzel/GA ecosystem.

These digests do *not* ingest real papers or perform summarization. They provide enough structure for the autonomous harness to:
1. Decide whether the work is relevant to the ecosystem.
2. Identify which claims are actionable.
3. Validate available evidence.
4. Determine whether a follow-up GitHub issue should be created for deeper exploration.

## Sub-Contracts

### EvidenceBundle
Represents a collection of supporting artifacts or evidence for a claim.

- `id` (string): Unique identifier.
- `description` (string): Description of the evidence.
- `uri` (string): Location of the evidence.

### ResearchClaim
Represents a specific, verifiable statement made within the research.

- `id` (string): Unique identifier.
- `claim` (string): The text of the claim.
- `evidence` (array of `EvidenceBundle`): The evidence backing the claim.

### MethodCandidate
Describes a technique or methodology proposed in the research.

- `name` (string): Name of the method.
- `description` (string): Description of how it works.

### DatasetCandidate
Describes a dataset introduced or used in the research.

- `name` (string): Name of the dataset.
- `uri` (string): Location of the dataset.

### BenchmarkCandidate
Describes a benchmark introduced or evaluated in the research.

- `name` (string): Name of the benchmark.
- `metric` (string): The metric evaluated.

### IssueDraftCandidate
Represents a potential follow-up action to be translated into a GitHub Issue.

- `title` (string): Proposed issue title.
- `description` (string): Proposed issue description.
- `labels` (array of strings): Recommended labels.

## Core Digests

### ResearchDigest
Captures the summary, claims, and relevance of a standard research paper.

- `id` (string): Unique UUID or hash.
- `title` (string): Title of the paper.
- `authors` (array of strings): List of authors.
- `source_uri` (string): DOI, Arxiv link, or URL.
- `publication_date` (string, ISO-8601): Date of publication.
- `source_type` (string): E.g., "conference_paper", "journal_article", "preprint".
- `research_area` (string): Broad categorization (e.g., "neuro-symbolic AI", "MCTS").
- `summary` (string): Abstract or high-level summary.
- `key_claims` (array of `ResearchClaim`): Major verifiable claims.
- `methods` (array of `MethodCandidate`): Methods proposed or utilized.
- `datasets` (array of `DatasetCandidate`): Datasets evaluated or introduced.
- `benchmarks` (array of `BenchmarkCandidate`): Benchmarks targeted.
- `limitations` (array of strings): Known limitations or boundaries of the research.
- `applicability_to_tars` (string): Assessment of relevance to the TARS agent system.
- `applicability_to_ix` (string): Assessment of relevance to the IX Rust skill engine.
- `applicability_to_demerzel` (string): Assessment of relevance to Demerzel governance.
- `applicability_to_ga` (string): Assessment of relevance to the GuitarAlchemist domain.
- `confidence` (number): 0.0 to 1.0 score indicating confidence in the extraction/findings.
- `novelty_score` (number): 0.0 to 1.0 score indicating the uniqueness of the work.
- `implementation_risk` (string): "low", "medium", "high", or "unknown".
- `suggested_followups` (array of `IssueDraftCandidate`): Next actionable steps.
- `provenance` (string): Reference tracing back to the ingestion tool/run.

### ThesisDigest
Similar to `ResearchDigest` but tailored for long-form academic theses (Master's, Ph.D.).

- Includes all fields from `ResearchDigest`.
- Typically contains a deeper breakdown of `key_claims` and more extensive `suggested_followups` due to the length and scope of the source.

### HarnessImprovementProposal
A concrete proposal to modify the ecosystem based on one or more digests.

- `id` (string): Unique UUID or hash.
- `title` (string): Proposed improvement title.
- `source_digest_ids` (array of strings): IDs of the `ResearchDigest` or `ThesisDigest` informing this proposal.
- `proposal_summary` (string): Executive summary of the change.
- `target_systems` (array of strings): Systems affected (e.g., "TARS", "IX").
- `confidence` (number): 0.0 to 1.0 score on the viability of the proposal.
- `provenance` (string): Tool or run that generated the proposal.
- `suggested_followups` (array of `IssueDraftCandidate`): Actionable tasks to implement the proposal.