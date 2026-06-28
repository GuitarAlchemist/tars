# Evidence Bundle Contracts

This document defines the data contracts (JSON schemas and F# type equivalents) for the source-grounded synthesis pipeline in the TARS second-brain harness.

These contracts enforce strict separation between LLM-generated summaries and the factual evidence backing them. By mandating explicit source references and surfacing staleness/contradiction markers, these contracts prevent plausible but unsupported claims from entering durable memory.

## 1. SourceSet
A bounded collection of artifacts provided as context to the LLM.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "SourceSet",
  "type": "object",
  "properties": {
    "setId": { "type": "string", "format": "uuid" },
    "description": { "type": "string" },
    "artifacts": {
      "type": "array",
      "items": { "$ref": "#/definitions/SourceReference" }
    }
  },
  "required": ["setId", "artifacts"]
}
```

```fsharp
type SourceSet = {
    SetId: Guid
    Description: string option
    Artifacts: SourceReference list
}
```

## 2. SourceReference
A pointer to a specific artifact serving as evidence.

```json
{
  "title": "SourceReference",
  "type": "object",
  "properties": {
    "artifactId": { "type": "string" },
    "uri": { "type": "string", "format": "uri" },
    "versionHash": { "type": "string" },
    "timestamp": { "type": "string", "format": "date-time" }
  },
  "required": ["artifactId", "uri", "timestamp"]
}
```

```fsharp
type SourceReference = {
    ArtifactId: string
    Uri: string
    VersionHash: string option
    Timestamp: DateTime
}
```

## 3. GroundedClaim
A specific finding synthesized by the LLM, strictly linked to its supporting sources. The claim is explicitly separated from the evidence.

```json
{
  "title": "GroundedClaim",
  "type": "object",
  "properties": {
    "claimId": { "type": "string", "format": "uuid" },
    "statement": { "type": "string" },
    "supportingEvidence": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "artifactId": { "type": "string" },
          "exactQuote": { "type": "string" }
        },
        "required": ["artifactId", "exactQuote"]
      }
    }
  },
  "required": ["claimId", "statement", "supportingEvidence"]
}
```

```fsharp
type EvidenceLink = {
    ArtifactId: string
    ExactQuote: string
}

type GroundedClaim = {
    ClaimId: Guid
    Statement: string
    SupportingEvidence: EvidenceLink list
}
```

## 4. StalenessWarning
Generated when a source referenced in a claim might be outdated.

```json
{
  "title": "StalenessWarning",
  "type": "object",
  "properties": {
    "artifactId": { "type": "string" },
    "reason": { "type": "string" },
    "newerVersionUri": { "type": "string", "format": "uri" }
  },
  "required": ["artifactId", "reason"]
}
```

```fsharp
type StalenessWarning = {
    ArtifactId: string
    Reason: string
    NewerVersionUri: string option
}
```

## 5. ContradictionCandidate
Raised when a new claim directly opposes an existing durable memory item.

```json
{
  "title": "ContradictionCandidate",
  "type": "object",
  "properties": {
    "newClaimId": { "type": "string", "format": "uuid" },
    "existingMemoryId": { "type": "string", "format": "uuid" },
    "description": { "type": "string" }
  },
  "required": ["newClaimId", "existingMemoryId", "description"]
}
```

```fsharp
type ContradictionCandidate = {
    NewClaimId: Guid
    ExistingMemoryId: Guid
    Description: string
}
```

## 6. EvidenceBundle
The aggregate package of synthesized claims and their metadata, ready for evaluation.

```json
{
  "title": "EvidenceBundle",
  "type": "object",
  "properties": {
    "bundleId": { "type": "string", "format": "uuid" },
    "sourceSetId": { "type": "string", "format": "uuid" },
    "claims": {
      "type": "array",
      "items": { "$ref": "#/definitions/GroundedClaim" }
    },
    "stalenessWarnings": {
      "type": "array",
      "items": { "$ref": "#/definitions/StalenessWarning" }
    },
    "contradictions": {
      "type": "array",
      "items": { "$ref": "#/definitions/ContradictionCandidate" }
    }
  },
  "required": ["bundleId", "sourceSetId", "claims"]
}
```

```fsharp
type EvidenceBundle = {
    BundleId: Guid
    SourceSetId: Guid
    Claims: GroundedClaim list
    StalenessWarnings: StalenessWarning list
    Contradictions: ContradictionCandidate list
}
```

## 7. MemoryPromotionDecision
The output of the IX/Seldon/Demerzel assessment pipeline, dictating whether a bundle is promoted to memory, rejected, or escalated.

```json
{
  "title": "MemoryPromotionDecision",
  "type": "object",
  "properties": {
    "bundleId": { "type": "string", "format": "uuid" },
    "status": {
      "type": "string",
      "enum": ["Accepted", "Rejected", "Escalated"]
    },
    "ixScore": { "type": "number" },
    "reasoning": { "type": "string" },
    "requiresHumanReview": { "type": "boolean" }
  },
  "required": ["bundleId", "status", "ixScore", "reasoning"]
}
```

```fsharp
type PromotionStatus =
    | Accepted
    | Rejected
    | Escalated

type MemoryPromotionDecision = {
    BundleId: Guid
    Status: PromotionStatus
    IxScore: float
    Reasoning: string
    RequiresHumanReview: bool
}
```
