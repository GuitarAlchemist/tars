namespace Tars.Evolution

/// JSON schemas for the structured LLM call sites in Tars.Evolution.
///
/// Each schema mirrors the "return ONLY this JSON" template in the prompt it
/// accompanies. They are colocated here rather than inline so schema drift from
/// the prose instruction is a one-file diff — drift between the prompt's example
/// and the enforced shape is the dominant structured-output bug class.
///
/// Request shaping lives in `Tars.Llm.ConstrainedDecoding.withJsonSchema`, not
/// here: Cortex and Kernel have parallel JsonMode sites that will want the same
/// helper and cannot reference Evolution. This module is schema strings only.
///
/// All schemas are authored strict-mode-compatible — `additionalProperties:false`
/// with every property listed in `required` — because `OpenAiCompatibleClient`
/// sends `json_schema` with `strict = true`, and OpenAI rejects schemas that omit
/// either. Genuinely optional fields are expressed as nullable types rather than
/// by dropping them from `required`, which is the only encoding strict mode allows.
module EvolutionSchemas =

    /// Engine.evaluateContradiction — "Respond in JSON: {"contradicts": .., "reason": ..}"
    let contradictionSchema =
        """{
  "type": "object",
  "properties": {
    "contradicts": { "type": "boolean" },
    "reason": { "type": "string" }
  },
  "required": ["contradicts", "reason"],
  "additionalProperties": false
}"""

    /// Engine.generateTask direct-LLM fallback — {"tasks":[{goal, constraints, validation_criteria}]}
    let taskGenerationSchema =
        """{
  "type": "object",
  "properties": {
    "tasks": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "goal": { "type": "string" },
          "constraints": { "type": "array", "items": { "type": "string" } },
          "validation_criteria": { "type": "string" }
        },
        "required": ["goal", "constraints", "validation_criteria"],
        "additionalProperties": false
      }
    }
  },
  "required": ["tasks"],
  "additionalProperties": false
}"""

    /// Evaluation.SemanticEvaluation.Evaluate — mirrors the jsonTemplate literal.
    let evaluationSchema =
        """{
  "type": "object",
  "properties": {
    "passed": { "type": "boolean" },
    "confidence": { "type": "number" },
    "summary": { "type": "string" },
    "issues": { "type": "array", "items": { "type": "string" } },
    "suggested_fixes": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["passed", "confidence", "summary", "issues", "suggested_fixes"],
  "additionalProperties": false
}"""

    /// Reflection.LlmReflectionAgent.ReflectAsync — type/score/comment/suggestion.
    let reflectionSchema =
        """{
  "type": "object",
  "properties": {
    "type": { "type": "string", "enum": ["Success", "Failure", "Optimization"] },
    "score": { "type": "number" },
    "comment": { "type": "string" },
    "suggestion": { "type": "string" }
  },
  "required": ["type", "score", "comment", "suggestion"],
  "additionalProperties": false
}"""

    /// Optimizer.LlmWorkflowOptimizer.OptimizeAsync — a whole Metascript `Workflow`
    /// (Tars.Metascript.Domain.Workflow), since this site returns a modified copy of
    /// the serialized workflow rather than a fixed report shape. Step fields that are
    /// `option` in the F# record are nullable here; strict mode forbids omitting them
    /// from `required`, so absence is encoded as null.
    let optimizerSchema =
        """{
  "type": "object",
  "properties": {
    "Name": { "type": "string" },
    "Description": { "type": "string" },
    "Version": { "type": "string" },
    "Inputs": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "Name": { "type": "string" },
          "Type": { "type": "string" },
          "Description": { "type": "string" }
        },
        "required": ["Name", "Type", "Description"],
        "additionalProperties": false
      }
    },
    "Steps": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "Id": { "type": "string" },
          "Type": { "type": "string" },
          "Agent": { "type": ["string", "null"] },
          "Tool": { "type": ["string", "null"] },
          "Instruction": { "type": ["string", "null"] },
          "Outputs": { "type": ["array", "null"], "items": { "type": "string" } },
          "Tools": { "type": ["array", "null"], "items": { "type": "string" } },
          "Params": { "type": ["object", "null"], "additionalProperties": { "type": "string" } }
        },
        "required": ["Id", "Type", "Agent", "Tool", "Instruction", "Outputs", "Tools", "Params"],
        "additionalProperties": false
      }
    }
  },
  "required": ["Name", "Description", "Version", "Inputs", "Steps"],
  "additionalProperties": false
}"""

    /// SymbolicReflector.ReflectOnTrace — trigger + observations + summary.
    /// `severity` is documented "(optional, for Anomaly)"; strict mode encodes that
    /// as a nullable required field rather than an absent one.
    let symbolicReflectionSchema =
        """{
  "type": "object",
  "properties": {
    "trigger_type": {
      "type": "string",
      "enum": ["TaskFailed", "TaskCompleted", "ContradictionDetected"]
    },
    "trigger_details": { "type": "string" },
    "observations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "enum": ["Performance", "Anomaly", "Contradiction", "Pattern"]
          },
          "description": { "type": "string" },
          "severity": { "type": ["string", "null"], "enum": ["Low", "Medium", "High", null] }
        },
        "required": ["type", "description", "severity"],
        "additionalProperties": false
      }
    },
    "summary": { "type": "string" }
  },
  "required": ["trigger_type", "trigger_details", "observations", "summary"],
  "additionalProperties": false
}"""
