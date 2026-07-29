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

    // NOTE: there is deliberately no optimizerSchema.
    //
    // Optimizer.OptimizeAsync returns a whole Tars.Metascript Workflow, and that
    // type cannot be expressed as a strict-mode JSON schema: `Params` is a
    // Map<string,string>, i.e. an open-ended object, and strict mode requires
    // `additionalProperties:false` with a closed property list on every object.
    //
    // The first attempt at one was actively harmful: it declared 8 of WorkflowStep's
    // 10 fields with additionalProperties:false, so a compliant model was *forbidden*
    // from emitting DependsOn and Context. DependsOn carries step ordering — the
    // constraint would have silently deleted workflow dependencies, and the prompt
    // shows the model a serialized workflow that does include those fields.
    //
    // That site keeps ResponseFormat.Json. A schema that constrains the model to a
    // shape its consumer rejects is worse than no schema.

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
