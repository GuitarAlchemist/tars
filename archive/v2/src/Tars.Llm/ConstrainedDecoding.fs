namespace Tars.Llm

/// Constrained decoding helpers: load grammars and create constrained LLM requests.
/// Bridges TARS grammar files (EBNF, JSON Schema) to the LLM request pipeline.
///
/// When targeting vLLM, EBNF grammars are sent via guided_decoding extra_body
/// using xgrammar or outlines backends. For other providers, grammars degrade
/// gracefully to prompt hints.
///
/// Architecture (from Probabilistic Grammars notebook):
///   Layer 1: Grammar/schema → this module loads and attaches grammars
///   Layer 2: Probabilistic policy → WeightedGrammar steers preferences
///   Layer 3: Semantic validation → GrammarGovernor checks meaning
///   Layer 4: Execution → only validated artifacts run
module ConstrainedDecoding =

    open System
    open System.IO

    // =========================================================================
    // Grammar loading
    // =========================================================================

    /// Load an EBNF grammar file from the grammars/ directory
    let loadEbnfGrammar (grammarsDir: string) (name: string) : Result<string, string> =
        let path =
            if name.EndsWith(".ebnf") then Path.Combine(grammarsDir, name)
            else Path.Combine(grammarsDir, name + ".ebnf")
        if File.Exists(path) then
            Ok (File.ReadAllText(path))
        else
            Error $"Grammar file not found: {path}"

    /// Load a JSON schema file
    let loadJsonSchema (path: string) : Result<string, string> =
        if File.Exists(path) then
            Ok (File.ReadAllText(path))
        else
            Error $"JSON schema not found: {path}"

    /// List available grammars in a directory
    let listGrammars (grammarsDir: string) : string list =
        if Directory.Exists(grammarsDir) then
            Directory.GetFiles(grammarsDir, "*.ebnf")
            |> Array.map Path.GetFileNameWithoutExtension
            |> Array.toList
        else []

    // =========================================================================
    // Request construction
    // =========================================================================

    /// Create an LLM request with EBNF grammar constraint
    let withEbnfGrammar (grammar: string) (req: LlmRequest) : LlmRequest =
        { req with ResponseFormat = Some (ResponseFormat.Constrained (Grammar.Ebnf grammar)) }

    /// Create an LLM request with JSON schema constraint
    let withJsonSchema (schema: string) (req: LlmRequest) : LlmRequest =
        { req with ResponseFormat = Some (ResponseFormat.Constrained (Grammar.JsonSchema schema)) }

    /// Create an LLM request with regex constraint
    let withRegex (pattern: string) (req: LlmRequest) : LlmRequest =
        { req with ResponseFormat = Some (ResponseFormat.Constrained (Grammar.Regex pattern)) }

    /// Load a named grammar and attach it to a request
    let withNamedGrammar (grammarsDir: string) (name: string) (req: LlmRequest) : Result<LlmRequest, string> =
        match loadEbnfGrammar grammarsDir name with
        | Ok grammar -> Ok (withEbnfGrammar grammar req)
        | Error err -> Error err

    // =========================================================================
    // Convenience: common constrained request patterns
    // =========================================================================

    /// Create a request that forces JSON output matching a schema
    let jsonConstrained (schema: string) (messages: LlmMessage list) : LlmRequest =
        { LlmRequest.Default with
            Messages = messages
            ResponseFormat = Some (ResponseFormat.Constrained (Grammar.JsonSchema schema)) }

    /// Create a request with EBNF grammar from string
    let ebnfConstrained (grammar: string) (messages: LlmMessage list) : LlmRequest =
        { LlmRequest.Default with
            Messages = messages
            ResponseFormat = Some (ResponseFormat.Constrained (Grammar.Ebnf grammar)) }

    /// Create a request constrained by the cortex DSL grammar
    let cortexConstrained (grammarsDir: string) (messages: LlmMessage list) : Result<LlmRequest, string> =
        let req = { LlmRequest.Default with Messages = messages }
        withNamedGrammar grammarsDir "cortex" req

    // =========================================================================
    // IR type schemas (from Probabilistic Grammars notebook)
    // =========================================================================

    /// JSON Schema for TARS IntentPlan IR (Layer 1 typed latent plan)
    let intentPlanSchema = """
{
  "type": "object",
  "properties": {
    "intent": { "type": "string", "description": "Inferred user intent" },
    "strategy": { "type": "string", "enum": ["chain_of_thought", "react", "tree_of_thoughts", "graph_of_thoughts"] },
    "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
    "steps": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "action": { "type": "string" },
          "tool": { "type": "string" },
          "expected_output": { "type": "string" }
        },
        "required": ["action"]
      }
    },
    "fallback": { "type": "string", "description": "What to do if the plan fails" }
  },
  "required": ["intent", "strategy", "confidence", "steps"]
}"""

    /// JSON Schema for TARS BeliefUpdate IR
    let beliefUpdateSchema = """
{
  "type": "object",
  "properties": {
    "belief_id": { "type": "string" },
    "operation": { "type": "string", "enum": ["assert", "retract", "revise", "strengthen", "weaken"] },
    "subject": { "type": "string" },
    "predicate": { "type": "string" },
    "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
    "evidence": { "type": "array", "items": { "type": "string" } },
    "source": { "type": "string" }
  },
  "required": ["operation", "subject", "predicate", "confidence"]
}"""

    /// JSON Schema for TARS RepairProposal IR
    let repairProposalSchema = """
{
  "type": "object",
  "properties": {
    "error_type": { "type": "string" },
    "diagnosis": { "type": "string" },
    "repair_action": { "type": "string", "enum": ["retry", "fallback", "escalate", "skip", "modify_params"] },
    "modified_params": { "type": "object" },
    "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
    "rationale": { "type": "string" }
  },
  "required": ["error_type", "diagnosis", "repair_action", "confidence"]
}"""
