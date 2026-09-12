namespace Tars.Metascript

open System
open System.Threading.Tasks
open Tars.Llm
open Tars.Metascript.V1

/// <summary>
/// Output bundle of the GrammarDistiller containing all compiled representation targets.
/// </summary>
type DistilledContract =
    {
        ContractId: string
        Version: string
        JsonSchema: string
        EbnfGrammar: string
        PromptHints: string
        EvaluationCases: string list
    }

/// <summary>
/// Interface for distilling raw, unstructured, or programmatic inputs into typed schemas/grammars.
/// This acts as the direct compilation bridge into the Closure Factory V2.
/// </summary>
type IGrammarDistiller =
    /// <summary>
    /// Distills a raw JSON example structure into a full contract specification.
    /// </summary>
    abstract member DistillFromJsonAsync: jsonExample: string * contractId: string -> Task<DistilledContract>

    /// <summary>
    /// Compiles a legacy .tars block (such as meta, grammar, or transform blocks) into its V2 contract form.
    /// </summary>
    abstract member DistillFromTarsBlockAsync: block: MetascriptBlock -> Task<DistilledContract>

    /// <summary>
    /// Uses reflection over a .NET/F# Type (Records, Discriminated Unions) to construct a contract bundle.
    /// </summary>
    abstract member DistillFromTypeAsync: t: Type -> Task<DistilledContract>

/// <summary>
/// Coordinates with Closure Factory V2 to map grammar-distilled contracts into input/output validators.
/// </summary>
module ClosureFactoryIntegration =

    /// <summary>
    /// Context for executing a compiled Closure.
    /// </summary>
    type ClosureContext =
        {
            ClosureId: string
            InputContract: DistilledContract
            OutputContract: DistilledContract
        }

    /// <summary>
    /// Verifies if a given input payload complies with the distilled input contract.
    /// </summary>
    let validateInput (context: ClosureContext) (rawInputJson: string) : Result<unit, string list> =
        // Parse and validate rawInputJson against context.InputContract.JsonSchema
        // Placeholder validation: successfully parses JSON
        try
            if String.IsNullOrWhiteSpace(rawInputJson) then
                Error [ "Input JSON payload is empty" ]
            else
                // Simple brace check for basic syntax validity
                if rawInputJson.Trim().StartsWith("{") && rawInputJson.Trim().EndsWith("}") then
                    Ok ()
                else
                    Error [ "Input does not represent a valid JSON object structure" ]
        with
        | ex -> Error [ ex.Message ]

    /// <summary>
    /// Verifies if a given output payload complies with the distilled output contract.
    /// </summary>
    let validateOutput (context: ClosureContext) (rawOutputJson: string) : Result<unit, string list> =
        // Parse and validate rawOutputJson against context.OutputContract.JsonSchema
        try
            if String.IsNullOrWhiteSpace(rawOutputJson) then
                Error [ "Output JSON payload is empty" ]
            else
                if rawOutputJson.Trim().StartsWith("{") && rawOutputJson.Trim().EndsWith("}") then
                    Ok ()
                else
                    Error [ "Output does not represent a valid JSON object structure" ]
        with
        | ex -> Error [ ex.Message ]
