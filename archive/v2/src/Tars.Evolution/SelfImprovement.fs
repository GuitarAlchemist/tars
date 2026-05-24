namespace Tars.Evolution

open System
open System.IO
open System.Threading.Tasks
open Tars.Llm
open Tars.Core.WorkflowOfThought
open System.Text.Json

module SelfImprovement =

    type ImprovementResult =
        | Success of Proposal: Mutation.Proposal * VariantPath: string
        | Failure of Error: string

    /// <summary>
    /// Analyzes a failure using an LLM and proposes a mutation to fix it.
    /// </summary>
    let analyzeAndPropose
        (llm: ILlmService)
        (targetFile: string)
        (lastError: string)
        (trace: string)
        (parentRunId: Guid)
        =
        task {
            if not (File.Exists targetFile) then
                return Failure(sprintf "Target file not found: %s" targetFile)
            else
                let content = File.ReadAllText targetFile

                let prompt =
                    $"""You are the TARS Neuro-Symbolic Self-Improvement Engine. 
A reasoning workflow failed. Your mission is to analyze the failure and propose a structured MUTATION to fix it.

WORKFLOW DEFINITION ({targetFile}):
{content}

FAILURE ERROR:
{lastError}

EXECUTION TRACE:
{trace}

CRITICAL: Output ONLY a single JSON object. 
Format:
{{
  "rationale": "one sentence explanation",
  "old_text": "exact text from the original file to replace",
  "new_text": "the new text to substitute"
}}
"""

                let req =
                    { ModelHint = None
                      Model = None
                      SystemPrompt = Some "You are a specialized code mutation agent. OUTPUT JSON ONLY."
                      MaxTokens = Some 2000
                      Temperature = Some 0.0
                      Stop = []
                      Messages = [ { Role = Role.User; Content = prompt } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = false
                      Seed = None
                      ContextWindow = None }

                let! resp = llm.CompleteAsync req

                // Robust JSON extraction
                let mutable text = resp.Text.Trim()
                let firstBrace = text.IndexOf('{')
                let lastBrace = text.LastIndexOf('}')

                if firstBrace >= 0 && lastBrace > firstBrace then
                    text <- text.Substring(firstBrace, lastBrace - firstBrace + 1)

                try
                    use doc = JsonDocument.Parse(text)
                    let root = doc.RootElement

                    let getProp (element: JsonElement) (name: string) =
                        let mutable prop = JsonElement()

                        if element.TryGetProperty(name, &prop) then
                            Some(prop.GetString())
                        else
                            None

                    let rationale =
                        getProp root "rationale" |> Option.defaultValue "No rationale provided"

                    let oldText =
                        getProp root "old_text"
                        |> Option.orElse (getProp root "old")
                        |> function
                            | Some s -> s
                            | None -> failwith "Missing 'old_text' or 'old'"

                    let newText =
                        getProp root "new_text"
                        |> Option.orElse (getProp root "new")
                        |> function
                            | Some s -> s
                            | None -> failwith "Missing 'new_text' or 'new'"

                    let target = Mutation.Target.Prompt targetFile
                    let op = Mutation.Operation.Replace(oldText, newText)

                    let mutationService = Mutation.MutationService(".")
                    let proposal = mutationService.Propose(target, op, rationale, parentRunId)
                    let result = mutationService.Apply(proposal)

                    if result.IsApplied then
                        return Success(proposal, result.VariantPath)
                    else
                        return Failure(result.Error |> Option.defaultValue "Unknown mutation error")
                with ex ->
                    return
                        Failure(sprintf "Failed to parse mutation proposal: %s. Raw response: %s" ex.Message resp.Text)
        }

    /// <summary>
    /// Log the improvement event to symbolic memory.
    /// </summary>
    let logImprovement (proposal: Mutation.Proposal) (variantPath: string) (success: bool) =
        async {
            let metadata =
                Map.ofList
                    [ "proposal_id", proposal.Id.ToString()
                      "target", sprintf "%A" proposal.Target
                      "variant", variantPath
                      "success", string success ]

            do!
                SymbolicMemory.logStrategy
                    (proposal.ParentRunId |> Option.defaultValue (Guid.NewGuid()))
                    proposal.Rationale
                    metadata
        }
