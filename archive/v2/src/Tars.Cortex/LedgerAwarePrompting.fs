namespace Tars.Cortex

open System
open Tars.Core
open Tars.Knowledge

/// Utilities for injecting knowledge ledger context into prompts
module LedgerAwarePrompting =

    /// Format a list of beliefs for injection into a prompt
    let formatBeliefs (beliefs: Belief list) =
        if beliefs.IsEmpty then ""
        else
            let lines =
                beliefs
                |> List.map (fun b ->
                    let conf = sprintf "%.0f%%" (b.Confidence * 100.0)
                    $"- {b.Subject} {b.Predicate} {b.Object} (Confidence: {conf})")
                |> String.concat "\n"

            $"\n\n[KNOWN FACTS FROM KNOWLEDGE LEDGER]\n{lines}\n"

    /// Enrich a prompt with relevant beliefs from the ledger
    let enrichPrompt (ledger: KnowledgeLedger) (prompt: string) =
        let relevant = ledger.GetRelevantBeliefs(prompt, limit = 15)
        let knowledgeContext = formatBeliefs relevant
        prompt + knowledgeContext

    /// Inject knowledge into an agent's system prompt or goal
    let injectKnowledge (ledger: KnowledgeLedger) (goal: string) =
        let relevant = ledger.GetRelevantBeliefs(goal, limit = 10)
        formatBeliefs relevant
