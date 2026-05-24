namespace Tars.Knowledge

open System
open Tars.Symbolic
open Tars.Core
open Tars.Llm

/// The Reflection Agent - Analyzes existing knowledge for internal consistency
/// "The system that remembers being wrong."
type ReflectionAgent(ledger: KnowledgeLedger, registry: IAgentRegistry option, llm: ILlmService option) =
    let agentId = AgentId.System

    new(ledger: KnowledgeLedger) = ReflectionAgent(ledger, None, None)

    /// Perform symbolic reflection on the entire ledger
    /// Scans for contradictions and metrics consistency
    member this.ReflectAsync() : Async<unit> =
        async {
            Logging.info "Starting symbolic reflection..."
            let sw = System.Diagnostics.Stopwatch.StartNew()

            // 1. Get all valid beliefs
            let allBeliefs = ledger.Query() |> Seq.toList
            let mutable contradictionsFound = 0

            // 2. Identify subjects with multiple beliefs
            let bySubject = allBeliefs |> List.groupBy (fun b -> b.Subject)

            for (subject, subjectBeliefs) in bySubject do
                if subjectBeliefs.Length > 1 then
                    // Cross-check all beliefs about this subject
                    for i in 0 .. subjectBeliefs.Length - 1 do
                        for j in i + 1 .. subjectBeliefs.Length - 1 do
                            let b1 = subjectBeliefs.[i]
                            let b2 = subjectBeliefs.[j]
                            
                            // Skip if already marked as contradicting
                            let alreadyContradicts = 
                                ledger.GetContradictions() 
                                |> Seq.exists (fun (c1, c2) -> 
                                    (c1.Id = b1.Id && c2.Id = b2.Id) || (c1.Id = b2.Id && c2.Id = b1.Id))
                            
                            if not alreadyContradicts then
                                let score = ConstraintScoring.scoreBeliefConsistency b1.TripleString [ b2.TripleString ]
                                
                                if score < 0.4 then
                                    let explanation = $"Heuristic contradiction detected during reflection. Score: {score:F2}"
                                    do! ledger.MarkContradiction(b1.Id, b2.Id, explanation, agentId) |> Async.AwaitTask |> Async.Ignore
                                    contradictionsFound <- contradictionsFound + 1
                                    Logging.warn $"[Reflection] Found contradiction: {b1.TripleString} vs {b2.TripleString}"

                                    // Operational Reflection: Penalize hallucinating agents
                                    match registry with
                                    | Some reg ->
                                        let penalize (b: Belief) = async {
                                            let aidOpt = 
                                                match b.Provenance.Source with
                                                | SourceType.Agent aid -> Some aid
                                                | _ -> b.Provenance.ExtractedBy
                                            
                                            match aidOpt with
                                            | Some aid ->
                                                let! agentOpt = reg.GetAgent(aid) |> Async.StartAsTask |> Async.AwaitTask
                                                match agentOpt with
                                                | Some agent ->
                                                    // Decrease fitness by 10%
                                                    let newFitness = agent.Fitness * 0.90
                                                    let updatedAgent = { agent with Fitness = newFitness }
                                                    do! reg.UpdateAgent(updatedAgent) |> Async.StartAsTask |> Async.AwaitTask |> Async.Ignore
                                                    Logging.info $"[Reflection] Penalized agent {agent.Name} (Fitness: {agent.Fitness:F2} -> {newFitness:F2})"
                                                | None -> ()
                                            | None -> ()
                                        }
                                        do! penalize b1
                                        do! penalize b2
                                    | None -> ()

            sw.Stop()
            Logging.info $"Reflection complete in {sw.ElapsedMilliseconds}ms. Found {contradictionsFound} new contradictions."
        }

    /// Perform Architectural Reflection (The "Architect's Eye")
    member this.ReflectOnArchitectureAsync() =
        async {
            match llm with
            | Some service -> 
                Logging.info "[Architect] Starting architectural reflection..."
                
                // 1. Gather context (recent beliefs)
                let allBeliefs = ledger.Query()
                
                // Filter for potentially relevant beliefs (this is a heuristic)
                let relevantBeliefs = 
                    allBeliefs
                    |> Seq.filter (fun b -> 
                        b.Confidence > 0.8 &&
                        // Focus on contradictions or things explicitly about architecture/tasks
                        (b.Predicate = RelationType.Contradicts || 
                         b.Tags |> List.contains "architecture" ||
                         b.Tags |> List.contains "insight" ||
                         b.Subject.Value.Contains("Tars")))
                    |> Seq.sortByDescending (fun b -> b.ValidFrom)
                    |> Seq.truncate 40
                    |> Seq.map (fun b -> $"- {b.TripleString}")
                    |> String.concat "\n"
                    
                if String.IsNullOrWhiteSpace relevantBeliefs then
                    Logging.info "[Architect] Not enough data to reflect on."
                else
                    let systemPrompt = "You are the Chief System Architect for TARS. Your goal is to identify structural weaknesses, circular dependencies, or high-leverage refactoring opportunities based on the Knowledge Ledger."
                    
                    let userMsg = 
                         $"Current System Beliefs:\n{relevantBeliefs}\n\nTask: Analyze these beliefs. Identify ONE critical architectural issue or improvement. Return it as a succinct task description (e.g. 'Refactor module X to decouple Y'). If everything looks good, return 'No action needed'."
                    
                    let request = 
                        { LlmRequest.Default with
                            SystemPrompt = Some systemPrompt
                            Messages = [ { Role = Role.User; Content = userMsg } ]
                            Temperature = Some 0.2 }
                            
                    try
                        let! response = service.CompleteAsync(request) |> Async.AwaitTask
                        let suggestion = response.Text.Trim()
                        
                        // Sanitize quotes to avoid JSON issues downstream
                        let suggestionSanitized = suggestion.Replace("\"", "'")
                        
                        if not (String.IsNullOrEmpty suggestionSanitized) && not (suggestionSanitized.Contains("No action needed")) then
                            // Check if we already have this suggestion to avoid duplicates
                            let exists = 
                                allBeliefs 
                                |> Seq.exists (fun b -> 
                                    b.Subject.Value = "TARS_Architecture" && 
                                    b.Object.Value = suggestionSanitized)
                                    
                            if not exists then
                                // Assert the suggestion as a belief
                                let belief = Belief.create "TARS_Architecture" (RelationType.Custom "requires_refactor") suggestionSanitized (Provenance.FromRun(RunId.New(), agentId))
                                do! ledger.Assert(belief, agentId) |> Async.AwaitTask |> Async.Ignore
                                Logging.info $"[Architect] Proposed: {suggestionSanitized}"
                            else
                                Logging.info $"[Architect] Skipping duplicate suggestion."
                    with ex ->
                        Logging.error $"[Architect] Failed: {ex.Message}" ex
            | None -> 
                Logging.warn "[Architect] No LLM available for architectural reflection."
        }

    /// Higher-order reflection: Auto-cleanup of low confidence or highly contradicted beliefs
    member this.CleanupAsync(confidenceThreshold: float) : Async<int> =
        async {
            let allBeliefs = ledger.Query() |> Seq.toList
            let mutable retractedCount = 0

            for belief in allBeliefs do
                if belief.Confidence < confidenceThreshold then
                    let reason = $"Confidence ({belief.Confidence:P0}) fell below threshold ({confidenceThreshold:P0})"
                    do! ledger.Retract(belief.Id, reason, agentId) |> Async.AwaitTask |> Async.Ignore
                    retractedCount <- retractedCount + 1

            return retractedCount
        }
