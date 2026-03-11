namespace Tars.Knowledge

open Tars.Symbolic

/// Result of verification
type VerificationDecision =
    | Accepted of confidence: float
    | Denied of reason: string
    | Conflict of conflictingBelief: Belief * score: float

/// The Verifier Agent - Gatekeeper of the Knowledge Ledger
/// "The Internet never writes beliefs directly."
type VerifierAgent(ledger: KnowledgeLedger) =
    let agentId = AgentId "verifier"

    /// Verify a proposed assertion against the existing knowledge graph
    member this.Verify(proposal: ProposedAssertion) : Async<VerificationDecision> =
        async {
            // 1. Context Retrieval: Get neighborhood of the subject
            // We need to fetch strictly valid beliefs to check consistency
            let contextBeliefs =
                // Neighborhood depth 1 is sufficient for immediate contradictions
                // Note: accessing ledger.Graph is thread-safe (returns copy/iterator protected by lock)
                ledger.Graph.GetNeighborhood(EntityId proposal.Subject, 1) |> Seq.toList

            // 2. Symbolic Invariant: Belief Consistency
            // We interpret the proposal as a belief string and context as list of strings
            // Format: "(Subject Predicate Object)"
            let proposalStr = $"({proposal.Subject} {proposal.Predicate} {proposal.Object})"
            let contextStrs = contextBeliefs |> List.map (fun b -> b.TripleString)

            // 3. Constraint Scoring
            // Use Tars.Symbolic to score consistency
            let consistencyScore =
                ConstraintScoring.scoreBeliefConsistency proposalStr contextStrs

            // 4. Decision Logic
            // Thresholds:
            // > 0.7: Safe (lowered slightly to allow for partial matches)
            // < 0.4: Definite contradiction
            // 0.4 - 0.7: Ambiguous / Low confidence

            if consistencyScore >= 0.7 then
                return Accepted consistencyScore
            elif consistencyScore < 0.4 then
                // In a future version, use ConstraintScoring/NeuralSymbolicFeedback to identify WHICH belief conflicted
                return Denied $"Inconsistent with existing knowledge (Score: {consistencyScore:F2})"
            else
                return Denied $"Ambiguous consistency (Score: {consistencyScore:F2})"
        }

    /// Process an evidence candidate and verify all its proposals
    member this.ProcessCandidate(candidate: EvidenceCandidate) : Async<(ProposedAssertion * VerificationDecision)[]> =
        async {
            let! decisions =
                candidate.ProposedAssertions
                |> List.map (fun p ->
                    async {
                        let! decision = this.Verify p
                        return (p, decision)
                    })
                |> Async.Parallel

            return decisions
        }
