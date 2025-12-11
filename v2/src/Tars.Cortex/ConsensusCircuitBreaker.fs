/// ConsensusCircuitBreaker - Multi-agent consensus before execution
/// Part of v2.2 Cognitive Patterns (Pattern 6)
namespace Tars.Cortex

open System
open System.Threading.Tasks
open System.Collections.Concurrent

/// Vote from an agent
type ConsensusVote =
    | Approve of reason: string
    | Reject of reason: string
    | Abstain

/// Result of a consensus round
type ConsensusResult =
    { ProposalId: Guid
      TotalVoters: int
      Approvals: int
      Rejections: int
      Abstentions: int
      Reached: bool
      Decision: string
      Reasons: string list }

/// Consensus strategy
type ConsensusStrategy =
    | Unanimous // All must approve
    | Majority // > 50% must approve
    | Supermajority // >= 2/3 must approve
    | Quorum of int // At least N approvals

/// Configuration for consensus circuit breaker
type ConsensusConfig =
    { Strategy: ConsensusStrategy
      TimeoutMs: int
      MinVoters: int
      OpenOnFailedConsensus: bool }

    static member Default =
        { Strategy = Majority
          TimeoutMs = 5000
          MinVoters = 2
          OpenOnFailedConsensus = true }

/// Consensus Circuit Breaker - Pattern 6 from research
/// Gates execution on multi-agent agreement
/// Useful for critical decisions that require multiple perspectives
type ConsensusCircuitBreaker(config: ConsensusConfig) =
    let activeProposals =
        ConcurrentDictionary<Guid, ConcurrentDictionary<string, ConsensusVote>>()

    let mutable isOpen = false
    let mutable consecutiveFailures = 0

    /// Start a new consensus round
    member this.Propose(proposalId: Guid) =
        activeProposals.TryAdd(proposalId, ConcurrentDictionary<string, ConsensusVote>())
        |> ignore

        proposalId

    /// Submit a vote for a proposal
    member this.Vote(proposalId: Guid, voterId: string, vote: ConsensusVote) =
        match activeProposals.TryGetValue(proposalId) with
        | true, votes -> votes.TryAdd(voterId, vote) |> ignore
        | false, _ -> ()

    /// Check if consensus has been reached
    member this.CheckConsensus(proposalId: Guid) : ConsensusResult =
        match activeProposals.TryGetValue(proposalId) with
        | true, votes ->
            let voteList = votes.Values |> Seq.toList

            let approvals =
                voteList
                |> List.filter (function
                    | Approve _ -> true
                    | _ -> false)
                |> List.length

            let rejections =
                voteList
                |> List.filter (function
                    | Reject _ -> true
                    | _ -> false)
                |> List.length

            let abstentions =
                voteList
                |> List.filter (function
                    | Abstain -> true
                    | _ -> false)
                |> List.length

            let total = voteList.Length

            let reached =
                if total < config.MinVoters then
                    false
                else
                    match config.Strategy with
                    | Unanimous -> rejections = 0 && approvals > 0
                    | Majority -> float approvals > float total / 2.0
                    | Supermajority -> float approvals >= float total * 2.0 / 3.0
                    | Quorum n -> approvals >= n

            let reasons =
                voteList
                |> List.choose (function
                    | Approve r -> Some $"✓ {r}"
                    | Reject r -> Some $"✗ {r}"
                    | Abstain -> None)

            { ProposalId = proposalId
              TotalVoters = total
              Approvals = approvals
              Rejections = rejections
              Abstentions = abstentions
              Reached = reached
              Decision = if reached then "APPROVED" else "PENDING/REJECTED"
              Reasons = reasons }
        | false, _ ->
            { ProposalId = proposalId
              TotalVoters = 0
              Approvals = 0
              Rejections = 0
              Abstentions = 0
              Reached = false
              Decision = "NO_PROPOSAL"
              Reasons = [] }

    /// Execute action only if consensus is reached
    member this.ExecuteWithConsensusAsync<'T>(proposalId: Guid, action: unit -> Task<'T>) : Task<Result<'T, string>> =
        task {
            if isOpen then
                return Error "Circuit breaker is OPEN due to failed consensus"
            else
                let result = this.CheckConsensus(proposalId)

                if result.Reached then
                    try
                        let! value = action ()
                        consecutiveFailures <- 0
                        return Ok value
                    with ex ->
                        consecutiveFailures <- consecutiveFailures + 1
                        return Error ex.Message
                else
                    consecutiveFailures <- consecutiveFailures + 1

                    if config.OpenOnFailedConsensus && consecutiveFailures >= 3 then
                        isOpen <- true

                    return Error $"Consensus not reached: {result.Approvals}/{result.TotalVoters} approvals"
        }

    /// Close the proposal and cleanup
    member this.Close(proposalId: Guid) =
        activeProposals.TryRemove(proposalId) |> ignore

    /// Reset the circuit breaker
    member this.Reset() =
        isOpen <- false
        consecutiveFailures <- 0

    /// Check if circuit is open
    member this.IsOpen = isOpen

module ConsensusCircuitBreaker =
    let createDefault () =
        ConsensusCircuitBreaker(ConsensusConfig.Default)

    let create config = ConsensusCircuitBreaker(config)

    /// Simple majority consensus helper
    let requireMajority voters action =
        let cb = createDefault ()
        let proposalId = cb.Propose(Guid.NewGuid())

        for (voterId, vote) in voters do
            cb.Vote(proposalId, voterId, vote)

        cb.ExecuteWithConsensusAsync(proposalId, action)
