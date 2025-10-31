namespace TarsEngine.FSharp.SelfImprovement

open System
open TarsEngine.FSharp.Core.Services.CrossAgentValidation

/// Shared types for recording cross-agent feedback cycles.
module CrossAgentFeedback =

    type FeedbackVerdict =
        | Approve
        | NeedsWork of string
        | Reject of string
        | Escalate of string

    type AgentFeedback =
        { AgentId: string
          Role: AgentRole
          Verdict: FeedbackVerdict
          Confidence: float option
          Notes: string option
          SuggestedActions: string list
          RecordedAt: DateTime }

    let verdictToString = function
        | Approve -> "approve"
        | NeedsWork _ -> "needs_work"
        | Reject _ -> "reject"
        | Escalate _ -> "escalate"

    let verdictDetail = function
        | Approve -> None
        | NeedsWork detail
        | Reject detail
        | Escalate detail -> Some detail

