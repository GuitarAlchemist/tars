namespace TarsEngine.FSharp.Core.Services

open System

/// Structures used for logging and critiquing reasoning output from autonomous agents.
module ReasoningTrace =

    type ReasoningEvent =
        { AgentId: string
          Step: string
          Message: string
          Score: float option
          Metadata: Map<string, obj>
          CreatedAt: DateTime }

    type ReasoningTrace =
        { CorrelationId: string
          Summary: string option
          Events: ReasoningEvent list }

    type CriticVerdict =
        | Accept
        | NeedsReview of string
        | Reject of string
