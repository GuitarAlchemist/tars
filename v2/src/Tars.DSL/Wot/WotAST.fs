namespace Tars.DSL.Wot

open System
open Tars.Core.WorkflowOfThought

type DslId = string

type NodeKind = Reason | Work

type DslNode =
  { Id: DslId
    Kind: NodeKind
    Name: string
    Inputs: string list
    Outputs: string list
    // Work
    Tool: string option
    Args: Map<string,obj> option
    Checks: WotCheck list
    // Reason
    Goal: string option
    Invariants: string list
    Constraints: string list
    Verdict: string option }

type DslPolicy =
  { AllowedTools: Set<string>
    MaxToolCalls: int
    MaxTokens: int
    MaxTimeMs: int }

type DslInputs = Map<string,string>

type DslWorkflow =
  { Name: string
    Version: string
    Risk: string
    Inputs: DslInputs
    Policy: DslPolicy
    Nodes: DslNode list
    Edges: (DslId * DslId) list }
