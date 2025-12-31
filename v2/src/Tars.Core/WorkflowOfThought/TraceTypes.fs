namespace Tars.Core.WorkflowOfThought

open System

[<RequireQualifiedAccess>]
type StepStatus = Ok | Error

type TraceEvent =
  { StepId: string
    Kind: string
    StartedAtUtc: DateTime
    EndedAtUtc: DateTime
    DurationMs: int64
    ToolName: string option
    ResolvedArgs: Map<string,string> option
    Outputs: string list
    Status: StepStatus
    Error: string option }

type CanonicalTraceEvent =
  { StepId: string
    Kind: string
    ToolName: string option
    // Sorted list of key-values for deterministic serialization
    ResolvedArgs: (string * string) list option
    Outputs: string list
    Status: StepStatus
    Error: string option }

module TraceEvent =
    let toCanonical (e: TraceEvent) =
        { StepId = e.StepId
          Kind = e.Kind
          ToolName = e.ToolName
          ResolvedArgs = e.ResolvedArgs |> Option.map (Map.toList >> List.sortBy fst)
          Outputs = e.Outputs
          Status = e.Status
          Error = e.Error }

type CanonicalGolden =
  { SchemaVersion: string
    Steps: CanonicalTraceEvent list
    Summary: {| ToolCalls: int; VerifyPassed: bool option; FirstError: string option; OutputKeys: string list; Mode: string |} }

// Diff Types
type PropertyDiff<'T> = { Old: 'T; New: 'T }

type StepDiff =
  | MissingInNew
  | ExtraInNew
  | Changed of changes: Map<string, string * string> // field -> (old, new) string rep

type GoldenDiff =
  { SummaryChanges: Map<string, string * string>
    StepChanges: Map<string, StepDiff> }
  member this.HasChanges = not (this.SummaryChanges.IsEmpty && this.StepChanges.IsEmpty)

module GoldenDiff =
    let private diffVal v1 v2 = if v1 = v2 then None else Some(string v1, string v2)
    
    let private diffLists l1 l2 =
        if l1 = l2 then None else Some(sprintf "%A" l1, sprintf "%A" l2)

    let compute (g1: CanonicalGolden) (g2: CanonicalGolden) : GoldenDiff =
        // 1. Summary Diff
        let s1, s2 = g1.Summary, g2.Summary
        let sumDiffs = 
            [ 
              if s1.ToolCalls <> s2.ToolCalls then yield "ToolCalls", (string s1.ToolCalls, string s2.ToolCalls)
              if s1.VerifyPassed <> s2.VerifyPassed then yield "VerifyPassed", (string s1.VerifyPassed, string s2.VerifyPassed)
              if s1.FirstError <> s2.FirstError then yield "FirstError", (string s1.FirstError, string s2.FirstError)
              if s1.Mode <> s2.Mode then yield "Mode", (s1.Mode, s2.Mode)
              match diffLists s1.OutputKeys s2.OutputKeys with Some d -> yield "OutputKeys", d | None -> ()
            ] |> Map.ofList

        // 2. Steps Diff
        let steps1 = g1.Steps |> List.map (fun s -> s.StepId, s) |> Map.ofList
        let steps2 = g2.Steps |> List.map (fun s -> s.StepId, s) |> Map.ofList
        
        let allKeys = Set.union (steps1.Keys |> Set.ofSeq) (steps2.Keys |> Set.ofSeq)
        
        let stepDiffs =
            allKeys
            |> Seq.choose (fun id ->
                match steps1.TryFind id, steps2.TryFind id with
                | Some _, None -> Some(id, MissingInNew)
                | None, Some _ -> Some(id, ExtraInNew)
                | None, None -> None // Should not happen
                | Some oldS, Some newS ->
                    if oldS = newS then None
                    else
                        let changes =
                            [ if oldS.Kind <> newS.Kind then yield "Kind", (oldS.Kind, newS.Kind)
                              if oldS.ToolName <> newS.ToolName then yield "ToolName", (string oldS.ToolName, string newS.ToolName)
                              if oldS.Status <> newS.Status then yield "Status", (string oldS.Status, string newS.Status)
                              // Comparing lists/objs as string rep for simplicity
                              match diffLists oldS.Outputs newS.Outputs with Some d -> yield "Outputs", d | None -> ()
                              match diffVal oldS.ResolvedArgs newS.ResolvedArgs with Some d -> yield "ResolvedArgs", d | None -> ()
                              if oldS.Error <> newS.Error then yield "Error", (string oldS.Error, string newS.Error)
                            ] |> Map.ofList
                        if changes.IsEmpty then None else Some(id, Changed changes)
            )
            |> Map.ofSeq

        { SummaryChanges = sumDiffs; StepChanges = stepDiffs }
