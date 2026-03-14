namespace Tars.Core.WorkflowOfThought

open System
open System.IO

/// <summary>
/// Domain model for mutation-driven self-improvement (Phase 15.4)
/// </summary>
[<RequireQualifiedAccess>]
module Mutation =

    /// <summary>
    /// Target types for system behavior modification.
    /// </summary>
    type Target =
        | Prompt of FilePath: string
        | RoutingConfig of Key: string
        | Pattern of WorkflowName: string * NodeId: string * Property: string

    /// <summary>
    /// Structured mutation operations.
    /// </summary>
    type Operation =
        | Append of Text: string
        | Replace of Old: string * New: string
        | SetValue of JsonProperty: string * NewValue: obj
        | LlmGuided of Instruction: string

    /// <summary>
    /// A proposal for a system change.
    /// </summary>
    type Proposal =
        { Id: Guid
          ParentRunId: Guid option
          Target: Target
          Op: Operation
          Rationale: string
          Timestamp: DateTimeOffset }

    /// <summary>
    /// Outcome of a mutation attempt.
    /// </summary>
    type Result =
        { ProposalId: Guid
          VariantPath: string
          IsApplied: bool
          Error: string option }

    /// <summary>
    /// Service for managing controlled mutations.
    /// </summary>
    type MutationService(baseDir: string) =
        let mutable variantsDir = Path.Combine(baseDir, ".wot", "variants")

        do
            if not (Directory.Exists variantsDir) then
                Directory.CreateDirectory variantsDir |> ignore

        member _.Propose(target, op, rationale, ?parentRunId) =
            { Id = Guid.NewGuid()
              ParentRunId = parentRunId
              Target = target
              Op = op
              Rationale = rationale
              Timestamp = DateTimeOffset.Now }

        member _.Apply(proposal: Proposal) : Result =
            try
                let (sourcePath, variantExt) =
                    match proposal.Target with
                    | Target.Prompt path -> (path, Path.GetExtension(path))
                    | Target.RoutingConfig _ -> ("src/Tars.Interface.Cli/appsettings.json", ".json")
                    | Target.Pattern(path, _, _) -> (path, ".trsx")

                if not (File.Exists sourcePath) then
                    { ProposalId = proposal.Id
                      VariantPath = ""
                      IsApplied = false
                      Error = Some(sprintf "Source not found: %s" sourcePath) }
                else
                    let content = File.ReadAllText sourcePath

                    let mutated =
                        match proposal.Op with
                        | Operation.Append t -> content + "\n\n" + t
                        | Operation.Replace(o, n) -> content.Replace(o, n)
                        | _ -> content

                    let variantId = proposal.Id.ToString("n").Substring(0, 8)
                    let fileName = Path.GetFileNameWithoutExtension(sourcePath)

                    let variantPath =
                        Path.Combine(variantsDir, sprintf "%s_%s%s" variantId fileName variantExt)

                    File.WriteAllText(variantPath, mutated)

                    { ProposalId = proposal.Id
                      VariantPath = variantPath
                      IsApplied = true
                      Error = None }
            with ex ->
                { ProposalId = proposal.Id
                  VariantPath = ""
                  IsApplied = false
                  Error = Some ex.Message }
