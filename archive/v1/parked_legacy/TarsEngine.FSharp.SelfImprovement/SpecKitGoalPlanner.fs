namespace TarsEngine.FSharp.SelfImprovement

open System
open System.IO
open System.Text.Json
open TarsEngine.FSharp.Core.Specs

/// Utilities for generating follow-up Spec Kit goals from iteration outcomes (Tier 4).
module SpecKitGoalPlanner =

    type BacklogEntry =
        { id: string
          title: string
          priority: string
          specPath: string
          createdAt: DateTime
          sourceSpec: string
          sourceRunId: string option
          status: string }

    let private serializerOptions =
        JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true)

    let private ensureDirectory (path: string) =
        Directory.CreateDirectory(path) |> ignore
        path

    let private backlogPath () =
        let root = Path.Combine(Environment.CurrentDirectory, ".specify")
        ensureDirectory root |> ignore
        Path.Combine(root, "backlog.json")

    let private autoSpecsRoot () =
        let specsRoot = Path.Combine(Environment.CurrentDirectory, ".specify", "specs")
        ensureDirectory specsRoot |> ignore
        let autoRoot = Path.Combine(specsRoot, "auto-generated")
        ensureDirectory autoRoot |> ignore
        autoRoot

    let private loadBacklog () =
        let path = backlogPath ()
        if File.Exists(path) then
            try
                let json = File.ReadAllText(path)
                JsonSerializer.Deserialize<BacklogEntry list>(json, serializerOptions)
                |> Option.ofObj
                |> Option.defaultValue []
            with _ -> []
        else
            []

    let private saveBacklog (entries: BacklogEntry list) =
        let path = backlogPath ()
        let json = JsonSerializer.Serialize(entries, serializerOptions)
        File.WriteAllText(path, json)

    let private priorityLabel priorityRank =
        match priorityRank with
        | rank when rank <= 0 -> "P0"
        | 1 -> "P1"
        | 2 -> "P2"
        | _ -> "P3"

    let private formatSpecContent (source: SpecKitWorkspace.SpecKitSelection) (goalId: string) (priority: string) =
        let summary = source.Feature.Summary
        let edgeCases =
            summary.EdgeCases
            |> List.map (fun item -> $"- {item}")
            |> String.concat Environment.NewLine

        let acceptanceCriteria =
            summary.UserStories
            |> List.collect (fun story -> story.AcceptanceCriteria)
            |> function
                | [] -> [ "1. Define measurable validation outcomes for the new goal." ]
                | existing -> existing
            |> String.concat Environment.NewLine

        [ sprintf "# Follow-up Feature: %s -> %s" summary.Title goalId
          ""
          sprintf "**Feature Branch**: `%s`" goalId
          sprintf "**Created**: %s" (DateTime.UtcNow.ToString("yyyy-MM-dd"))
          "**Status**: Draft"
          ""
          "### Goal Overview"
          ""
          sprintf "- Source Specification: `%s`" source.Feature.Id
          sprintf "- Selected Task: `%s`" source.Task.Description
          sprintf "- Priority: `%s`" priority
          sprintf "- Rationale: Auto-generated after harness iteration on `%s` to address outstanding work." summary.Title
          ""
          "### Acceptance Scenarios"
          acceptanceCriteria
          ""
          "### Edge Cases"
          edgeCases
          ""
          "```metascript"
          "SPAWN GOAL_MANAGER 1 DIRECTIVE"
          sprintf "DEFINE goal \"%s - %s\"" summary.Title source.Task.Description
          sprintf "ATTRIBUTE priority %s" priority
          sprintf "ATTRIBUTE source_spec %s" source.Feature.Id
          "ATTRIBUTE follow_up \"true\""
          "```"
          ""
          "```expectations"
          "goals=1"
          sprintf "priority=%s" priority
          sprintf "parent=\"%s\"" source.Feature.Id
          "```" ]
        |> String.concat Environment.NewLine
        |> fun text -> text.Trim()

    let generateGoal (selection: SpecKitWorkspace.SpecKitSelection) (runId: Guid option) =
        let idSeed = $"auto-{DateTime.UtcNow:yyyyMMddHHmmss}-{Guid.NewGuid():N}"
        let goalId =
            if idSeed.Length > 32 then idSeed.Substring(0, 32) else idSeed
        let priority = priorityLabel selection.PriorityRank
        let specContent = formatSpecContent selection goalId priority
        let specRoot = autoSpecsRoot ()
        let specPath = Path.Combine(specRoot, $"{goalId}.md")
        File.WriteAllText(specPath, specContent)

        let entry: BacklogEntry =
            { id = goalId
              title = $"Auto follow-up: {selection.Task.Description}"
              priority = priority
              specPath = specPath
              createdAt = DateTime.UtcNow
              sourceSpec = selection.Feature.Id
              sourceRunId = runId |> Option.map string
              status = "pending" }

        let backlog = loadBacklog ()
        saveBacklog (entry :: backlog)
        entry

    let recordGoal (selection: SpecKitWorkspace.SpecKitSelection) (runId: Guid option) =
        generateGoal selection runId |> ignore

    let private formatRemediationSpec (specId: string) (iterationId: Guid) (artifactPath: string) (actions: (string * string list) list) =
        let iterationLabel = iterationId.ToString("N").Substring(0, 8).ToUpperInvariant()

        let renderedActions =
            actions
            |> List.collect (fun (role, steps) ->
                let header = sprintf "- %s:" role
                match steps with
                | [] -> [ header; "  - Review remediation artifact and capture next steps." ]
                | _ ->
                    header
                    :: (steps |> List.map (fun step -> $"  - {step}")))
            |> function
                | [] -> [ "- Review remediation artifact and capture next steps." ]
                | items -> items
            |> String.concat Environment.NewLine

        [ sprintf "# Remediation Ticket: %s" specId
          ""
          sprintf "**Feature Branch**: `trem-%s-%s`" (DateTime.UtcNow.ToString("yyyyMMddHHmmss")) iterationLabel
          sprintf "**Created**: %s" (DateTime.UtcNow.ToString("yyyy-MM-dd"))
          "**Status**: Pending"
          ""
          "## Context"
          sprintf "- Source Spec: `%s`" specId
          sprintf "- Iteration Id: `%s`" (iterationId.ToString("D"))
          sprintf "- Remediation Artifact: `%s`" artifactPath
          ""
          "## Required Actions"
          renderedActions
          ""
          "## Validation"
          "1. Execute agreed remediation steps."
          "2. Update Tier4 backlog and close the corresponding TREM task."
          ""
          "```metascript"
          "SPAWN RemediationLead 1 DIRECTIVE"
          "SPAWN EvidenceScribe 1 DEMOCRATIC"
          "CONNECT RemediationLead EvidenceScribe audit"
          "METRIC remediation_urgency 1.00"
          "REPEAT remediation-followup 1"
          "```"
          ""
          "```expectations"
          "remediation=\"true\""
          "priority=P0"
          sprintf "parent=\"%s\"" specId
          "```" ]
        |> String.concat Environment.NewLine

    let recordRemediationTicket
        (specId: string)
        (iterationId: Guid)
        (artifactFullPath: string)
        (artifactDisplayPath: string)
        (actions: (string * string list) list) =

        if String.IsNullOrWhiteSpace(specId) then
            ()
        elif String.IsNullOrWhiteSpace(artifactFullPath) || not (File.Exists(artifactFullPath)) then
            ()
        else
            let iterationKey = iterationId.ToString("N")

            let backlog = loadBacklog ()
            let alreadyTracked =
                backlog
                |> List.exists (fun entry ->
                    String.Equals(entry.sourceSpec, specId, StringComparison.OrdinalIgnoreCase)
                    && entry.sourceRunId = Some iterationKey)

            if not alreadyTracked then
                let idSeed = $"trem-{DateTime.UtcNow:yyyyMMddHHmmss}-{iterationKey.Substring(0, 8)}"
                let entryId =
                    if idSeed.Length > 32 then idSeed.Substring(0, 32) else idSeed

                let specRoot = autoSpecsRoot ()
                let specPath = Path.Combine(specRoot, $"{entryId}.md")
                let artifactPathForSpec =
                    if String.IsNullOrWhiteSpace artifactDisplayPath then
                        artifactFullPath
                    else
                        artifactDisplayPath

                let specContent = formatRemediationSpec specId iterationId artifactPathForSpec actions
                File.WriteAllText(specPath, specContent)

                let title =
                    let iterationLabel = iterationKey.Substring(0, 8).ToUpperInvariant()
                    $"Auto remediation: {specId} ({iterationLabel})"

                let entry: BacklogEntry =
                    { id = entryId
                      title = title
                      priority = "P0"
                      specPath = specPath
                      createdAt = DateTime.UtcNow
                      sourceSpec = specId
                      sourceRunId = Some iterationKey
                      status = "pending" }

                saveBacklog (entry :: backlog)
