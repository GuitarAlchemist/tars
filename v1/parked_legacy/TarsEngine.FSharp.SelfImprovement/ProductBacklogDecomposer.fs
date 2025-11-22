namespace TarsEngine.FSharp.SelfImprovement

open System

/// Converts Spec Kit features into structured agile backlog entries.
module ProductBacklogDecomposer =

    open SpecKitWorkspace
    open TarsEngine.FSharp.Core.Specs

    type BacklogItemType =
        | Epic
        | Story
        | Spike

    type BacklogStatus =
        | New
        | Ready
        | InProgress
        | Blocked
        | Duplicate
        | Done

    type BacklogItem =
        { Id: Guid
          Title: string
          Description: string
          ItemType: BacklogItemType
          Priority: string
          StoryPoints: string option
          Tags: string list
          Status: BacklogStatus
          Participants: TeamConfiguration.AgentParticipant list
          EpicId: Guid option
          AcceptanceCriteria: string list }

    type DecompositionTrace =
        { EpicId: Guid
          EpicTitle: string
          StoryIds: Guid list
          SpikeIds: Guid list }

    type DecompositionResult =
        { FeatureId: string
          FeatureTitle: string
          Epics: BacklogItem list
          Stories: BacklogItem list
          Spikes: BacklogItem list
          Relationships: DecompositionTrace list }

    let private normalisePriority (label: string option) =
        match label with
        | Some value when not (String.IsNullOrWhiteSpace value) -> value.Trim().ToUpperInvariant()
        | _ -> "P3"

    let private storyPointsLabel (text: string) =
        let wordCount =
            text.Split([| ' '; '\t'; '\r'; '\n' |], StringSplitOptions.RemoveEmptyEntries)
            |> Array.length

        match wordCount with
        | count when count <= 12 -> Some "XS"
        | count when count <= 24 -> Some "S"
        | count when count <= 40 -> Some "M"
        | count when count <= 65 -> Some "L"
        | count when count <= 100 -> Some "XL"
        | count when count > 100 -> Some "XXL"
        | _ -> None

    let private itemTypeLabel = function
        | BacklogItemType.Epic -> "epic"
        | BacklogItemType.Story -> "story"
        | BacklogItemType.Spike -> "spike"

    let private assignParticipants teamRegistry tags itemType =
        match teamRegistry with
        | Some registry -> TeamConfiguration.assignParticipants registry tags itemType
        | None -> []

    let private buildItem
        teamRegistry
        title
        description
        itemType
        priority
        storyPoints
        status
        tags
        epicId
        acceptance =
        let participants = assignParticipants teamRegistry tags (itemTypeLabel itemType)

        { Id = Guid.NewGuid()
          Title = title
          Description = description
          ItemType = itemType
          Priority = priority
          StoryPoints = storyPoints
          Tags = tags
          Status = status
          Participants = participants
          EpicId = epicId
          AcceptanceCriteria = acceptance }

    let private isSpikeCandidate (text: string) =
        let lowered = text.ToLowerInvariant()
        [ "investigate"
          "research"
          "spike"
          "prototype"
          "explore"
          "analysis"
          "benchmark"
          "audit"
          "proof"
          "assessment"
          "discovery" ]
        |> List.exists lowered.Contains

    let private storiesForEpic teamRegistry (epic: BacklogItem) (userStory: SpecKitUserStory) =
        userStory.AcceptanceCriteria
        |> List.mapi (fun idx criterion ->
            let title = $"{userStory.Title} :: Story {idx + 1}"
            buildItem
                teamRegistry
                title
                criterion
                BacklogItemType.Story
                (normalisePriority userStory.Priority)
                (storyPointsLabel criterion)
                BacklogStatus.Ready
                [ "spec-kit"; epic.Title; "story" ]
                (Some epic.Id)
                [ criterion ])

    let private spikesForEdgeCases teamRegistry (edgeCases: string list) (epicOpt: BacklogItem option) =
        edgeCases
        |> List.map (fun edge ->
            let title =
                match epicOpt with
                | Some epic -> $"{epic.Title} :: Edge case coverage"
                | None -> "Edge case analysis"

            buildItem
                teamRegistry
                title
                edge
                BacklogItemType.Spike
                "P2"
                (storyPointsLabel edge)
                BacklogStatus.Ready
                [ "spec-kit"; "edge-case" ]
                (epicOpt |> Option.map (fun epic -> epic.Id))
                [ edge ])

    let private normalizeStoryTag (tag: string option) =
        tag |> Option.map (fun value -> value.Trim().ToUpperInvariant())

    let private backlogFromTasks teamRegistry (feature: SpecKitFeature) (epicByTag: Map<string, BacklogItem>) =
        feature.Tasks
        |> List.filter (fun task -> not (task.Status.Equals("done", StringComparison.OrdinalIgnoreCase)))
        |> List.fold
            (fun (stories, spikes) task ->
                let basePriority = normalisePriority task.Priority
                let description = task.Description.Trim()
                let tags =
                    [ "spec-kit"
                      feature.Id
                      task.Phase |> Option.defaultValue "Uncategorised" ]
                let status =
                    match task.Status.Trim().ToLowerInvariant() with
                    | "blocked" -> BacklogStatus.Blocked
                    | "inprogress"
                    | "in-progress"
                    | "doing" -> BacklogStatus.InProgress
                    | "ready" -> BacklogStatus.Ready
                    | "duplicate" -> BacklogStatus.Duplicate
                    | _ -> BacklogStatus.Ready

                let assignEpic item =
                    match normalizeStoryTag task.StoryTag with
                    | Some tag ->
                        match epicByTag |> Map.tryFind tag with
                        | Some epic -> { item with EpicId = Some epic.Id; Tags = item.Tags @ [ epic.Title ] }
                        | None -> item
                    | None -> item

                if isSpikeCandidate description then
                    let spike =
                        buildItem
                            teamRegistry
                            description
                            description
                            BacklogItemType.Spike
                            basePriority
                            (storyPointsLabel description)
                            status
                            (tags @ [ "spike" ])
                            None
                            []
                        |> assignEpic
                    (stories, spike :: spikes)
                else
                    let story =
                        buildItem
                            teamRegistry
                            description
                            description
                            BacklogItemType.Story
                            basePriority
                            (storyPointsLabel description)
                            status
                            tags
                            None
                            []
                        |> assignEpic
                    (story :: stories, spikes))
            ([], [])

    let private epicStatusFromStory (story: SpecKitUserStory) =
        if story.AcceptanceCriteria.IsEmpty then BacklogStatus.New else BacklogStatus.Ready

    let decomposeFeatureWithTeams (feature: SpecKitFeature) (teamRegistry: TeamConfiguration.TeamRegistry option) =
        let summary = feature.Summary

        let epics =
            if summary.UserStories.IsEmpty then
                [ buildItem
                      teamRegistry
                      summary.Title
                      "Auto-generated epic from Spec Kit summary."
                      BacklogItemType.Epic
                      (normalisePriority None)
                      (Some "Epic")
                      BacklogStatus.New
                      [ "spec-kit"; feature.Id ]
                      None
                      summary.EdgeCases ]
            else
                summary.UserStories
                |> List.mapi (fun index story ->
                    let description =
                        if story.AcceptanceCriteria.IsEmpty then
                            $"Auto-generated epic for Spec Kit user story #{index + 1}."
                        else
                            story.AcceptanceCriteria |> String.concat Environment.NewLine

                    buildItem
                        teamRegistry
                        story.Title
                        description
                        BacklogItemType.Epic
                        (normalisePriority story.Priority)
                        (Some "Epic")
                        (epicStatusFromStory story)
                        [ "spec-kit"; feature.Id; $"US{index + 1}" ]
                        None
                        story.AcceptanceCriteria)

        let epicTagMap =
            epics
            |> List.mapi (fun index epic -> $"US{index + 1}", epic)
            |> Map.ofList

        let acceptanceStories =
            summary.UserStories
            |> List.mapi (fun index story ->
                let tag = $"US{index + 1}"
                epicTagMap
                |> Map.tryFind tag
                |> Option.map (fun epic -> storiesForEpic teamRegistry epic story)
                |> Option.defaultValue [])
            |> List.collect id

        let edgeSpikes =
            match epics with
            | [] -> []
            | epic :: _ -> spikesForEdgeCases teamRegistry summary.EdgeCases (Some epic)

        let taskStories, taskSpikes = backlogFromTasks teamRegistry feature epicTagMap

        let stories =
            acceptanceStories @ taskStories
            |> List.distinctBy (fun item -> item.Id)

        let spikes =
            edgeSpikes @ taskSpikes
            |> List.distinctBy (fun item -> item.Id)

        let relationships =
            epics
            |> List.map (fun epic ->
                let storyIds =
                    stories
                    |> List.filter (fun item -> item.EpicId = Some epic.Id && item.ItemType = BacklogItemType.Story)
                    |> List.map (fun item -> item.Id)

                let spikeIds =
                    spikes
                    |> List.filter (fun item -> item.EpicId = Some epic.Id)
                    |> List.map (fun item -> item.Id)

                { EpicId = epic.Id
                  EpicTitle = epic.Title
                  StoryIds = storyIds
                  SpikeIds = spikeIds })

        { FeatureId = feature.Id
          FeatureTitle = summary.Title
          Epics = epics
          Stories = stories
          Spikes = spikes
          Relationships = relationships }

    let decomposeFeature feature =
        decomposeFeatureWithTeams feature None
