namespace TarsEngine.FSharp.SelfImprovement

open System
open System.IO
open YamlDotNet.Serialization
open YamlDotNet.Serialization.NamingConventions

/// Loads YAML-defined team participant specifications for autonomous agents.
module TeamConfiguration =

    [<CLIMutable>]
    type ParticipantSpec =
        { Id: string
          Role: string
          Metascript: string
          Description: string
          Capabilities: string[] }

    [<CLIMutable>]
    type AssignmentSpec =
        { Tags: string[]
          ItemTypes: string[]
          Participants: string[] }

    [<CLIMutable>]
    type TeamSpec =
        { Name: string
          Description: string
          Participants: ParticipantSpec[]
          Assignments: AssignmentSpec[] }

    [<CLIMutable>]
    type TeamConfig =
        { Teams: TeamSpec[] }

    type AgentParticipant =
        { Id: string
          Role: string
          Metascript: string option
          Description: string option
          Capabilities: string list }

    type AssignmentRule =
        { Tags: string list
          ItemTypes: string list
          Participants: AgentParticipant list }

    type TeamDefinition =
        { Name: string
          Description: string option
          Participants: AgentParticipant list
          Assignments: AssignmentRule list }

    type TeamRegistry =
        { Teams: TeamDefinition list }

    let private deserializer =
        DeserializerBuilder()
            .WithNamingConvention(CamelCaseNamingConvention.Instance)
            .IgnoreUnmatchedProperties()
            .Build()

    let private normaliseList (values: string[]) =
        if isNull values then []
        else
            values
            |> Array.toList
        |> List.map (fun value -> value.Trim())
        |> List.filter (String.IsNullOrWhiteSpace >> not)

    let private toParticipant (spec: ParticipantSpec) =
        let metascript =
            spec.Metascript
            |> Option.ofObj
            |> Option.bind (fun value ->
                let trimmed = value.Trim()
                if String.IsNullOrWhiteSpace(trimmed) then None else Some trimmed)

        let description =
            spec.Description
            |> Option.ofObj
            |> Option.bind (fun value ->
                let trimmed = value.Trim()
                if String.IsNullOrWhiteSpace(trimmed) then None else Some trimmed)

        let capabilities =
            spec.Capabilities
            |> normaliseList

        { Id = spec.Id
          Role = spec.Role
          Metascript = metascript
          Description = description
          Capabilities = capabilities }

    let private buildTeamDefinition (spec: TeamSpec) =
        let participantMap =
            (if isNull spec.Participants then [||] else spec.Participants)
            |> Array.map (fun participant -> participant.Id, toParticipant participant)
            |> Array.toList
            |> Map.ofList

        let assignments =
            (if isNull spec.Assignments then [||] else spec.Assignments)
            |> Array.toList
            |> List.map (fun assignment ->
                let referencedParticipants =
                    assignment.Participants
                    |> normaliseList
                    |> List.choose (fun participantId -> participantMap |> Map.tryFind participantId)

                { Tags = normaliseList assignment.Tags
                  ItemTypes =
                    normaliseList assignment.ItemTypes
                    |> fun items -> if List.isEmpty items then [ "any" ] else items
                  Participants = referencedParticipants })

        let teamDescription =
            spec.Description
            |> Option.ofObj
            |> Option.bind (fun value ->
                let trimmed = value.Trim()
                if String.IsNullOrWhiteSpace(trimmed) then None else Some trimmed)

        { Name = spec.Name
          Description = teamDescription
          Participants = participantMap |> Map.toList |> List.map snd
          Assignments = assignments }

    let loadFromText (yaml: string) =
        let parsed = deserializer.Deserialize<TeamConfig>(yaml)
        let teams =
            parsed.Teams
            |> Option.ofObj
            |> Option.map Array.toList
            |> Option.defaultValue []
            |> List.map buildTeamDefinition
        { Teams = teams }

    let loadFromFile (path: string) =
        if not (File.Exists(path)) then
            None
        else
            File.ReadAllText(path)
            |> loadFromText
            |> Some

    let loadFromDirectory (directory: string) =
        if not (Directory.Exists(directory)) then
            None
        else
            let files =
                Directory.GetFiles(directory, "*.yml")
                |> Array.append (Directory.GetFiles(directory, "*.yaml"))

            files
            |> Array.choose loadFromFile
            |> Array.toList
            |> function
                | [] -> None
                | registries ->
                    registries
                    |> List.collect (fun registry -> registry.Teams)
                    |> fun teams -> Some { Teams = teams }

    let private matchesTags (ruleTags: string list) (itemTags: string list) =
        if ruleTags.IsEmpty then true
        else
            ruleTags
            |> List.exists (fun tag -> itemTags |> List.exists (fun itemTag -> itemTag.Equals(tag, StringComparison.OrdinalIgnoreCase)))

    let private matchesItemType (ruleTypes: string list) (itemType: string) =
        ruleTypes
        |> List.exists (fun ruleType ->
            ruleType.Equals("any", StringComparison.OrdinalIgnoreCase)
            || ruleType.Equals(itemType, StringComparison.OrdinalIgnoreCase))

    let assignParticipants (registry: TeamRegistry) (tags: string list) (itemType: string) =
        registry.Teams
        |> List.collect (fun team ->
            team.Assignments
            |> List.filter (fun assignment -> matchesItemType assignment.ItemTypes itemType && matchesTags assignment.Tags tags)
            |> List.collect (fun assignment -> assignment.Participants))
        |> List.distinctBy (fun participant -> participant.Id)
