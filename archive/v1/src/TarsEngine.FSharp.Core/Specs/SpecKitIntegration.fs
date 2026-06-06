namespace TarsEngine.FSharp.Core.Specs

open System
open System.IO
open System.Text
open System.Text.Json
open System.Text.RegularExpressions

/// Simplified representation of a Spec Kit style feature specification.
type SpecKitUserStory =
    { Title: string
      Priority: string option
      AcceptanceCriteria: string list }

type SpecKitSummary =
    { Title: string
      Status: string option
      FeatureBranch: string option
      Created: string option
      UserStories: SpecKitUserStory list
      EdgeCases: string list }

module SpecKitParser =

    let private trimAfter (prefix: string) (line: string) =
        let index = line.IndexOf(prefix, StringComparison.OrdinalIgnoreCase)
        if index < 0 then None
        else
            let value = line.Substring(index + prefix.Length).Trim()
            if String.IsNullOrWhiteSpace(value) then None else Some value

    let private tryMatch (pattern: string) (line: string) =
        let m = Regex.Match(line, pattern, RegexOptions.IgnoreCase)
        if m.Success && m.Groups.Count > 1 then
            let value = m.Groups.[1].Value.Trim()
            if String.IsNullOrWhiteSpace(value) then None else Some value
        else
            None

    let private parseMarkdown (content: string) =
        let lines =
            content.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)
            |> Array.map (fun line -> line.Trim())

        let title =
            lines
            |> Array.tryPick (fun line ->
                if line.StartsWith("# Feature Specification", StringComparison.OrdinalIgnoreCase) then
                    line |> trimAfter ":" |> Option.orElseWith (fun () ->
                        // Handle headline without colon: "# Feature Specification – Foo"
                        line
                        |> tryMatch @"#\s*Feature\s*Specification[:\-–]\s*(.+)")
                else
                    None)
            |> Option.defaultValue "Unnamed Feature"

        let featureBranch =
            lines
            |> Array.tryPick (fun line ->
                if line.StartsWith("**Feature Branch**", StringComparison.OrdinalIgnoreCase) then
                    trimAfter ":" line
                else None)

        let created =
            lines
            |> Array.tryPick (fun line ->
                if line.StartsWith("**Created**", StringComparison.OrdinalIgnoreCase) then
                    trimAfter ":" line
                else None)

        let status =
            lines
            |> Array.tryPick (fun line ->
                if line.StartsWith("**Status**", StringComparison.OrdinalIgnoreCase) then
                    trimAfter ":" line
                else None)

        let rec collectUserStories index stories =
            if index >= lines.Length then stories
            else
                let line = lines.[index]
                if line.StartsWith("### User Story", StringComparison.OrdinalIgnoreCase) then
                    let title = line |> trimAfter "-" |> Option.defaultValue "Untitled User Story"
                    let priority = line |> tryMatch @"\(Priority:\s*(.+?)\)"

                    let rec gatherAcceptance i acc =
                        if i >= lines.Length then (List.rev acc, i)
                        else
                            let l = lines.[i]
                            if l.StartsWith("### ", StringComparison.OrdinalIgnoreCase)
                               || l.StartsWith("## ", StringComparison.OrdinalIgnoreCase) then
                                (List.rev acc, i)
                            elif Regex.IsMatch(l, @"^\d+\.\s") then
                                gatherAcceptance (i + 1) (l :: acc)
                            else
                                gatherAcceptance (i + 1) acc

                    let acceptance, nextIndex = gatherAcceptance (index + 1) []
                    let story =
                        { Title = title
                          Priority = priority
                          AcceptanceCriteria = acceptance }
                    collectUserStories nextIndex (story :: stories)
                else
                    collectUserStories (index + 1) stories

        let userStories = collectUserStories 0 [] |> List.rev

        let edgeCases =
            let startIndex =
                lines
                |> Array.tryFindIndex (fun line -> line.StartsWith("### Edge Cases", StringComparison.OrdinalIgnoreCase))
            match startIndex with
            | None -> []
            | Some idx ->
                let rec gather i acc =
                    if i >= lines.Length then List.rev acc
                    else
                        let line = lines.[i]
                        if line.StartsWith("## ", StringComparison.OrdinalIgnoreCase) then
                            List.rev acc
                        elif line.StartsWith("- ") then
                            gather (i + 1) (line.Substring(2).Trim() :: acc)
                        elif Regex.IsMatch(line, @"^\d+\.\s") then
                            gather (i + 1) (Regex.Replace(line, @"^\d+\.\s*", "").Trim() :: acc)
                        else
                            gather (i + 1) acc

                gather (idx + 1) []

        { Title = title
          Status = status
          FeatureBranch = featureBranch
          Created = created
          UserStories = userStories
          EdgeCases = edgeCases }

    let private parseJson (content: string) =
        use document = JsonDocument.Parse(content)
        let root = document.RootElement

        let tryProperty (element: JsonElement) (name: string) =
            let mutable property = Unchecked.defaultof<JsonElement>
            if element.TryGetProperty(name, &property) then Some property else None

        let tryString name =
            tryProperty root name
            |> Option.filter (fun prop -> prop.ValueKind = JsonValueKind.String)
            |> Option.bind (fun prop -> prop.GetString() |> Option.ofObj)

        let userStories =
            match tryProperty root "userStories" with
            | Some prop when prop.ValueKind = JsonValueKind.Array ->
                prop.EnumerateArray()
                |> Seq.choose (fun story ->
                    if story.ValueKind <> JsonValueKind.Object then None
                    else
                        let title =
                            tryProperty story "title"
                            |> Option.filter (fun v -> v.ValueKind = JsonValueKind.String)
                            |> Option.bind (fun v -> v.GetString() |> Option.ofObj)

                        match title with
                        | None -> None
                        | Some actualTitle when String.IsNullOrWhiteSpace(actualTitle) -> None
                        | Some actualTitle ->
                            let priority =
                                tryProperty story "priority"
                                |> Option.filter (fun v -> v.ValueKind = JsonValueKind.String)
                                |> Option.bind (fun v -> v.GetString() |> Option.ofObj)

                            let acceptance =
                                match tryProperty story "acceptance" with
                                | Some v when v.ValueKind = JsonValueKind.Array ->
                                    v.EnumerateArray()
                                    |> Seq.choose (fun item ->
                                        if item.ValueKind = JsonValueKind.String then
                                            item.GetString() |> Option.ofObj
                                        else
                                            None)
                                    |> Seq.toList
                                | _ -> []

                            Some
                                { Title = actualTitle
                                  Priority = priority
                                  AcceptanceCriteria = acceptance })
                |> Seq.toList
            | _ -> []

        let edgeCases =
            match tryProperty root "edgeCases" with
            | Some prop when prop.ValueKind = JsonValueKind.Array ->
                prop.EnumerateArray()
                |> Seq.choose (fun item ->
                    if item.ValueKind = JsonValueKind.String then item.GetString() |> Option.ofObj else None)
                |> Seq.toList
            | _ -> []

        { Title = tryString "title" |> Option.defaultValue "Unnamed Feature"
          Status = tryString "status"
          FeatureBranch = tryString "featureBranch"
          Created = tryString "created"
          UserStories = userStories
          EdgeCases = edgeCases }

    let parse (content: string) (extension: string) =
        if String.Equals(extension, ".json", StringComparison.OrdinalIgnoreCase) then
            parseJson content
        else
            parseMarkdown content

    let loadFromFile (path: string) =
        let extension = Path.GetExtension(path)
        let content = File.ReadAllText(path, Encoding.UTF8)
        parse content extension
