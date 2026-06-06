namespace TarsEngine.FSharp.SelfImprovement

open System
open System.IO
open System.Text
open System.Text.RegularExpressions
open Microsoft.Extensions.Logging

/// Utilities for creating and maintaining Spec Kit roadmap files under `.specify/specs`.
module RoadmapMaintenance =

    type RoadmapStory =
        { Title: string
          Priority: string
          Acceptance: string list }

    type RoadmapSpecTemplate =
        { Id: string
          Title: string
          Status: string
          Created: DateTime option
          Stories: RoadmapStory list
          EdgeCases: string list
          Metascript: string option
          Expectations: (string * string) list }

    type RoadmapSpecInfo =
        { SpecDirectory: string
          SpecPath: string
          TasksPath: string option }

    let private specsRoot () =
        Path.Combine(Environment.CurrentDirectory, ".specify", "specs")

    let private ensureDirectory (logger: ILogger) (path: string) =
        if not (Directory.Exists(path)) then
            logger.LogInformation("Creating directory {Path}", path)
            Directory.CreateDirectory(path) |> ignore

    let private writeFileIfMissing (logger: ILogger) (path: string) (content: string) =
        if File.Exists(path) then
            logger.LogDebug("Skipping generation for {Path}; file already exists.", path)
        else
            logger.LogInformation("Writing new roadmap file {Path}", path)
            File.WriteAllText(path, content, Encoding.UTF8)

    let private renderSpec (template: RoadmapSpecTemplate) =
        let created = template.Created |> Option.defaultValue DateTime.UtcNow

        let headerLines =
            [ sprintf "# Feature Specification: %s" template.Title
              ""
              "**Feature Branch**: "
              sprintf "**Created**: %s" (created.ToString("yyyy-MM-dd"))
              sprintf "**Status**: %s" template.Status
              "" ]

        let storyLines =
            template.Stories
            |> List.collect (fun story ->
                let priority = if String.IsNullOrWhiteSpace(story.Priority) then "P3" else story.Priority
                let acceptance =
                    match story.Acceptance with
                    | [] -> [ "1. Define measurable validation criteria." ]
                    | items -> items |> List.mapi (fun idx item -> sprintf "%d. %s" (idx + 1) item)
                let baseLines =
                    [ sprintf "### User Story - %s (Priority: %s)" story.Title priority
                      ""
                      "**Acceptance Scenarios**:" ]
                baseLines @ acceptance @ [ "" ])

        let edgeCaseLines =
            match template.EdgeCases with
            | [] -> [ "### Edge Cases"; "- None documented."; "" ]
            | cases ->
                let rendered = cases |> List.map (fun case -> "- " + case)
                ("### Edge Cases" :: rendered) @ [ "" ]

        let metascriptLines =
            template.Metascript
            |> Option.map (fun block ->
                [ "```metascript"
                  block.Trim()
                  "```"
                  "" ])
            |> Option.defaultValue []

        let expectationLines =
            match template.Expectations with
            | [] -> []
            | expectations ->
                let rendered = expectations |> List.map (fun (key, value) -> $"{key}={value}")
                [ "```expectations" ] @ rendered @ [ "```"; "" ]

        (headerLines @ storyLines @ edgeCaseLines @ metascriptLines @ expectationLines)
        |> String.concat Environment.NewLine
        |> fun text -> text.Trim() + Environment.NewLine

    let private renderTasks (tasks: string list option) =
        match tasks with
        | None -> None
        | Some entries ->
            let rendered =
                entries
                |> List.map (fun line ->
                    if line.TrimStart().StartsWith("- [") then line
                    else $"- [ ] {line}")
            Some(
                ([ "## Phase 1"; "" ] @ rendered)
                |> String.concat Environment.NewLine
                |> fun text -> text.Trim() + Environment.NewLine)

    let ensureRoadmap (logger: ILogger) (template: RoadmapSpecTemplate) (tasks: string list option) =
        if String.IsNullOrWhiteSpace(template.Id) then
            invalidArg (nameof template.Id) "Roadmap template requires a non-empty Id."

        let root = specsRoot ()
        ensureDirectory logger root

        let specDirectory = Path.Combine(root, template.Id)
        ensureDirectory logger specDirectory

        let specPath = Path.Combine(specDirectory, "spec.md")
        let specContent = renderSpec template
        writeFileIfMissing logger specPath specContent

        let tasksPath =
            match renderTasks tasks with
            | None -> None
            | Some content ->
                let path = Path.Combine(specDirectory, "tasks.md")
                writeFileIfMissing logger path content
                Some path

        { SpecDirectory = specDirectory
          SpecPath = specPath
          TasksPath = tasksPath }

    let setTaskStatus (logger: ILogger) (specId: string) (taskId: string) (completed: bool) =
        if String.IsNullOrWhiteSpace(specId) then
            invalidArg (nameof specId) "Spec identifier must not be empty."
        if String.IsNullOrWhiteSpace(taskId) then
            invalidArg (nameof taskId) "Task identifier must not be empty."

        let tasksPath = Path.Combine(specsRoot (), specId, "tasks.md")
        if not (File.Exists(tasksPath)) then
            logger.LogWarning("Cannot update task status because {Path} does not exist.", tasksPath)
            false
        else
            let lines = File.ReadAllLines(tasksPath)
            let pattern = Regex($@"^- \[( |x|X)\].*{Regex.Escape(taskId)}", RegexOptions.IgnoreCase)
            let mutable changed = false

            let updated =
                lines
                |> Array.map (fun line ->
                    if pattern.IsMatch(line) then
                        changed <- true
                        let statusChar = if completed then "x" else " "
                        Regex.Replace(line, "^- \[( |x|X)\]", $"- [{statusChar}]", RegexOptions.IgnoreCase)
                    else line)

            if changed then
                logger.LogInformation("Updating task {TaskId} in {Path} => completed={Completed}", taskId, tasksPath, completed)
                File.WriteAllLines(tasksPath, updated, Encoding.UTF8)
                true
            else
                logger.LogWarning("Task id {TaskId} not found in {Path}.", taskId, tasksPath)
                false
