namespace Tars.Cortex

open System
open System.IO

/// Bridge between TARS internal execution and Ralph Loop mechanism.
/// Manages Ralph loop state files, enabling TARS to programmatically
/// start/stop/check Ralph loops for iterative self-improvement.
module RalphBridge =

    /// Ralph loop state parsed from .ralph-loop.local.md
    type RalphState =
        { Active: bool
          Iteration: int
          MaxIterations: int option
          CompletionPromise: string option
          SessionId: string option
          StartedAt: DateTime option
          Prompt: string }

    let private stateFileName = ".claude/ralph-loop.local.md"

    /// Find the .claude directory, walking up from the given path.
    let private findClaudeDir (startDir: string) =
        let rec walk dir =
            let candidate = Path.Combine(dir, ".claude")
            if Directory.Exists(candidate) then Some candidate
            else
                let parent = Directory.GetParent(dir)
                if isNull parent then None
                else walk parent.FullName
        walk startDir

    /// Get the Ralph state file path for the current project.
    let stateFilePath (projectDir: string) =
        match findClaudeDir projectDir with
        | Some claudeDir -> Path.Combine(Path.GetDirectoryName(claudeDir), stateFileName)
        | None -> Path.Combine(projectDir, stateFileName)

    /// Check if a Ralph loop is currently active.
    let isActive (projectDir: string) : bool =
        let path = stateFilePath projectDir
        File.Exists(path)

    /// Parse a Ralph loop state file.
    let parseState (content: string) : RalphState =
        let lines = content.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)
        let mutable inFrontmatter = false
        let mutable active = false
        let mutable iteration = 0
        let mutable maxIter = None
        let mutable promise = None
        let mutable sessionId = None
        let mutable startedAt = None
        let mutable promptLines = ResizeArray<string>()
        let mutable pastFrontmatter = false

        for line in lines do
            let trimmed = line.Trim()
            if trimmed = "---" then
                if inFrontmatter then
                    inFrontmatter <- false
                    pastFrontmatter <- true
                else
                    inFrontmatter <- true
            elif inFrontmatter then
                let parts = trimmed.Split([| ':' |], 2)
                if parts.Length = 2 then
                    let key = parts.[0].Trim()
                    let value = parts.[1].Trim().Trim('"')
                    match key with
                    | "active" -> active <- value = "true"
                    | "iteration" -> iteration <- try int value with _ -> 0
                    | "max_iterations" ->
                        match value with
                        | "null" | "" -> ()
                        | v -> maxIter <- try Some (int v) with _ -> None
                    | "completion_promise" ->
                        match value with
                        | "null" | "" -> ()
                        | v -> promise <- Some v
                    | "session_id" ->
                        if value <> "null" && value <> "" then
                            sessionId <- Some value
                    | "started_at" ->
                        match DateTime.TryParse(value) with
                        | true, dt -> startedAt <- Some dt
                        | _ -> ()
                    | _ -> ()
            elif pastFrontmatter then
                promptLines.Add(line)

        { Active = active
          Iteration = iteration
          MaxIterations = maxIter
          CompletionPromise = promise
          SessionId = sessionId
          StartedAt = startedAt
          Prompt = String.Join("\n", promptLines).Trim() }

    /// Read the current Ralph loop state, if any.
    let readState (projectDir: string) : RalphState option =
        let path = stateFilePath projectDir
        if File.Exists(path) then
            let content = File.ReadAllText(path)
            Some (parseState content)
        else
            None

    /// Write a Ralph loop state file to start a loop.
    let startLoop
        (projectDir: string)
        (prompt: string)
        (maxIterations: int option)
        (completionPromise: string option)
        : unit =
        let path = stateFilePath projectDir
        let dir = Path.GetDirectoryName(path)
        if not (Directory.Exists(dir)) then
            Directory.CreateDirectory(dir) |> ignore

        let maxIterStr =
            match maxIterations with
            | Some n -> string n
            | None -> "null"
        let promiseStr =
            match completionPromise with
            | Some p -> sprintf "\"%s\"" p
            | None -> "null"

        let lines =
            [ "---"
              "active: true"
              "iteration: 1"
              "session_id: null"
              sprintf "max_iterations: %s" maxIterStr
              sprintf "completion_promise: %s" promiseStr
              sprintf "started_at: \"%s\"" (DateTime.UtcNow.ToString("o"))
              "---"
              ""
              prompt
              "" ]
        let content = String.Join("\n", lines)
        File.WriteAllText(path, content)

    /// Stop an active Ralph loop by removing the state file.
    let stopLoop (projectDir: string) : bool =
        let path = stateFilePath projectDir
        if File.Exists(path) then
            File.Delete(path)
            true
        else
            false

    /// Increment the iteration counter in an active loop.
    let incrementIteration (projectDir: string) : int option =
        match readState projectDir with
        | Some state when state.Active ->
            let newIteration = state.Iteration + 1
            let path = stateFilePath projectDir
            let content = File.ReadAllText(path)
            let updated =
                content.Replace(
                    sprintf "iteration: %d" state.Iteration,
                    sprintf "iteration: %d" newIteration)
            File.WriteAllText(path, updated)
            Some newIteration
        | _ -> None

    /// Check if the loop should terminate based on iteration count.
    let shouldTerminate (state: RalphState) : bool =
        match state.MaxIterations with
        | Some max -> state.Iteration >= max
        | None -> false

    /// Generate a TARS-specific Ralph prompt from meta-cognitive analysis.
    let generateTarsPrompt
        (gaps: Tars.Core.MetaCognition.CapabilityGap list)
        (focus: string option)
        : string =
        let gapSection =
            if gaps.IsEmpty then
                "No specific gaps detected. Look for improvements in test coverage and code quality."
            else
                gaps
                |> List.truncate 5
                |> List.map (fun g ->
                    sprintf "- **%s**: %.0f%% failure rate (%d samples) — %A"
                        g.Domain (g.FailureRate * 100.0) g.SampleSize g.SuggestedRemedy)
                |> String.concat "\n"

        let focusSection =
            match focus with
            | Some f -> sprintf "\n## Focus Area\n%s\n" f
            | None -> ""

        String.Join("\n",
            [ "# TARS Self-Improvement Iteration"
              ""
              "## Current Capability Gaps"
              gapSection
              focusSection
              "## Protocol"
              "1. Build: `dotnet build` — fix any errors"
              "2. Test: `dotnet test` — all tests must pass"
              "3. Pick the highest-priority gap above"
              "4. Make the minimal fix"
              "5. Add a test proving the fix"
              "6. Run `dotnet run --project src/Tars.Interface.Cli -- meta analyze` to verify improvement"
              ""
              "## Completion"
              "When all gaps have failure rates below 30% and all tests pass:"
              "<promise>TARS GAPS RESOLVED</promise>"
              "" ])

    /// Create a Ralph prompt for a generic TARS task.
    let generateTaskPrompt
        (goal: string)
        (completionCriteria: string)
        (promise: string)
        : string =
        String.Join("\n",
            [ "# TARS Iterative Task"
              ""
              "## Goal"
              goal
              ""
              "## Success Criteria"
              completionCriteria
              ""
              "## Protocol"
              "1. Build and test first: `dotnet build && dotnet test`"
              "2. Work toward the goal incrementally"
              "3. Verify your changes don't break anything"
              "4. Each iteration should make measurable progress"
              ""
              "## Completion"
              "When the success criteria are met:"
              sprintf "<promise>%s</promise>" promise
              "" ])
