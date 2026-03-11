namespace Tars.Tools.Standard

open System
open System.IO
open System.Diagnostics
open Tars.Tools

module GitTools =

    /// Helper to run git commands
    let private runGit (workDir: string) (args: string) =
        task {
            try
                let psi = ProcessStartInfo()
                psi.FileName <- "git"
                psi.Arguments <- args
                psi.WorkingDirectory <- workDir
                psi.RedirectStandardOutput <- true
                psi.RedirectStandardError <- true
                psi.UseShellExecute <- false
                psi.CreateNoWindow <- true

                use proc = Process.Start(psi)
                let! stdout = proc.StandardOutput.ReadToEndAsync()
                let! stderr = proc.StandardError.ReadToEndAsync()
                proc.WaitForExit()

                if proc.ExitCode = 0 then
                    return Ok(stdout.Trim())
                else
                    return Error(stderr.Trim())
            with ex ->
                return Error(ex.Message)
        }

    let private parseWriteCodeArgs (args: string) =
        try
            let doc = System.Text.Json.JsonDocument.Parse(args)
            let root = doc.RootElement
            let path = root.GetProperty("path").GetString()
            let content = root.GetProperty("content").GetString()
            Ok(path, content)
        with ex ->
            Error ex.Message

    [<TarsToolAttribute("write_code",
                        "Writes code to a file. Creates directories if needed. Input JSON: { \"path\": \"relative/path.fs\", \"content\": \"code content\" }")>]
    let writeCode (args: string) =
        task {
            match parseWriteCodeArgs args with
            | Error msg -> return $"write_code error: {msg}"
            | Ok(path, content) ->
                try
                    let fullPath = Path.GetFullPath(path)
                    let dir = Path.GetDirectoryName(fullPath)

                    if not (Directory.Exists dir) then
                        Directory.CreateDirectory(dir) |> ignore

                    // Backup existing file if it exists
                    if File.Exists fullPath then
                        let backup = fullPath + ".bak"
                        File.Copy(fullPath, backup, true)

                    File.WriteAllText(fullPath, content)
                    printfn $"📝 TARS wrote code to: {fullPath}"
                    return $"Successfully wrote {content.Length} characters to {fullPath}"
                with ex ->
                    return $"write_code error: {ex.Message}"
        }

    let private parseGitCommitArgs (args: string) =
        try
            let doc = System.Text.Json.JsonDocument.Parse(args)
            let root = doc.RootElement

            let mutable msgProp = Unchecked.defaultof<System.Text.Json.JsonElement>

            let message =
                if root.TryGetProperty("message", &msgProp) then
                    msgProp.GetString()
                else
                    "TARS: Auto-generated improvement"

            let mutable pathsProp = Unchecked.defaultof<System.Text.Json.JsonElement>

            let paths =
                if root.TryGetProperty("paths", &pathsProp) then
                    [| for elem in pathsProp.EnumerateArray() -> elem.GetString() |]
                else
                    [| "." |]

            Ok(message, paths)
        with ex ->
            Error ex.Message

    [<TarsToolAttribute("git_commit",
                        "Stages and commits files. Input JSON: { \"message\": \"commit message\", \"paths\": [\"file1.fs\", \"file2.fs\"] }")>]
    let gitCommit (args: string) =
        task {
            match parseGitCommitArgs args with
            | Error msg -> return $"git_commit error: {msg}"
            | Ok(message, paths) ->
                try
                    let workDir = Directory.GetCurrentDirectory()

                    // Stage files
                    for path in paths do
                        let! stageResult = runGit workDir $"add \"{path}\""

                        match stageResult with
                        | Error e -> printfn $"Warning: git add {path}: {e}"
                        | Ok _ -> ()

                    // Commit
                    let! commitResult = runGit workDir $"commit -m \"[TARS] {message}\""

                    match commitResult with
                    | Ok output ->
                        printfn $"🔧 TARS committed: {message}"
                        return $"Committed successfully: {output}"
                    | Error e -> return $"git_commit: {e}"
                with ex ->
                    return $"git_commit error: {ex.Message}"
        }

    [<TarsToolAttribute("git_status", "Shows git status of the repository. No input required.")>]
    let gitStatus (_: string) =
        task {
            try
                let workDir = Directory.GetCurrentDirectory()
                let! result = runGit workDir "status --short"

                match result with
                | Ok output ->
                    if String.IsNullOrWhiteSpace(output) then
                        return "Working directory clean"
                    else
                        return output
                | Error e -> return $"git status error: {e}"
            with ex ->
                return $"git status error: {ex.Message}"
        }

    [<TarsToolAttribute("git_diff", "Shows unstaged changes. Input JSON: { \"path\": \"optional/file.fs\" }")>]
    let gitDiff (args: string) =
        task {
            try
                let workDir = Directory.GetCurrentDirectory()

                let path = ToolHelpers.parseStringArg args "path"

                let! result = runGit workDir $"diff {path}"

                match result with
                | Ok output ->
                    if String.IsNullOrWhiteSpace(output) then
                        return "No changes"
                    else if
                        // Truncate large diffs
                        output.Length > 5000
                    then
                        return output.Substring(0, 5000) + "\n... (truncated)"
                    else
                        return output
                | Error e -> return $"git diff error: {e}"
            with ex ->
                return $"git diff error: {ex.Message}"
        }
