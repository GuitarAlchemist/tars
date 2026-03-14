/// Git Integration - Per-project git operations
/// Enables projects to track changes and integrate with remote repositories
module Tars.Core.GitIntegration

open System
open System.Diagnostics
open System.IO
open System.Threading.Tasks

// ============================================================================
// Git Types
// ============================================================================

/// Git operation result
type GitResult =
    | Success of output: string
    | Error of message: string

/// Git commit info
type GitCommit =
    { Hash: string
      ShortHash: string
      Author: string
      Date: DateTime
      Message: string }

/// Git branch info
type GitBranch =
    { Name: string
      IsRemote: bool
      IsCurrent: bool }

/// Git status entry
type GitStatusEntry =
    { Path: string
      Status: string } // "M" modified, "A" added, "D" deleted, "?" untracked

/// Git repository state
type GitRepoState =
    { IsRepo: bool
      CurrentBranch: string option
      HasUncommittedChanges: bool
      RemoteUrl: string option }

// ============================================================================
// Git Command Executor
// ============================================================================

/// Execute a git command and return result
let executeGit (workingDir: string) (args: string) : Task<GitResult> =
    task {
        try
            let psi = ProcessStartInfo()
            psi.FileName <- "git"
            psi.Arguments <- args
            psi.WorkingDirectory <- workingDir
            psi.RedirectStandardOutput <- true
            psi.RedirectStandardError <- true
            psi.UseShellExecute <- false
            psi.CreateNoWindow <- true

            use proc = Process.Start(psi)
            let! output = proc.StandardOutput.ReadToEndAsync()
            let! error = proc.StandardError.ReadToEndAsync()
            do! proc.WaitForExitAsync()

            if proc.ExitCode = 0 then
                return Success(output.Trim())
            else
                return
                    Error(
                        if String.IsNullOrEmpty error then
                            output.Trim()
                        else
                            error.Trim()
                    )
        with ex ->
            return Error ex.Message
    }

// ============================================================================
// Git Manager
// ============================================================================

type GitManager(workingDir: string) =

    /// Check if directory is a git repository
    member _.IsRepository() : Task<bool> =
        task {
            match! executeGit workingDir "rev-parse --is-inside-work-tree" with
            | Success "true" -> return true
            | _ -> return false
        }

    /// Initialize a new git repository
    member _.Init() : Task<GitResult> = executeGit workingDir "init"

    /// Clone a repository
    member _.Clone(url: string, targetDir: string) : Task<GitResult> =
        let parentDir = Path.GetDirectoryName(targetDir)
        let dirName = Path.GetFileName(targetDir)
        executeGit parentDir $"clone {url} {dirName}"

    /// Get current branch
    member _.GetCurrentBranch() : Task<string option> =
        task {
            match! executeGit workingDir "rev-parse --abbrev-ref HEAD" with
            | Success branch -> return Some branch
            | Error _ -> return None
        }

    /// Get status
    member _.GetStatus() : Task<GitStatusEntry list> =
        task {
            match! executeGit workingDir "status --porcelain" with
            | Success output when not (String.IsNullOrWhiteSpace output) ->
                return
                    output.Split('\n', StringSplitOptions.RemoveEmptyEntries)
                    |> Array.map (fun line ->
                        { Path = line.Substring(3).Trim()
                          Status = line.Substring(0, 2).Trim() })
                    |> Array.toList
            | _ -> return []
        }

    /// Check for uncommitted changes
    member this.HasChanges() : Task<bool> =
        task {
            let! status = this.GetStatus()
            return not status.IsEmpty
        }

    /// Stage all changes
    member _.StageAll() : Task<GitResult> = executeGit workingDir "add -A"

    /// Stage specific files
    member _.Stage(paths: string list) : Task<GitResult> =
        let pathsArg = paths |> List.map (sprintf "\"%s\"") |> String.concat " "
        executeGit workingDir $"add {pathsArg}"

    /// Commit with message
    member _.Commit(message: string) : Task<GitResult> =
        executeGit workingDir $"commit -m \"{message}\""

    /// Create a new branch
    member _.CreateBranch(name: string) : Task<GitResult> =
        executeGit workingDir $"checkout -b {name}"

    /// Switch to branch
    member _.Checkout(branch: string) : Task<GitResult> =
        executeGit workingDir $"checkout {branch}"

    /// Push to remote
    member _.Push(remote: string, branch: string) : Task<GitResult> =
        executeGit workingDir $"push {remote} {branch}"

    /// Pull from remote
    member _.Pull(remote: string, branch: string) : Task<GitResult> =
        executeGit workingDir $"pull {remote} {branch}"

    /// Get recent commits
    member _.GetRecentCommits(count: int) : Task<GitCommit list> =
        task {
            match! executeGit workingDir $"log -n {count} --pretty=format:\"%%H|%%h|%%an|%%ai|%%s\"" with
            | Success output when not (String.IsNullOrWhiteSpace output) ->
                return
                    output.Split('\n', StringSplitOptions.RemoveEmptyEntries)
                    |> Array.map (fun line ->
                        let parts = line.Split('|')

                        if parts.Length >= 5 then
                            Some
                                { Hash = parts.[0]
                                  ShortHash = parts.[1]
                                  Author = parts.[2]
                                  Date = DateTime.Parse(parts.[3])
                                  Message = parts.[4] }
                        else
                            None)
                    |> Array.choose id
                    |> Array.toList
            | _ -> return []
        }

    /// Get remote URL
    member _.GetRemoteUrl(remote: string) : Task<string option> =
        task {
            match! executeGit workingDir $"remote get-url {remote}" with
            | Success url -> return Some url
            | Error _ -> return None
        }

    /// Get full repo state
    member this.GetRepoState() : Task<GitRepoState> =
        task {
            let! isRepo = this.IsRepository()

            if not isRepo then
                return
                    { IsRepo = false
                      CurrentBranch = None
                      HasUncommittedChanges = false
                      RemoteUrl = None }
            else
                let! branch = this.GetCurrentBranch()
                let! hasChanges = this.HasChanges()
                let! remoteUrl = this.GetRemoteUrl("origin")

                return
                    { IsRepo = true
                      CurrentBranch = branch
                      HasUncommittedChanges = hasChanges
                      RemoteUrl = remoteUrl }
        }

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a git manager for a path
let createManager path = GitManager(path)

/// Quick init for new project
let initRepo path =
    let manager = GitManager(path)
    manager.Init() |> Async.AwaitTask |> Async.RunSynchronously

/// Quick commit all changes
let commitAll path message =
    task {
        let manager = GitManager(path)
        let! _ = manager.StageAll()
        return! manager.Commit(message)
    }
    |> Async.AwaitTask
    |> Async.RunSynchronously
