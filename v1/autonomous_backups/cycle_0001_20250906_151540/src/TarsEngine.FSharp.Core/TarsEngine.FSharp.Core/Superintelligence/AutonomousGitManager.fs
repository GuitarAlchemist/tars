namespace TarsEngine.FSharp.Core.Superintelligence

open System
open System.IO
open System.Diagnostics
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// Git operation result
type GitOperationResult = {
    Success: bool
    Output: string
    ErrorOutput: string
    ExitCode: int
    Command: string
    Duration: TimeSpan
}

/// Autonomous improvement branch
type ImprovementBranch = {
    Name: string
    BaseCommit: string
    Purpose: string
    CreatedAt: DateTime
    Status: string // "active", "merged", "abandoned"
}

/// Autonomous Git Manager for Tier 3 superintelligence
type AutonomousGitManager(repoPath: string, logger: ILogger<AutonomousGitManager>) =
    
    let mutable activeBranches = []
    
    /// Execute Git command safely
    let executeGitCommand (command: string) =
        task {
            let startTime = DateTime.UtcNow
            
            try
                let processInfo = ProcessStartInfo(
                    FileName = "git",
                    Arguments = command,
                    WorkingDirectory = repoPath,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                )
                
                use process = new Process(StartInfo = processInfo)
                process.Start() |> ignore
                
                let! output = process.StandardOutput.ReadToEndAsync()
                let! errorOutput = process.StandardError.ReadToEndAsync()
                
                process.WaitForExit()
                let duration = DateTime.UtcNow - startTime
                
                let result = {
                    Success = process.ExitCode = 0
                    Output = output.Trim()
                    ErrorOutput = errorOutput.Trim()
                    ExitCode = process.ExitCode
                    Command = command
                    Duration = duration
                }
                
                if result.Success then
                    logger.LogDebug("Git command succeeded: {Command} ({Duration}ms)", command, duration.TotalMilliseconds)
                else
                    logger.LogWarning("Git command failed: {Command} - {Error}", command, errorOutput)
                
                return result
                
            with
            | ex ->
                let duration = DateTime.UtcNow - startTime
                logger.LogError(ex, "Git command exception: {Command}", command)
                
                return {
                    Success = false
                    Output = ""
                    ErrorOutput = ex.Message
                    ExitCode = -1
                    Command = command
                    Duration = duration
                }
        }
    
    /// Check if repository is clean
    member _.IsRepositoryClean() =
        task {
            let! result = executeGitCommand "status --porcelain"
            return result.Success && String.IsNullOrWhiteSpace(result.Output)
        }
    
    /// Get current branch name
    member _.GetCurrentBranch() =
        task {
            let! result = executeGitCommand "branch --show-current"
            return if result.Success then Some result.Output else None
        }
    
    /// Create autonomous improvement branch
    member _.CreateImprovementBranch(purpose: string, improvementId: string) =
        task {
            logger.LogInformation("Creating autonomous improvement branch for: {Purpose}", purpose)
            
            // Ensure we're on main/master branch
            let! currentBranch = executeGitCommand "branch --show-current"
            if not currentBranch.Success then
                return Error "Failed to determine current branch"
            
            // Get current commit hash
            let! commitResult = executeGitCommand "rev-parse HEAD"
            if not commitResult.Success then
                return Error "Failed to get current commit hash"
            
            let baseCommit = commitResult.Output
            let timestamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss")
            let branchName = sprintf "autonomous/%s-%s-%s" (purpose.Replace(" ", "-").ToLower()) improvementId timestamp
            
            // Create and checkout new branch
            let! createResult = executeGitCommand (sprintf "checkout -b %s" branchName)
            if not createResult.Success then
                return Error (sprintf "Failed to create branch: %s" createResult.ErrorOutput)
            
            let branch = {
                Name = branchName
                BaseCommit = baseCommit
                Purpose = purpose
                CreatedAt = DateTime.UtcNow
                Status = "active"
            }
            
            activeBranches <- branch :: activeBranches
            
            logger.LogInformation("Created autonomous branch: {BranchName}", branchName)
            return Ok branch
        }
    
    /// Apply code changes to repository
    member _.ApplyCodeChanges(changes: (string * string) list) =
        task {
            logger.LogInformation("Applying {ChangeCount} code changes autonomously", changes.Length)
            
            let mutable appliedChanges = []
            let mutable errors = []
            
            for (filePath, content) in changes do
                try
                    let fullPath = Path.Combine(repoPath, filePath)
                    let directory = Path.GetDirectoryName(fullPath)
                    
                    // Ensure directory exists
                    if not (Directory.Exists(directory)) then
                        Directory.CreateDirectory(directory) |> ignore
                    
                    // Write file content
                    do! File.WriteAllTextAsync(fullPath, content)
                    appliedChanges <- filePath :: appliedChanges
                    
                    logger.LogDebug("Applied changes to: {FilePath}", filePath)
                    
                with
                | ex ->
                    let error = sprintf "Failed to apply changes to %s: %s" filePath ex.Message
                    errors <- error :: errors
                    logger.LogError(ex, "Failed to apply changes to {FilePath}", filePath)
            
            if errors.IsEmpty then
                return Ok (List.rev appliedChanges)
            else
                return Error (String.concat "; " errors)
        }
    
    /// Stage and commit changes
    member _.CommitChanges(message: string, author: string option) =
        task {
            logger.LogInformation("Committing autonomous changes: {Message}", message)
            
            // Stage all changes
            let! stageResult = executeGitCommand "add -A"
            if not stageResult.Success then
                return Error (sprintf "Failed to stage changes: %s" stageResult.ErrorOutput)
            
            // Check if there are changes to commit
            let! statusResult = executeGitCommand "status --porcelain"
            if statusResult.Success && String.IsNullOrWhiteSpace(statusResult.Output) then
                return Error "No changes to commit"
            
            // Prepare commit command
            let authorFlag = 
                match author with
                | Some a -> sprintf "--author=\"%s\"" a
                | None -> ""
            
            let commitCommand = sprintf "commit %s -m \"%s\"" authorFlag message
            
            // Execute commit
            let! commitResult = executeGitCommand commitCommand
            if not commitResult.Success then
                return Error (sprintf "Failed to commit changes: %s" commitResult.ErrorOutput)
            
            logger.LogInformation("Successfully committed autonomous changes")
            return Ok commitResult.Output
        }
    
    /// Run tests on current branch
    member _.RunTests(testCommand: string option) =
        task {
            let command = testCommand |> Option.defaultValue "dotnet test"
            logger.LogInformation("Running tests: {Command}", command)
            
            let startTime = DateTime.UtcNow
            
            try
                let processInfo = ProcessStartInfo(
                    FileName = "cmd",
                    Arguments = sprintf "/c %s" command,
                    WorkingDirectory = repoPath,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                )
                
                use process = new Process(StartInfo = processInfo)
                process.Start() |> ignore
                
                let! output = process.StandardOutput.ReadToEndAsync()
                let! errorOutput = process.StandardError.ReadToEndAsync()
                
                process.WaitForExit()
                let duration = DateTime.UtcNow - startTime
                
                let success = process.ExitCode = 0
                
                if success then
                    logger.LogInformation("Tests passed ({Duration}ms)", duration.TotalMilliseconds)
                else
                    logger.LogWarning("Tests failed ({Duration}ms): {Error}", duration.TotalMilliseconds, errorOutput)
                
                return {
                    Success = success
                    Output = output
                    ErrorOutput = errorOutput
                    ExitCode = process.ExitCode
                    Command = command
                    Duration = duration
                }
                
            with
            | ex ->
                let duration = DateTime.UtcNow - startTime
                logger.LogError(ex, "Test execution failed: {Command}", command)
                
                return {
                    Success = false
                    Output = ""
                    ErrorOutput = ex.Message
                    ExitCode = -1
                    Command = command
                    Duration = duration
                }
        }
    
    /// Merge branch if tests pass
    member _.MergeBranchIfValid(branch: ImprovementBranch, testCommand: string option) =
        task {
            logger.LogInformation("Validating and merging branch: {BranchName}", branch.Name)
            
            // Run tests on the branch
            let! testResult = this.RunTests(testCommand)
            if not testResult.Success then
                return Error (sprintf "Tests failed, cannot merge: %s" testResult.ErrorOutput)
            
            // Switch to main branch
            let! checkoutResult = executeGitCommand "checkout main"
            if not checkoutResult.Success then
                // Try master if main doesn't exist
                let! masterResult = executeGitCommand "checkout master"
                if not masterResult.Success then
                    return Error "Failed to checkout main/master branch"
            
            // Merge the improvement branch
            let mergeMessage = sprintf "auto: merge %s - %s" branch.Name branch.Purpose
            let! mergeResult = executeGitCommand (sprintf "merge %s -m \"%s\"" branch.Name mergeMessage)
            if not mergeResult.Success then
                return Error (sprintf "Failed to merge branch: %s" mergeResult.ErrorOutput)
            
            // Delete the merged branch
            let! deleteResult = executeGitCommand (sprintf "branch -d %s" branch.Name)
            if deleteResult.Success then
                logger.LogInformation("Deleted merged branch: {BranchName}", branch.Name)
            
            // Update branch status
            activeBranches <- 
                activeBranches 
                |> List.map (fun b -> if b.Name = branch.Name then { b with Status = "merged" } else b)
            
            logger.LogInformation("Successfully merged autonomous improvement: {BranchName}", branch.Name)
            return Ok mergeResult.Output
        }
    
    /// Rollback branch if validation fails
    member _.RollbackBranch(branch: ImprovementBranch, reason: string) =
        task {
            logger.LogWarning("Rolling back branch {BranchName}: {Reason}", branch.Name, reason)
            
            // Switch to main branch
            let! checkoutResult = executeGitCommand "checkout main"
            if not checkoutResult.Success then
                let! masterResult = executeGitCommand "checkout master"
                if not masterResult.Success then
                    return Error "Failed to checkout main/master branch for rollback"
            
            // Delete the failed branch
            let! deleteResult = executeGitCommand (sprintf "branch -D %s" branch.Name)
            if not deleteResult.Success then
                logger.LogWarning("Failed to delete rolled back branch: {Error}", deleteResult.ErrorOutput)
            
            // Update branch status
            activeBranches <- 
                activeBranches 
                |> List.map (fun b -> if b.Name = branch.Name then { b with Status = "abandoned" } else b)
            
            logger.LogInformation("Rolled back autonomous branch: {BranchName}", branch.Name)
            return Ok "Branch rolled back successfully"
        }
    
    /// Execute complete autonomous improvement workflow
    member this.ExecuteAutonomousImprovement(purpose: string, improvementId: string, changes: (string * string) list, testCommand: string option) =
        task {
            logger.LogInformation("Starting autonomous improvement workflow: {Purpose}", purpose)
            
            try
                // Ensure repository is clean
                let! isClean = this.IsRepositoryClean()
                if not isClean then
                    return Error "Repository is not clean, cannot start autonomous improvement"
                
                // Create improvement branch
                let! branchResult = this.CreateImprovementBranch(purpose, improvementId)
                match branchResult with
                | Error err -> return Error err
                | Ok branch ->
                    
                    // Apply code changes
                    let! changesResult = this.ApplyCodeChanges(changes)
                    match changesResult with
                    | Error err -> 
                        let! _ = this.RollbackBranch(branch, err)
                        return Error err
                    | Ok appliedFiles ->
                        
                        // Commit changes
                        let commitMessage = sprintf "auto: %s [%s] — autonomous improvement iteration" purpose improvementId
                        let! commitResult = this.CommitChanges(commitMessage, Some "TARS Autonomous System <tars@autonomous.ai>")
                        match commitResult with
                        | Error err ->
                            let! _ = this.RollbackBranch(branch, err)
                            return Error err
                        | Ok commitOutput ->
                            
                            // Validate with tests and merge if successful
                            let! mergeResult = this.MergeBranchIfValid(branch, testCommand)
                            match mergeResult with
                            | Error err ->
                                let! _ = this.RollbackBranch(branch, err)
                                return Error err
                            | Ok mergeOutput ->
                                
                                logger.LogInformation("Autonomous improvement completed successfully: {Purpose}", purpose)
                                return Ok {|
                                    Branch = branch
                                    AppliedFiles = appliedFiles
                                    CommitOutput = commitOutput
                                    MergeOutput = mergeOutput
                                |}
            with
            | ex ->
                logger.LogError(ex, "Autonomous improvement workflow failed: {Purpose}", purpose)
                return Error (sprintf "Workflow exception: %s" ex.Message)
        }
    
    /// Get autonomous improvement statistics
    member _.GetImprovementStatistics() =
        let totalBranches = activeBranches.Length
        let mergedCount = activeBranches |> List.filter (fun b -> b.Status = "merged") |> List.length
        let abandonedCount = activeBranches |> List.filter (fun b -> b.Status = "abandoned") |> List.length
        let activeCount = activeBranches |> List.filter (fun b -> b.Status = "active") |> List.length
        
        {|
            TotalBranches = totalBranches
            MergedBranches = mergedCount
            AbandonedBranches = abandonedCount
            ActiveBranches = activeCount
            SuccessRate = if totalBranches > 0 then float mergedCount / float totalBranches else 0.0
            RecentBranches = activeBranches |> List.sortByDescending (fun b -> b.CreatedAt) |> List.truncate 5
        |}
    
    /// Initialize the system
    member _.Initialize() =
        logger.LogInformation("Autonomous Git Manager initialized for repository: {RepoPath}", repoPath)
