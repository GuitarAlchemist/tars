namespace TarsEngine.TreeOfThought

open System
open System.IO
open System.Text

/// Represents the result of applying a fix
type FixApplicationResult = {
    Fix: CodeFix
    Success: bool
    ErrorMessage: string option
    BackupPath: string option
}

/// Represents the result of applying multiple fixes
type BatchFixApplicationResult = {
    Results: FixApplicationResult list
    TotalApplied: int
    TotalFailed: int
    ApplicationTime: TimeSpan
}

/// Fix application functionality for Tree of Thought reasoning
module FixApplication =
    
    /// Create a backup of a file before applying fixes
    let createBackup (filePath: string) : string option =
        try
            if File.Exists(filePath) then
                let backupPath = filePath + ".backup." + DateTime.Now.ToString("yyyyMMdd_HHmmss")
                File.Copy(filePath, backupPath)
                Some backupPath
            else
                None
        with
        | ex -> 
            None
    
    /// Apply a single fix to a file
    let applySingleFix (fix: CodeFix) (shouldCreateBackup: bool) : FixApplicationResult =
        try
            let backupPath = if shouldCreateBackup then createBackup fix.TargetFile else None
            
            if not (File.Exists(fix.TargetFile)) then
                {
                    Fix = fix
                    Success = false
                    ErrorMessage = Some $"Target file not found: {fix.TargetFile}"
                    BackupPath = backupPath
                }
            else
                let lines = File.ReadAllLines(fix.TargetFile)
                
                if fix.TargetLine <= 0 || fix.TargetLine > lines.Length then
                    {
                        Fix = fix
                        Success = false
                        ErrorMessage = Some $"Invalid line number: {fix.TargetLine}"
                        BackupPath = backupPath
                    }
                else
                    let targetLineIndex = fix.TargetLine - 1
                    let originalLine = lines.[targetLineIndex]
                    
                    // Apply the fix
                    let newLine = 
                        if String.IsNullOrEmpty(fix.OriginalText) then
                            // Insert text at specified column
                            if fix.TargetColumn <= originalLine.Length then
                                originalLine.Insert(fix.TargetColumn, fix.ReplacementText)
                            else
                                originalLine + fix.ReplacementText
                        else
                            // Replace original text with replacement text
                            originalLine.Replace(fix.OriginalText, fix.ReplacementText)
                    
                    // Update the line in the array
                    lines.[targetLineIndex] <- newLine
                    
                    // Write back to file
                    File.WriteAllLines(fix.TargetFile, lines)
                    
                    {
                        Fix = fix
                        Success = true
                        ErrorMessage = None
                        BackupPath = backupPath
                    }
        with
        | ex ->
            {
                Fix = fix
                Success = false
                ErrorMessage = Some ex.Message
                BackupPath = None
            }
    
    /// Apply multiple fixes to files
    let applyFixes (fixes: CodeFix list) (createBackups: bool) : BatchFixApplicationResult =
        let startTime = DateTime.Now
        
        let results = 
            fixes
            |> List.map (fun fix -> applySingleFix fix createBackups)
        
        let endTime = DateTime.Now
        
        let successful = results |> List.filter (fun r -> r.Success) |> List.length
        let failed = results |> List.filter (fun r -> not r.Success) |> List.length
        
        {
            Results = results
            TotalApplied = successful
            TotalFailed = failed
            ApplicationTime = endTime - startTime
        }
    
    /// Apply fixes with conflict resolution
    let applyFixesWithConflictResolution (fixes: CodeFix list) (createBackups: bool) : BatchFixApplicationResult =
        // Group fixes by file to handle conflicts
        let fixesByFile = 
            fixes
            |> List.groupBy (fun fix -> fix.TargetFile)
            |> Map.ofList
        
        let startTime = DateTime.Now
        let mutable allResults = []
        
        // Process each file separately
        for (filePath, fileFixes) in Map.toList fixesByFile do
            // Sort fixes by line number (descending) and column (descending) to avoid offset issues
            let sortedFixes = 
                fileFixes 
                |> List.sortBy (fun fix -> -fix.TargetLine, -fix.TargetColumn)
            
            // Apply fixes one by one
            for fix in sortedFixes do
                let result = applySingleFix fix createBackups
                allResults <- result :: allResults
        
        let endTime = DateTime.Now
        
        let successful = allResults |> List.filter (fun r -> r.Success) |> List.length
        let failed = allResults |> List.filter (fun r -> not r.Success) |> List.length
        
        {
            Results = List.rev allResults
            TotalApplied = successful
            TotalFailed = failed
            ApplicationTime = endTime - startTime
        }
    
    /// Restore files from backups
    let restoreFromBackups (results: FixApplicationResult list) : int =
        let mutable restoredCount = 0
        
        for result in results do
            match result.BackupPath with
            | Some backupPath when File.Exists(backupPath) ->
                try
                    File.Copy(backupPath, result.Fix.TargetFile, true)
                    File.Delete(backupPath)
                    restoredCount <- restoredCount + 1
                with
                | _ -> ()
            | _ -> ()
        
        restoredCount
    
    /// Validate that fixes were applied correctly
    let validateFixApplication (results: FixApplicationResult list) : (FixApplicationResult * bool) list =
        results
        |> List.map (fun result ->
            if result.Success then
                try
                    // Check if the file contains the expected replacement text
                    let content = File.ReadAllText(result.Fix.TargetFile)
                    let isValid = 
                        if String.IsNullOrEmpty(result.Fix.ReplacementText) then
                            // For deletions, check that original text is gone
                            not (content.Contains(result.Fix.OriginalText))
                        else
                            // For replacements/insertions, check that new text is present
                            content.Contains(result.Fix.ReplacementText)
                    
                    (result, isValid)
                with
                | _ -> (result, false)
            else
                (result, false)
        )
    
    /// Get application summary
    let getApplicationSummary (result: BatchFixApplicationResult) : string =
        let successRate = 
            if result.Results.Length > 0 then
                float result.TotalApplied / float result.Results.Length
            else 0.0
        
        $"Fix application completed in {result.ApplicationTime.TotalMilliseconds:F0}ms. " +
        $"Applied: {result.TotalApplied}, Failed: {result.TotalFailed}, Success rate: {successRate:P1}"
    
    /// Clean up backup files older than specified days
    let cleanupOldBackups (directory: string) (daysOld: int) : int =
        let mutable cleanedCount = 0
        
        try
            let cutoffDate = DateTime.Now.AddDays(-float daysOld)
            let backupFiles = Directory.GetFiles(directory, "*.backup.*", SearchOption.AllDirectories)
            
            for backupFile in backupFiles do
                let fileInfo = FileInfo(backupFile)
                if fileInfo.CreationTime < cutoffDate then
                    try
                        File.Delete(backupFile)
                        cleanedCount <- cleanedCount + 1
                    with
                    | _ -> ()
        with
        | _ -> ()
        
        cleanedCount
    
    /// Preview fixes without applying them
    let previewFixes (fixes: CodeFix list) : string list =
        fixes
        |> List.map (fun fix ->
            $"File: {fix.TargetFile}, Line: {fix.TargetLine}, Column: {fix.TargetColumn}\n" +
            $"Description: {fix.Description}\n" +
            $"Original: '{fix.OriginalText}' -> Replacement: '{fix.ReplacementText}'\n" +
            $"Confidence: {fix.Confidence:P1}, Category: {fix.Category}\n"
        )
