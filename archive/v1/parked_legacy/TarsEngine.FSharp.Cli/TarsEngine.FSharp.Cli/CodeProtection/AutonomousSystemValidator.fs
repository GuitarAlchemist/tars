namespace TarsEngine.FSharp.Cli.CodeProtection

open System
open System.IO
open System.Text.RegularExpressions
open System.Text.Json
open System.Collections.Generic

/// Validator for autonomous system modifications to prevent corruption
module AutonomousSystemValidator =
    
    /// Configuration for autonomous system validation
    type ValidationConfig = {
        MaxFilesPerOperation: int
        MaxLinesPerFile: int
        AllowedFileExtensions: string list
        ForbiddenPatterns: string list
        RequiredTestCoverage: float
        MaxExecutionTimeMinutes: int
    }
    
    /// Default validation configuration
    let defaultConfig = {
        MaxFilesPerOperation = 10
        MaxLinesPerFile = 500
        AllowedFileExtensions = [".fs"; ".fsx"; ".md"; ".txt"; ".json"; ".yml"; ".yaml"]
        ForbiddenPatterns = [
            @"TODO:\s*Implement\s+real\s+functionality"
            @"//\s*REAL:\s*Implement\s+actual\s+logic\s+here"
            @"//\s*HONEST:\s*Cannot\s+generate"
            @"NotImplementedException"
            @"throw\s+new\s+Exception"
            @"System\.Environment\.Exit"
            @"Process\.Start"
            @"File\.Delete.*\.exe"
            @"Registry\."
            @"unsafe\s*{"
        ]
        RequiredTestCoverage = 0.8
        MaxExecutionTimeMinutes = 30
    }
    
    /// Result of autonomous operation validation
    type ValidationResult = {
        IsValid: bool
        Errors: string list
        Warnings: string list
        ModifiedFiles: string list
        TestCoverage: float option
        ExecutionTime: TimeSpan option
    }
    
    /// Validate a proposed file modification
    let validateFileModification (config: ValidationConfig) (filePath: string) (originalContent: string) (newContent: string) : ValidationResult =
        let errors = ResizeArray<string>()
        let warnings = ResizeArray<string>()
        
        // Check file extension
        let extension = Path.GetExtension(filePath).ToLowerInvariant()
        if not (config.AllowedFileExtensions |> List.contains extension) then
            errors.Add($"File extension '{extension}' not allowed for autonomous modification")
        
        // Check file size
        let newLines = newContent.Split('\n').Length
        if newLines > config.MaxLinesPerFile then
            errors.Add($"File too large: {newLines} lines (max: {config.MaxLinesPerFile})")
        
        // Check for forbidden patterns
        config.ForbiddenPatterns
        |> List.iter (fun pattern ->
            if Regex.IsMatch(newContent, pattern, RegexOptions.IgnoreCase) then
                errors.Add($"Forbidden pattern detected: {pattern}")
        )
        
        // Check for suspicious changes
        let originalLines = originalContent.Split('\n')
        let newContentLines = newContent.Split('\n')
        let deletedLines = originalLines.Length - newContentLines.Length
        
        if deletedLines > originalLines.Length / 2 then
            warnings.Add($"Large deletion detected: {deletedLines} lines removed")
        
        // Check for compilation-breaking patterns
        let compilationBreakers = [
            @"^\s*module\s+\w+\s*$"  // Module declarations
            @"^\s*namespace\s+\w+"   // Namespace declarations
            @"^\s*open\s+\w+"        // Open statements
            @"^\s*type\s+\w+\s*="    // Type definitions
        ]
        
        compilationBreakers
        |> List.iter (fun pattern ->
            let originalMatches = Regex.Matches(originalContent, pattern, RegexOptions.Multiline).Count
            let newMatches = Regex.Matches(newContent, pattern, RegexOptions.Multiline).Count
            
            if newMatches < originalMatches then
                errors.Add($"Critical code structure removed: {pattern}")
        )
        
        {
            IsValid = errors.Count = 0
            Errors = errors |> Seq.toList
            Warnings = warnings |> Seq.toList
            ModifiedFiles = [filePath]
            TestCoverage = None
            ExecutionTime = None
        }
    
    /// Validate an autonomous operation batch
    let validateOperationBatch (config: ValidationConfig) (operations: (string * string * string) list) : ValidationResult =
        let allErrors = ResizeArray<string>()
        let allWarnings = ResizeArray<string>()
        let modifiedFiles = ResizeArray<string>()
        
        // Check operation count
        if operations.Length > config.MaxFilesPerOperation then
            allErrors.Add($"Too many files in operation: {operations.Length} (max: {config.MaxFilesPerOperation})")
        
        // Validate each operation
        operations
        |> List.iter (fun (filePath, originalContent, newContent) ->
            let result = validateFileModification config filePath originalContent newContent
            allErrors.AddRange(result.Errors)
            allWarnings.AddRange(result.Warnings)
            modifiedFiles.AddRange(result.ModifiedFiles)
        )
        
        // Check for critical file modifications
        let criticalFiles = [
            "Program.fs"
            "Types.fs"
            "CommandRegistry.fs"
            "CliApplication.fs"
            ".fsproj"
        ]
        
        operations
        |> List.iter (fun (filePath, _, _) ->
            let fileName = Path.GetFileName(filePath)
            if criticalFiles |> List.exists (fun critical -> fileName.Contains(critical)) then
                allWarnings.Add($"Critical file modification: {fileName}")
        )
        
        {
            IsValid = allErrors.Count = 0
            Errors = allErrors |> Seq.toList
            Warnings = allWarnings |> Seq.toList
            ModifiedFiles = modifiedFiles |> Seq.distinct |> Seq.toList
            TestCoverage = None
            ExecutionTime = None
        }
    
    /// Create a safe sandbox for autonomous operations
    let createSandbox (baseDir: string) : Result<string, string> =
        try
            let sandboxDir = Path.Combine(baseDir, "autonomous_sandbox", DateTime.Now.ToString("yyyyMMdd_HHmmss"))
            Directory.CreateDirectory(sandboxDir) |> ignore
            
            // Copy source files to sandbox
            let sourceDir = Path.Combine(baseDir, "src")
            if Directory.Exists(sourceDir) then
                let sandboxSrc = Path.Combine(sandboxDir, "src")
                
                let rec copyDirectory (source: string) (target: string) =
                    Directory.CreateDirectory(target) |> ignore
                    
                    Directory.GetFiles(source)
                    |> Array.iter (fun file ->
                        let fileName = Path.GetFileName(file)
                        let targetFile = Path.Combine(target, fileName)
                        File.Copy(file, targetFile, true)
                    )
                    
                    Directory.GetDirectories(source)
                    |> Array.iter (fun dir ->
                        let dirName = Path.GetFileName(dir)
                        let targetDir = Path.Combine(target, dirName)
                        copyDirectory dir targetDir
                    )
                
                copyDirectory sourceDir sandboxSrc
            
            Ok sandboxDir
        with
        | ex -> Error $"Sandbox creation failed: {ex.Message}"
    
    /// Test autonomous changes in sandbox
    let testInSandbox (sandboxDir: string) (operations: (string * string * string) list) : Result<ValidationResult, string> =
        try
            // Apply changes in sandbox
            operations
            |> List.iter (fun (filePath, _, newContent) ->
                let sandboxPath = filePath.Replace("src", Path.Combine(sandboxDir, "src"))
                let sandboxDir = Path.GetDirectoryName(sandboxPath)
                Directory.CreateDirectory(sandboxDir) |> ignore
                File.WriteAllText(sandboxPath, newContent)
            )
            
            // Try to compile in sandbox
            let startTime = DateTime.Now
            let buildProcess = System.Diagnostics.Process.Start(
                System.Diagnostics.ProcessStartInfo(
                    FileName = "dotnet",
                    Arguments = "build",
                    WorkingDirectory = sandboxDir,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                )
            )
            
            buildProcess.WaitForExit(30000) |> ignore // 30 second timeout
            let executionTime = DateTime.Now - startTime
            
            let buildOutput = buildProcess.StandardOutput.ReadToEnd()
            let buildErrors = buildProcess.StandardError.ReadToEnd()
            
            let errors = ResizeArray<string>()
            let warnings = ResizeArray<string>()
            
            if buildProcess.ExitCode <> 0 then
                errors.Add($"Compilation failed in sandbox: {buildErrors}")
            
            if buildOutput.Contains("warning") then
                warnings.Add("Compilation warnings detected")
            
            Ok {
                IsValid = errors.Count = 0
                Errors = errors |> Seq.toList
                Warnings = warnings |> Seq.toList
                ModifiedFiles = operations |> List.map (fun (path, _, _) -> path)
                TestCoverage = None
                ExecutionTime = Some executionTime
            }
        with
        | ex -> Error $"Sandbox testing failed: {ex.Message}"
    
    /// Generate validation report
    let generateValidationReport (result: ValidationResult) : string =
        let report = System.Text.StringBuilder()
        
        report.AppendLine("=== AUTONOMOUS SYSTEM VALIDATION REPORT ===") |> ignore
        report.AppendLine("Validation Date: " + DateTime.Now.ToString()) |> ignore
        report.AppendLine("Status: " + (if result.IsValid then "✅ VALID" else "❌ INVALID")) |> ignore
        report.AppendLine() |> ignore
        
        if not result.Errors.IsEmpty then
            report.AppendLine("❌ ERRORS:") |> ignore
            result.Errors |> List.iter (fun error -> report.AppendLine("  - " + error) |> ignore)
            report.AppendLine() |> ignore
        
        if not result.Warnings.IsEmpty then
            report.AppendLine("⚠️ WARNINGS:") |> ignore
            result.Warnings |> List.iter (fun warning -> report.AppendLine("  - " + warning) |> ignore)
            report.AppendLine() |> ignore
        
        report.AppendLine("Modified Files: " + result.ModifiedFiles.Length.ToString()) |> ignore
        result.ModifiedFiles |> List.iter (fun file -> report.AppendLine("  - " + file) |> ignore)
        
        match result.ExecutionTime with
        | Some time -> report.AppendLine("Execution Time: " + time.TotalSeconds.ToString("F2") + " seconds") |> ignore
        | None -> ()
        
        report.ToString()
