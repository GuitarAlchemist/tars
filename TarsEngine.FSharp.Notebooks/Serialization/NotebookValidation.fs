namespace TarsEngine.FSharp.Notebooks.Serialization

open System
open System.Text.RegularExpressions
open TarsEngine.FSharp.Notebooks.Types

/// <summary>
/// Validation functions for Jupyter notebooks
/// </summary>

/// Validation error types
type ValidationError = {
    Code: string
    Message: string
    Severity: ValidationSeverity
    Location: ValidationLocation option
}

/// Validation severity levels
and ValidationSeverity = 
    | Error
    | Warning
    | Info

/// Location of validation issue
and ValidationLocation = {
    CellIndex: int option
    LineNumber: int option
    ColumnNumber: int option
    Path: string option
}

/// Validation result
type ValidationResult = {
    IsValid: bool
    Errors: ValidationError list
    Warnings: ValidationError list
    Info: ValidationError list
}

/// Notebook validator
type NotebookValidator() =
    
    /// Validate a complete notebook
    member _.ValidateNotebook(notebook: JupyterNotebook) : ValidationResult =
        let errors = ResizeArray<ValidationError>()
        let warnings = ResizeArray<ValidationError>()
        let info = ResizeArray<ValidationError>()
        
        // Validate notebook structure
        this.ValidateNotebookStructure(notebook, errors, warnings, info)
        
        // Validate metadata
        this.ValidateMetadata(notebook.Metadata, errors, warnings, info)
        
        // Validate cells
        notebook.Cells |> List.iteri (fun i cell ->
            this.ValidateCell(cell, i, errors, warnings, info)
        )
        
        {
            IsValid = errors.Count = 0
            Errors = errors |> List.ofSeq
            Warnings = warnings |> List.ofSeq
            Info = info |> List.ofSeq
        }
    
    /// Validate notebook structure
    member private _.ValidateNotebookStructure(
        notebook: JupyterNotebook,
        errors: ResizeArray<ValidationError>,
        warnings: ResizeArray<ValidationError>,
        info: ResizeArray<ValidationError>) =
        
        // Check notebook format version
        match notebook.NbFormat with
        | 4 -> () // Valid
        | v when v < 4 ->
            warnings.Add({
                Code = "OUTDATED_FORMAT"
                Message = $"Notebook format version {v} is outdated. Consider upgrading to version 4."
                Severity = Warning
                Location = None
            })
        | v ->
            warnings.Add({
                Code = "UNKNOWN_FORMAT"
                Message = $"Unknown notebook format version {v}. Expected version 4."
                Severity = Warning
                Location = None
            })
        
        // Check if notebook has cells
        if notebook.Cells.IsEmpty then
            info.Add({
                Code = "EMPTY_NOTEBOOK"
                Message = "Notebook contains no cells."
                Severity = Info
                Location = None
            })
    
    /// Validate notebook metadata
    member private _.ValidateMetadata(
        metadata: NotebookMetadata,
        errors: ResizeArray<ValidationError>,
        warnings: ResizeArray<ValidationError>,
        info: ResizeArray<ValidationError>) =
        
        // Check for title
        match metadata.Title with
        | None ->
            info.Add({
                Code = "NO_TITLE"
                Message = "Notebook has no title."
                Severity = Info
                Location = None
            })
        | Some title when String.IsNullOrWhiteSpace(title) ->
            warnings.Add({
                Code = "EMPTY_TITLE"
                Message = "Notebook title is empty or whitespace."
                Severity = Warning
                Location = None
            })
        | _ -> ()
        
        // Check kernel spec
        match metadata.KernelSpec with
        | None ->
            warnings.Add({
                Code = "NO_KERNEL_SPEC"
                Message = "Notebook has no kernel specification."
                Severity = Warning
                Location = None
            })
        | Some kernelSpec ->
            if String.IsNullOrWhiteSpace(kernelSpec.Name) then
                errors.Add({
                    Code = "INVALID_KERNEL_NAME"
                    Message = "Kernel specification has no name."
                    Severity = Error
                    Location = None
                })
    
    /// Validate a single cell
    member private this.ValidateCell(
        cell: NotebookCell,
        cellIndex: int,
        errors: ResizeArray<ValidationError>,
        warnings: ResizeArray<ValidationError>,
        info: ResizeArray<ValidationError>) =
        
        let location = { CellIndex = Some cellIndex; LineNumber = None; ColumnNumber = None; Path = None }
        
        match cell with
        | CodeCell codeData ->
            this.ValidateCodeCell(codeData, location, errors, warnings, info)
        | MarkdownCell markdownData ->
            this.ValidateMarkdownCell(markdownData, location, errors, warnings, info)
        | RawCell rawData ->
            this.ValidateRawCell(rawData, location, errors, warnings, info)
    
    /// Validate code cell
    member private _.ValidateCodeCell(
        codeData: CodeCellData,
        location: ValidationLocation,
        errors: ResizeArray<ValidationError>,
        warnings: ResizeArray<ValidationError>,
        info: ResizeArray<ValidationError>) =
        
        // Check for empty code cells
        if codeData.Source.IsEmpty || codeData.Source |> List.forall String.IsNullOrWhiteSpace then
            info.Add({
                Code = "EMPTY_CODE_CELL"
                Message = "Code cell is empty."
                Severity = Info
                Location = Some location
            })
        
        // Check for very long lines
        codeData.Source |> List.iteri (fun lineIndex line ->
            if line.Length > 120 then
                warnings.Add({
                    Code = "LONG_LINE"
                    Message = $"Line {lineIndex + 1} is very long ({line.Length} characters). Consider breaking it up."
                    Severity = Warning
                    Location = Some { location with LineNumber = Some (lineIndex + 1) }
                })
        )
        
        // Check for potential issues in Python code
        if codeData.Source |> List.exists (fun line -> line.Contains("import *")) then
            warnings.Add({
                Code = "WILDCARD_IMPORT"
                Message = "Wildcard imports (import *) can make code harder to understand and debug."
                Severity = Warning
                Location = Some location
            })
    
    /// Validate markdown cell
    member private _.ValidateMarkdownCell(
        markdownData: MarkdownCellData,
        location: ValidationLocation,
        errors: ResizeArray<ValidationError>,
        warnings: ResizeArray<ValidationError>,
        info: ResizeArray<ValidationError>) =
        
        // Check for empty markdown cells
        if markdownData.Source.IsEmpty || markdownData.Source |> List.forall String.IsNullOrWhiteSpace then
            info.Add({
                Code = "EMPTY_MARKDOWN_CELL"
                Message = "Markdown cell is empty."
                Severity = Info
                Location = Some location
            })
        
        // Check for broken links (basic check)
        let content = String.Join("\n", markdownData.Source)
        let linkPattern = @"\[([^\]]+)\]\(([^)]+)\)"
        let matches = Regex.Matches(content, linkPattern)
        
        for m in matches do
            let url = m.Groups.[2].Value
            if url.StartsWith("http") && not (url.Contains("://")) then
                warnings.Add({
                    Code = "MALFORMED_URL"
                    Message = $"Potentially malformed URL: {url}"
                    Severity = Warning
                    Location = Some location
                })
    
    /// Validate raw cell
    member private _.ValidateRawCell(
        rawData: RawCellData,
        location: ValidationLocation,
        errors: ResizeArray<ValidationError>,
        warnings: ResizeArray<ValidationError>,
        info: ResizeArray<ValidationError>) =
        
        // Check for empty raw cells
        if rawData.Source.IsEmpty || rawData.Source |> List.forall String.IsNullOrWhiteSpace then
            info.Add({
                Code = "EMPTY_RAW_CELL"
                Message = "Raw cell is empty."
                Severity = Info
                Location = Some location
            })

/// Validation utilities
module ValidationUtils =
    
    /// Create a validation error
    let createError code message location =
        {
            Code = code
            Message = message
            Severity = Error
            Location = location
        }
    
    /// Create a validation warning
    let createWarning code message location =
        {
            Code = code
            Message = message
            Severity = Warning
            Location = location
        }
    
    /// Create a validation info
    let createInfo code message location =
        {
            Code = code
            Message = message
            Severity = Info
            Location = location
        }
    
    /// Create a location
    let createLocation cellIndex lineNumber columnNumber path =
        {
            CellIndex = cellIndex
            LineNumber = lineNumber
            ColumnNumber = columnNumber
            Path = path
        }
    
    /// Format validation result as string
    let formatValidationResult (result: ValidationResult) : string =
        let sb = System.Text.StringBuilder()
        
        if result.IsValid then
            sb.AppendLine("✅ Notebook validation passed") |> ignore
        else
            sb.AppendLine("❌ Notebook validation failed") |> ignore
        
        if not result.Errors.IsEmpty then
            sb.AppendLine("\n🔴 Errors:") |> ignore
            for error in result.Errors do
                sb.AppendLine($"  [{error.Code}] {error.Message}") |> ignore
        
        if not result.Warnings.IsEmpty then
            sb.AppendLine("\n🟡 Warnings:") |> ignore
            for warning in result.Warnings do
                sb.AppendLine($"  [{warning.Code}] {warning.Message}") |> ignore
        
        if not result.Info.IsEmpty then
            sb.AppendLine("\n🔵 Info:") |> ignore
            for info in result.Info do
                sb.AppendLine($"  [{info.Code}] {info.Message}") |> ignore
        
        sb.ToString()
    
    /// Get validation summary
    let getValidationSummary (result: ValidationResult) : string =
        let errorCount = result.Errors.Length
        let warningCount = result.Warnings.Length
        let infoCount = result.Info.Length
        
        if result.IsValid then
            $"✅ Valid ({warningCount} warnings, {infoCount} info)"
        else
            $"❌ Invalid ({errorCount} errors, {warningCount} warnings, {infoCount} info)"
