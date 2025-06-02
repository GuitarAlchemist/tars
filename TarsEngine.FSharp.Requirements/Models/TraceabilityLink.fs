namespace TarsEngine.FSharp.Requirements.Models

open System
open System.Text.Json.Serialization

/// <summary>
/// Type of traceability link between requirements and code
/// </summary>
type LinkType =
    | Implementation  // Code that implements the requirement
    | Test           // Test that validates the requirement
    | Documentation  // Documentation that describes the requirement
    | Configuration  // Configuration that supports the requirement
    | Dependency     // Code that the requirement depends on
    | Usage          // Code that uses functionality from the requirement

/// <summary>
/// Traceability link between requirements and code elements
/// Real implementation for requirement-to-code mapping
/// </summary>
[<CLIMutable>]
type TraceabilityLink = {
    /// <summary>Unique identifier for the link</summary>
    Id: string
    
    /// <summary>ID of the requirement</summary>
    RequirementId: string
    
    /// <summary>Source file path</summary>
    SourceFile: string
    
    /// <summary>Line number in the source file (optional)</summary>
    LineNumber: int option
    
    /// <summary>End line number for multi-line elements (optional)</summary>
    EndLineNumber: int option
    
    /// <summary>Code element name (function, class, method, etc.)</summary>
    CodeElement: string
    
    /// <summary>Type of code element (class, method, function, etc.)</summary>
    ElementType: string
    
    /// <summary>Type of link</summary>
    LinkType: LinkType
    
    /// <summary>Confidence score of the link (0.0 to 1.0)</summary>
    Confidence: float
    
    /// <summary>How the link was created (manual, automatic, etc.)</summary>
    CreationMethod: string
    
    /// <summary>Description of the link</summary>
    Description: string option
    
    /// <summary>Code snippet or excerpt</summary>
    CodeSnippet: string option
    
    /// <summary>Tags for categorization</summary>
    Tags: string list
    
    /// <summary>Date link was created</summary>
    CreatedAt: DateTime
    
    /// <summary>Date link was last updated</summary>
    UpdatedAt: DateTime
    
    /// <summary>User who created the link</summary>
    CreatedBy: string
    
    /// <summary>User who last updated the link</summary>
    UpdatedBy: string
    
    /// <summary>Version number for change tracking</summary>
    Version: int
    
    /// <summary>Whether the link is still valid</summary>
    IsValid: bool
    
    /// <summary>Last validation date</summary>
    LastValidated: DateTime option
    
    /// <summary>Additional metadata</summary>
    Metadata: Map<string, string>
}

/// <summary>
/// Traceability analysis result
/// </summary>
[<CLIMutable>]
type TraceabilityAnalysis = {
    /// <summary>Requirement ID</summary>
    RequirementId: string
    
    /// <summary>Total number of links</summary>
    TotalLinks: int
    
    /// <summary>Number of implementation links</summary>
    ImplementationLinks: int
    
    /// <summary>Number of test links</summary>
    TestLinks: int
    
    /// <summary>Number of documentation links</summary>
    DocumentationLinks: int
    
    /// <summary>Coverage percentage</summary>
    Coverage: float
    
    /// <summary>Whether requirement is fully traced</summary>
    IsFullyTraced: bool
    
    /// <summary>Missing link types</summary>
    MissingLinkTypes: LinkType list
    
    /// <summary>Analysis timestamp</summary>
    AnalyzedAt: DateTime
}

module TraceabilityLinkHelpers =
    
    /// <summary>
    /// Create a new traceability link
    /// </summary>
    let create (requirementId: string) (sourceFile: string) (codeElement: string) (elementType: string) (linkType: LinkType) (createdBy: string) =
        {
            Id = Guid.NewGuid().ToString()
            RequirementId = requirementId
            SourceFile = sourceFile
            LineNumber = None
            EndLineNumber = None
            CodeElement = codeElement
            ElementType = elementType
            LinkType = linkType
            Confidence = 1.0
            CreationMethod = "manual"
            Description = None
            CodeSnippet = None
            Tags = []
            CreatedAt = DateTime.UtcNow
            UpdatedAt = DateTime.UtcNow
            CreatedBy = createdBy
            UpdatedBy = createdBy
            Version = 1
            IsValid = true
            LastValidated = Some DateTime.UtcNow
            Metadata = Map.empty
        }
    
    /// <summary>
    /// Create automatic traceability link with confidence score
    /// </summary>
    let createAutomatic (requirementId: string) (sourceFile: string) (lineNumber: int) (codeElement: string) (elementType: string) (linkType: LinkType) (confidence: float) (createdBy: string) =
        {
            Id = Guid.NewGuid().ToString()
            RequirementId = requirementId
            SourceFile = sourceFile
            LineNumber = Some lineNumber
            EndLineNumber = None
            CodeElement = codeElement
            ElementType = elementType
            LinkType = linkType
            Confidence = confidence
            CreationMethod = "automatic"
            Description = None
            CodeSnippet = None
            Tags = []
            CreatedAt = DateTime.UtcNow
            UpdatedAt = DateTime.UtcNow
            CreatedBy = createdBy
            UpdatedBy = createdBy
            Version = 1
            IsValid = true
            LastValidated = Some DateTime.UtcNow
            Metadata = Map.empty
        }
    
    /// <summary>
    /// Update traceability link
    /// </summary>
    let update (link: TraceabilityLink) (updatedBy: string) (updateFn: TraceabilityLink -> TraceabilityLink) =
        let updated = updateFn link
        { updated with 
            UpdatedAt = DateTime.UtcNow
            UpdatedBy = updatedBy
            Version = link.Version + 1 }
    
    /// <summary>
    /// Mark link as invalid
    /// </summary>
    let markInvalid (link: TraceabilityLink) (updatedBy: string) =
        { link with 
            IsValid = false
            UpdatedAt = DateTime.UtcNow
            UpdatedBy = updatedBy
            Version = link.Version + 1 }
    
    /// <summary>
    /// Validate link (check if code element still exists)
    /// </summary>
    let validate (link: TraceabilityLink) =
        // In a real implementation, this would check if the file exists
        // and if the code element is still present at the specified location
        try
            if System.IO.File.Exists(link.SourceFile) then
                { link with 
                    IsValid = true
                    LastValidated = Some DateTime.UtcNow }
            else
                { link with 
                    IsValid = false
                    LastValidated = Some DateTime.UtcNow }
        with
        | _ ->
            { link with 
                IsValid = false
                LastValidated = Some DateTime.UtcNow }
    
    /// <summary>
    /// Get link type description
    /// </summary>
    let getLinkTypeDescription (linkType: LinkType) =
        match linkType with
        | Implementation -> "Implementation - Code that implements the requirement"
        | Test -> "Test - Test that validates the requirement"
        | Documentation -> "Documentation - Documentation that describes the requirement"
        | Configuration -> "Configuration - Configuration that supports the requirement"
        | Dependency -> "Dependency - Code that the requirement depends on"
        | Usage -> "Usage - Code that uses functionality from the requirement"
    
    /// <summary>
    /// Analyze traceability for a requirement
    /// </summary>
    let analyzeTraceability (requirementId: string) (links: TraceabilityLink list) =
        let requirementLinks = links |> List.filter (fun l -> l.RequirementId = requirementId && l.IsValid)
        let implementationLinks = requirementLinks |> List.filter (fun l -> l.LinkType = Implementation) |> List.length
        let testLinks = requirementLinks |> List.filter (fun l -> l.LinkType = Test) |> List.length
        let documentationLinks = requirementLinks |> List.filter (fun l -> l.LinkType = Documentation) |> List.length
        
        let hasImplementation = implementationLinks > 0
        let hasTests = testLinks > 0
        let hasDocumentation = documentationLinks > 0
        
        let coverage = 
            let totalExpected = 3.0 // Implementation, Test, Documentation
            let actualCoverage = 
                (if hasImplementation then 1.0 else 0.0) +
                (if hasTests then 1.0 else 0.0) +
                (if hasDocumentation then 1.0 else 0.0)
            (actualCoverage / totalExpected) * 100.0
        
        let missingTypes = [
            if not hasImplementation then yield Implementation
            if not hasTests then yield Test
            if not hasDocumentation then yield Documentation
        ]
        
        {
            RequirementId = requirementId
            TotalLinks = requirementLinks.Length
            ImplementationLinks = implementationLinks
            TestLinks = testLinks
            DocumentationLinks = documentationLinks
            Coverage = coverage
            IsFullyTraced = coverage >= 100.0
            MissingLinkTypes = missingTypes
            AnalyzedAt = DateTime.UtcNow
        }
    
    /// <summary>
    /// Get traceability summary for multiple requirements
    /// </summary>
    let getTraceabilitySummary (requirements: string list) (links: TraceabilityLink list) =
        let analyses = requirements |> List.map (fun reqId -> analyzeTraceability reqId links)
        let totalRequirements = analyses.Length
        let fullyTraced = analyses |> List.filter (fun a -> a.IsFullyTraced) |> List.length
        let averageCoverage = if totalRequirements > 0 then (analyses |> List.sumBy (fun a -> a.Coverage)) / float totalRequirements else 0.0
        
        {|
            TotalRequirements = totalRequirements
            FullyTraced = fullyTraced
            PartiallyTraced = totalRequirements - fullyTraced
            AverageCoverage = averageCoverage
            TraceabilityRate = if totalRequirements > 0 then (float fullyTraced / float totalRequirements) * 100.0 else 0.0
        |}
