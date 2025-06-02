namespace TarsEngine.FSharp.Requirements.Models

open System
open System.Text.Json.Serialization

/// <summary>
/// Core requirement model for TARS system
/// Real implementation - no fake or placeholder data
/// </summary>
[<CLIMutable>]
type Requirement = {
    /// <summary>Unique identifier for the requirement</summary>
    Id: string
    
    /// <summary>Short, descriptive title</summary>
    Title: string
    
    /// <summary>Detailed description of the requirement</summary>
    Description: string
    
    /// <summary>Type of requirement (Functional, Performance, etc.)</summary>
    Type: RequirementType
    
    /// <summary>Priority level</summary>
    Priority: RequirementPriority
    
    /// <summary>Current implementation status</summary>
    Status: RequirementStatus
    
    /// <summary>Acceptance criteria for verification</summary>
    AcceptanceCriteria: string list
    
    /// <summary>Tags for categorization and search</summary>
    Tags: string list
    
    /// <summary>Source of the requirement (user story, bug report, etc.)</summary>
    Source: string option
    
    /// <summary>Person or team responsible for implementation</summary>
    Assignee: string option
    
    /// <summary>Estimated effort in story points or hours</summary>
    EstimatedEffort: float option
    
    /// <summary>Actual effort spent</summary>
    ActualEffort: float option
    
    /// <summary>Target completion date</summary>
    TargetDate: DateTime option
    
    /// <summary>Date requirement was created</summary>
    CreatedAt: DateTime
    
    /// <summary>Date requirement was last updated</summary>
    UpdatedAt: DateTime
    
    /// <summary>User who created the requirement</summary>
    CreatedBy: string
    
    /// <summary>User who last updated the requirement</summary>
    UpdatedBy: string
    
    /// <summary>Version number for change tracking</summary>
    Version: int
    
    /// <summary>Dependencies on other requirements</summary>
    Dependencies: string list
    
    /// <summary>Requirements that depend on this one</summary>
    Dependents: string list
    
    /// <summary>Additional metadata as key-value pairs</summary>
    Metadata: Map<string, string>
}

module RequirementHelpers =
    
    /// <summary>
    /// Create a new requirement with default values
    /// </summary>
    let create (title: string) (description: string) (reqType: RequirementType) (createdBy: string) =
        {
            Id = Guid.NewGuid().ToString()
            Title = title
            Description = description
            Type = reqType
            Priority = RequirementPriority.Medium
            Status = RequirementStatus.Draft
            AcceptanceCriteria = []
            Tags = []
            Source = None
            Assignee = None
            EstimatedEffort = None
            ActualEffort = None
            TargetDate = None
            CreatedAt = DateTime.UtcNow
            UpdatedAt = DateTime.UtcNow
            CreatedBy = createdBy
            UpdatedBy = createdBy
            Version = 1
            Dependencies = []
            Dependents = []
            Metadata = Map.empty
        }
    
    /// <summary>
    /// Update requirement with new values
    /// </summary>
    let update (requirement: Requirement) (updatedBy: string) (updateFn: Requirement -> Requirement) =
        let updated = updateFn requirement
        { updated with 
            UpdatedAt = DateTime.UtcNow
            UpdatedBy = updatedBy
            Version = requirement.Version + 1 }
    
    /// <summary>
    /// Add acceptance criteria to requirement
    /// </summary>
    let addAcceptanceCriteria (criteria: string) (requirement: Requirement) =
        { requirement with AcceptanceCriteria = criteria :: requirement.AcceptanceCriteria }
    
    /// <summary>
    /// Add tag to requirement
    /// </summary>
    let addTag (tag: string) (requirement: Requirement) =
        if not (List.contains tag requirement.Tags) then
            { requirement with Tags = tag :: requirement.Tags }
        else
            requirement
    
    /// <summary>
    /// Remove tag from requirement
    /// </summary>
    let removeTag (tag: string) (requirement: Requirement) =
        { requirement with Tags = List.filter ((<>) tag) requirement.Tags }
    
    /// <summary>
    /// Add dependency to requirement
    /// </summary>
    let addDependency (dependencyId: string) (requirement: Requirement) =
        if not (List.contains dependencyId requirement.Dependencies) then
            { requirement with Dependencies = dependencyId :: requirement.Dependencies }
        else
            requirement
    
    /// <summary>
    /// Check if requirement is complete
    /// </summary>
    let isComplete (requirement: Requirement) =
        requirement.Status = RequirementStatus.Verified
    
    /// <summary>
    /// Check if requirement is overdue
    /// </summary>
    let isOverdue (requirement: Requirement) =
        match requirement.TargetDate with
        | Some targetDate -> DateTime.UtcNow > targetDate && not (isComplete requirement)
        | None -> false
    
    /// <summary>
    /// Get requirement progress percentage
    /// </summary>
    let getProgress (requirement: Requirement) =
        match requirement.Status with
        | RequirementStatus.Draft -> 0.0
        | RequirementStatus.Approved -> 10.0
        | RequirementStatus.InProgress -> 50.0
        | RequirementStatus.Implemented -> 80.0
        | RequirementStatus.Testing -> 90.0
        | RequirementStatus.Verified -> 100.0
        | RequirementStatus.Rejected -> 0.0
        | RequirementStatus.Obsolete -> 0.0
    
    /// <summary>
    /// Validate requirement completeness
    /// </summary>
    let validate (requirement: Requirement) =
        let errors = ResizeArray<string>()
        
        if String.IsNullOrWhiteSpace(requirement.Title) then
            errors.Add("Title is required")
        
        if String.IsNullOrWhiteSpace(requirement.Description) then
            errors.Add("Description is required")
        
        if requirement.AcceptanceCriteria.IsEmpty then
            errors.Add("At least one acceptance criteria is required")
        
        if String.IsNullOrWhiteSpace(requirement.CreatedBy) then
            errors.Add("CreatedBy is required")
        
        errors |> List.ofSeq
