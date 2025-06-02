namespace TarsEngine.FSharp.Requirements.Validation

open System
open System.Text.RegularExpressions
open TarsEngine.FSharp.Requirements.Models

/// <summary>
/// Validation result for requirements
/// </summary>
type ValidationResult = {
    IsValid: bool
    Errors: string list
    Warnings: string list
}

/// <summary>
/// Validation rules for requirements
/// </summary>
type ValidationRule = {
    Name: string
    Description: string
    Validator: Requirement -> ValidationResult
    IsRequired: bool
}

/// <summary>
/// Requirement validator with configurable rules
/// Real implementation for requirement validation
/// </summary>
type RequirementValidator() =
    
    /// <summary>
    /// Create validation result
    /// </summary>
    let createResult (isValid: bool) (errors: string list) (warnings: string list) =
        { IsValid = isValid; Errors = errors; Warnings = warnings }
    
    /// <summary>
    /// Create success result
    /// </summary>
    let success = createResult true [] []
    
    /// <summary>
    /// Create error result
    /// </summary>
    let error (message: string) = createResult false [message] []
    
    /// <summary>
    /// Create warning result
    /// </summary>
    let warning (message: string) = createResult true [] [message]
    
    /// <summary>
    /// Combine validation results
    /// </summary>
    let combineResults (results: ValidationResult list) =
        let allErrors = results |> List.collect (fun r -> r.Errors)
        let allWarnings = results |> List.collect (fun r -> r.Warnings)
        let isValid = results |> List.forall (fun r -> r.IsValid)
        createResult isValid allErrors allWarnings
    
    /// <summary>
    /// Built-in validation rules
    /// </summary>
    let builtInRules = [
        {
            Name = "TitleRequired"
            Description = "Title must not be empty"
            IsRequired = true
            Validator = fun req ->
                if String.IsNullOrWhiteSpace(req.Title) then
                    error "Title is required"
                else
                    success
        }
        
        {
            Name = "TitleLength"
            Description = "Title should be between 5 and 100 characters"
            IsRequired = false
            Validator = fun req ->
                let length = req.Title.Length
                if length < 5 then
                    warning "Title is very short (less than 5 characters)"
                elif length > 100 then
                    error "Title is too long (more than 100 characters)"
                else
                    success
        }
        
        {
            Name = "DescriptionRequired"
            Description = "Description must not be empty"
            IsRequired = true
            Validator = fun req ->
                if String.IsNullOrWhiteSpace(req.Description) then
                    error "Description is required"
                else
                    success
        }
        
        {
            Name = "DescriptionLength"
            Description = "Description should be at least 20 characters"
            IsRequired = false
            Validator = fun req ->
                if req.Description.Length < 20 then
                    warning "Description is very short (less than 20 characters)"
                else
                    success
        }
        
        {
            Name = "AcceptanceCriteriaRequired"
            Description = "At least one acceptance criteria is required"
            IsRequired = true
            Validator = fun req ->
                if req.AcceptanceCriteria.IsEmpty then
                    error "At least one acceptance criteria is required"
                else
                    success
        }
        
        {
            Name = "AcceptanceCriteriaQuality"
            Description = "Acceptance criteria should be specific and measurable"
            IsRequired = false
            Validator = fun req ->
                let vagueCriteria = req.AcceptanceCriteria |> List.filter (fun ac ->
                    let lower = ac.ToLowerInvariant()
                    lower.Contains("should") || lower.Contains("might") || lower.Contains("could")
                )
                if not vagueCriteria.IsEmpty then
                    let vagueCriteriaText = String.Join(", ", vagueCriteria)
                    warning $"Some acceptance criteria use vague language: {vagueCriteriaText}"
                else
                    success
        }
        
        {
            Name = "CreatedByRequired"
            Description = "CreatedBy must not be empty"
            IsRequired = true
            Validator = fun req ->
                if String.IsNullOrWhiteSpace(req.CreatedBy) then
                    error "CreatedBy is required"
                else
                    success
        }
        
        {
            Name = "UpdatedByRequired"
            Description = "UpdatedBy must not be empty"
            IsRequired = true
            Validator = fun req ->
                if String.IsNullOrWhiteSpace(req.UpdatedBy) then
                    error "UpdatedBy is required"
                else
                    success
        }
        
        {
            Name = "VersionConsistency"
            Description = "Version should be positive"
            IsRequired = true
            Validator = fun req ->
                if req.Version <= 0 then
                    error "Version must be positive"
                else
                    success
        }
        
        {
            Name = "DateConsistency"
            Description = "UpdatedAt should be after CreatedAt"
            IsRequired = true
            Validator = fun req ->
                if req.UpdatedAt < req.CreatedAt then
                    error "UpdatedAt cannot be before CreatedAt"
                else
                    success
        }
        
        {
            Name = "TargetDateRealistic"
            Description = "Target date should be in the future for non-completed requirements"
            IsRequired = false
            Validator = fun req ->
                match req.TargetDate with
                | Some targetDate when req.Status <> RequirementStatus.Verified ->
                    if targetDate < DateTime.UtcNow then
                        warning "Target date is in the past"
                    else
                        success
                | _ -> success
        }
        
        {
            Name = "EffortEstimateRealistic"
            Description = "Effort estimates should be reasonable"
            IsRequired = false
            Validator = fun req ->
                match req.EstimatedEffort with
                | Some effort when effort <= 0.0 ->
                    warning "Estimated effort should be positive"
                | Some effort when effort > 1000.0 ->
                    warning "Estimated effort seems very high (>1000 hours)"
                | _ -> success
        }
        
        {
            Name = "TagsFormat"
            Description = "Tags should follow naming conventions"
            IsRequired = false
            Validator = fun req ->
                let invalidTags = req.Tags |> List.filter (fun tag ->
                    String.IsNullOrWhiteSpace(tag) || 
                    tag.Contains(" ") || 
                    not (Regex.IsMatch(tag, @"^[a-zA-Z0-9\-_]+$"))
                )
                if not invalidTags.IsEmpty then
                    let invalidTagsText = String.Join(", ", invalidTags)
                    warning $"Some tags have invalid format: {invalidTagsText}"
                else
                    success
        }
    ]
    
    /// <summary>
    /// Custom validation rules
    /// </summary>
    let mutable customRules: ValidationRule list = []
    
    /// <summary>
    /// Add custom validation rule
    /// </summary>
    member this.AddRule(rule: ValidationRule) =
        customRules <- rule :: customRules
    
    /// <summary>
    /// Remove custom validation rule
    /// </summary>
    member this.RemoveRule(ruleName: string) =
        customRules <- customRules |> List.filter (fun r -> r.Name <> ruleName)
    
    /// <summary>
    /// Get all validation rules
    /// </summary>
    member this.GetAllRules() =
        builtInRules @ customRules
    
    /// <summary>
    /// Validate requirement against all rules
    /// </summary>
    member this.ValidateRequirement(requirement: Requirement) =
        let allRules = this.GetAllRules()
        let results = allRules |> List.map (fun rule -> rule.Validator requirement)
        combineResults results
    
    /// <summary>
    /// Validate requirement against specific rules
    /// </summary>
    member this.ValidateRequirementWithRules(requirement: Requirement, ruleNames: string list) =
        let allRules = this.GetAllRules()
        let selectedRules = allRules |> List.filter (fun r -> List.contains r.Name ruleNames)
        let results = selectedRules |> List.map (fun rule -> rule.Validator requirement)
        combineResults results
    
    /// <summary>
    /// Validate only required rules
    /// </summary>
    member this.ValidateRequiredRules(requirement: Requirement) =
        let allRules = this.GetAllRules()
        let requiredRules = allRules |> List.filter (fun r -> r.IsRequired)
        let results = requiredRules |> List.map (fun rule -> rule.Validator requirement)
        combineResults results
    
    /// <summary>
    /// Validate multiple requirements
    /// </summary>
    member this.ValidateRequirements(requirements: Requirement list) =
        requirements |> List.map (fun req -> (req.Id, this.ValidateRequirement(req)))
    
    /// <summary>
    /// Get validation summary for multiple requirements
    /// </summary>
    member this.GetValidationSummary(requirements: Requirement list) =
        let validationResults = this.ValidateRequirements(requirements)
        let totalCount = validationResults.Length
        let validCount = validationResults |> List.filter (fun (_, result) -> result.IsValid) |> List.length
        let invalidCount = totalCount - validCount
        let totalErrors = validationResults |> List.collect (fun (_, result) -> result.Errors) |> List.length
        let totalWarnings = validationResults |> List.collect (fun (_, result) -> result.Warnings) |> List.length
        
        {|
            TotalRequirements = totalCount
            ValidRequirements = validCount
            InvalidRequirements = invalidCount
            TotalErrors = totalErrors
            TotalWarnings = totalWarnings
            ValidationRate = if totalCount > 0 then (float validCount / float totalCount) * 100.0 else 0.0
            Results = validationResults
        |}

module RequirementValidatorHelpers =
    
    /// <summary>
    /// Create a simple validation rule
    /// </summary>
    let createRule (name: string) (description: string) (isRequired: bool) (validator: Requirement -> bool) (errorMessage: string) =
        {
            Name = name
            Description = description
            IsRequired = isRequired
            Validator = fun req ->
                if validator req then
                    { IsValid = true; Errors = []; Warnings = [] }
                else
                    { IsValid = false; Errors = [errorMessage]; Warnings = [] }
        }
    
    /// <summary>
    /// Create a warning rule
    /// </summary>
    let createWarningRule (name: string) (description: string) (validator: Requirement -> bool) (warningMessage: string) =
        {
            Name = name
            Description = description
            IsRequired = false
            Validator = fun req ->
                if validator req then
                    { IsValid = true; Errors = []; Warnings = [] }
                else
                    { IsValid = true; Errors = []; Warnings = [warningMessage] }
        }
    
    /// <summary>
    /// Validate requirement ID format
    /// </summary>
    let validateIdFormat (requirement: Requirement) =
        let isValidFormat = Regex.IsMatch(requirement.Id, @"^[A-Z]{2,5}-\d{1,6}$")
        if not isValidFormat then
            Some "Requirement ID should follow format: PREFIX-NUMBER (e.g., REQ-123)"
        else
            None
