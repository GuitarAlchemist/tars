namespace TarsEngine.FSharp.Core.Metascript

/// Module for generating metascripts
module Generator =
    open System
    open System.Text.RegularExpressions
    open Types
    
    /// Represents a placeholder replacement
    type PlaceholderReplacement = {
        /// The placeholder name
        Name: string
        /// The replacement value
        Value: string
    }
    
    /// Represents a metascript generation options
    type GenerationOptions = {
        /// Whether to validate placeholders
        ValidatePlaceholders: bool
        /// Whether to validate dependencies
        ValidateDependencies: bool
        /// Whether to apply transformations
        ApplyTransformations: bool
        /// Whether to include metadata
        IncludeMetadata: bool
        /// The author of the generated metascript
        Author: string
        /// The category of the generated metascript
        Category: string
        /// The tags of the generated metascript
        Tags: string list
    }
    
    /// Creates default generation options
    let defaultOptions = {
        ValidatePlaceholders = true
        ValidateDependencies = true
        ApplyTransformations = true
        IncludeMetadata = true
        Author = "System"
        Category = "Generated"
        Tags = []
    }
    
    /// Validates a placeholder replacement
    let validatePlaceholderReplacement (placeholder: Placeholder) (replacement: PlaceholderReplacement) =
        if placeholder.Required && String.IsNullOrWhiteSpace(replacement.Value) then
            Error $"Placeholder '{placeholder.Name}' is required but no value was provided."
        elif placeholder.ValidationPattern.IsSome && not (String.IsNullOrWhiteSpace(replacement.Value)) then
            let pattern = placeholder.ValidationPattern.Value
            let regex = Regex(pattern)
            if not (regex.IsMatch(replacement.Value)) then
                let errorMessage = 
                    match placeholder.ValidationErrorMessage with
                    | Some msg -> msg
                    | None -> $"Value '{replacement.Value}' for placeholder '{placeholder.Name}' does not match the required pattern '{pattern}'."
                Error errorMessage
            else
                Ok replacement
        else
            Ok replacement
    
    /// Validates placeholder replacements
    let validatePlaceholderReplacements (template: Template) (replacements: PlaceholderReplacement list) =
        let placeholderMap = template.Placeholders |> List.map (fun p -> p.Name, p) |> Map.ofList
        let replacementMap = replacements |> List.map (fun r -> r.Name, r) |> Map.ofList
        
        let missingRequiredPlaceholders =
            template.Placeholders
            |> List.filter (fun p -> p.Required)
            |> List.filter (fun p -> not (replacementMap.ContainsKey(p.Name)))
            |> List.map (fun p -> p.Name)
        
        if not (List.isEmpty missingRequiredPlaceholders) then
            Error $"Missing required placeholders: {String.Join(", ", missingRequiredPlaceholders)}"
        else
            let validationResults =
                replacements
                |> List.map (fun replacement ->
                    match placeholderMap.TryGetValue(replacement.Name) with
                    | true, placeholder -> validatePlaceholderReplacement placeholder replacement
                    | false, _ -> Ok replacement)
            
            let errors =
                validationResults
                |> List.choose (function
                    | Error msg -> Some msg
                    | Ok _ -> None)
            
            if not (List.isEmpty errors) then
                Error (String.Join(Environment.NewLine, errors))
            else
                Ok replacements
    
    /// Replaces placeholders in content
    let replacePlaceholders (content: string) (replacements: PlaceholderReplacement list) =
        let mutable result = content
        
        for replacement in replacements do
            result <- result.Replace($"{{{{{replacement.Name}}}}}", replacement.Value)
        
        result
    
    /// Applies transformations to content
    let applyTransformations (content: string) (transformations: Transformation list) (language: string) =
        let mutable result = content
        
        for transformation in transformations |> List.filter (fun t -> t.Language = language) do
            let regex = Regex(transformation.Pattern)
            result <- regex.Replace(result, transformation.Replacement)
        
        result
    
    /// Generates a metascript from a template
    let generateFromTemplate (template: Template) (replacements: PlaceholderReplacement list) (options: GenerationOptions) (transformations: Transformation list) =
        try
            // Validate placeholders if required
            let validatedReplacements =
                if options.ValidatePlaceholders then
                    match validatePlaceholderReplacements template replacements with
                    | Ok validated -> validated
                    | Error msg -> failwith msg
                else
                    replacements
            
            // Replace placeholders in content
            let content = replacePlaceholders template.Content validatedReplacements
            
            // Apply transformations if required
            let transformedContent =
                if options.ApplyTransformations then
                    applyTransformations content transformations template.Language
                else
                    content
            
            // Create placeholder map
            let placeholderMap =
                validatedReplacements
                |> List.map (fun r -> r.Name, r.Value)
                |> Map.ofList
            
            // Create metascript
            let metascript = {
                Name = template.Name
                Description = template.Description
                Content = transformedContent
                Language = template.Language
                Category = options.Category
                Tags = options.Tags
                Author = options.Author
                CreationDate = DateTime.UtcNow
                LastModifiedDate = DateTime.UtcNow
                Version = "1.0.0"
                Placeholders = placeholderMap
                Dependencies = template.Dependencies
                Template = Some template
                Components = []
                Transformations = if options.ApplyTransformations then transformations else []
                Metadata = if options.IncludeMetadata then Map.empty else Map.empty
            }
            
            Ok metascript
        with
        | ex -> Error ex.Message
    
    /// Generates a metascript from components
    let generateFromComponents (components: Component list) (replacements: PlaceholderReplacement list) (options: GenerationOptions) (transformations: Transformation list) =
        try
            // Validate placeholders for each component if required
            let validatedComponents =
                if options.ValidatePlaceholders then
                    components
                    |> List.map (fun component ->
                        let componentReplacements = replacements |> List.filter (fun r -> component.Placeholders |> List.exists (fun p -> p.Name = r.Name))
                        match validatePlaceholderReplacements { template with Placeholders = component.Placeholders; Dependencies = component.Dependencies } componentReplacements with
                        | Ok _ -> Ok component
                        | Error msg -> Error (component.Name, msg))
                    |> List.choose (function
                        | Ok component -> Some component
                        | Error (name, msg) -> failwith $"Validation failed for component '{name}': {msg}")
                else
                    components
            
            // Combine component contents
            let combinedContent =
                validatedComponents
                |> List.map (fun component -> replacePlaceholders component.Content replacements)
                |> String.concat Environment.NewLine
            
            // Apply transformations if required
            let transformedContent =
                if options.ApplyTransformations then
                    let language = if List.isEmpty validatedComponents then "text" else validatedComponents.[0].Language
                    applyTransformations combinedContent transformations language
                else
                    combinedContent
            
            // Create placeholder map
            let placeholderMap =
                replacements
                |> List.map (fun r -> r.Name, r.Value)
                |> Map.ofList
            
            // Create metascript
            let metascript = {
                Name = if List.isEmpty validatedComponents then "Generated Metascript" else validatedComponents.[0].Name
                Description = if List.isEmpty validatedComponents then "Generated from components" else validatedComponents.[0].Description
                Content = transformedContent
                Language = if List.isEmpty validatedComponents then "text" else validatedComponents.[0].Language
                Category = options.Category
                Tags = options.Tags
                Author = options.Author
                CreationDate = DateTime.UtcNow
                LastModifiedDate = DateTime.UtcNow
                Version = "1.0.0"
                Placeholders = placeholderMap
                Dependencies = validatedComponents |> List.collect (fun c -> c.Dependencies)
                Template = None
                Components = validatedComponents
                Transformations = if options.ApplyTransformations then transformations else []
                Metadata = if options.IncludeMetadata then Map.empty else Map.empty
            }
            
            Ok metascript
        with
        | ex -> Error ex.Message
    
    /// Creates a placeholder
    let createPlaceholder name description defaultValue required validationPattern validationErrorMessage =
        { Name = name
          Description = description
          DefaultValue = defaultValue
          Required = required
          ValidationPattern = validationPattern
          ValidationErrorMessage = validationErrorMessage }
    
    /// Creates a dependency
    let createDependency name description version url required =
        { Name = name
          Description = description
          Version = version
          Url = url
          Required = required }
    
    /// Creates a template
    let createTemplate name description content placeholders dependencies language category tags author =
        { Name = name
          Description = description
          Content = content
          Placeholders = placeholders
          Dependencies = dependencies
          Language = language
          Category = category
          Tags = tags
          Author = author
          CreationDate = DateTime.UtcNow
          LastModifiedDate = DateTime.UtcNow
          Version = "1.0.0" }
    
    /// Creates a component
    let createComponent name description content placeholders dependencies language category tags author =
        { Name = name
          Description = description
          Content = content
          Placeholders = placeholders
          Dependencies = dependencies
          Language = language
          Category = category
          Tags = tags
          Author = author
          CreationDate = DateTime.UtcNow
          LastModifiedDate = DateTime.UtcNow
          Version = "1.0.0" }
    
    /// Creates a transformation
    let createTransformation name description pattern replacement language category tags author =
        { Name = name
          Description = description
          Pattern = pattern
          Replacement = replacement
          Language = language
          Category = category
          Tags = tags
          Author = author
          CreationDate = DateTime.UtcNow
          LastModifiedDate = DateTime.UtcNow
          Version = "1.0.0" }
    
    /// Creates a placeholder replacement
    let createPlaceholderReplacement name value =
        { Name = name
          Value = value }
