namespace TarsEngine.FSharp.Core.Metascript

/// Module containing types for metascripts
module Types =
    open System
    open System.Collections.Generic
    
    /// Represents a metascript placeholder
    type Placeholder = {
        /// The name of the placeholder
        Name: string
        /// The description of the placeholder
        Description: string
        /// The default value of the placeholder
        DefaultValue: string option
        /// Whether the placeholder is required
        Required: bool
        /// The validation pattern for the placeholder
        ValidationPattern: string option
        /// The validation error message
        ValidationErrorMessage: string option
    }
    
    /// Represents a metascript dependency
    type Dependency = {
        /// The name of the dependency
        Name: string
        /// The description of the dependency
        Description: string
        /// The version of the dependency
        Version: string option
        /// The URL of the dependency
        Url: string option
        /// Whether the dependency is required
        Required: bool
    }
    
    /// Represents a metascript template
    type Template = {
        /// The name of the template
        Name: string
        /// The description of the template
        Description: string
        /// The content of the template
        Content: string
        /// The placeholders in the template
        Placeholders: Placeholder list
        /// The dependencies of the template
        Dependencies: Dependency list
        /// The language of the template
        Language: string
        /// The category of the template
        Category: string
        /// The tags of the template
        Tags: string list
        /// The author of the template
        Author: string
        /// The creation date of the template
        CreationDate: DateTime
        /// The last modified date of the template
        LastModifiedDate: DateTime
        /// The version of the template
        Version: string
    }
    
    /// Represents a metascript example
    type Example = {
        /// The name of the example
        Name: string
        /// The description of the example
        Description: string
        /// The content of the example
        Content: string
        /// The language of the example
        Language: string
        /// The category of the example
        Category: string
        /// The tags of the example
        Tags: string list
        /// The author of the example
        Author: string
        /// The creation date of the example
        CreationDate: DateTime
        /// The last modified date of the example
        LastModifiedDate: DateTime
        /// The version of the example
        Version: string
    }
    
    /// Represents a metascript component
    type Component = {
        /// The name of the component
        Name: string
        /// The description of the component
        Description: string
        /// The content of the component
        Content: string
        /// The placeholders in the component
        Placeholders: Placeholder list
        /// The dependencies of the component
        Dependencies: Dependency list
        /// The language of the component
        Language: string
        /// The category of the component
        Category: string
        /// The tags of the component
        Tags: string list
        /// The author of the component
        Author: string
        /// The creation date of the component
        CreationDate: DateTime
        /// The last modified date of the component
        LastModifiedDate: DateTime
        /// The version of the component
        Version: string
    }
    
    /// Represents a metascript transformation
    type Transformation = {
        /// The name of the transformation
        Name: string
        /// The description of the transformation
        Description: string
        /// The pattern to match
        Pattern: string
        /// The replacement text
        Replacement: string
        /// The language the transformation applies to
        Language: string
        /// The category of the transformation
        Category: string
        /// The tags of the transformation
        Tags: string list
        /// The author of the transformation
        Author: string
        /// The creation date of the transformation
        CreationDate: DateTime
        /// The last modified date of the transformation
        LastModifiedDate: DateTime
        /// The version of the transformation
        Version: string
    }
    
    /// Represents a metascript
    type Metascript = {
        /// The name of the metascript
        Name: string
        /// The description of the metascript
        Description: string
        /// The content of the metascript
        Content: string
        /// The language of the metascript
        Language: string
        /// The category of the metascript
        Category: string
        /// The tags of the metascript
        Tags: string list
        /// The author of the metascript
        Author: string
        /// The creation date of the metascript
        CreationDate: DateTime
        /// The last modified date of the metascript
        LastModifiedDate: DateTime
        /// The version of the metascript
        Version: string
        /// The placeholders in the metascript
        Placeholders: Map<string, string>
        /// The dependencies of the metascript
        Dependencies: Dependency list
        /// The template used to generate the metascript
        Template: Template option
        /// The components used in the metascript
        Components: Component list
        /// The transformations applied to the metascript
        Transformations: Transformation list
        /// Additional metadata for the metascript
        Metadata: Map<string, obj>
    }
    
    /// Represents a metascript execution result
    type ExecutionResult = {
        /// The metascript that was executed
        Metascript: Metascript
        /// Whether the execution was successful
        Success: bool
        /// The output of the execution
        Output: string
        /// The error message if the execution failed
        ErrorMessage: string option
        /// The execution time in milliseconds
        ExecutionTime: int64
        /// The start time of the execution
        StartTime: DateTime
        /// The end time of the execution
        EndTime: DateTime
        /// Additional metadata for the execution
        Metadata: Map<string, obj>
    }
    
    /// Represents a metascript validation result
    type ValidationResult = {
        /// The metascript that was validated
        Metascript: Metascript
        /// Whether the validation was successful
        Success: bool
        /// The validation errors
        Errors: string list
        /// The validation warnings
        Warnings: string list
        /// The validation time in milliseconds
        ValidationTime: int64
        /// Additional metadata for the validation
        Metadata: Map<string, obj>
    }
