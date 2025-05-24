namespace TarsEngine.FSharp.Core.CodeAnalysis

/// Module containing types for code analysis
module Types =
    /// Represents a code pattern
    type Pattern = {
        /// The name of the pattern
        Name: string
        /// The description of the pattern
        Description: string
        /// The regular expression pattern
        Regex: string
        /// The severity of the pattern (0-1)
        Severity: float
        /// The category of the pattern
        Category: string
        /// The language the pattern applies to
        Language: string
    }
    
    /// Represents a code match
    type Match = {
        /// The pattern that matched
        Pattern: Pattern
        /// The matched text
        Text: string
        /// The line number where the match was found
        LineNumber: int
        /// The column number where the match was found
        ColumnNumber: int
        /// The file path where the match was found
        FilePath: string
    }
    
    /// Represents a code transformation
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
    }
    
    /// Represents a code analysis report
    type Report = {
        /// The matches found in the code
        Matches: Match list
        /// The transformations applied to the code
        Transformations: (Transformation * string) list
        /// The summary of the analysis
        Summary: string
        /// The overall score (0-1)
        Score: float
    }
    
    /// Represents a code analysis configuration
    type Configuration = {
        /// The patterns to look for
        Patterns: Pattern list
        /// The transformations to apply
        Transformations: Transformation list
        /// The file extensions to analyze
        FileExtensions: string list
        /// The directories to exclude
        ExcludeDirectories: string list
        /// The files to exclude
        ExcludeFiles: string list
    }
