namespace TarsEngine.DSL

/// <summary>
/// Represents a unique warning code for a diagnostic message.
/// </summary>
type WarningCode =
    // Deprecated features
    | DeprecatedBlockType = 1001
    | DeprecatedProperty = 1002
    | DeprecatedPropertyValue = 1003
    
    // Problematic patterns
    | ProblematicNesting = 2001
    | DeepNesting = 2002
    | UnusedProperty = 2003
    | DuplicateProperty = 2004
    | MissingRequiredProperty = 2005
    | TypeMismatch = 2006
    
    // Style issues
    | NamingConvention = 3001
    | InconsistentIndentation = 3002
    | InconsistentLineEndings = 3003
    
    // Performance issues
    | LargeContentBlock = 4001
    | TooManyProperties = 4002
    | TooManyBlocks = 4003
    
    // Security issues
    | InsecureProperty = 5001
    | InsecureValue = 5002
