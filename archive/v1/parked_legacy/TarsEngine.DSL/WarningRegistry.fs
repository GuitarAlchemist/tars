namespace TarsEngine.DSL

open System.Collections.Generic

/// <summary>
/// Module for managing warnings in the TARS DSL parser.
/// </summary>
module WarningRegistry =
    /// <summary>
    /// Registry of warning codes with descriptions.
    /// </summary>
    let private warningDescriptions = Dictionary<WarningCode, string>()
    
    /// <summary>
    /// Registry of enabled warnings.
    /// </summary>
    let private enabledWarnings = HashSet<WarningCode>()
    
    /// <summary>
    /// Initialize the warning registry with default values.
    /// </summary>
    let initialize() =
        // Clear existing registries
        warningDescriptions.Clear()
        enabledWarnings.Clear()
        
        // Add warning descriptions
        warningDescriptions.Add(WarningCode.DeprecatedBlockType, "Block type is deprecated and will be removed in a future version.")
        warningDescriptions.Add(WarningCode.DeprecatedProperty, "Property is deprecated and will be removed in a future version.")
        warningDescriptions.Add(WarningCode.DeprecatedPropertyValue, "Property value is deprecated and will be removed in a future version.")
        
        warningDescriptions.Add(WarningCode.ProblematicNesting, "Block nesting pattern is problematic and may cause issues.")
        warningDescriptions.Add(WarningCode.DeepNesting, "Block is nested too deeply, which may make the code difficult to read and maintain.")
        warningDescriptions.Add(WarningCode.UnusedProperty, "Property is defined but not used.")
        warningDescriptions.Add(WarningCode.DuplicateProperty, "Property is defined multiple times.")
        warningDescriptions.Add(WarningCode.MissingRequiredProperty, "Required property is missing.")
        warningDescriptions.Add(WarningCode.TypeMismatch, "Property value type does not match expected type.")
        
        warningDescriptions.Add(WarningCode.NamingConvention, "Name does not follow naming conventions.")
        warningDescriptions.Add(WarningCode.InconsistentIndentation, "Indentation is inconsistent.")
        warningDescriptions.Add(WarningCode.InconsistentLineEndings, "Line endings are inconsistent.")
        
        warningDescriptions.Add(WarningCode.LargeContentBlock, "Content block is very large, which may impact performance.")
        warningDescriptions.Add(WarningCode.TooManyProperties, "Block has too many properties, which may impact performance.")
        warningDescriptions.Add(WarningCode.TooManyBlocks, "Program has too many blocks, which may impact performance.")
        
        warningDescriptions.Add(WarningCode.InsecureProperty, "Property may pose a security risk.")
        warningDescriptions.Add(WarningCode.InsecureValue, "Property value may pose a security risk.")
        
        // Enable all warnings by default
        for code in warningDescriptions.Keys do
            enabledWarnings.Add(code) |> ignore
    
    // Initialize the registry
    do initialize()
    
    /// <summary>
    /// Get the description for a warning code.
    /// </summary>
    /// <param name="code">The warning code.</param>
    /// <returns>The description for the warning code, or a default message if the code is not registered.</returns>
    let getDescription (code: WarningCode) =
        match warningDescriptions.TryGetValue(code) with
        | true, description -> description
        | false, _ -> sprintf "Warning code %d is not registered." (int code)
    
    /// <summary>
    /// Check if a warning is enabled.
    /// </summary>
    /// <param name="code">The warning code.</param>
    /// <returns>True if the warning is enabled, false otherwise.</returns>
    let isEnabled (code: WarningCode) =
        enabledWarnings.Contains(code)
    
    /// <summary>
    /// Enable a warning.
    /// </summary>
    /// <param name="code">The warning code.</param>
    let enableWarning (code: WarningCode) =
        enabledWarnings.Add(code) |> ignore
    
    /// <summary>
    /// Disable a warning.
    /// </summary>
    /// <param name="code">The warning code.</param>
    let disableWarning (code: WarningCode) =
        enabledWarnings.Remove(code) |> ignore
    
    /// <summary>
    /// Enable all warnings.
    /// </summary>
    let enableAllWarnings() =
        for code in warningDescriptions.Keys do
            enabledWarnings.Add(code) |> ignore
    
    /// <summary>
    /// Disable all warnings.
    /// </summary>
    let disableAllWarnings() =
        enabledWarnings.Clear()
    
    /// <summary>
    /// Get all registered warning codes.
    /// </summary>
    /// <returns>A list of all registered warning codes.</returns>
    let getAllWarningCodes() =
        warningDescriptions.Keys |> Seq.toList
    
    /// <summary>
    /// Get all enabled warning codes.
    /// </summary>
    /// <returns>A list of all enabled warning codes.</returns>
    let getEnabledWarningCodes() =
        enabledWarnings |> Seq.toList
    
    /// <summary>
    /// Get all disabled warning codes.
    /// </summary>
    /// <returns>A list of all disabled warning codes.</returns>
    let getDisabledWarningCodes() =
        warningDescriptions.Keys
        |> Seq.filter (fun code -> not (enabledWarnings.Contains(code)))
        |> Seq.toList
