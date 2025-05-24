namespace TarsEngine.DSL

open System
open Ast

/// <summary>
/// Module for generating warnings in the TARS DSL parser.
/// </summary>
module WarningGenerator =
    /// <summary>
    /// Generate a diagnostic message.
    /// </summary>
    /// <param name="severity">The severity of the diagnostic message.</param>
    /// <param name="code">The warning code of the diagnostic message.</param>
    /// <param name="message">The message of the diagnostic message.</param>
    /// <param name="line">The line number where the diagnostic message occurred.</param>
    /// <param name="column">The column number where the diagnostic message occurred.</param>
    /// <param name="lineContent">The line content where the diagnostic message occurred.</param>
    /// <param name="suggestions">Suggestions for fixing the issue.</param>
    /// <returns>A diagnostic message.</returns>
    let generateDiagnostic severity code message line column lineContent suggestions =
        {
            Severity = severity
            Code = code
            Message = message
            Line = line
            Column = column
            LineContent = lineContent
            Suggestions = suggestions
        }
    
    /// <summary>
    /// Generate a warning for a deprecated block type.
    /// </summary>
    /// <param name="blockType">The deprecated block type.</param>
    /// <param name="alternativeBlockType">The alternative block type to use instead.</param>
    /// <param name="line">The line number where the deprecated block type occurred.</param>
    /// <param name="column">The column number where the deprecated block type occurred.</param>
    /// <param name="lineContent">The line content where the deprecated block type occurred.</param>
    /// <returns>A diagnostic message for the deprecated block type.</returns>
    let generateDeprecatedBlockTypeWarning blockType alternativeBlockType line column lineContent =
        let message = sprintf "Block type '%s' is deprecated and will be removed in a future version." blockType
        let suggestions = 
            match alternativeBlockType with
            | Some alternative -> [sprintf "Use '%s' instead." alternative]
            | None -> ["Consider using a different block type."]
        
        generateDiagnostic DiagnosticSeverity.Warning WarningCode.DeprecatedBlockType message line column lineContent suggestions
    
    /// <summary>
    /// Generate a warning for a deprecated property.
    /// </summary>
    /// <param name="blockType">The block type containing the deprecated property.</param>
    /// <param name="propertyName">The deprecated property name.</param>
    /// <param name="alternativePropertyName">The alternative property name to use instead.</param>
    /// <param name="line">The line number where the deprecated property occurred.</param>
    /// <param name="column">The column number where the deprecated property occurred.</param>
    /// <param name="lineContent">The line content where the deprecated property occurred.</param>
    /// <returns>A diagnostic message for the deprecated property.</returns>
    let generateDeprecatedPropertyWarning blockType propertyName alternativePropertyName line column lineContent =
        let message = sprintf "Property '%s' in block type '%s' is deprecated and will be removed in a future version." propertyName blockType
        let suggestions = 
            match alternativePropertyName with
            | Some alternative -> [sprintf "Use '%s' instead." alternative]
            | None -> ["Consider using a different property."]
        
        generateDiagnostic DiagnosticSeverity.Warning WarningCode.DeprecatedProperty message line column lineContent suggestions
    
    /// <summary>
    /// Generate a warning for a deprecated property value.
    /// </summary>
    /// <param name="blockType">The block type containing the deprecated property value.</param>
    /// <param name="propertyName">The property name containing the deprecated value.</param>
    /// <param name="propertyValue">The deprecated property value.</param>
    /// <param name="alternativePropertyValue">The alternative property value to use instead.</param>
    /// <param name="line">The line number where the deprecated property value occurred.</param>
    /// <param name="column">The column number where the deprecated property value occurred.</param>
    /// <param name="lineContent">The line content where the deprecated property value occurred.</param>
    /// <returns>A diagnostic message for the deprecated property value.</returns>
    let generateDeprecatedPropertyValueWarning blockType propertyName propertyValue alternativePropertyValue line column lineContent =
        let message = sprintf "Value '%A' for property '%s' in block type '%s' is deprecated and will be removed in a future version." propertyValue propertyName blockType
        let suggestions = 
            match alternativePropertyValue with
            | Some alternative -> [sprintf "Use '%A' instead." alternative]
            | None -> ["Consider using a different value."]
        
        generateDiagnostic DiagnosticSeverity.Warning WarningCode.DeprecatedPropertyValue message line column lineContent suggestions
    
    /// <summary>
    /// Generate a warning for problematic nesting.
    /// </summary>
    /// <param name="parentBlockType">The parent block type.</param>
    /// <param name="childBlockType">The child block type.</param>
    /// <param name="reason">The reason why the nesting is problematic.</param>
    /// <param name="line">The line number where the problematic nesting occurred.</param>
    /// <param name="column">The column number where the problematic nesting occurred.</param>
    /// <param name="lineContent">The line content where the problematic nesting occurred.</param>
    /// <returns>A diagnostic message for the problematic nesting.</returns>
    let generateProblematicNestingWarning parentBlockType childBlockType reason line column lineContent =
        let message = sprintf "Nesting block type '%s' inside block type '%s' is problematic: %s" childBlockType parentBlockType reason
        let suggestions = ["Consider restructuring your code to avoid this nesting pattern."]
        
        generateDiagnostic DiagnosticSeverity.Warning WarningCode.ProblematicNesting message line column lineContent suggestions
    
    /// <summary>
    /// Generate a warning for deep nesting.
    /// </summary>
    /// <param name="blockType">The block type that is nested too deeply.</param>
    /// <param name="nestingDepth">The current nesting depth.</param>
    /// <param name="maxRecommendedDepth">The maximum recommended nesting depth.</param>
    /// <param name="line">The line number where the deep nesting occurred.</param>
    /// <param name="column">The column number where the deep nesting occurred.</param>
    /// <param name="lineContent">The line content where the deep nesting occurred.</param>
    /// <returns>A diagnostic message for the deep nesting.</returns>
    let generateDeepNestingWarning blockType nestingDepth maxRecommendedDepth line column lineContent =
        let message = sprintf "Block type '%s' is nested %d levels deep, which exceeds the recommended maximum of %d levels." blockType nestingDepth maxRecommendedDepth
        let suggestions = ["Consider restructuring your code to reduce nesting depth."; "Extract nested blocks into separate blocks."]
        
        generateDiagnostic DiagnosticSeverity.Warning WarningCode.DeepNesting message line column lineContent suggestions
    
    /// <summary>
    /// Generate a warning for an unused property.
    /// </summary>
    /// <param name="blockType">The block type containing the unused property.</param>
    /// <param name="propertyName">The unused property name.</param>
    /// <param name="line">The line number where the unused property occurred.</param>
    /// <param name="column">The column number where the unused property occurred.</param>
    /// <param name="lineContent">The line content where the unused property occurred.</param>
    /// <returns>A diagnostic message for the unused property.</returns>
    let generateUnusedPropertyWarning blockType propertyName line column lineContent =
        let message = sprintf "Property '%s' in block type '%s' is defined but not used." propertyName blockType
        let suggestions = ["Remove the unused property."; "Use the property in your code."]
        
        generateDiagnostic DiagnosticSeverity.Warning WarningCode.UnusedProperty message line column lineContent suggestions
    
    /// <summary>
    /// Generate a warning for a duplicate property.
    /// </summary>
    /// <param name="blockType">The block type containing the duplicate property.</param>
    /// <param name="propertyName">The duplicate property name.</param>
    /// <param name="line">The line number where the duplicate property occurred.</param>
    /// <param name="column">The column number where the duplicate property occurred.</param>
    /// <param name="lineContent">The line content where the duplicate property occurred.</param>
    /// <returns>A diagnostic message for the duplicate property.</returns>
    let generateDuplicatePropertyWarning blockType propertyName line column lineContent =
        let message = sprintf "Property '%s' in block type '%s' is defined multiple times." propertyName blockType
        let suggestions = ["Remove the duplicate property definition."; "Rename one of the properties."]
        
        generateDiagnostic DiagnosticSeverity.Warning WarningCode.DuplicateProperty message line column lineContent suggestions
    
    /// <summary>
    /// Generate a warning for a missing required property.
    /// </summary>
    /// <param name="blockType">The block type missing a required property.</param>
    /// <param name="propertyName">The missing required property name.</param>
    /// <param name="line">The line number where the block with the missing property occurred.</param>
    /// <param name="column">The column number where the block with the missing property occurred.</param>
    /// <param name="lineContent">The line content where the block with the missing property occurred.</param>
    /// <returns>A diagnostic message for the missing required property.</returns>
    let generateMissingRequiredPropertyWarning blockType propertyName line column lineContent =
        let message = sprintf "Required property '%s' is missing in block type '%s'." propertyName blockType
        let suggestions = [sprintf "Add the required property '%s'." propertyName]
        
        generateDiagnostic DiagnosticSeverity.Warning WarningCode.MissingRequiredProperty message line column lineContent suggestions
    
    /// <summary>
    /// Generate a warning for a type mismatch.
    /// </summary>
    /// <param name="blockType">The block type containing the type mismatch.</param>
    /// <param name="propertyName">The property name with the type mismatch.</param>
    /// <param name="expectedType">The expected property value type.</param>
    /// <param name="actualType">The actual property value type.</param>
    /// <param name="line">The line number where the type mismatch occurred.</param>
    /// <param name="column">The column number where the type mismatch occurred.</param>
    /// <param name="lineContent">The line content where the type mismatch occurred.</param>
    /// <returns>A diagnostic message for the type mismatch.</returns>
    let generateTypeMismatchWarning blockType propertyName expectedType actualType line column lineContent =
        let message = sprintf "Property '%s' in block type '%s' has type '%s', but expected type '%s'." propertyName blockType actualType expectedType
        let suggestions = [sprintf "Change the property value to type '%s'." expectedType]
        
        generateDiagnostic DiagnosticSeverity.Warning WarningCode.TypeMismatch message line column lineContent suggestions
    
    /// <summary>
    /// Generate a warning for a naming convention violation.
    /// </summary>
    /// <param name="kind">The kind of name (block, property, etc.).</param>
    /// <param name="name">The name that violates the convention.</param>
    /// <param name="convention">The naming convention that was violated.</param>
    /// <param name="line">The line number where the naming convention violation occurred.</param>
    /// <param name="column">The column number where the naming convention violation occurred.</param>
    /// <param name="lineContent">The line content where the naming convention violation occurred.</param>
    /// <returns>A diagnostic message for the naming convention violation.</returns>
    let generateNamingConventionWarning kind name convention line column lineContent =
        let message = sprintf "%s name '%s' does not follow the naming convention: %s" kind name convention
        let suggestions = [sprintf "Rename to follow the convention: %s" convention]
        
        generateDiagnostic DiagnosticSeverity.Warning WarningCode.NamingConvention message line column lineContent suggestions
