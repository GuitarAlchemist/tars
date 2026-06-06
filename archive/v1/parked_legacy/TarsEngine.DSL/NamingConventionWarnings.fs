namespace TarsEngine.DSL

open System
open System.Text.RegularExpressions
open Ast

/// <summary>
/// Module for handling naming convention warnings.
/// </summary>
module NamingConventionWarnings =
    /// <summary>
    /// Check if a block name follows the naming convention.
    /// Block names should be in camelCase.
    /// </summary>
    /// <param name="blockName">The block name to check.</param>
    /// <returns>A tuple of (followsConvention, convention).</returns>
    let checkBlockName (blockName: string) =
        if String.IsNullOrEmpty(blockName) then
            (true, "")
        else
            let camelCasePattern = "^[a-z][a-zA-Z0-9]*$"
            let followsConvention = Regex.IsMatch(blockName, camelCasePattern)
            (followsConvention, "Block names should be in camelCase (e.g., myBlock).")
    
    /// <summary>
    /// Check if a property name follows the naming convention.
    /// Property names should be in snake_case.
    /// </summary>
    /// <param name="propertyName">The property name to check.</param>
    /// <returns>A tuple of (followsConvention, convention).</returns>
    let checkPropertyName (propertyName: string) =
        if String.IsNullOrEmpty(propertyName) then
            (true, "")
        else
            let snakeCasePattern = "^[a-z][a-z0-9_]*$"
            let followsConvention = Regex.IsMatch(propertyName, snakeCasePattern)
            (followsConvention, "Property names should be in snake_case (e.g., my_property).")
    
    /// <summary>
    /// Check a block for naming convention violations and generate warnings if needed.
    /// </summary>
    /// <param name="block">The block to check.</param>
    /// <param name="line">The line number where the block starts.</param>
    /// <param name="column">The column number where the block starts.</param>
    /// <param name="lineContent">The line content where the block starts.</param>
    /// <returns>A list of diagnostic messages for naming convention violations.</returns>
    let checkBlock (block: TarsBlock) line column lineContent =
        let warnings = ResizeArray<Diagnostic>()
        
        // Check block name
        match block.Name with
        | Some blockName ->
            let (followsConvention, convention) = checkBlockName blockName
            
            if not followsConvention && WarningRegistry.isEnabled WarningCode.NamingConvention then
                warnings.Add(WarningGenerator.generateNamingConventionWarning "Block" blockName convention line column lineContent)
        | None -> ()
        
        // Check property names
        for KeyValue(propertyName, _) in block.Properties do
            let (followsConvention, convention) = checkPropertyName propertyName
            
            if not followsConvention && WarningRegistry.isEnabled WarningCode.NamingConvention then
                warnings.Add(WarningGenerator.generateNamingConventionWarning "Property" propertyName convention line column lineContent)
        
        warnings |> Seq.toList
