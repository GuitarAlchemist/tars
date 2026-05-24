namespace TarsEngine.DSL

open System
open Ast

/// <summary>
/// Module for testing the warning system.
/// </summary>
module TestWarnings =
    /// <summary>
    /// Test code with various warning triggers.
    /// </summary>
    let testCode = """
// This is a test file with various warning triggers

// @suppress-warning: DeprecatedBlockType
DESCRIBE {
    name: "This block type is deprecated",
    description: "This should trigger a warning, but it's suppressed"
}

SPAWN_AGENT {
    name: "This block type is deprecated",
    description: "This should trigger a warning"
}

CONFIG {
    // Missing required property api_version
    name: "Test Config"
}

PROMPT {
    // This property is deprecated
    text: "This property is deprecated",
    content_type: "text" // This property value is deprecated
}

CONFIG {
    name: "Test Config",
    api_version: "1.0", // This property value is deprecated
    
    // Problematic nesting
    PROMPT {
        content: "This nesting is problematic"
    }
}

// Deep nesting
AGENT {
    name: "Test Agent",
    agent_type: "standard",
    
    TASK {
        name: "Test Task",
        description: "This is a test task",
        
        ACTION {
            name: "Test Action",
            description: "This is a test action",
            
            VARIABLE {
                name: "Test Variable",
                value: 42,
                
                // This is too deeply nested
                FUNCTION {
                    name: "Test Function",
                    parameters: "param1, param2"
                }
            }
        }
    }
}

// Naming convention violations
VARIABLE BadName {
    bad_name: "This block name violates naming conventions",
    value: 42
}

VARIABLE {
    name: "Test Variable",
    badProperty: "This property name violates naming conventions",
    value: 42
}
"""
    
    /// <summary>
    /// Test the warning system.
    /// </summary>
    let testWarnings() =
        // Parse warning suppression comments
        WarningSuppression.parseWarningSuppressionComments testCode
        
        // Parse the code
        let program = Parser.parse testCode
        
        // Check for warnings
        let warnings = ResizeArray<Diagnostic>()
        
        // Check for deprecated block types
        for block in program.Blocks do
            let blockType = 
                match block.Type with
                | BlockType.Unknown name -> name
                | _ -> block.Type.ToString()
            
            let line = 1 // In a real implementation, this would be the actual line number
            let column = 1 // In a real implementation, this would be the actual column number
            let lineContent = blockType // In a real implementation, this would be the actual line content
            
            // Check for deprecated block types
            let deprecatedBlockTypeWarnings = DeprecatedBlockTypeWarnings.checkBlock block line column lineContent
            warnings.AddRange(deprecatedBlockTypeWarnings)
            
            // Check for deprecated properties
            let deprecatedPropertyWarnings = DeprecatedPropertyWarnings.checkBlock block line column lineContent
            warnings.AddRange(deprecatedPropertyWarnings)
            
            // Check for deprecated property values
            let deprecatedPropertyValueWarnings = DeprecatedPropertyValueWarnings.checkBlock block line column lineContent
            warnings.AddRange(deprecatedPropertyValueWarnings)
            
            // Check for problematic nesting
            let problematicNestingWarnings = ProblematicNestingWarnings.checkBlock block line column lineContent
            warnings.AddRange(problematicNestingWarnings)
            
            // Check for missing required properties
            let missingRequiredPropertyWarnings = MissingRequiredPropertyWarnings.checkBlock block line column lineContent
            warnings.AddRange(missingRequiredPropertyWarnings)
            
            // Check for naming convention violations
            let namingConventionWarnings = NamingConventionWarnings.checkBlock block line column lineContent
            warnings.AddRange(namingConventionWarnings)
        
        // Check for deep nesting
        let deepNestingWarnings = DeepNestingWarnings.checkProgram program 1 1 "program"
        warnings.AddRange(deepNestingWarnings)
        
        // Filter suppressed warnings
        let filteredWarnings = WarningSuppression.filterSuppressedWarnings (warnings |> Seq.toList)
        
        // Print warnings
        printfn "Found %d warnings (%d suppressed):" filteredWarnings.Length (warnings.Count - filteredWarnings.Length)
        
        for warning in filteredWarnings do
            printfn "  %A: %s" warning.Code warning.Message
            printfn "    Severity: %A" warning.Severity
            printfn "    Line: %d, Column: %d" warning.Line warning.Column
            printfn "    Suggestions:"
            
            for suggestion in warning.Suggestions do
                printfn "      - %s" suggestion
            
            printfn ""
        
        // Return the warnings
        filteredWarnings
