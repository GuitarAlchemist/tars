namespace TarsEngine.DSL

open System
open Ast

/// Module for testing the template block parser
module TestTemplateBlockParser =
    /// Test the template block parser with a simple TARS program
    let testTemplateBlockParser() =
        let sampleCode = """
// Define a template
TEMPLATE button {
    type: "button",
    style: "primary",
    text: "Click me",
    action: "submit"
}

// Use the template
USE_TEMPLATE button {
    text: "Submit",
    action: "submit_form"
}

// Use the template with different properties
USE_TEMPLATE button {
    text: "Cancel",
    action: "cancel_form",
    style: "secondary"
}

// Define a template with nested blocks
TEMPLATE form {
    title: "My Form",
    
    PROMPT {
        text: "Please fill out the form"
    }
    
    USE_TEMPLATE button {
        text: "Submit",
        action: "submit_form"
    }
}

// Use the template with nested blocks
USE_TEMPLATE form {
    title: "Contact Form"
}
"""
        
        try
            printfn "Parsing with original parser..."
            // Parse with the original parser
            let originalResult = Parser.parse sampleCode
            
            printfn "Parsing with FParsec-based parser..."
            // Parse with the FParsec-based parser
            let fparsecResult = FParsecParser.parse sampleCode
            
            // Print the results
            printfn "Original parser blocks: %d" originalResult.Blocks.Length
            printfn "FParsec parser blocks: %d" fparsecResult.Blocks.Length
            
            // Compare each block
            for i in 0 .. min (originalResult.Blocks.Length - 1) (fparsecResult.Blocks.Length - 1) do
                let originalBlock = originalResult.Blocks.[i]
                let fparsecBlock = fparsecResult.Blocks.[i]
                
                printfn "Block %d:" i
                printfn "  Original: Type=%A, Name=%A, Properties=%d" originalBlock.Type originalBlock.Name originalBlock.Properties.Count
                printfn "  FParsec:  Type=%A, Name=%A, Properties=%d" fparsecBlock.Type fparsecBlock.Name fparsecBlock.Properties.Count
                
                // Compare properties
                for KeyValue(key, value) in originalBlock.Properties do
                    match fparsecBlock.Properties.TryFind key with
                    | Some fparsecValue ->
                        if value <> fparsecValue then
                            printfn "    Property '%s': Original=%A, FParsec=%A" key value fparsecValue
                    | None ->
                        printfn "    Property '%s' missing in FParsec result" key
                
                // Compare nested blocks
                printfn "  Original nested blocks: %d" originalBlock.NestedBlocks.Length
                printfn "  FParsec nested blocks: %d" fparsecBlock.NestedBlocks.Length
            
            // Return the results
            (originalResult, fparsecResult)
        with
        | ex ->
            printfn "Error: %s" ex.Message
            printfn "Stack trace: %s" ex.StackTrace
            
            // Return empty results
            let emptyProgram = { Blocks = [] }
            (emptyProgram, emptyProgram)
            
    /// Test the template block parser with a more complex TARS program
    let testComplexTemplateBlockParser() =
        let complexCode = """
// Define a template with parameters
TEMPLATE button {
    type: "button",
    style: "primary",
    text: "Click me",
    action: "submit",
    disabled: false,
    icon: null
}

// Define a template with nested blocks and parameters
TEMPLATE form {
    title: "My Form",
    description: "A form template",
    
    PROMPT {
        text: "Please fill out the form"
    }
    
    VARIABLE formData {
        value: {}
    }
    
    FUNCTION submit {
        parameters: "data",
        
        RETURN {
            value: @data
        }
    }
    
    USE_TEMPLATE button {
        text: "Submit",
        action: "submit_form"
    }
    
    USE_TEMPLATE button {
        text: "Cancel",
        action: "cancel_form",
        style: "secondary"
    }
}

// Use the form template
USE_TEMPLATE form {
    title: "Contact Form",
    description: "Contact us"
}

// Use the form template with different properties
USE_TEMPLATE form {
    title: "Registration Form",
    description: "Create an account"
}

// Define a template that uses another template
TEMPLATE wizard {
    title: "Wizard",
    steps: ["Step 1", "Step 2", "Step 3"],
    currentStep: 0,
    
    USE_TEMPLATE form {
        title: "Wizard Form",
        description: "Step ${currentStep + 1}"
    }
    
    FUNCTION nextStep {
        VARIABLE newStep {
            value: @currentStep + 1
        }
        
        RETURN {
            value: @newStep
        }
    }
    
    FUNCTION previousStep {
        VARIABLE newStep {
            value: @currentStep - 1
        }
        
        RETURN {
            value: @newStep
        }
    }
    
    USE_TEMPLATE button {
        text: "Next",
        action: "next_step",
        disabled: @currentStep >= @steps.length - 1
    }
    
    USE_TEMPLATE button {
        text: "Previous",
        action: "previous_step",
        disabled: @currentStep <= 0,
        style: "secondary"
    }
}

// Use the wizard template
USE_TEMPLATE wizard {
    title: "Registration Wizard",
    steps: ["Personal Info", "Account Info", "Confirmation"]
}
"""
        
        try
            printfn "Parsing complex program with original parser..."
            // Parse with the original parser
            let originalResult = Parser.parse complexCode
            
            printfn "Parsing complex program with FParsec-based parser..."
            // Parse with the FParsec-based parser
            let fparsecResult = FParsecParser.parse complexCode
            
            // Print the results
            printfn "Original parser blocks: %d" originalResult.Blocks.Length
            printfn "FParsec parser blocks: %d" fparsecResult.Blocks.Length
            
            // Return the results
            (originalResult, fparsecResult)
        with
        | ex ->
            printfn "Error: %s" ex.Message
            printfn "Stack trace: %s" ex.StackTrace
            
            // Return empty results
            let emptyProgram = { Blocks = [] }
            (emptyProgram, emptyProgram)
