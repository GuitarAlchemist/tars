DESCRIBE {
    name: "Result Handling Pattern"
    purpose: "Replace exception-based error handling with Result types"
    roadmap_priority: "High - addresses failwith usage in codebase"
}

PATTERN result_error_handling {
    input: "Functions that use failwith or raise"
    output: "Result<'T, 'E> based error handling"
    
    transformation: {
        // Before: failwith "error message"
        // After: Error "error message"
        
        // Before: let result = riskyOperation()
        // After: match riskyOperation() with | Ok value -> ... | Error err -> ...
    }
}

FSHARP {
    type TarsError = 
        | ValidationError of string
        | ProcessingError of string
        | SystemError of string
    
    type TarsResult<'T> = Result<'T, TarsError>
    
    let bind f = function 
        | Ok value -> f value 
        | Error err -> Error err
    
    let map f = function 
        | Ok value -> Ok (f value) 
        | Error err -> Error err
    
    let (>>=) result f = bind f result
    let (<!>) f result = map f result
}