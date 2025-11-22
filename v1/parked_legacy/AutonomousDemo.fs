open System

module DemoModule =
    
    // TODO: Implement real functionality
    let processData (input: string) =
        Console.WriteLine("Processing: " + input)
        try
            let result = input.ToUpper()
            result
        with
        | ex -> 
            printfn "Error: %s" ex.Message
            ""
    
    let calculateScore (value: int) =
        if value > 0 then
            value * 2
        else
            0 // Default value
